"""
Transformer Training Script for BreakHis Dataset.

Supports two model variants:
  1. hybrid  — EfficientNet-B3 backbone + Transformer encoder head (CNN+ViT)
  2. vit     — Pure ViT-Tiny (patch 16, embed 384, depth 6)

Usage:
    # Hybrid CNN+ViT, binary
    python src/train_transformer.py --model hybrid --task binary --data_dir <PATH>

    # Pure ViT-Tiny, binary
    python src/train_transformer.py --model vit --task binary --data_dir <PATH>

    # Quick sanity check
    python src/train_transformer.py --model hybrid --task binary --data_dir <PATH> --subset 200 --epochs 3
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import get_dataloaders, set_seed
from src.models.transformer import get_hybrid_vit, get_vit_tiny
from src.utils.metrics import (
    compute_metrics, plot_training_curves, plot_confusion_matrix,
    plot_roc_curve, count_parameters, compute_flops,
    measure_inference_time, save_results_csv, save_epoch_log
)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TASK_CLASS_NAMES = {
    'binary': ['Benign', 'Malignant'],
    'multi': ['IDC', 'ILC', 'Fibroadenoma']
}


def parse_args():
    parser = argparse.ArgumentParser(description='Transformer Models on BreakHis')
    parser.add_argument('--model', type=str, default='hybrid', choices=['hybrid', 'vit'],
                        help='Model variant: hybrid (CNN+ViT) or vit (ViT-Tiny)')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multi'],
                        help='Classification task: binary or multi (3-class)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the root BreakHis dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (monitors val AUC)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='LR warmup epochs (important for transformers)')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze CNN backbone in hybrid mode')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Training & Evaluation (with AMP and LR Warmup)
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_probs = []
    softmax = nn.Softmax(dim=1)

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        probs = softmax(outputs.float())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        running_loss / total,
        correct / total,
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


# ─────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Model name string
    model_display = 'CNN+ViT-Hybrid' if args.model == 'hybrid' else 'ViT-Tiny'
    task_name = f"{model_display}_{args.task}"
    output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)

    class_names = TASK_CLASS_NAMES[args.task]
    num_classes = len(class_names)
    print(f"\nModel: {model_display} | Task: {args.task} | Classes: {class_names}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        subset_size=args.subset,
        num_workers=args.num_workers
    )

    # ── Build Model ──
    if args.model == 'hybrid':
        model = get_hybrid_vit(
            num_classes=num_classes,
            d_model=256, num_heads=4, num_layers=2,
            dropout=args.dropout,
            freeze_backbone=args.freeze_backbone,
        )
    else:
        model = get_vit_tiny(
            num_classes=num_classes,
            dropout=args.dropout,
        )
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\n{model_display} Parameters — Total: {total_params:,} | Trainable: {trainable_params:,}")

    # ── Training Setup ──
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc': [], 'lr': []
    }

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(output_dir, 'best_model.pth')

    print(f"\n{'='*60}")
    print(f"Starting {model_display} Training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"Warmup: {args.warmup_epochs} epochs | Mixed Precision: Enabled")
    print(f"{'='*60}\n")

    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # LR Warmup (linear ramp-up)
        if epoch <= args.warmup_epochs:
            warmup_factor = epoch / args.warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * warmup_factor

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        val_loss, val_acc, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        try:
            from sklearn.metrics import roc_auc_score
            if num_classes == 2:
                val_auc = float(np.nan_to_num(roc_auc_score(val_labels, val_probs[:, 1])))
            else:
                val_auc = float(np.nan_to_num(
                    roc_auc_score(val_labels, val_probs, multi_class='ovr', average='macro')
                ))
        except ValueError:
            val_auc = 0.0

        # Step scheduler only after warmup
        if epoch > args.warmup_epochs:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.6f} | GPU: {gpu_mem:.2f}GB | Time: {epoch_time:.1f}s"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ↑ New best val AUC: {best_val_auc:.4f} — model saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience}).")
                break

    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time:.1f}s")
    print(f"Best epoch: {best_epoch} | Best val AUC: {best_val_auc:.4f}")

    save_epoch_log(history, output_dir, task_name)
    plot_training_curves(history, output_dir, task_name)

    # ── Final Test ──
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)

    test_loss, test_acc, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device, num_classes
    )

    test_metrics = compute_metrics(test_labels, test_probs, num_classes, class_names)

    print(f"\nTest Loss:      {test_loss:.4f}")
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Test AUC:       {test_metrics['auc']:.4f}")
    print(f"\nClassification Report:\n{test_metrics['classification_report']}")

    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, output_dir, task_name)
    plot_roc_curve(test_labels, test_probs, num_classes, class_names, output_dir, task_name)

    avg_inf_time, std_inf_time = measure_inference_time(model, device)
    print(f"\nInference time per image: {avg_inf_time:.2f} ± {std_inf_time:.2f} ms")

    gpu_mem_peak = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0

    summary = {
        'Model': model_display,
        'Task': args.task,
        'Num_Classes': num_classes,
        'Accuracy': round(test_metrics['accuracy'], 4),
        'Precision': round(test_metrics['precision'], 4),
        'Recall': round(test_metrics['recall'], 4),
        'F1': round(test_metrics['f1'], 4),
        'AUC': round(test_metrics['auc'], 4),
        'Total_Params': total_params,
        'Best_Epoch': best_epoch,
        'Total_Epochs_Run': len(history['train_loss']),
        'Total_Training_Time_s': round(total_training_time, 1),
        'Inference_Time_ms': round(avg_inf_time, 2),
        'GPU_Memory_Peak_GB': round(gpu_mem_peak, 2),
        'Batch_Size': args.batch_size,
        'LR': args.lr,
        'Warmup_Epochs': args.warmup_epochs,
        'Seed': args.seed
    }
    save_results_csv(summary, output_dir, task_name)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k:.<30s} {v}")
    print("="*60)
    print(f"\nAll outputs saved to: {os.path.abspath(output_dir)}")
    print("Done!")


if __name__ == '__main__':
    main()
