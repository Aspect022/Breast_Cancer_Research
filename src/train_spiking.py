"""
Spiking CNN (LIF-SNN) Training Script for BreakHis Dataset.

Trains a native Spiking CNN with Leaky Integrate-and-Fire neurons
for breast cancer histopathology classification.

Usage:
    # Binary classification
    python src/train_spiking.py --task binary --data_dir <PATH_TO_BREAKHIS>

    # 3-class classification
    python src/train_spiking.py --task multi --data_dir <PATH>

    # Quick sanity check
    python src/train_spiking.py --task binary --data_dir <PATH> --subset 200 --epochs 3
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

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import get_dataloaders, set_seed
from src.models.spiking import get_spiking_cnn
from src.utils.metrics import (
    compute_metrics, plot_training_curves, plot_confusion_matrix,
    plot_roc_curve, count_parameters, measure_inference_time,
    save_results_csv, save_epoch_log
)


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

TASK_CLASS_NAMES = {
    'binary': ['Benign', 'Malignant'],
    'multi': ['IDC', 'ILC', 'Fibroadenoma']
}


def parse_args():
    parser = argparse.ArgumentParser(description='Spiking CNN (LIF-SNN) on BreakHis')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multi'],
                        help='Classification task: binary or multi (3-class)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to the root BreakHis dataset directory')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save results, plots, and models')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (smaller due to SNN time-step loop)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=8e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (monitors val AUC)')
    parser.add_argument('--num_steps', type=int, default=10,
                        help='SNN time steps for rate coding (10-20)')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='LIF membrane decay constant')
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='LIF spike threshold')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--subset', type=int, default=None,
                        help='Use only N images for a quick sanity check')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ─────────────────────────────────────────────
# Training & Evaluation Loops
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    """Runs a single training epoch for the Spiking CNN."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping — important for SNN surrogate gradients
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    """Evaluates model on a given loader. Returns loss, accuracy, and probabilities."""
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

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        probs = softmax(outputs.float())
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)

    return epoch_loss, epoch_acc, all_labels, all_probs


# ─────────────────────────────────────────────
# Main Training Pipeline
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Reproducibility ──
    set_seed(args.seed)

    # ── Device ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Paths ──
    task_name = f"SpikingCNN-LIF_{args.task}"
    output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)

    # ── Data ──
    class_names = TASK_CLASS_NAMES[args.task]
    num_classes = len(class_names)
    print(f"\nTask: {args.task} | Classes: {class_names} | Num classes: {num_classes}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        subset_size=args.subset,
        num_workers=args.num_workers
    )

    # ── Model ──
    model = get_spiking_cnn(
        num_classes=num_classes,
        num_steps=args.num_steps,
        beta=args.beta,
        threshold=args.threshold,
        dropout=args.dropout,
    )
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nSpiking CNN Parameters — Total: {total_params:,} | Trainable: {trainable_params:,}")
    print(f"SNN Config — Time steps: {args.num_steps} | Beta: {args.beta} | Threshold: {args.threshold}")

    # ── Training Setup ──
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training Loop ──
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
    print(f"Starting Spiking CNN Training: {args.epochs} epochs, batch_size={args.batch_size}")
    print(f"Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})")
    print(f"Scheduler: CosineAnnealingLR | Early Stopping: patience={args.patience}")
    print(f"Gradient Clipping: {args.grad_clip}")
    print(f"{'='*60}\n")

    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip
        )

        # Validate
        val_loss, val_acc, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, num_classes
        )

        # Compute validation AUC
        try:
            from sklearn.metrics import roc_auc_score
            if num_classes == 2:
                val_auc = float(np.nan_to_num(
                    roc_auc_score(val_labels, val_probs[:, 1])
                ))
            else:
                val_auc = float(np.nan_to_num(
                    roc_auc_score(val_labels, val_probs, multi_class='ovr', average='macro')
                ))
        except ValueError:
            val_auc = 0.0

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        gpu_mem = 0
        if device.type == 'cuda':
            gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.6f} | GPU: {gpu_mem:.2f}GB | "
            f"Time: {epoch_time:.1f}s"
        )

        # Early stopping on Validation AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ↑ New best val AUC: {best_val_auc:.4f} — model saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (patience={args.patience}).")
                break

    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time:.1f}s ({total_training_time/60:.1f} min)")
    print(f"Best epoch: {best_epoch} | Best val AUC: {best_val_auc:.4f}")

    # ── Save epoch log ──
    save_epoch_log(history, output_dir, task_name)

    # ── Plot training curves ──
    plot_training_curves(history, output_dir, task_name)

    # ── Load best model for final evaluation ──
    print(f"\nLoading best model from epoch {best_epoch}...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    # ── Spike Statistics ──
    print("\n" + "="*60)
    print("SPIKE STATISTICS (on first test batch)")
    print("="*60)
    test_batch = next(iter(test_loader))
    test_images = test_batch[0][:8].to(device)  # Use 8 images
    spike_stats = model.get_spike_stats(test_images)
    for key, value in spike_stats.items():
        print(f"  {key}: {value:.4f}")

    # ── Final Test Evaluation ──
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

    # ── Plots ──
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, output_dir, task_name)
    plot_roc_curve(test_labels, test_probs, num_classes, class_names, output_dir, task_name)

    # ── Inference Time ──
    avg_inf_time, std_inf_time = measure_inference_time(model, device)
    print(f"\nInference time per image: {avg_inf_time:.2f} ± {std_inf_time:.2f} ms")

    # ── GPU Memory ──
    gpu_mem_peak = 0
    if device.type == 'cuda':
        gpu_mem_peak = torch.cuda.max_memory_allocated(device) / 1e9

    # ── Save Summary CSV ──
    summary = {
        'Model': 'SpikingCNN-LIF',
        'Task': args.task,
        'Num_Classes': num_classes,
        'SNN_TimeSteps': args.num_steps,
        'SNN_Beta': args.beta,
        'SNN_Threshold': args.threshold,
        'Accuracy': round(test_metrics['accuracy'], 4),
        'Precision': round(test_metrics['precision'], 4),
        'Recall': round(test_metrics['recall'], 4),
        'F1': round(test_metrics['f1'], 4),
        'AUC': round(test_metrics['auc'], 4),
        'Avg_Sparsity': round(spike_stats['avg_sparsity'], 4),
        'Avg_Firing_Rate': round(spike_stats['avg_firing_rate'], 4),
        'Total_Params': total_params,
        'Best_Epoch': best_epoch,
        'Total_Epochs_Run': len(history['train_loss']),
        'Total_Training_Time_s': round(total_training_time, 1),
        'Inference_Time_ms': round(avg_inf_time, 2),
        'GPU_Memory_Peak_GB': round(gpu_mem_peak, 2),
        'Batch_Size': args.batch_size,
        'LR': args.lr,
        'Weight_Decay': args.weight_decay,
        'Seed': args.seed
    }
    save_results_csv(summary, output_dir, task_name)

    # ── Final Summary ──
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
