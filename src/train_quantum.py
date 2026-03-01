"""
Quantum-Classical Hybrid Training Script for BreakHis Dataset.

Trains an EfficientNet-B3 + PQC (PennyLane) hybrid model for
breast cancer histopathology classification.

Usage:
    # Binary classification
    python src/train_quantum.py --task binary --data_dir <PATH_TO_BREAKHIS>

    # 3-class classification
    python src/train_quantum.py --task multi --data_dir <PATH>

    # Quick sanity check
    python src/train_quantum.py --task binary --data_dir <PATH> --subset 200 --epochs 3

    # Force classical simulation (no PennyLane)
    python src/train_quantum.py --task binary --data_dir <PATH> --no-pennylane
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
from src.models.quantum import get_quantum_hybrid
from src.utils.metrics import (
    compute_metrics, plot_training_curves, plot_confusion_matrix,
    plot_roc_curve, count_parameters, measure_inference_time,
    save_results_csv, save_epoch_log
)


TASK_CLASS_NAMES = {
    'binary': ['Benign', 'Malignant'],
    'multi': ['IDC', 'ILC', 'Fibroadenoma']
}


def parse_args():
    parser = argparse.ArgumentParser(description='Quantum-Classical Hybrid on BreakHis')
    parser.add_argument('--task', type=str, default='binary', choices=['binary', 'multi'])
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Smaller batch size (quantum simulation is heavy)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (AdamW, joint optimization)')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--n_qubits', type=int, default=8,
                        help='Number of PQC qubits')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of PQC variational layers')
    parser.add_argument('--no_pennylane', action='store_true',
                        help='Force classical simulation instead of PennyLane')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze EfficientNet-B3 backbone')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping (recommended for quantum)')
    parser.add_argument('--subset', type=int, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()

        # No AMP for quantum — PennyLane needs float32
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

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


def main():
    args = parse_args()
    set_seed(args.seed)

    # Quantum circuits run on CPU; backbone can use GPU
    # For simplicity, run entire model on CPU if using PennyLane
    device = torch.device('cuda' if torch.cuda.is_available() and args.no_pennylane else 'cpu')
    print(f"Using device: {device}")
    if not args.no_pennylane:
        print("Note: PennyLane circuits run on CPU. Using CPU for full model.")

    mode_str = 'Classical-Sim' if args.no_pennylane else 'PennyLane'
    task_name = f"CNN+PQC-{mode_str}_{args.task}"
    output_dir = os.path.join(args.output_dir, task_name)
    os.makedirs(output_dir, exist_ok=True)

    class_names = TASK_CLASS_NAMES[args.task]
    num_classes = len(class_names)
    print(f"\nModel: CNN+PQC ({mode_str}) | Task: {args.task}")
    print(f"Qubits: {args.n_qubits} | PQC Layers: {args.n_layers}")

    train_loader, val_loader, test_loader, _ = get_dataloaders(
        data_dir=args.data_dir,
        task=args.task,
        batch_size=args.batch_size,
        subset_size=args.subset,
        num_workers=args.num_workers
    )

    model = get_quantum_hybrid(
        num_classes=num_classes,
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        use_pennylane=not args.no_pennylane,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)

    total_params, trainable_params = count_parameters(model)
    print(f"\nParameters — Total: {total_params:,} | Trainable: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

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
    print(f"Starting Quantum Hybrid Training: {args.epochs} epochs")
    print(f"{'='*60}\n")

    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, args.grad_clip
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

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train: {train_acc:.4f} | Val: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  ↑ New best val AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    total_training_time = time.time() - training_start_time
    print(f"\nTraining completed in {total_training_time:.1f}s")

    save_epoch_log(history, output_dir, task_name)
    plot_training_curves(history, output_dir, task_name)

    print(f"\nLoading best model...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))

    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)

    test_loss, test_acc, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device, num_classes
    )
    test_metrics = compute_metrics(test_labels, test_probs, num_classes, class_names)

    print(f"\nTest Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test F1-Score:  {test_metrics['f1']:.4f}")
    print(f"Test AUC:       {test_metrics['auc']:.4f}")
    print(f"\nClassification Report:\n{test_metrics['classification_report']}")

    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, output_dir, task_name)
    plot_roc_curve(test_labels, test_probs, num_classes, class_names, output_dir, task_name)

    avg_inf_time, std_inf_time = measure_inference_time(model, device)

    summary = {
        'Model': f'CNN+PQC ({mode_str})',
        'Task': args.task,
        'Num_Classes': num_classes,
        'N_Qubits': args.n_qubits,
        'N_PQC_Layers': args.n_layers,
        'Accuracy': round(test_metrics['accuracy'], 4),
        'Precision': round(test_metrics['precision'], 4),
        'Recall': round(test_metrics['recall'], 4),
        'F1': round(test_metrics['f1'], 4),
        'AUC': round(test_metrics['auc'], 4),
        'Total_Params': total_params,
        'Best_Epoch': best_epoch,
        'Total_Training_Time_s': round(total_training_time, 1),
        'Inference_Time_ms': round(avg_inf_time, 2),
        'Batch_Size': args.batch_size,
        'LR': args.lr,
        'Seed': args.seed
    }
    save_results_csv(summary, output_dir, task_name)

    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k:.<30s} {v}")
    print("="*60)
    print(f"\nOutputs saved to: {os.path.abspath(output_dir)}")
    print("Done!")


if __name__ == '__main__':
    main()
