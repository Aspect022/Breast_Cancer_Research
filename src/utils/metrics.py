import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server/headless environments
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize


# ─────────────────────────────────────────────
# Metric Computation
# ─────────────────────────────────────────────

def compute_metrics(y_true, y_probs, num_classes, class_names=None):
    """
    Computes all required classification metrics.
    
    Args:
        y_true: Ground truth labels (numpy array).
        y_probs: Predicted probabilities (numpy array, shape [N, num_classes]).
        num_classes: Number of classes.
        class_names: Optional list of class name strings.
        
    Returns:
        Dictionary of computed metrics.
    """
    y_pred = np.argmax(y_probs, axis=1)
    
    avg = 'binary' if num_classes == 2 else 'macro'
    pos_label = 1 if num_classes == 2 else None
    
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    
    # AUC calculation
    if num_classes == 2:
        # For binary: use probability of the positive class
        auc_score = roc_auc_score(y_true, y_probs[:, 1])
    else:
        # For multi-class: One-vs-Rest macro AUC
        auc_score = roc_auc_score(
            y_true, y_probs, multi_class='ovr', average='macro'
        )
    
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc_score,
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs
    }


# ─────────────────────────────────────────────
# Plotting Utilities
# ─────────────────────────────────────────────

def plot_training_curves(history, output_dir, task_name):
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss', markersize=3)
    axes[0].set_title(f'{task_name} — Loss per Epoch', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=3)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Val Acc', markersize=3)
    axes[1].set_title(f'{task_name} — Accuracy per Epoch', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f'{task_name}_training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_confusion_matrix(cm, class_names, output_dir, task_name):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title(f'{task_name} — Confusion Matrix', fontsize=13)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    path = os.path.join(output_dir, f'{task_name}_confusion_matrix.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_roc_curve(y_true, y_probs, num_classes, class_names, output_dir, task_name):
    """Plot and save ROC curve(s)."""
    fig, ax = plt.subplots(figsize=(8, 7))
    
    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.4f})')
    else:
        # One-vs-Rest for each class
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            label = class_names[i] if class_names else f'Class {i}'
            ax.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc_val:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'{task_name} — ROC Curve', fontsize=13)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{task_name}_roc_curve.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


# ─────────────────────────────────────────────
# Model Profiling Utilities
# ─────────────────────────────────────────────

def count_parameters(model):
    """Returns total and trainable parameter count."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model, input_size=(1, 3, 224, 224)):
    """
    Computes FLOPs (MACs) using ptflops.
    Returns (macs_str, params_str) or a fallback message.
    """
    try:
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, (3, 224, 224), as_strings=True,
            print_per_layer_stat=False, verbose=False
        )
        return macs, params
    except ImportError:
        return "ptflops not installed", "N/A"
    except Exception as e:
        return f"Error: {e}", "N/A"


def measure_inference_time(model, device, input_size=(1, 3, 224, 224), num_runs=100):
    """Measures average inference time per image in milliseconds."""
    model.eval()
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    return avg_time, std_time


# ─────────────────────────────────────────────
# CSV Logging
# ─────────────────────────────────────────────

def save_results_csv(results_dict, output_dir, task_name):
    """
    Saves a flat summary CSV with all key metrics.
    Columns: Model, Task, Accuracy, F1, AUC, Precision, Recall, 
             Params, FLOPs, BestEpoch, TotalTime_s, InferenceTime_ms
    """
    path = os.path.join(output_dir, f'{task_name}_results.csv')
    df = pd.DataFrame([results_dict])
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def save_epoch_log(history, output_dir, task_name):
    """Saves per-epoch training log as CSV."""
    path = os.path.join(output_dir, f'{task_name}_epoch_log.csv')
    df = pd.DataFrame(history)
    df.index.name = 'epoch'
    df.index = df.index + 1  # 1-indexed epochs
    df.to_csv(path)
    print(f"Saved: {path}")
    return path
