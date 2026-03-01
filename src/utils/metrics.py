"""
Research-Grade Metrics for Breast Cancer Histopathology Classification.

Contains ALL metrics required for a paradigm comparison paper:

I.   Core Classification: Accuracy, Precision, Recall, F1, AUC-ROC
II.  Medical-Grade: Specificity, FNR, NPV, PPV, Balanced Accuracy, MCC
III. Training Dynamics: Loss/Accuracy curves, convergence epoch
IV.  Computational: Params, FLOPs, training time, inference time, GPU memory
V.   Paradigm-Specific: Spike stats (SNN), attention params (Transformer),
     quantum params (PQC), gradient norms
VI.  Comparative: Master comparison table
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc, balanced_accuracy_score, matthews_corrcoef,
)
from sklearn.preprocessing import label_binarize


# ═══════════════════════════════════════════════
# I. CORE + II. MEDICAL-GRADE METRICS
# ═══════════════════════════════════════════════

def compute_metrics(y_true, y_probs, num_classes, class_names=None):
    """
    Computes ALL classification metrics (core + medical-grade).

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

    # ── Core Metrics ──
    acc = accuracy_score(y_true, y_pred)

    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)

    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)

    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    # ── AUC-ROC ──
    try:
        if num_classes == 2:
            auc_score = roc_auc_score(y_true, y_probs[:, 1])
        else:
            auc_score = roc_auc_score(
                y_true, y_probs, multi_class='ovr', average='macro'
            )
    except ValueError:
        auc_score = 0.0

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    # ── Medical-Grade Metrics ──
    # Balanced Accuracy (handles imbalanced datasets)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    # Matthews Correlation Coefficient (MCC)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Specificity, FNR, NPV (computed from confusion matrix)
    specificity_per_class = []
    fnr_per_class = []
    npv_per_class = []

    for i in range(num_classes):
        # True Positives, False Positives, False Negatives, True Negatives
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        # Specificity = TN / (TN + FP)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificity_per_class.append(spec)

        # False Negative Rate = FN / (FN + TP)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fnr_per_class.append(fnr)

        # Negative Predictive Value = TN / (TN + FN)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        npv_per_class.append(npv)

    specificity_macro = np.mean(specificity_per_class)
    fnr_macro = np.mean(fnr_per_class)
    npv_macro = np.mean(npv_per_class)

    # For binary: extract the malignant (positive class) metrics specifically
    if num_classes == 2:
        sensitivity = recall_per_class[1]  # Malignant recall
        specificity = specificity_per_class[1]
        fnr = fnr_per_class[1]  # Malignant FNR — critical metric for cancer
        ppv = precision_per_class[1]  # Positive Predictive Value
        npv = npv_per_class[0]  # TN / (TN + FN) for benign-as-negative
    else:
        sensitivity = recall_macro
        specificity = specificity_macro
        fnr = fnr_macro
        ppv = precision_macro
        npv = npv_macro

    return {
        # Core
        'accuracy': acc,
        'precision': precision_macro,
        'precision_weighted': precision_weighted,
        'precision_per_class': precision_per_class.tolist(),
        'recall': recall_macro,
        'recall_weighted': recall_weighted,
        'recall_per_class': recall_per_class.tolist(),
        'f1': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class.tolist(),
        'auc': auc_score,

        # Medical-Grade
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fnr': fnr,                    # False Negative Rate (critical for cancer)
        'ppv': ppv,                    # Positive Predictive Value
        'npv': npv,                    # Negative Predictive Value
        'balanced_accuracy': balanced_acc,
        'mcc': mcc,                    # Matthews Correlation Coefficient

        # Per-class details
        'specificity_per_class': specificity_per_class,
        'fnr_per_class': fnr_per_class,
        'npv_per_class': npv_per_class,

        # Raw data
        'confusion_matrix': cm,
        'classification_report': report,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
    }


def print_medical_metrics(metrics, class_names=None, num_classes=2):
    """Pretty-print all medical-grade metrics."""
    print(f"\n{'─'*50}")
    print("MEDICAL-GRADE METRICS")
    print(f"{'─'*50}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Sensitivity:       {metrics['sensitivity']:.4f}")
    print(f"  Specificity:       {metrics['specificity']:.4f}")
    print(f"  FNR (miss rate):   {metrics['fnr']:.4f}")
    print(f"  PPV (Precision):   {metrics['ppv']:.4f}")
    print(f"  NPV:               {metrics['npv']:.4f}")
    print(f"  F1 (macro):        {metrics['f1']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"  AUC-ROC:           {metrics['auc']:.4f}")
    print(f"  MCC:               {metrics['mcc']:.4f}")

    if class_names:
        print(f"\n  Per-class F1:       {dict(zip(class_names, metrics['f1_per_class']))}")
        print(f"  Per-class Recall:   {dict(zip(class_names, metrics['recall_per_class']))}")
        print(f"  Per-class Spec:     {dict(zip(class_names, metrics['specificity_per_class']))}")

    print(f"\n{metrics['classification_report']}")


# ═══════════════════════════════════════════════
# III. TRAINING DYNAMICS
# ═══════════════════════════════════════════════

def get_convergence_epoch(history, metric='val_auc'):
    """
    Determines convergence epoch (where validation metric stabilizes).
    Uses the epoch of best validation AUC.
    """
    if metric in history and len(history[metric]) > 0:
        values = history[metric]
        return int(np.argmax(values)) + 1  # 1-indexed
    return len(history.get('train_loss', []))


def plot_training_curves(history, output_dir, task_name):
    """Plot training/validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', markersize=3)
    axes[0].plot(epochs, history['val_loss'], 'r-o', label='Val Loss', markersize=3)
    axes[0].set_title(f'{task_name} — Loss', fontsize=13)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy curves
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc', markersize=3)
    axes[1].plot(epochs, history['val_acc'], 'r-o', label='Val Acc', markersize=3)
    axes[1].set_title(f'{task_name} — Accuracy', fontsize=13)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # AUC curve (if available)
    if 'val_auc' in history:
        axes[2].plot(epochs, history['val_auc'], 'g-o', label='Val AUC', markersize=3)
        convergence = get_convergence_epoch(history)
        axes[2].axvline(x=convergence, color='orange', linestyle='--',
                       label=f'Best @ Ep {convergence}', alpha=0.7)
        axes[2].set_title(f'{task_name} — Validation AUC', fontsize=13)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('AUC')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f'{task_name}_training_curves.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_confusion_matrix(cm, class_names, output_dir, task_name):
    """Plot both raw count and normalized confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f'{task_name} — CM (Counts)', fontsize=13)
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    # Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f'{task_name} — CM (Normalized)', fontsize=13)
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    path = os.path.join(output_dir, f'{task_name}_confusion_matrix.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")
    return path


def plot_roc_curve(y_true, y_probs, num_classes, class_names, output_dir, task_name):
    """Plot ROC curve(s)."""
    fig, ax = plt.subplots(figsize=(8, 7))

    if num_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc_val:.4f})')
    else:
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


# ═══════════════════════════════════════════════
# IV. COMPUTATIONAL METRICS
# ═══════════════════════════════════════════════

def count_parameters(model):
    """Returns (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model, input_size=(1, 3, 224, 224)):
    """Computes FLOPs (MACs) using ptflops."""
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
            times.append((end - start) * 1000)

    return np.mean(times), np.std(times)


def get_gpu_memory_peak(device):
    """Returns peak GPU memory in GB."""
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1e9
    return 0.0


# ═══════════════════════════════════════════════
# V. PARADIGM-SPECIFIC METRICS
# ═══════════════════════════════════════════════

def compute_gradient_norms(model):
    """
    Computes L2 norm of gradients for stability monitoring.
    Call AFTER loss.backward() but BEFORE optimizer.step().

    Returns dict with total_grad_norm and per-layer norms.
    """
    total_norm = 0.0
    layer_norms = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            layer_norms[name] = param_norm
            total_norm += param_norm ** 2

    total_norm = total_norm ** 0.5

    return {
        'total_grad_norm': total_norm,
        'layer_grad_norms': layer_norms,
    }


def count_attention_params(model):
    """
    For Transformer models: counts params in attention vs backbone.
    Returns dict with attention_params and backbone_params.
    """
    attn_params = 0
    backbone_params = 0
    other_params = 0

    for name, param in model.named_parameters():
        n = param.numel()
        if 'features' in name:
            backbone_params += n
        elif any(k in name for k in ['attn', 'transformer', 'mha', 'cls_token', 'pos_embed']):
            attn_params += n
        else:
            other_params += n

    return {
        'backbone_params': backbone_params,
        'attention_params': attn_params,
        'head_params': other_params,
    }


def count_quantum_params(model):
    """
    For Quantum models: counts classical vs quantum trainable params.
    """
    classical_params = 0
    quantum_params = 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = param.numel()
        if 'quantum_layer' in name or 'params' in name.lower() and 'quantum' in name.lower():
            quantum_params += n
        else:
            classical_params += n

    return {
        'classical_params': classical_params,
        'quantum_params': quantum_params,
    }


# ═══════════════════════════════════════════════
# VI. CSV LOGGING
# ═══════════════════════════════════════════════

def save_results_csv(results_dict, output_dir, task_name):
    """Saves a flat summary CSV with all key metrics."""
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
    df.index = df.index + 1
    df.to_csv(path)
    print(f"Saved: {path}")
    return path


def build_comparison_row(model_name, metrics, total_params, flops_str,
                         inference_ms, gpu_mem_gb, training_time_s,
                         convergence_epoch, paradigm_extras=None):
    """
    Build a single row for the master comparison table.

    Args:
        model_name: Display name.
        metrics: Output of compute_metrics().
        total_params: Total parameter count.
        flops_str: FLOPs string from compute_flops().
        inference_ms: Average inference time in ms.
        gpu_mem_gb: Peak GPU memory in GB.
        training_time_s: Total training time in seconds.
        convergence_epoch: Best/convergence epoch.
        paradigm_extras: Dict of paradigm-specific metrics (optional).

    Returns:
        Dict suitable for pd.DataFrame row.
    """
    row = {
        'Model': model_name,
        'Accuracy': round(metrics['accuracy'], 4),
        'Balanced_Acc': round(metrics['balanced_accuracy'], 4),
        'Sensitivity': round(metrics['sensitivity'], 4),
        'Specificity': round(metrics['specificity'], 4),
        'F1_macro': round(metrics['f1'], 4),
        'F1_weighted': round(metrics['f1_weighted'], 4),
        'AUC': round(metrics['auc'], 4),
        'MCC': round(metrics['mcc'], 4),
        'FNR': round(metrics['fnr'], 4),
        'PPV': round(metrics['ppv'], 4),
        'NPV': round(metrics['npv'], 4),
        'Precision': round(metrics['precision'], 4),
        'Recall': round(metrics['recall'], 4),
        'Params': total_params,
        'FLOPs': flops_str,
        'Inference_ms': round(inference_ms, 2),
        'GPU_Mem_GB': round(gpu_mem_gb, 2),
        'Train_Time_s': round(training_time_s, 1),
        'Convergence_Ep': convergence_epoch,
    }

    # Merge paradigm-specific extras
    if paradigm_extras:
        row.update(paradigm_extras)

    return row
