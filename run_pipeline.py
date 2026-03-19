"""
Unified Pipeline — Runs all models with 5-Fold Cross-Validation.

Reads configuration from config.yaml and trains each enabled model
across K folds with patient-grouped splits. Produces per-fold results,
aggregated stats, and a final comparison table with research-grade metrics.

Usage:
    python run_pipeline.py                        # Uses config.yaml defaults
    python run_pipeline.py --config my_config.yaml
    python run_pipeline.py --models efficientnet spiking   # Run specific models only
"""

import os
import sys
import time
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.data.dataset import get_kfold_splits, set_seed, get_multidataset_dataloaders
from src.models.efficientnet import get_efficientnet_b3
from src.models.spiking import get_spiking_cnn
from src.models.transformer import (
    get_hybrid_vit, get_vit_tiny,
    get_swin_tiny, get_swin_small, get_swin_v2_small,
    get_convnext_tiny, get_convnext_small,
    get_deit_tiny, get_deit_small, get_deit_base,
)
from src.models.quantum import (
    get_quantum_hybrid,
    get_qenn,
    get_vectorized_quantum_circuit,
)
from src.models.fusion import (
    get_dual_branch_fusion,
    get_quantum_enhanced_fusion,
)
from src.utils.metrics import (
    compute_metrics, print_medical_metrics, get_convergence_epoch,
    plot_training_curves, plot_confusion_matrix, plot_roc_curve,
    count_parameters, compute_flops, measure_inference_time, get_gpu_memory_peak,
    count_attention_params, count_quantum_params,
    save_results_csv, save_epoch_log, build_comparison_row,
)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

def load_config(config_path="config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


TASK_CLASS_NAMES = {
    'binary': ['Benign', 'Malignant'],
    'multi': ['IDC', 'ILC', 'Fibroadenoma']
}


# ─────────────────────────────────────────────
# Model Builders
# ─────────────────────────────────────────────

def build_model(model_name, num_classes, model_cfg):
    """Build model. Returns (model, display_name, paradigm_type)."""
    
    # ── Baseline CNN ─────────────────────────────────────────
    if model_name == 'efficientnet':
        return get_efficientnet_b3(num_classes), 'EfficientNet-B3', 'cnn'

    # ── Spiking Neural Network (Legacy) ──────────────────────
    elif model_name == 'spiking':
        model = get_spiking_cnn(
            num_classes=num_classes,
            num_steps=model_cfg.get('num_steps', 10),
            beta=model_cfg.get('beta', 0.9),
            threshold=model_cfg.get('threshold', 1.0),
            dropout=model_cfg.get('dropout', 0.5),
        )
        return model, 'SpikingCNN-LIF', 'snn'

    # ── Transformer Models ───────────────────────────────────
    elif model_name == 'hybrid_vit':
        model = get_hybrid_vit(
            num_classes=num_classes,
            d_model=model_cfg.get('d_model', 256),
            num_heads=model_cfg.get('num_heads', 4),
            num_layers=model_cfg.get('num_layers', 2),
            dropout=model_cfg.get('dropout', 0.1),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'CNN+ViT-Hybrid', 'transformer'

    elif model_name == 'vit_tiny':
        model = get_vit_tiny(
            num_classes=num_classes,
            dropout=model_cfg.get('dropout', 0.1),
        )
        return model, 'ViT-Tiny', 'transformer'

    # Swin Transformer variants
    elif model_name == 'swin_tiny':
        model = get_swin_tiny(
            num_classes=num_classes,
            img_size=model_cfg.get('img_size', 224),
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.3),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'Swin-Tiny', 'transformer'

    elif model_name == 'swin_small':
        model = get_swin_small(
            num_classes=num_classes,
            img_size=model_cfg.get('img_size', 224),
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.3),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'Swin-Small', 'transformer'

    elif model_name == 'swin_v2_small':
        model = get_swin_v2_small(
            num_classes=num_classes,
            img_size=model_cfg.get('img_size', 256),
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.3),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'Swin-V2-Small', 'transformer'

    # ConvNeXt variants
    elif model_name == 'convnext_tiny':
        model = get_convnext_tiny(
            num_classes=num_classes,
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.3),
            drop_path_rate=model_cfg.get('drop_path_rate', 0.1),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'ConvNeXt-Tiny', 'transformer'

    elif model_name == 'convnext_small':
        model = get_convnext_small(
            num_classes=num_classes,
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.3),
            drop_path_rate=model_cfg.get('drop_path_rate', 0.1),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'ConvNeXt-Small', 'transformer'

    # DeiT variants
    elif model_name == 'deit_tiny':
        model = get_deit_tiny(
            num_classes=num_classes,
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.1),
            use_distillation=model_cfg.get('use_distillation', True),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'DeiT-Tiny', 'transformer'

    elif model_name == 'deit_small':
        model = get_deit_small(
            num_classes=num_classes,
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.1),
            use_distillation=model_cfg.get('use_distillation', True),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'DeiT-Small', 'transformer'

    elif model_name == 'deit_base':
        model = get_deit_base(
            num_classes=num_classes,
            pretrained=model_cfg.get('pretrained', True),
            dropout=model_cfg.get('dropout', 0.1),
            use_distillation=model_cfg.get('use_distillation', True),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, 'DeiT-Base', 'transformer'

    # ── Quantum Models ───────────────────────────────────────
    elif model_name == 'quantum':
        model = get_quantum_hybrid(
            num_classes=num_classes,
            n_qubits=model_cfg.get('n_qubits', 8),
            n_layers=model_cfg.get('n_layers', 2),
            use_pennylane=model_cfg.get('use_pennylane', True),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
            dropout=model_cfg.get('dropout', 0.3),
        )
        return model, 'CNN+PQC', 'quantum'

    # QENN variants (all rotation configurations)
    elif model_name.startswith('qenn_'):
        rotation_config = model_name.replace('qenn_', '')
        model = get_qenn(
            num_classes=num_classes,
            n_qubits=model_cfg.get('n_qubits', 8),
            n_layers=model_cfg.get('n_layers', 2),
            rotation_config=model_cfg.get('rotation_config', rotation_config),
            entanglement=model_cfg.get('entanglement', 'cyclic'),
            dropout=model_cfg.get('dropout', 0.3),
            freeze_backbone=model_cfg.get('freeze_backbone', False),
        )
        return model, f'QENN-{rotation_config.upper()}', 'quantum'

    # ── Fusion Models ────────────────────────────────────────
    elif model_name == 'dual_branch_fusion':
        model = get_dual_branch_fusion(
            num_classes=num_classes,
            swin_variant=model_cfg.get('swin_variant', 'tiny'),
            convnext_variant=model_cfg.get('convnext_variant', 'tiny'),
            dropout=model_cfg.get('dropout', 0.3),
            entropy_weight=model_cfg.get('entropy_weight', 0.01),
            freeze_backbones=model_cfg.get('freeze_backbones', False),
        )
        return model, 'DualBranch-Fusion', 'fusion'

    elif model_name == 'quantum_enhanced_fusion':
        model = get_quantum_enhanced_fusion(
            num_classes=num_classes,
            swin_variant=model_cfg.get('swin_variant', 'tiny'),
            convnext_variant=model_cfg.get('convnext_variant', 'tiny'),
            n_qubits=model_cfg.get('n_qubits', 8),
            n_layers=model_cfg.get('n_layers', 2),
            rotation_config=model_cfg.get('rotation_config', 'ry_only'),
            entanglement=model_cfg.get('entanglement', 'cyclic'),
            dropout=model_cfg.get('dropout', 0.3),
            entropy_weight=model_cfg.get('entropy_weight', 0.01),
            freeze_backbones=model_cfg.get('freeze_backbones', False),
        )
        return model, 'Quantum-Enhanced-Fusion', 'fusion_quantum'

    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────
# Training & Evaluation
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_amp=True, scaler=None, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
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
def evaluate(model, loader, criterion, device, use_amp=True):
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

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
        else:
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


def compute_val_auc(val_labels, val_probs, num_classes):
    try:
        from sklearn.metrics import roc_auc_score
        if num_classes == 2:
            return float(np.nan_to_num(roc_auc_score(val_labels, val_probs[:, 1])))
        else:
            return float(np.nan_to_num(
                roc_auc_score(val_labels, val_probs, multi_class='ovr', average='macro')
            ))
    except ValueError:
        return 0.0


# ─────────────────────────────────────────────
# Single-Fold Training
# ─────────────────────────────────────────────

def train_fold(model, model_cfg, train_cfg, train_loader, val_loader,
               device, output_dir, fold_idx, display_name):
    lr = model_cfg.get('lr', 1e-4)
    wd = model_cfg.get('weight_decay', 1e-4)
    epochs = train_cfg.get('epochs', 30)
    patience = train_cfg.get('patience', 7)
    grad_clip = train_cfg.get('grad_clip', 1.0)
    use_amp = model_cfg.get('use_amp', True)
    warmup_epochs = model_cfg.get('warmup_epochs', 0)

    num_classes = list(model.parameters())[-1].shape[0]

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler() if use_amp else None

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'val_auc': [], 'lr': []
    }

    best_val_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_path = os.path.join(output_dir, f'best_model_fold{fold_idx + 1}.pth')

    fold_start = time.time()

    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = lr * (epoch / warmup_epochs)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_amp=use_amp, scaler=scaler, grad_clip=grad_clip
        )

        val_loss, val_acc, val_labels, val_probs = evaluate(
            model, val_loader, criterion, device, use_amp=use_amp
        )

        val_auc = compute_val_auc(val_labels, val_probs, num_classes)

        if epoch > warmup_epochs:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)

        print(
            f"    Ep {epoch:02d}/{epochs} | "
            f"T-Acc: {train_acc:.4f} | V-Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"    Early stop at epoch {epoch}")
                break

    fold_time = time.time() - fold_start
    convergence = get_convergence_epoch(history)

    fold_name = f"{display_name}_fold{fold_idx + 1}"
    save_epoch_log(history, output_dir, fold_name)
    plot_training_curves(history, output_dir, fold_name)

    return {
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'fold_time_s': fold_time,
        'best_path': best_path,
        'convergence_epoch': convergence,
        'history': history,
    }


# ─────────────────────────────────────────────
# Test Evaluation
# ─────────────────────────────────────────────

def evaluate_on_test(model, best_path, test_loader, device, num_classes,
                     class_names, display_name, output_dir, fold_idx, use_amp):
    model.load_state_dict(torch.load(best_path, weights_only=True, map_location=device))

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_labels, test_probs = evaluate(
        model, test_loader, criterion, device, use_amp=use_amp
    )

    test_metrics = compute_metrics(test_labels, test_probs, num_classes, class_names)

    fold_name = f"{display_name}_fold{fold_idx + 1}"
    plot_confusion_matrix(test_metrics['confusion_matrix'], class_names, output_dir, fold_name)
    plot_roc_curve(test_labels, test_probs, num_classes, class_names, output_dir, fold_name)

    # Print medical metrics for visibility
    print_medical_metrics(test_metrics, class_names, num_classes)

    return test_metrics


# ─────────────────────────────────────────────
# Paradigm-Specific Metrics
# ─────────────────────────────────────────────

def get_paradigm_extras(model, paradigm_type, test_loader, device, model_cfg):
    """Collect paradigm-specific metrics."""
    extras = {}

    if paradigm_type == 'snn':
        # Spike statistics
        try:
            test_batch = next(iter(test_loader))
            test_images = test_batch[0][:4].to(device)
            spike_stats = model.get_spike_stats(test_images)
            extras['Avg_Firing_Rate'] = round(spike_stats['avg_firing_rate'], 4)
            extras['Avg_Sparsity'] = round(spike_stats['avg_sparsity'], 4)
            extras['SNN_TimeSteps'] = model_cfg.get('num_steps', 10)
            extras['SNN_Beta'] = model_cfg.get('beta', 0.9)
            print(f"    ⚡ Firing Rate: {spike_stats['avg_firing_rate']:.4f} | "
                  f"Sparsity: {spike_stats['avg_sparsity']:.4f}")
        except Exception as e:
            print(f"    ⚠ Could not compute spike stats: {e}")

    elif paradigm_type == 'transformer':
        attn_info = count_attention_params(model)
        extras['Backbone_Params'] = attn_info['backbone_params']
        extras['Attention_Params'] = attn_info['attention_params']
        extras['Head_Params'] = attn_info['head_params']
        print(f"    🔷 Backbone: {attn_info['backbone_params']:,} | "
              f"Attention: {attn_info['attention_params']:,} | "
              f"Head: {attn_info['head_params']:,}")

    elif paradigm_type == 'quantum':
        q_info = count_quantum_params(model)
        extras['Classical_Params'] = q_info['classical_params']
        extras['Quantum_Params'] = q_info['quantum_params']
        extras['N_Qubits'] = model_cfg.get('n_qubits', 8)
        extras['PQC_Layers'] = model_cfg.get('n_layers', 2)
        # Circuit depth = n_layers × (3 rotations + 1 CNOT ring) per layer
        n_q = model_cfg.get('n_qubits', 8)
        n_l = model_cfg.get('n_layers', 2)
        circuit_depth = n_l * (3 * n_q + n_q)  # rotations + CNOTs
        extras['Circuit_Depth'] = circuit_depth
        print(f"    🟢 Classical: {q_info['classical_params']:,} | "
              f"Quantum: {q_info['quantum_params']} | "
              f"Circuit Depth: {circuit_depth}")

    return extras


# ─────────────────────────────────────────────
# Main Model Runner
# ─────────────────────────────────────────────

def run_model_pipeline(model_name, cfg, all_results):
    data_cfg = cfg['data']
    cv_cfg = cfg['cv']
    train_cfg = cfg['training']
    model_cfg = cfg['models'][model_name]
    out_cfg = cfg['output']

    task = data_cfg['task']
    class_names = TASK_CLASS_NAMES[task]
    num_classes = len(class_names)
    n_folds = cv_cfg['n_folds']
    seed = data_cfg.get('seed', 42)

    force_cpu = (model_name == 'quantum' and model_cfg.get('use_pennylane', True))
    device = torch.device('cpu' if force_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))

    model_proto, display_name, paradigm_type = build_model(model_name, num_classes, model_cfg)
    total_params, trainable_params = count_parameters(model_proto)
    del model_proto

    model_output_dir = os.path.join(out_cfg['output_dir'], f"{display_name}_{task}")
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  MODEL: {display_name} | Paradigm: {paradigm_type.upper()}")
    print(f"  Task: {task} | Device: {device}")
    print(f"  Params: {total_params:,} total, {trainable_params:,} trainable")
    print(f"  {n_folds}-Fold CV | Max Epochs: {train_cfg['epochs']}")
    print(f"{'═'*70}")

    fold_rows = []
    model_start_time = time.time()

    for fold_idx, train_loader, val_loader, test_loader, nc in get_kfold_splits(
        data_dir=data_cfg['data_dir'],
        task=task,
        n_folds=n_folds,
        batch_size=model_cfg.get('batch_size', 32),
        subset_size=data_cfg.get('subset', None),
        num_workers=data_cfg.get('num_workers', 4),
        seed=seed,
    ):
        print(f"\n  ── Fold {fold_idx + 1}/{n_folds} ──")

        model, _, _ = build_model(model_name, num_classes, model_cfg)
        model = model.to(device)

        # Train
        fold_result = train_fold(
            model, model_cfg, train_cfg, train_loader, val_loader,
            device, model_output_dir, fold_idx, display_name
        )

        # Test
        test_metrics = evaluate_on_test(
            model, fold_result['best_path'], test_loader, device,
            num_classes, class_names, display_name, model_output_dir, fold_idx,
            use_amp=model_cfg.get('use_amp', True),
        )

        # Paradigm extras
        paradigm_extras = get_paradigm_extras(model, paradigm_type, test_loader, device, model_cfg)

        # Inference time
        avg_inf, std_inf = measure_inference_time(model, device)
        gpu_mem = get_gpu_memory_peak(device)

        fold_entry = {
            'Model': display_name,
            'Paradigm': paradigm_type,
            'Fold': fold_idx + 1,
            'Best_Epoch': fold_result['best_epoch'],
            'Convergence_Ep': fold_result['convergence_epoch'],
            'Val_AUC': round(fold_result['best_val_auc'], 4),
            'Test_Acc': round(test_metrics['accuracy'], 4),
            'Test_BalAcc': round(test_metrics['balanced_accuracy'], 4),
            'Test_Sensitivity': round(test_metrics['sensitivity'], 4),
            'Test_Specificity': round(test_metrics['specificity'], 4),
            'Test_F1': round(test_metrics['f1'], 4),
            'Test_AUC': round(test_metrics['auc'], 4),
            'Test_MCC': round(test_metrics['mcc'], 4),
            'Test_FNR': round(test_metrics['fnr'], 4),
            'Test_PPV': round(test_metrics['ppv'], 4),
            'Test_NPV': round(test_metrics['npv'], 4),
            'Inference_ms': round(avg_inf, 2),
            'GPU_Mem_GB': round(gpu_mem, 2),
            'Fold_Time_s': round(fold_result['fold_time_s'], 1),
        }
        fold_entry.update(paradigm_extras)
        fold_rows.append(fold_entry)

        print(f"    → AUC: {test_metrics['auc']:.4f} | F1: {test_metrics['f1']:.4f} | "
              f"MCC: {test_metrics['mcc']:.4f} | FNR: {test_metrics['fnr']:.4f}")

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    total_model_time = time.time() - model_start_time

    # ── Per-fold CSV ──
    fold_df = pd.DataFrame(fold_rows)
    fold_csv = os.path.join(model_output_dir, f"{display_name}_fold_results.csv")
    fold_df.to_csv(fold_csv, index=False)
    print(f"\n  Saved: {fold_csv}")

    # ── Aggregate (mean ± std) ──
    metric_cols = [c for c in fold_df.columns if c.startswith('Test_')]
    agg = {'Model': display_name, 'Paradigm': paradigm_type, 'Task': task,
           'Folds': n_folds, 'Total_Params': total_params}

    for col in metric_cols:
        clean = col.replace('Test_', '')
        agg[f'Mean_{clean}'] = round(fold_df[col].mean(), 4)
        agg[f'Std_{clean}'] = round(fold_df[col].std(), 4)

    agg['Mean_Inference_ms'] = round(fold_df['Inference_ms'].mean(), 2)
    agg['Total_Time_s'] = round(total_model_time, 1)
    agg['LR'] = model_cfg.get('lr', 1e-4)
    agg['Batch_Size'] = model_cfg.get('batch_size', 32)

    # Add paradigm extras to aggregate (take from last fold)
    for key in fold_rows[-1]:
        if key not in agg and key not in ['Fold', 'Best_Epoch', 'Convergence_Ep',
                                           'Val_AUC', 'Fold_Time_s', 'Model',
                                           'Paradigm', 'Inference_ms', 'GPU_Mem_GB'] \
                and not key.startswith('Test_'):
            agg[key] = fold_rows[-1][key]

    all_results.append(agg)

    print(f"\n  ╔═══ {display_name} SUMMARY ({n_folds}-Fold) ═══╗")
    print(f"  ║ Accuracy:    {agg['Mean_Acc']:.4f} ± {agg['Std_Acc']:.4f}")
    print(f"  ║ F1 (macro):  {agg['Mean_F1']:.4f} ± {agg['Std_F1']:.4f}")
    print(f"  ║ AUC:         {agg['Mean_AUC']:.4f} ± {agg['Std_AUC']:.4f}")
    print(f"  ║ MCC:         {agg['Mean_MCC']:.4f} ± {agg['Std_MCC']:.4f}")
    print(f"  ║ Sensitivity: {agg['Mean_Sensitivity']:.4f} ± {agg['Std_Sensitivity']:.4f}")
    print(f"  ║ Specificity: {agg['Mean_Specificity']:.4f} ± {agg['Std_Specificity']:.4f}")
    print(f"  ║ FNR:         {agg['Mean_FNR']:.4f} ± {agg['Std_FNR']:.4f}")
    print(f"  ║ NPV:         {agg['Mean_NPV']:.4f} ± {agg['Std_NPV']:.4f}")
    print(f"  ║ Time:        {agg['Total_Time_s']:.0f}s")
    print(f"  ╚{'═'*40}╝")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Unified Training Pipeline')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Specific models to run (overrides config)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset to use: breakhis, wbcd, seer (overrides config)')
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Override dataset if specified
    if args.dataset:
        cfg['data']['dataset'] = args.dataset
        if args.dataset == 'wbcd':
            cfg['data']['data_dir'] = 'data/WBCD'
        elif args.dataset == 'seer':
            cfg['data']['data_dir'] = 'data/SEER'
        elif args.dataset == 'breakhis':
            cfg['data']['data_dir'] = 'data/BreaKHis_v1'

    # Updated model order with all new architectures
    model_order = [
        # Baseline
        'efficientnet',
        'spiking',  # Legacy
        'hybrid_vit',
        'vit_tiny',
        
        # Transformer zoo (Phase 1-2)
        'swin_tiny',
        'swin_small',
        'swin_v2_small',
        'convnext_tiny',
        'convnext_small',
        'deit_tiny',
        'deit_small',
        'deit_base',
        
        # Fusion models (Phase 2)
        'dual_branch_fusion',
        
        # Quantum models (Phase 3)
        'quantum',  # PennyLane legacy
        'qenn_ry',
        'qenn_ry_rz',
        'qenn_u3',
        'qenn_rx_ry_rz',
        'quantum_enhanced_fusion',
    ]

    if args.models:
        models_to_run = [m for m in args.models if m in model_order]
    else:
        models_to_run = [m for m in model_order if cfg['models'].get(m, {}).get('enabled', False)]

    if not models_to_run:
        print("No models enabled! Check config.yaml or use --models flag.")
        return

    # A100 GPU Optimizations
    if torch.cuda.is_available():
        gpu_cfg = cfg.get('gpu', {})
        if gpu_cfg.get('use_tf32', True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if gpu_cfg.get('benchmark_mode', True):
            torch.backends.cudnn.benchmark = True
        if gpu_cfg.get('deterministic', False):
            torch.backends.cudnn.deterministic = True
        
        print(f"\n🚀 A100 GPU Optimizations Enabled:")
        print(f"   TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
    
    set_seed(cfg['data'].get('seed', 42))

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║   BREAST CANCER HISTOPATHOLOGY — PARADIGM COMPARISON       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Task:   {cfg['data']['task']:.<50s}║")
    print(f"║  Folds:  {cfg['cv']['n_folds']:.<50d}║")
    print(f"║  Epochs: {cfg['training']['epochs']:.<50d}║")
    print(f"║  Models: {', '.join(models_to_run):.<50s}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    pipeline_start = time.time()
    all_results = []

    for model_name in models_to_run:
        try:
            run_model_pipeline(model_name, cfg, all_results)
        except Exception as e:
            print(f"\n  ✗ ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    pipeline_time = time.time() - pipeline_start

    # ── Final Comparison Table ──
    if all_results:
        output_dir = cfg['output']['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        comparison_df = pd.DataFrame(all_results)
        comparison_path = os.path.join(output_dir, f"comparison_{cfg['data']['task']}.csv")
        comparison_df.to_csv(comparison_path, index=False)

        print(f"\n\n{'═'*70}")
        print("  MASTER COMPARISON TABLE")
        print(f"{'═'*70}")

        # Display key columns only for readability
        key_cols = ['Model', 'Mean_Acc', 'Mean_F1', 'Mean_AUC', 'Mean_MCC',
                    'Mean_Sensitivity', 'Mean_Specificity', 'Mean_FNR',
                    'Total_Params', 'Mean_Inference_ms', 'Total_Time_s']
        display_cols = [c for c in key_cols if c in comparison_df.columns]
        print(comparison_df[display_cols].to_string(index=False))

        print(f"\nFull results saved: {comparison_path}")
        print(f"\nTotal pipeline time: {pipeline_time:.0f}s ({pipeline_time/60:.1f} min)")
    else:
        print("\nNo results to compare.")

    print("\n✓ Pipeline complete!")


if __name__ == '__main__':
    main()
