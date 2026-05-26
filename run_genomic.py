"""Unified genomic experiment runner.

Initial implemented scope:
    - GEO TNBC-style binary pCR/RD classification
    - Synthetic smoke-test mode
    - G-Baseline-MLP
    - G-Baseline-Trees

Planned next models are configured in config_genomics_a100.yaml and can be added
to the registry as their modules are implemented.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.data.genomics import (
    GenomicExpressionDataset,
    load_genomic_table,
    make_stratified_holdout_and_folds,
    make_synthetic_genomic_table,
)
from src.models.genomics import build_tree_stack, get_genomic_mlp
from src.utils.genomics_metrics import (
    compute_binary_genomic_metrics,
    plot_binary_calibration,
)


DISPLAY_NAMES = {
    "g_baseline_mlp": "G-Baseline-MLP",
    "g_baseline_trees": "G-Baseline-Trees",
}


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_gpu(cfg: Dict) -> torch.device:
    gpu_cfg = cfg.get("gpu", {})
    if torch.cuda.is_available():
        if gpu_cfg.get("use_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if gpu_cfg.get("benchmark_mode", True):
            torch.backends.cudnn.benchmark = True
        if gpu_cfg.get("deterministic", False):
            torch.backends.cudnn.deterministic = True
        print(f"CUDA: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("CUDA unavailable; using CPU.")
    return torch.device("cpu")


def infer_synthetic_gene_count(model_cfg: Dict) -> int:
    key = model_cfg.get("input_feature_set", "genes_500")
    if key.endswith("1000"):
        return 1000
    if key.endswith("100"):
        return 100
    return 500


def load_table_for_model(cfg: Dict, model_cfg: Dict, synthetic: bool) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    if synthetic:
        n_genes = infer_synthetic_gene_count(model_cfg)
        return make_synthetic_genomic_table(
            n_samples=180,
            n_genes=n_genes,
            seed=int(cfg["data"].get("seed", 42)),
        )
    return load_genomic_table(cfg, model_cfg)


def maybe_apply_smote(x_train: np.ndarray, y_train: np.ndarray, enabled: bool, seed: int):
    if not enabled:
        return x_train, y_train
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=seed, k_neighbors=min(5, max(1, np.bincount(y_train).min() - 1)))
        return smote.fit_resample(x_train, y_train)
    except Exception as exc:
        print(f"  SMOTE skipped: {exc}")
        return x_train, y_train


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    ds = GenomicExpressionDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=torch.cuda.is_available())


def class_pos_weight(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(int), minlength=2).astype(float)
    if counts[1] == 0:
        return torch.tensor(1.0, dtype=torch.float32, device=device)
    return torch.tensor(counts[0] / counts[1], dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    use_amp: bool,
    grad_clip: float,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    y_true = []
    y_prob = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        y_true.append(y.detach().cpu().numpy())
        y_prob.append(torch.sigmoid(logits.detach()).cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_prob_np = np.concatenate(y_prob)
    metrics = compute_binary_genomic_metrics(y_true_np, y_prob_np)
    return total_loss / len(loader.dataset), metrics["accuracy"]


@torch.no_grad()
def evaluate_mlp(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    model.eval()
    total_loss = 0.0
    y_true = []
    y_prob = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_amp:
            with autocast("cuda"):
                logits = model(x)
                loss = criterion(logits, y)
        else:
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        y_true.append(y.cpu().numpy())
        y_prob.append(torch.sigmoid(logits.float()).cpu().numpy())

    y_true_np = np.concatenate(y_true)
    y_prob_np = np.concatenate(y_prob)
    metrics = compute_binary_genomic_metrics(y_true_np, y_prob_np)
    return total_loss / len(loader.dataset), y_true_np, y_prob_np, metrics


def run_mlp_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Dict,
    model_cfg: Dict,
    output_dir: str,
    fold_idx: int,
    device: torch.device,
) -> Dict:
    seed = int(cfg["data"].get("seed", 42)) + fold_idx
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    x_train, y_train = maybe_apply_smote(
        x_train,
        y_train,
        enabled=bool(cfg["training"].get("smote_training", True)),
        seed=seed,
    )

    batch_size = int(model_cfg.get("batch_size", 64))
    train_loader = make_loader(x_train, y_train, batch_size=batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, batch_size=batch_size, shuffle=False)
    test_loader = make_loader(x_test, y_test, batch_size=batch_size, shuffle=False)

    model = get_genomic_mlp(
        input_dim=x.shape[1],
        hidden=model_cfg.get("hidden", [256, 128, 64]),
        dropout=float(model_cfg.get("dropout", 0.3)),
    ).to(device)

    pos_weight = class_pos_weight(y_train, device) if cfg["training"].get("class_weight_auto", True) else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(
        model.parameters(),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(cfg["training"].get("epochs", 150))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(cfg["training"].get("use_amp", True)) and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    patience = int(cfg["training"].get("patience", 20))
    grad_clip = float(cfg["training"].get("grad_clip", 1.0))

    best_auc = -1.0
    best_epoch = 0
    patience_counter = 0
    best_path = os.path.join(output_dir, f"best_model_fold{fold_idx + 1}.pth")
    history = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, grad_clip
        )
        val_loss, _, _, val_metrics = evaluate_mlp(model, val_loader, criterion, device, use_amp)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_metrics["accuracy"],
                "val_auroc": val_metrics["auroc"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if val_metrics["auroc"] > best_auc:
            best_auc = val_metrics["auroc"]
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    pd.DataFrame(history).to_csv(
        os.path.join(output_dir, f"fold{fold_idx + 1}_epoch_log.csv"),
        index=False,
    )

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    test_loss, test_y, test_prob, test_metrics = evaluate_mlp(model, test_loader, criterion, device, use_amp)
    plot_binary_calibration(test_y, test_prob, output_dir, f"fold{fold_idx + 1}")

    return {
        "Fold": fold_idx + 1,
        "Best_Epoch": best_epoch,
        "Val_AUC": best_auc,
        "Test_Loss": test_loss,
        **{f"Test_{k}": v for k, v in test_metrics.items()},
    }


def run_tree_fold(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: Dict,
    model_cfg: Dict,
    fold_idx: int,
) -> Dict:
    seed = int(cfg["data"].get("seed", 42)) + fold_idx
    local_cfg = dict(model_cfg)
    local_cfg["seed"] = seed

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    x_test, y_test = x[test_idx], y[test_idx]

    x_train, y_train = maybe_apply_smote(
        x_train,
        y_train,
        enabled=bool(cfg["training"].get("smote_training", True)),
        seed=seed,
    )

    model = build_tree_stack(local_cfg)
    model.fit(x_train, y_train)

    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    val_metrics = compute_binary_genomic_metrics(y_val, val_prob)
    test_metrics = compute_binary_genomic_metrics(y_test, test_prob)

    return {
        "Fold": fold_idx + 1,
        "Best_Epoch": 0,
        "Val_AUC": val_metrics["auroc"],
        "Test_Loss": np.nan,
        **{f"Test_{k}": v for k, v in test_metrics.items()},
    }


def clean_fold_row(row: Dict) -> Dict:
    cleaned = {}
    for key, value in row.items():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        if isinstance(value, float):
            cleaned[key] = round(value, 6)
        else:
            cleaned[key] = value
    return cleaned


def aggregate_rows(display_name: str, model_key: str, fold_rows: List[Dict], total_time: float) -> Dict:
    df = pd.DataFrame(fold_rows)
    agg = {
        "Model": display_name,
        "Model_Key": model_key,
        "Folds": len(fold_rows),
        "Total_Time_s": round(total_time, 2),
    }
    metric_cols = [c for c in df.columns if c.startswith("Test_") or c == "Val_AUC"]
    for col in metric_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[f"Mean_{col}"] = round(float(df[col].mean()), 6)
            agg[f"Std_{col}"] = round(float(df[col].std()), 6)
    return agg


def enabled_models(cfg: Dict) -> List[str]:
    implemented = {"g_baseline_mlp", "g_baseline_trees"}
    return [
        name
        for name, model_cfg in cfg.get("models", {}).items()
        if name in implemented and model_cfg.get("enabled", False)
    ]


def run_model(model_key: str, cfg: Dict, device: torch.device, synthetic: bool) -> Dict:
    if model_key not in {"g_baseline_mlp", "g_baseline_trees"}:
        raise NotImplementedError(f"{model_key} is configured but not implemented yet.")

    model_cfg = dict(cfg["models"][model_key])
    display_name = DISPLAY_NAMES[model_key]

    x_df, y_ser, patient_ids = load_table_for_model(cfg, model_cfg, synthetic=synthetic)
    x = x_df.to_numpy(dtype=np.float32)
    y = y_ser.to_numpy(dtype=np.int64)

    test_idx, folds = make_stratified_holdout_and_folds(
        y=y,
        patient_ids=patient_ids,
        n_folds=int(cfg["cv"].get("n_folds", 5)),
        test_holdout=float(cfg["cv"].get("test_holdout", 0.15)),
        seed=int(cfg["data"].get("seed", 42)),
    )

    suffix = "synthetic" if synthetic else cfg["data"].get("dataset", "geo_tnbc")
    output_root = cfg.get("output", {}).get("output_dir", "outputs_genomics")
    output_dir = os.path.join(output_root, f"{display_name}_{suffix}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n=== {display_name} ===")
    print(f"Samples: {len(y)} | Genes: {x.shape[1]} | Positives: {int(y.sum())} | Output: {output_dir}")

    start = time.time()
    fold_rows = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx + 1}/{len(folds)}")
        if model_key == "g_baseline_mlp":
            row = run_mlp_fold(
                x, y, train_idx, val_idx, test_idx, cfg, model_cfg, output_dir, fold_idx, device
            )
        else:
            row = run_tree_fold(x, y, train_idx, val_idx, test_idx, cfg, model_cfg, fold_idx)
        row = clean_fold_row(row)
        fold_rows.append(row)
        print(
            f"    Val AUC: {row['Val_AUC']:.4f} | "
            f"Test AUC: {row['Test_auroc']:.4f} | "
            f"Test Acc: {row['Test_accuracy']:.4f} | "
            f"FNR: {row['Test_fnr']:.4f}"
        )

    total_time = time.time() - start
    fold_df = pd.DataFrame(fold_rows)
    fold_path = os.path.join(output_dir, f"{display_name}_fold_results.csv")
    fold_df.to_csv(fold_path, index=False)
    print(f"  Saved: {fold_path}")

    agg = aggregate_rows(display_name, model_key, fold_rows, total_time)
    summary_path = os.path.join(output_dir, f"{display_name}_summary.csv")
    pd.DataFrame([agg]).to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    return agg


def parse_args():
    parser = argparse.ArgumentParser(description="Run genomic breast cancer experiments.")
    parser.add_argument("--config", default="config_genomics_a100.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--synthetic", action="store_true", help="Run on generated data for smoke testing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.dataset:
        cfg["data"]["dataset"] = args.dataset
    set_seed(int(cfg["data"].get("seed", 42)))
    device = configure_gpu(cfg)

    if args.models:
        model_keys = args.models
    else:
        model_keys = enabled_models(cfg)

    if not model_keys:
        raise SystemExit("No implemented genomic models selected.")

    all_results = []
    for model_key in model_keys:
        try:
            result = run_model(model_key, cfg, device, synthetic=args.synthetic)
            all_results.append(result)
        except NotImplementedError as exc:
            print(f"Skipping {model_key}: {exc}")
        except Exception as exc:
            print(f"ERROR running {model_key}: {exc}")
            raise

    if all_results:
        output_root = cfg.get("output", {}).get("output_dir", "outputs_genomics")
        os.makedirs(output_root, exist_ok=True)
        suffix = "synthetic" if args.synthetic else cfg["data"].get("dataset", "geo_tnbc")
        comparison_path = os.path.join(output_root, f"comparison_{suffix}.csv")
        pd.DataFrame(all_results).to_csv(comparison_path, index=False)
        print(f"\nSaved comparison: {comparison_path}")


if __name__ == "__main__":
    main()

