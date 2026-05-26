"""GEO TNBC expression table loading for genomic experiments."""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class GenomicExpressionDataset(Dataset):
    """Simple patient-level expression dataset."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _coerce_index(df: pd.DataFrame) -> pd.DataFrame:
    """Use a sample/patient id column as index when present."""
    for col in ["sample_id", "patient_id", "SampleID", "ID", "id"]:
        if col in df.columns:
            return df.set_index(col)

    first = df.columns[0]
    if first.lower().startswith("unnamed") or not pd.api.types.is_numeric_dtype(df[first]):
        return df.set_index(first)

    return df


def _read_expression_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _coerce_index(df)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    df = df.fillna(df.median(numeric_only=True))
    return df.astype("float32")


def _map_label_value(value) -> int:
    if pd.isna(value):
        raise ValueError("Missing label value")
    if isinstance(value, str):
        normalized = value.strip().lower()
        positive = {"pcr", "pathologic complete response", "complete_response", "1", "yes", "true"}
        negative = {"rd", "residual disease", "residual_disease", "0", "no", "false"}
        if normalized in positive:
            return 1
        if normalized in negative:
            return 0
    return int(value)


def _read_labels_csv(path: str) -> pd.Series:
    df = pd.read_csv(path)
    df = _coerce_index(df)
    label_col = None
    for candidate in ["label", "pcr", "response", "target", "y"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        if len(df.columns) != 1:
            raise ValueError(
                f"Could not infer label column in {path}. Expected one of "
                "label, pcr, response, target, y."
            )
        label_col = df.columns[0]

    return df[label_col].map(_map_label_value).astype("int64")


def _resolve_feature_path(cfg: Dict, model_cfg: Dict) -> str:
    data_cfg = cfg["data"]
    geo_cfg = data_cfg.get("geo_tnbc", {})
    processed_dir = geo_cfg.get("processed_dir", os.path.join(data_cfg["data_dir"], "processed"))
    feature_key = model_cfg.get("input_feature_set", "genes_500")
    feature_sets = geo_cfg.get("feature_sets", {})
    filename = feature_sets.get(feature_key, f"combined_325_{feature_key}.csv")
    return os.path.join(processed_dir, filename)


def load_genomic_table(cfg: Dict, model_cfg: Dict) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load expression, labels, and patient IDs for GEO TNBC experiments."""
    data_cfg = cfg["data"]
    geo_cfg = data_cfg.get("geo_tnbc", {})
    processed_dir = geo_cfg.get("processed_dir", os.path.join(data_cfg["data_dir"], "processed"))

    expression_path = _resolve_feature_path(cfg, model_cfg)
    labels_path = os.path.join(processed_dir, geo_cfg.get("labels_file", "labels_pcr_rd.csv"))

    if not os.path.exists(expression_path):
        raise FileNotFoundError(f"Expression file not found: {expression_path}")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    x = _read_expression_csv(expression_path)
    y = _read_labels_csv(labels_path)

    common = x.index.intersection(y.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between expression and label files.")

    x = x.loc[common].sort_index()
    y = y.loc[common].sort_index()
    patient_ids = x.index.astype(str).to_numpy()
    return x, y, patient_ids


def make_synthetic_genomic_table(
    n_samples: int = 160,
    n_genes: int = 500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Create a small expression-like binary dataset for smoke tests."""
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, 0.35, size=n_samples)
    x = rng.normal(0.0, 1.0, size=(n_samples, n_genes)).astype("float32")

    signal_genes = min(20, n_genes)
    x[:, :signal_genes] += y[:, None] * rng.normal(0.9, 0.15, size=(n_samples, signal_genes))
    x[:, signal_genes : signal_genes * 2] -= y[:, None] * rng.normal(
        0.6, 0.15, size=(n_samples, signal_genes)
    )

    sample_ids = np.array([f"SYN_{i:04d}" for i in range(n_samples)])
    columns = [f"GENE_{i:04d}" for i in range(n_genes)]
    x_df = pd.DataFrame(x, index=sample_ids, columns=columns)
    y_ser = pd.Series(y.astype("int64"), index=sample_ids, name="label")
    return x_df, y_ser, sample_ids

