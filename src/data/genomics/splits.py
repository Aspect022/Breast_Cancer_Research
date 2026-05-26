"""Cross-validation split helpers for patient-level genomics."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def make_stratified_holdout_and_folds(
    y: np.ndarray,
    patient_ids: np.ndarray,
    n_folds: int = 5,
    test_holdout: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, List[Tuple[np.ndarray, np.ndarray]]]:
    """Create a fixed test holdout and stratified folds over train/val patients."""
    indices = np.arange(len(y))
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_holdout,
        random_state=seed,
        stratify=y,
    )

    trainval_patients = set(patient_ids[trainval_idx])
    test_patients = set(patient_ids[test_idx])
    overlap = trainval_patients.intersection(test_patients)
    if overlap:
        raise ValueError(f"Patient leakage into test set: {sorted(overlap)[:5]}")

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    trainval_y = y[trainval_idx]
    for train_rel, val_rel in splitter.split(trainval_idx, trainval_y):
        train_idx = trainval_idx[train_rel]
        val_idx = trainval_idx[val_rel]

        train_patients = set(patient_ids[train_idx])
        val_patients = set(patient_ids[val_idx])
        fold_overlap = train_patients.intersection(val_patients)
        if fold_overlap:
            raise ValueError(f"Patient leakage inside fold: {sorted(fold_overlap)[:5]}")

        folds.append((train_idx, val_idx))

    return test_idx, folds

