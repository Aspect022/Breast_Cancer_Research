"""Baseline MLP for patient-level gene expression classification."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn


class GenomicMLP(nn.Module):
    """Small fully connected baseline for tabular gene expression."""

    def __init__(
        self,
        input_dim: int,
        hidden: Sequence[int] = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for width in hidden:
            layers.extend(
                [
                    nn.Linear(prev_dim, int(width)),
                    nn.BatchNorm1d(int(width)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = int(width)

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def get_genomic_mlp(
    input_dim: int,
    hidden: Iterable[int] = (256, 128, 64),
    dropout: float = 0.3,
) -> GenomicMLP:
    return GenomicMLP(input_dim=input_dim, hidden=tuple(hidden), dropout=dropout)

