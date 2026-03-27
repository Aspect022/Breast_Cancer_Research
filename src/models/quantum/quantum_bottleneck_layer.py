"""
Local GPU-based quantum bottleneck layer for TBCA-Fusion.

Placement 2: Post-backbone, pre-fusion.

This version removes PennyLane entirely and uses the project's
VectorizedQuantumCircuit so the bottleneck can run batched on GPU.
"""

import math
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from .vectorized_circuit import VectorizedQuantumCircuit


class QuantumBottleneckLayer(nn.Module):
    """
    Per-branch quantum feature transform with residual reconstruction.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation_config: str = "ry_only",
        entanglement: str = "cyclic",
        dropout: float = 0.3,
    ):
        super().__init__()

        self.classical_compress = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_qubits),
        )

        self.quantum_circuit = VectorizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=rotation_config,
            entanglement=entanglement,
        )

        self.quantum_norm = nn.LayerNorm(n_qubits)

        self.classical_expand = nn.Sequential(
            nn.Linear(n_qubits, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)
        self._init_quantum_weights()

    def _init_quantum_weights(self):
        for param in self.quantum_circuit.params.values():
            nn.init.normal_(param, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = self.classical_compress(x)
        compressed = torch.tanh(compressed) * math.pi

        quantum_out = self.quantum_circuit(compressed)
        quantum_out = self.quantum_norm(quantum_out)
        quantum_out = quantum_out.to(x.dtype)

        expanded = self.classical_expand(quantum_out)
        return self.layer_norm(x + expanded)


class MultiBranchQuantumBottleneck(nn.Module):
    """
    Applies independent quantum bottlenecks to selected TBCA branches.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 768,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation_config: str = "ry_only",
        entanglement: str = "cyclic",
        dropout: float = 0.3,
        apply_to: Optional[List[str]] = None,
    ):
        super().__init__()

        self.apply_to = apply_to or ["swin", "convnext", "efficientnet"]

        self.swin_bottleneck = None
        self.convnext_bottleneck = None
        self.efficientnet_bottleneck = None

        if "swin" in self.apply_to:
            self.swin_bottleneck = QuantumBottleneckLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_qubits=n_qubits,
                n_layers=n_layers,
                rotation_config=rotation_config,
                entanglement=entanglement,
                dropout=dropout,
            )

        if "convnext" in self.apply_to:
            self.convnext_bottleneck = QuantumBottleneckLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_qubits=n_qubits,
                n_layers=n_layers,
                rotation_config=rotation_config,
                entanglement=entanglement,
                dropout=dropout,
            )

        if "efficientnet" in self.apply_to:
            self.efficientnet_bottleneck = QuantumBottleneckLayer(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_qubits=n_qubits,
                n_layers=n_layers,
                rotation_config=rotation_config,
                entanglement=entanglement,
                dropout=dropout,
            )

    def apply_to_swin(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.swin_bottleneck is None else self.swin_bottleneck(x)

    def apply_to_convnext(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.convnext_bottleneck is None else self.convnext_bottleneck(x)

    def apply_to_efficientnet(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.efficientnet_bottleneck is None else self.efficientnet_bottleneck(x)

    def forward(
        self,
        swin_feat: torch.Tensor,
        convnext_feat: torch.Tensor,
        effnet_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.apply_to_swin(swin_feat),
            self.apply_to_convnext(convnext_feat),
            self.apply_to_efficientnet(effnet_feat),
        )


def get_quantum_bottleneck(
    input_dim: int = 768,
    hidden_dim: int = 768,
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = "ry_only",
    entanglement: str = "cyclic",
    dropout: float = 0.3,
    multi_branch: bool = True,
    apply_to: Optional[List[str]] = None,
):
    if multi_branch:
        return MultiBranchQuantumBottleneck(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=rotation_config,
            entanglement=entanglement,
            dropout=dropout,
            apply_to=apply_to,
        )

    return QuantumBottleneckLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
        dropout=dropout,
    )
