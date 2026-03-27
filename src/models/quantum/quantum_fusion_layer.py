"""
Local GPU-based quantum fusion layer for TBCA-Fusion.

Placement 3: Post-fusion, pre-classification.

This implementation is fully native PyTorch and uses the project's
VectorizedQuantumCircuit instead of PennyLane, so training stays batched
and GPU-friendly.
"""

import math
import torch
import torch.nn as nn
from .vectorized_circuit import VectorizedQuantumCircuit


class QuantumFusionLayer(nn.Module):
    """
    Quantum feature refinement over the fused representation.

    Flow:
        fused features -> compress to qubits -> angle normalization
        -> vectorized quantum circuit -> normalize -> expand -> residual
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
        """Near-identity init to keep early training stable."""
        for param in self.quantum_circuit.params.values():
            nn.init.normal_(param, mean=0.0, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compressed = self.classical_compress(x)

        # Keep angle encoding bounded for stable training.
        compressed = torch.tanh(compressed) * math.pi

        quantum_out = self.quantum_circuit(compressed)
        quantum_out = self.quantum_norm(quantum_out)
        quantum_out = quantum_out.to(x.dtype)

        expanded = self.classical_expand(quantum_out)
        return self.layer_norm(x + expanded)


def get_quantum_fusion_layer(
    input_dim: int = 768,
    hidden_dim: int = 768,
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = "ry_only",
    entanglement: str = "cyclic",
    dropout: float = 0.3,
) -> QuantumFusionLayer:
    return QuantumFusionLayer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
        dropout=dropout,
    )
