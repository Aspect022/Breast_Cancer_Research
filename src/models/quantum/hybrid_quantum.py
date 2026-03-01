"""
Quantum–Classical Hybrid Model for Breast Cancer Histopathology Classification.

Architecture (from plan.md):
    EfficientNet-B3 (feature extractor, no classifier head)
        → Global Average Pool → 1536-dim feature vector
        → Linear(1536 → 8)  [classical down-projection]
        → PQC Layer (8 qubits):
            • Angle encoding: RY(x_i) per qubit
            • 2–3 Strongly Entangling Layers:
                RX(θ), RY(θ), RZ(θ) + CNOT ring entanglement
            • PauliZ measurement on all 8 qubits → 8 expectation values
        → Linear(8 → num_classes)

If PennyLane is not installed, falls back to a classical-simulation
of the quantum circuit using standard PyTorch operations.

Reference: plan.md §1 Quantum–Classical Hybrid Design
           EEG project quantum_snn.py architecture patterns
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional

# Try to import PennyLane
try:
    import pennylane as qml
    PENNYLANE_AVAILABLE = True
except ImportError:
    PENNYLANE_AVAILABLE = False


# ─────────────────────────────────────────────
# PennyLane Quantum Circuit (when available)
# ─────────────────────────────────────────────

if PENNYLANE_AVAILABLE:
    class PennyLanePQC(nn.Module):
        """
        Parameterized Quantum Circuit using PennyLane with PyTorch interface.

        - 8 qubits
        - Angle encoding via RY gates
        - Strongly Entangling Layers (RX, RY, RZ + CNOT ring)
        - PauliZ measurements on all qubits

        Args:
            n_qubits: Number of qubits (default: 8).
            n_layers: Number of variational layers (default: 2).
        """
        def __init__(self, n_qubits: int = 8, n_layers: int = 2):
            super().__init__()
            self.n_qubits = n_qubits
            self.n_layers = n_layers

            # PennyLane device (default.qubit supports autograd)
            self.dev = qml.device("default.qubit", wires=n_qubits)

            # Trainable parameters: 3 rotations per qubit per layer
            self.params = nn.Parameter(
                torch.randn(n_layers, n_qubits, 3) * 0.1
            )

            # Build QNode
            self.qnode = qml.QNode(self._circuit, self.dev, interface="torch")

        def _circuit(self, inputs, weights):
            """
            Quantum circuit definition.

            Args:
                inputs: Classical features (8 values) for angle encoding.
                weights: Trainable rotation parameters (n_layers, n_qubits, 3).
            """
            # Angle encoding
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)

            # Strongly Entangling Layers
            for layer in range(self.n_layers):
                # Parameterized rotations
                for i in range(self.n_qubits):
                    qml.RX(weights[layer, i, 0], wires=i)
                    qml.RY(weights[layer, i, 1], wires=i)
                    qml.RZ(weights[layer, i, 2], wires=i)

                # CNOT ring entanglement: q0→q1→q2→...→q7→q0
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

            # Measure PauliZ expectation on all qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass through the quantum circuit.

            Args:
                x: Input features (batch, n_qubits).

            Returns:
                Expectation values (batch, n_qubits).
            """
            batch_size = x.size(0)
            results = []

            for i in range(batch_size):
                # Run each sample through the circuit
                result = self.qnode(x[i], self.params)
                results.append(torch.stack(result))

            return torch.stack(results)


# ─────────────────────────────────────────────
# Classical Fallback (simulates PQC behavior)
# ─────────────────────────────────────────────

class ClassicalPQCSimulation(nn.Module):
    """
    Classical simulation of PQC behavior for environments without PennyLane.

    Uses learnable rotation matrices and nonlinear transformations to
    approximate the effect of quantum feature mapping.

    Args:
        n_qubits: Number of simulated qubits (default: 8).
        n_layers: Number of variational layers (default: 2).
    """
    def __init__(self, n_qubits: int = 8, n_layers: int = 2):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers

        # Simulate angle encoding + variational layers
        layers = []
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(n_qubits, n_qubits, bias=False),
                nn.Tanh(),  # Simulates bounded quantum expectation values [-1, 1]
            ])
        self.transform = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch, n_qubits).

        Returns:
            Simulated expectation values (batch, n_qubits).
        """
        # Apply angle encoding (sinusoidal nonlinearity)
        encoded = torch.sin(x * math.pi)

        # Variational transformation
        out = self.transform(encoded)

        # Bound to [-1, 1] like PauliZ expectations
        return torch.tanh(out)


# ─────────────────────────────────────────────
# Quantum–Classical Hybrid Model
# ─────────────────────────────────────────────

class QuantumClassicalHybrid(nn.Module):
    """
    EfficientNet-B3 + Parameterized Quantum Circuit Hybrid.

    Architecture:
        EfficientNet-B3 → GAP → 1536-dim
            → Linear(1536 → 8)
            → PQC (8 qubits, 2 layers)  OR  Classical simulation
            → Linear(8 → num_classes)

    Args:
        num_classes: Number of output classes.
        n_qubits: Number of qubits (default: 8).
        n_layers: Number of PQC variational layers (default: 2).
        use_pennylane: Force PennyLane usage (default: auto-detect).
        freeze_backbone: Freeze EfficientNet weights (default: False).
        dropout: Dropout before final projection (default: 0.3).
    """
    def __init__(
        self,
        num_classes: int = 2,
        n_qubits: int = 8,
        n_layers: int = 2,
        use_pennylane: Optional[bool] = None,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_qubits = n_qubits

        # ── CNN Backbone ──
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # ── Classical down-projection ──
        self.pre_quantum = nn.Sequential(
            nn.Linear(1536, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_qubits),
        )

        # ── Quantum Layer ──
        if use_pennylane is None:
            use_pennylane = PENNYLANE_AVAILABLE

        if use_pennylane and PENNYLANE_AVAILABLE:
            self.quantum_layer = PennyLanePQC(n_qubits=n_qubits, n_layers=n_layers)
            self.using_pennylane = True
            print(f"✅ Using PennyLane PQC ({n_qubits} qubits, {n_layers} layers)")
        else:
            self.quantum_layer = ClassicalPQCSimulation(n_qubits=n_qubits, n_layers=n_layers)
            self.using_pennylane = False
            if use_pennylane:
                print("⚠️  PennyLane not available. Falling back to classical simulation.")
            else:
                print(f"ℹ️  Using classical PQC simulation ({n_qubits} qubits, {n_layers} layers)")

        # ── Post-quantum classification head ──
        self.post_quantum = nn.Linear(n_qubits, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Logits (B, num_classes).
        """
        # CNN feature extraction
        feat = self.features(x)                      # (B, 1536, 7, 7)
        feat = self.avgpool(feat).squeeze(-1).squeeze(-1)  # (B, 1536)

        # Classical down-projection to n_qubits
        feat = self.pre_quantum(feat)                 # (B, 8)

        # Quantum circuit (or classical simulation)
        quantum_out = self.quantum_layer(feat)         # (B, 8)

        # Classification
        logits = self.post_quantum(quantum_out)        # (B, num_classes)
        return logits

    def get_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the 8-dim quantum feature vector for analysis."""
        with torch.no_grad():
            feat = self.features(x)
            feat = self.avgpool(feat).squeeze(-1).squeeze(-1)
            feat = self.pre_quantum(feat)
            return self.quantum_layer(feat)


# ─────────────────────────────────────────────
# Factory Function
# ─────────────────────────────────────────────

def get_quantum_hybrid(
    num_classes: int = 2,
    n_qubits: int = 8,
    n_layers: int = 2,
    use_pennylane: Optional[bool] = None,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
) -> QuantumClassicalHybrid:
    """
    Factory for Quantum-Classical Hybrid model.

    Args:
        num_classes: 2 for binary, 3 for multi-class.
        n_qubits: Number of PQC qubits (8 recommended per plan.md).
        n_layers: Number of PQC layers (2-3 recommended).
        use_pennylane: True to force PennyLane, False for classical sim, None for auto.
        freeze_backbone: Freeze EfficientNet weights.
        dropout: Dropout in pre-quantum projection.

    Returns:
        QuantumClassicalHybrid model instance.
    """
    return QuantumClassicalHybrid(
        num_classes=num_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        use_pennylane=use_pennylane,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
    )
