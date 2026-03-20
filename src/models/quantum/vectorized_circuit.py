"""
GPU-Accelerated Vectorized Quantum Circuits for Breast Cancer Classification.

Implements custom vectorized quantum circuit simulation using PyTorch,
optimized for GPU execution on A100 servers.

Reference:
- Implementation based on EEG project quantum patterns
- Optimized for 100× speedup over PennyLane CPU simulation

Rotation Gate Variants:
1. RY-only: Simplest, fastest, easiest to train
2. RY+RZ: Medium complexity, better expressivity
3. U3: Full single-qubit rotation (θ, φ, λ)
4. RX+RY+RZ: Maximum expressivity, harder to train

Entanglement Strategies:
1. Cyclic CNOT ring: q0→q1→q2→...→qn→q0
2. Full entanglement: All-to-all connectivity
3. Tree entanglement: Hierarchical structure

Target Performance:
- Forward pass: <10ms (vs ~500ms PennyLane)
- 5-fold CV training: <2 hours (vs 13.4 hours PennyLane)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

@dataclass
class QuantumCircuitConfig:
    """Configuration for quantum circuit."""
    n_qubits: int = 8
    n_layers: int = 2
    rotation_config: str = 'ry_only'  # 'ry_only', 'ry_rz', 'u3', 'rx_ry_rz'
    entanglement: str = 'cyclic'  # 'cyclic', 'full', 'tree', 'none'
    use_amp: bool = False  # Automatic mixed precision


ROTATION_CONFIGS = {
    'ry_only': {'axes': ['y'], 'n_params_per_qubit': 1},
    'ry_rz': {'axes': ['y', 'z'], 'n_params_per_qubit': 2},
    'u3': {'axes': ['u3'], 'n_params_per_qubit': 3, 'use_u3': True},
    'rx_ry_rz': {'axes': ['x', 'y', 'z'], 'n_params_per_qubit': 3},
}

ENTANGLEMENT_TYPES = ['cyclic', 'full', 'tree', 'none']


# ─────────────────────────────────────────────
# Quantum Gate Operations (Vectorized)
# ─────────────────────────────────────────────

class VectorizedQuantumGates:
    """
    Vectorized quantum gate operations for GPU acceleration.
    
    All operations are batched and vectorized for efficient GPU execution.
    """
    
    @staticmethod
    def create_pauli_matrices(device: torch.device) -> Dict[str, torch.Tensor]:
        """Create Pauli matrices on device."""
        return {
            'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64, device=device),
            'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64, device=device),
            'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64, device=device),
        }

    @staticmethod
    def _apply_single_qubit_gate(state: torch.Tensor, gate: torch.Tensor, qubit_idx: int) -> torch.Tensor:
        batch_size = state.shape[0]
        n_qubits = int(math.log2(state.shape[1]))
        
        # Reshape to (batch, 2, 2, ..., 2)
        shape = [batch_size] + [2] * n_qubits
        state = state.view(*shape)
        
        # Move target qubit axis to position 1
        dims = list(range(n_qubits + 1))
        qubit_axis = n_qubits - qubit_idx
        dims[1], dims[qubit_axis] = dims[qubit_axis], dims[1]
        state = state.permute(*dims)
        
        # (batch, 2, rest)
        rest_size = (2 ** n_qubits) // 2
        state = state.reshape(batch_size, 2, rest_size)
        
        # Matrix multiply: gate @ state
        state = torch.bmm(gate, state)
        
        # Reshape back
        shape_after = [batch_size, 2] + [2] * (n_qubits - 1)
        state = state.view(*shape_after)
        
        # Permute back
        inverse_dims = list(range(n_qubits + 1))
        inverse_dims[1], inverse_dims[qubit_axis] = inverse_dims[qubit_axis], inverse_dims[1]
        state = state.permute(*inverse_dims)
        
        return state.reshape(batch_size, 2 ** n_qubits)
    
    @staticmethod
    def apply_ry(state: torch.Tensor, angle: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply RY gate to specified qubit."""
        cos_half = torch.cos(angle / 2)
        sin_half = torch.sin(angle / 2)
        gate = torch.stack([
            torch.stack([cos_half, -sin_half], dim=-1),
            torch.stack([sin_half, cos_half], dim=-1)
        ], dim=-2).to(state.dtype).to(state.device)
        return VectorizedQuantumGates._apply_single_qubit_gate(state, gate, qubit)
    
    @staticmethod
    def apply_rx(state: torch.Tensor, angle: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply RX gate to specified qubit."""
        cos_half = torch.cos(angle / 2).to(torch.complex64)
        sin_half = -1j * torch.sin(angle / 2)
        gate = torch.stack([
            torch.stack([cos_half, sin_half], dim=-1),
            torch.stack([sin_half, cos_half], dim=-1)
        ], dim=-2).to(state.device)
        return VectorizedQuantumGates._apply_single_qubit_gate(state, gate, qubit)
    
    @staticmethod
    def apply_rz(state: torch.Tensor, angle: torch.Tensor, qubit: int) -> torch.Tensor:
        """Apply RZ gate to specified qubit."""
        phase_pos = torch.exp(-1j * angle / 2)
        phase_neg = torch.exp(1j * angle / 2)
        zeros = torch.zeros_like(angle, dtype=torch.complex64)
        gate = torch.stack([
            torch.stack([phase_pos, zeros], dim=-1),
            torch.stack([zeros, phase_neg], dim=-1)
        ], dim=-2).to(state.device)
        return VectorizedQuantumGates._apply_single_qubit_gate(state, gate, qubit)
    
    @staticmethod
    def apply_cnot(state: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """Apply CNOT gate (controlled-X)."""
        batch_size = state.shape[0]
        n_qubits = int(math.log2(state.shape[1]))
        
        shape = [batch_size] + [2] * n_qubits
        state = state.view(*shape)
        
        control_axis = n_qubits - control
        target_axis = n_qubits - target
        
        idx_c1_t0 = [slice(None)] * (n_qubits + 1)
        idx_c1_t1 = [slice(None)] * (n_qubits + 1)
        idx_c1_t0[control_axis] = 1
        idx_c1_t0[target_axis] = 0
        idx_c1_t1[control_axis] = 1
        idx_c1_t1[target_axis] = 1
        
        state_c1_t0 = state[tuple(idx_c1_t0)].clone()
        state_c1_t1 = state[tuple(idx_c1_t1)].clone()
        
        state_clone = state.clone()
        state_clone[tuple(idx_c1_t0)] = state_c1_t1
        state_clone[tuple(idx_c1_t1)] = state_c1_t0
        
        return state_clone.reshape(batch_size, 2 ** n_qubits)


# ─────────────────────────────────────────────
# Vectorized Quantum Circuit
# ─────────────────────────────────────────────

class VectorizedQuantumCircuit(nn.Module):
    """
    GPU-Accelerated Vectorized Quantum Circuit.
    
    Implements variational quantum circuit with configurable rotation gates
    and entanglement strategies. Fully vectorized for GPU acceleration.
    
    Args:
        n_qubits: Number of qubits (default: 8).
        n_layers: Number of variational layers (default: 2).
        rotation_config: Rotation gate configuration.
        entanglement: Entanglement strategy.
        measure_observables: Observables to measure (default: PauliZ on all qubits).
    """
    
    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation_config: str = 'ry_only',
        entanglement: str = 'cyclic',
        measure_observables: Optional[List[str]] = None,
    ):
        super().__init__()
        
        if n_qubits < 1 or n_qubits > 12:
            raise ValueError(f"n_qubits must be 1-12, got {n_qubits}")
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation_config = rotation_config
        self.entanglement = entanglement
        self.state_dim = 2 ** n_qubits
        
        # Validate rotation config
        if rotation_config not in ROTATION_CONFIGS:
            raise ValueError(f"Unknown rotation_config: {rotation_config}. "
                           f"Choose from: {list(ROTATION_CONFIGS.keys())}")
        
        # Validate entanglement
        if entanglement not in ENTANGLEMENT_TYPES:
            raise ValueError(f"Unknown entanglement: {entanglement}. "
                           f"Choose from: {ENTANGLEMENT_TYPES}")
        
        config = ROTATION_CONFIGS[rotation_config]
        self.rotation_axes = config['axes']
        self.use_u3 = config.get('use_u3', False)
        
        # Initialize trainable parameters
        self._init_parameters()
        
        # Gate operations
        self.gates = VectorizedQuantumGates()
        
        # Cache for Pauli matrices (created on first forward pass)
        self._pauli_cache = None
    
    def _init_parameters(self):
        """Initialize variational parameters."""
        self.params = nn.ParameterDict()
        
        n_params = ROTATION_CONFIGS[self.rotation_config]['n_params_per_qubit']
        
        for layer in range(self.n_layers):
            for axis_idx, axis in enumerate(self.rotation_axes):
                param_name = f'theta_{layer}_{axis}'
                self.params[param_name] = nn.Parameter(
                    torch.randn(self.n_qubits) * 0.1
                )
    
    def _initialize_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize |0⟩^⊗n state for batch."""
        state = torch.zeros(batch_size, self.state_dim, dtype=torch.complex64, device=device)
        state[:, 0] = 1.0 + 0.0j  # |00...0⟩
        return state
    
    def _apply_angle_encoding(self, state: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply angle encoding using RY gates.
        
        Args:
            state: Initial quantum state
            x: Input features (batch, n_qubits)
        """
        batch_size = x.shape[0]
        
        for i in range(self.n_qubits):
            state = self.gates.apply_ry(state, x[:, i], i)
        
        return state
    
    def _apply_variational_layer(self, state: torch.Tensor, layer: int) -> torch.Tensor:
        """Apply one variational layer."""
        # Apply rotation gates
        for axis_idx, axis in enumerate(self.rotation_axes):
            param_name = f'theta_{layer}_{axis}'
            angles = self.params[param_name]  # (n_qubits,)
            
            for qubit in range(self.n_qubits):
                if axis == 'y':
                    state = self.gates.apply_ry(state, angles[qubit:qubit+1].expand(state.shape[0]), qubit)
                elif axis == 'x':
                    state = self.gates.apply_rx(state, angles[qubit:qubit+1].expand(state.shape[0]), qubit)
                elif axis == 'z':
                    state = self.gates.apply_rz(state, angles[qubit:qubit+1].expand(state.shape[0]), qubit)
        
        # Apply entanglement
        if self.entanglement == 'cyclic':
            state = self._apply_cyclic_cnot(state)
        elif self.entanglement == 'full':
            state = self._apply_full_entanglement(state)
        elif self.entanglement == 'tree':
            state = self._apply_tree_entanglement(state)
        
        return state
    
    def _apply_cyclic_cnot(self, state: torch.Tensor) -> torch.Tensor:
        """Apply CNOT ring: q0→q1→q2→...→qn→q0."""
        for i in range(self.n_qubits):
            state = self.gates.apply_cnot(state, control=i, target=(i + 1) % self.n_qubits)
        return state
    
    def _apply_full_entanglement(self, state: torch.Tensor) -> torch.Tensor:
        """Apply all-to-all CNOT entanglement."""
        for i in range(self.n_qubits):
            for j in range(i + 1, self.n_qubits):
                state = self.gates.apply_cnot(state, control=i, target=j)
        return state
    
    def _apply_tree_entanglement(self, state: torch.Tensor) -> torch.Tensor:
        """Apply tree-structured entanglement."""
        # Root qubit 0 entangles with all others
        for i in range(1, self.n_qubits):
            state = self.gates.apply_cnot(state, control=0, target=i)
        return state
    
    def _measure_pauli_z(self, state: torch.Tensor) -> torch.Tensor:
        """
        Measure Pauli-Z expectation on all qubits.
        
        Returns:
            Expectation values (batch, n_qubits)
        """
        batch_size = state.shape[0]
        n_qubits = self.n_qubits
        
        # Probabilities of each computational basis state
        probs = torch.abs(state) ** 2  # (batch, state_dim)
        
        # Reshape to (batch, 2, 2, ..., 2)
        shape = [batch_size] + [2] * n_qubits
        probs = probs.view(*shape)
        
        expectations = []
        for i in range(n_qubits):
            # Sum over all axes except the batch (0) and specific qubit (n_qubits - i)
            qubit_axis = n_qubits - i
            axes_to_sum = [dim for dim in range(1, n_qubits + 1) if dim != qubit_axis]
            if axes_to_sum:
                marginal_probs = torch.sum(probs, dim=axes_to_sum)
            else:
                marginal_probs = probs
                
            # Expectation is P(0) - P(1)
            exp_z = marginal_probs[:, 0] - marginal_probs[:, 1]
            expectations.append(exp_z)
        
        return torch.stack(expectations, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through quantum circuit.
        
        Args:
            x: Input features (batch, n_qubits) - angle-encoded
        
        Returns:
            Measurement expectations (batch, n_qubits)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize |0⟩^⊗n state
        state = self._initialize_state(batch_size, device)
        
        # Angle encoding
        state = self._apply_angle_encoding(state, x)
        
        # Variational layers
        for layer in range(self.n_layers):
            state = self._apply_variational_layer(state, layer)
        
        # Measure Pauli-Z on all qubits
        measurements = self._measure_pauli_z(state)
        
        return measurements


# ─────────────────────────────────────────────
# Quantum-Classical Hybrid Models
# ─────────────────────────────────────────────

class QENN(nn.Module):
    """
    Quantum-Enhanced Neural Network for Breast Cancer Classification.
    
    Architecture:
        EfficientNet-B3 → GAP → 1536-dim
            → Classical compression (1536 → n_qubits)
            → Vectorized Quantum Circuit
            → Classical expansion (n_qubits → 256)
            → Classification head (256 → num_classes)
    
    Args:
        num_classes: Number of output classes.
        n_qubits: Number of qubits (default: 8).
        n_layers: Number of quantum layers (default: 2).
        rotation_config: Rotation gate configuration.
        entanglement: Entanglement strategy.
        dropout: Dropout rate.
        freeze_backbone: Freeze EfficientNet backbone.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation_config: str = 'ry_only',
        entanglement: str = 'cyclic',
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.rotation_config = rotation_config
        self.entanglement = entanglement
        
        # Classical backbone (EfficientNet-B3)
        from torchvision import models
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
        
        # Classical compression
        self.compression = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_qubits),
        )
        
        # Quantum circuit
        self.quantum_circuit = VectorizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=rotation_config,
            entanglement=entanglement,
        )
        
        # Classical expansion and classification
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.features(x)
        features = self.avgpool(features).squeeze(-1).squeeze(-1)  # (B, 1536)
        
        # Compress to n_qubits
        compressed = self.compression(features)  # (B, n_qubits)
        
        # Quantum circuit
        quantum_out = self.quantum_circuit(compressed)  # (B, n_qubits)
        
        # Classification
        logits = self.classifier(quantum_out)
        
        return logits
    
    def get_quantum_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract quantum features for analysis."""
        with torch.no_grad():
            features = self.features(x)
            features = self.avgpool(features).squeeze(-1).squeeze(-1)
            compressed = self.compression(features)
            return self.quantum_circuit(compressed)


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_qenn(
    num_classes: int = 2,
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = 'ry_only',
    entanglement: str = 'cyclic',
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> QENN:
    """
    Factory for QENN (Quantum-Enhanced Neural Network).
    
    Args:
        num_classes: Number of output classes.
        n_qubits: Number of qubits (8 recommended).
        n_layers: Number of quantum layers (2-3 recommended).
        rotation_config: 'ry_only', 'ry_rz', 'u3', or 'rx_ry_rz'.
        entanglement: 'cyclic', 'full', 'tree', or 'none'.
        dropout: Dropout rate.
        freeze_backbone: Freeze EfficientNet backbone.
    
    Returns:
        QENN model for breast cancer classification.
    """
    return QENN(
        num_classes=num_classes,
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )


def get_vectorized_quantum_circuit(
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = 'ry_only',
    entanglement: str = 'cyclic',
) -> VectorizedQuantumCircuit:
    """
    Factory for standalone vectorized quantum circuit.
    
    Args:
        n_qubits: Number of qubits.
        n_layers: Number of variational layers.
        rotation_config: Rotation gate configuration.
        entanglement: Entanglement strategy.
    
    Returns:
        VectorizedQuantumCircuit module.
    """
    return VectorizedQuantumCircuit(
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
    )
