"""
Quantum Models for Breast Cancer Histopathology Classification.

This module provides quantum-classical hybrid architectures:
- Vectorized Quantum Circuit (GPU-accelerated)
- QENN (Quantum-Enhanced Neural Network)
- CNN+PQC (PennyLane fallback)
"""

from .vectorized_circuit import (
    VectorizedQuantumCircuit,
    QENN,
    get_qenn,
    get_vectorized_quantum_circuit,
    VectorizedQuantumGates,
    QuantumCircuitConfig,
    ROTATION_CONFIGS,
    ENTANGLEMENT_TYPES,
)

from .hybrid_quantum import (
    get_quantum_hybrid,
    QuantumClassicalHybrid,
    PENNYLANE_AVAILABLE,
)

__all__ = [
    # Vectorized Quantum Circuit
    'VectorizedQuantumCircuit',
    'VectorizedQuantumGates',
    'QENN',
    'get_qenn',
    'get_vectorized_quantum_circuit',
    'QuantumCircuitConfig',
    'ROTATION_CONFIGS',
    'ENTANGLEMENT_TYPES',
    
    # PennyLane Hybrid (fallback)
    'get_quantum_hybrid',
    'QuantumClassicalHybrid',
    'PENNYLANE_AVAILABLE',
]
