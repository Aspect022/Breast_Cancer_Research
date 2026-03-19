"""
Fusion Models for Breast Cancer Histopathology Classification.

This module provides multi-architecture fusion models:
- Dual-Branch Dynamic Fusion (Swin + ConvNeXt)
- Quantum-Enhanced Fusion
"""

from .gating import (
    GatingNetwork,
    EntropyRegularization,
    FeatureAlignment,
    entropy_regularization,
)

from .dual_branch import (
    DualBranchDynamicFusion,
    QuantumEnhancedFusion,
    get_dual_branch_fusion,
    get_quantum_enhanced_fusion,
)

__all__ = [
    # Gating
    'GatingNetwork',
    'EntropyRegularization',
    'FeatureAlignment',
    'entropy_regularization',
    
    # Fusion Models
    'DualBranchDynamicFusion',
    'QuantumEnhancedFusion',
    'get_dual_branch_fusion',
    'get_quantum_enhanced_fusion',
]
