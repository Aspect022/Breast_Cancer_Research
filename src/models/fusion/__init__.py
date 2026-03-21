"""
Fusion Models for Breast Cancer Histopathology Classification.

This module provides multi-architecture fusion models:
- Dual-Branch Dynamic Fusion (Swin + ConvNeXt)
- Quantum-Enhanced Fusion
- Triple-Branch Cross-Attention Fusion (TBCA-Fusion)
- Class-Balanced Quantum-Classical Fusion (CB-QCCF)
- Multi-Scale Quantum Fusion (MSQF)
- Teacher-Student Distillation with Ensemble (TSD-Ensemble)

Reference: PROPOSED_ARCHITECTURES.md, QUANTUM_ARCHITECTURES.md
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

from .triple_branch import (
    TripleBranchCrossAttention,
    CrossAttention,
    SelfAttention,
    get_triple_branch_fusion,
)

from .class_balanced_quantum import (
    ClassBalancedQuantumClassicalFusion,
    ClassBalancedLoss,
    get_cb_qccf,
)

from .multi_scale_quantum import (
    MultiScaleQuantumFusion,
    ScaleQuantumModule,
    MultiScaleAttentionFusion,
    get_multi_scale_quantum_fusion,
)

from .ensemble_distillation import (
    EnsembleDistillation,
    EnsembleDistillationLoss,
    get_ensemble_distillation,
)

__all__ = [
    # Gating
    'GatingNetwork',
    'EntropyRegularization',
    'FeatureAlignment',
    'entropy_regularization',

    # Dual-Branch Fusion
    'DualBranchDynamicFusion',
    'QuantumEnhancedFusion',
    'get_dual_branch_fusion',
    'get_quantum_enhanced_fusion',

    # Triple-Branch Cross-Attention Fusion
    'TripleBranchCrossAttention',
    'CrossAttention',
    'SelfAttention',
    'get_triple_branch_fusion',

    # Class-Balanced Quantum-Classical Fusion
    'ClassBalancedQuantumClassicalFusion',
    'ClassBalancedLoss',
    'get_cb_qccf',

    # Multi-Scale Quantum Fusion
    'MultiScaleQuantumFusion',
    'ScaleQuantumModule',
    'MultiScaleAttentionFusion',
    'get_multi_scale_quantum_fusion',

    # Ensemble Distillation
    'EnsembleDistillation',
    'EnsembleDistillationLoss',
    'get_ensemble_distillation',
]
