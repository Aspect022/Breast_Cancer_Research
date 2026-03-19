"""
Transformer Models for Breast Cancer Histopathology Classification.

This module provides various transformer architectures:
- Swin Transformer (Tiny, Small, V2)
- ConvNeXt (Tiny, Small, Base)
- DeiT (Tiny, Small, Base)
- Hybrid ViT (CNN + Transformer head)
- Pure ViT (Tiny)
"""

from .swin import (
    get_swin_tiny,
    get_swin_small,
    get_swin_base,
    get_swin_v2_small,
    SwinTransformerWrapper,
)

from .convnext import (
    get_convnext_tiny,
    get_convnext_small,
    get_convnext_base,
    ConvNeXtWrapper,
)

from .deit import (
    get_deit_tiny,
    get_deit_small,
    get_deit_base,
    DeiTWithDistillation,
    DeiTWithoutDistillation,
)

from .hybrid_vit import (
    get_hybrid_vit,
    get_vit_tiny,
    HybridCNNViT,
    PureViTTiny,
)

__all__ = [
    # Swin Transformer
    'get_swin_tiny',
    'get_swin_small',
    'get_swin_base',
    'get_swin_v2_small',
    'SwinTransformerWrapper',
    
    # ConvNeXt
    'get_convnext_tiny',
    'get_convnext_small',
    'get_convnext_base',
    'ConvNeXtWrapper',
    
    # DeiT
    'get_deit_tiny',
    'get_deit_small',
    'get_deit_base',
    'DeiTWithDistillation',
    'DeiTWithoutDistillation',
    
    # Hybrid ViT
    'get_hybrid_vit',
    'get_vit_tiny',
    'HybridCNNViT',
    'PureViTTiny',
]
