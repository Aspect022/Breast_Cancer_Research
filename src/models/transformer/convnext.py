"""
ConvNeXt Variants for Breast Cancer Histopathology Classification.

Implements ConvNeXt Tiny and Small using timm library.

Reference: 
- "A ConvNet for the 2020s" (CVPR 2022)

ConvNeXt modernizes ResNet-style architectures to match transformer design:
- Large kernel sizes (7×7 depthwise conv)
- Inverted bottleneck structure
- Fewer activation functions
- Separate normalization and parameterization

Variants:
- ConvNeXt-Tiny: 29M params, target acc: 83-85%, faster training
- ConvNeXt-Small: 50M params, target acc: 84-86%
"""

import torch
import torch.nn as nn
from typing import Optional
import timm


class ConvNeXtWrapper(nn.Module):
    """
    ConvNeXt wrapper for breast cancer classification.
    
    Uses timm's implementation with ImageNet pretrained weights.
    
    Args:
        num_classes: Number of output classes (2 for binary, 3 for multi-class).
        variant: Model variant - 'tiny' or 'small'.
        pretrained: Use ImageNet pretrained weights (default: True).
        dropout: Dropout rate for classification head (default: 0.3).
        freeze_backbone: Freeze backbone weights (default: False).
        drop_path_rate: Stochastic depth rate (default: 0.1).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        variant: str = 'tiny',
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        
        # Model configuration mapping
        MODEL_CONFIGS = {
            'tiny': {
                'model_name': 'convnext_tiny',
                'embed_dim': 768,
                'depths': [3, 3, 9, 3],
                'dims': [96, 192, 384, 768],
            },
            'small': {
                'model_name': 'convnext_small',
                'embed_dim': 768,
                'depths': [3, 3, 27, 3],
                'dims': [96, 192, 384, 768],
            },
            'base': {
                'model_name': 'convnext_base',
                'embed_dim': 1024,
                'depths': [3, 3, 27, 3],
                'dims': [128, 256, 512, 1024],
            },
            'large': {
                'model_name': 'convnext_large',
                'embed_dim': 1536,
                'depths': [3, 3, 27, 3],
                'dims': [192, 384, 768, 1536],
            },
        }
        
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[variant]
        
        # Create ConvNeXt model
        try:
            self.backbone = timm.create_model(
                config['model_name'],
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool='avg',  # Use global average pooling
                drop_path_rate=drop_path_rate,
            )
            print(f"✅ Loaded ConvNeXt-{variant.upper()}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained ConvNeXt-{variant}: {e}")
            print("  Creating model without pretrained weights...")
            self.backbone = timm.create_model(
                config['model_name'],
                pretrained=False,
                num_classes=0,
                global_pool='avg',
                drop_path_rate=drop_path_rate,
            )
        
        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"  ℹ️  Backbone frozen - only training classification head")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, 224, 224).
        
        Returns:
            Logits (B, num_classes).
        """
        # Extract features using ConvNeXt backbone
        features = self.backbone(x)  # (B, feature_dim)
        
        # Classification
        logits = self.classifier(features)
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector for fusion."""
        return self.backbone(x)


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_convnext_tiny(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    drop_path_rate: float = 0.1,
) -> ConvNeXtWrapper:
    """
    Factory for ConvNeXt Tiny.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
        drop_path_rate: Stochastic depth rate.
    
    Returns:
        ConvNeXt-Tiny model (29M params, target acc: 83-85%).
    """
    return ConvNeXtWrapper(
        num_classes=num_classes,
        variant='tiny',
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
    )


def get_convnext_small(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    drop_path_rate: float = 0.1,
) -> ConvNeXtWrapper:
    """
    Factory for ConvNeXt Small.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
        drop_path_rate: Stochastic depth rate.
    
    Returns:
        ConvNeXt-Small model (50M params, target acc: 84-86%).
    """
    return ConvNeXtWrapper(
        num_classes=num_classes,
        variant='small',
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
    )


def get_convnext_base(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
    drop_path_rate: float = 0.2,
) -> ConvNeXtWrapper:
    """
    Factory for ConvNeXt Base.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
        drop_path_rate: Stochastic depth rate (higher for larger models).
    
    Returns:
        ConvNeXt-Base model (89M params, target acc: 86-88%).
    """
    return ConvNeXtWrapper(
        num_classes=num_classes,
        variant='base',
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        drop_path_rate=drop_path_rate,
    )
