"""
DeiT (Data-efficient Image Transformers) Variants for Breast Cancer Classification.

Implements DeiT Tiny, Small, and Base with distillation using timm library.

Reference: 
- "Training data-efficient image transformers & distillation through attention" (ICML 2021)

DeiT introduces:
- Distillation token for knowledge distillation from CNN teachers
- Data-efficient training strategies
- Strong augmentations (RandAugment, Mixup, CutMix)

Variants:
- DeiT-Tiny: 5.7M params, target acc: 78-80%
- DeiT-Small: 22M params, target acc: 82-84%
- DeiT-Base: 87M params, target acc: 84-86%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import timm


class DeiTWithDistillation(nn.Module):
    """
    DeiT wrapper with distillation support for breast cancer classification.
    
    Uses timm's implementation with ImageNet pretrained weights.
    
    Args:
        num_classes: Number of output classes (2 for binary, 3 for multi-class).
        variant: Model variant - 'tiny', 'small', or 'base'.
        pretrained: Use ImageNet pretrained weights (default: True).
        dropout: Dropout rate (default: 0.1).
        drop_path_rate: Stochastic depth rate (default: 0.1).
        freeze_backbone: Freeze backbone weights (default: False).
        use_distillation: Use distillation token (default: True).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        variant: str = 'small',
        pretrained: bool = True,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        freeze_backbone: bool = False,
        use_distillation: bool = True,
    ):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        self.use_distillation = use_distillation
        
        # Model configuration mapping
        MODEL_CONFIGS = {
            'tiny': {
                'model_name': 'deit_tiny_distilled_patch16_224',
                'embed_dim': 192,
                'depth': 12,
                'num_heads': 3,
            },
            'small': {
                'model_name': 'deit_small_distilled_patch16_224',
                'embed_dim': 384,
                'depth': 12,
                'num_heads': 6,
            },
            'base': {
                'model_name': 'deit_base_distilled_patch16_224',
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
            },
        }
        
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[variant]
        
        # Create DeiT model
        try:
            self.backbone = timm.create_model(
                config['model_name'],
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool='',
                drop_path_rate=drop_path_rate,
            )
            print(f"✅ Loaded DeiT-{variant.upper()} (distilled)")
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained DeiT-{variant}: {e}")
            print("  Creating model without pretrained weights...")
            self.backbone = timm.create_model(
                config['model_name'],
                pretrained=False,
                num_classes=0,
                global_pool='',
                drop_path_rate=drop_path_rate,
            )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Classification head (simpler than Swin/ConvNeXt due to token structure)
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        
        # Distillation head (if using distillation)
        if use_distillation:
            self.distill_head = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Dropout(dropout),
                nn.Linear(feature_dim, num_classes),
            )
        else:
            self.distill_head = None
        
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
            Logits (B, num_classes). If using distillation, returns main logits.
        """
        # Extract features using DeiT backbone
        # DeiT returns features for all tokens including CLS and distillation tokens
        features = self.backbone(x)  # (B, num_tokens, feature_dim)
        
        # Get CLS token features (first token)
        cls_features = features[:, 0]  # (B, feature_dim)
        
        # Classification from CLS token
        logits = self.classifier(cls_features)
        
        return logits
    
    def forward_with_distillation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with distillation output.
        
        Args:
            x: Input images (B, 3, 224, 224).
        
        Returns:
            (main_logits, distill_logits) for distillation training.
        """
        features = self.backbone(x)
        cls_features = features[:, 0]
        
        # Main classification
        main_logits = self.classifier(cls_features)
        
        # Distillation output
        if self.distill_head is not None:
            # Distillation token is typically the second token
            if features.shape[1] > 1:
                distill_features = features[:, 1]
                distill_logits = self.distill_head(distill_features)
            else:
                distill_logits = main_logits
        else:
            distill_logits = main_logits
        
        return main_logits, distill_logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS token feature vector for fusion."""
        features = self.backbone(x)
        return features[:, 0]  # Return CLS token


class DeiTWithoutDistillation(nn.Module):
    """
    DeiT wrapper without distillation (simpler variant).
    
    Uses standard ViT-style architecture without distillation token.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        variant: str = 'small',
        pretrained: bool = True,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        
        # Model configuration (without distillation)
        MODEL_CONFIGS = {
            'tiny': {
                'model_name': 'deit_tiny_patch16_224',
                'embed_dim': 192,
            },
            'small': {
                'model_name': 'deit_small_patch16_224',
                'embed_dim': 384,
            },
            'base': {
                'model_name': 'deit_base_patch16_224',
                'embed_dim': 768,
            },
        }
        
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[variant]
        
        # Create DeiT model without distillation
        try:
            self.backbone = timm.create_model(
                config['model_name'],
                pretrained=pretrained,
                num_classes=0,
                global_pool='',
                drop_path_rate=drop_path_rate,
            )
            print(f"✅ Loaded DeiT-{variant.upper()} (no distillation)")
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained DeiT-{variant}: {e}")
            self.backbone = timm.create_model(
                config['model_name'],
                pretrained=False,
                num_classes=0,
                global_pool='',
                drop_path_rate=drop_path_rate,
            )
        
        feature_dim = self.backbone.num_features
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, num_classes),
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        cls_features = features[:, 0]
        return self.classifier(cls_features)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return features[:, 0]


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_deit_tiny(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.1,
    use_distillation: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory for DeiT Tiny.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate.
        use_distillation: Use distillation token.
        freeze_backbone: Freeze backbone weights.
    
    Returns:
        DeiT-Tiny model (5.7M params, target acc: 78-80%).
    """
    if use_distillation:
        return DeiTWithDistillation(
            num_classes=num_classes,
            variant='tiny',
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            use_distillation=True,
        )
    else:
        return DeiTWithoutDistillation(
            num_classes=num_classes,
            variant='tiny',
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )


def get_deit_small(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.1,
    use_distillation: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory for DeiT Small.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate.
        use_distillation: Use distillation token.
        freeze_backbone: Freeze backbone weights.
    
    Returns:
        DeiT-Small model (22M params, target acc: 82-84%).
    """
    if use_distillation:
        return DeiTWithDistillation(
            num_classes=num_classes,
            variant='small',
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            use_distillation=True,
        )
    else:
        return DeiTWithoutDistillation(
            num_classes=num_classes,
            variant='small',
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )


def get_deit_base(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.1,
    use_distillation: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """
    Factory for DeiT Base.
    
    Args:
        num_classes: Number of output classes.
        pretrained: Use ImageNet-21K pretrained weights.
        dropout: Dropout rate.
        use_distillation: Use distillation token.
        freeze_backbone: Freeze backbone weights.
    
    Returns:
        DeiT-Base model (87M params, target acc: 84-86%).
    """
    if use_distillation:
        return DeiTWithDistillation(
            num_classes=num_classes,
            variant='base',
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            use_distillation=True,
        )
    else:
        return DeiTWithoutDistillation(
            num_classes=num_classes,
            variant='base',
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
