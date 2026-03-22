"""
Swin Transformer Variants for Breast Cancer Histopathology Classification.

Implements Swin Transformer Tiny, Small, and V2 using timm library.

Reference: 
- "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021)
- "Swin Transformer V2: Scaling Up Capacity and Resolution" (CVPR 2022)

Variants:
- Swin-Tiny: 29M params, window=7, target acc: 83-85%
- Swin-Small: 50M params, window=7, target acc: 85-87%
- Swin-V2-Small: 50M params, window=16, target acc: 86-88%
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
import timm


class SwinTransformerWrapper(nn.Module):
    """
    Swin Transformer wrapper for breast cancer classification.
    
    Uses timm's implementation with ImageNet pretrained weights.
    
    Args:
        num_classes: Number of output classes (2 for binary, 3 for multi-class).
        variant: Model variant - 'tiny', 'small', or 'v2'.
        img_size: Input image size (default: 224 for tiny/small, 256 for v2).
        pretrained: Use ImageNet pretrained weights (default: True).
        dropout: Dropout rate for classification head (default: 0.3).
        freeze_backbone: Freeze transformer backbone weights (default: False).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        variant: str = 'tiny',
        img_size: int = 224,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.variant = variant
        self.num_classes = num_classes
        
        # Model configuration mapping
        MODEL_CONFIGS = {
            'tiny': {
                'model_name': 'swin_tiny_patch4_window7_224',
                'img_size': 224,
                'embed_dim': 96,
                'depths': [2, 2, 6, 2],
                'num_heads': [3, 6, 12, 24],
            },
            'small': {
                'model_name': 'swin_small_patch4_window7_224',
                'img_size': 224,
                'embed_dim': 96,
                'depths': [2, 2, 18, 2],
                'num_heads': [3, 6, 12, 24],
            },
            'base': {
                'model_name': 'swin_base_patch4_window7_224',
                'img_size': 224,
                'embed_dim': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
            },
            'v2_small': {
                'model_name': 'swinv2_small_window16_256',
                'img_size': 256,
                'embed_dim': 96,
                'depths': [2, 2, 18, 2],
                'num_heads': [3, 6, 12, 24],
            },
            'v2_base': {
                'model_name': 'swinv2_base_window16_256',
                'img_size': 256,
                'embed_dim': 128,
                'depths': [2, 2, 18, 2],
                'num_heads': [4, 8, 16, 32],
            },
        }
        
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unknown variant: {variant}. Choose from: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[variant]
        
        # Adjust img_size if not specified
        if img_size != config['img_size'] and img_size == 224:
            img_size = config['img_size']
        
        # Create Swin Transformer model
        try:
            self.backbone = timm.create_model(
                config['model_name'],
                pretrained=pretrained,
                img_size=img_size,
                num_classes=0,  # Remove classification head
                global_pool='avg',  # Use global average pooling
            )
            print(f"✅ Loaded Swin-{variant.upper()} (img_size={img_size})")
        except Exception as e:
            print(f"⚠️  Warning: Could not load pretrained Swin-{variant}: {e}")
            print("  Creating model without pretrained weights...")
            self.backbone = timm.create_model(
                config['model_name'],
                pretrained=False,
                img_size=img_size,
                num_classes=0,
                global_pool='avg',
            )
        
        # Get feature dimension from backbone
        feature_dim = self.backbone.num_features
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 256),
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
            x: Input images (B, 3, H, W) where H, W = 224 or 256 depending on variant.

        Returns:
            Logits (B, num_classes).
        """
        # Extract features using Swin backbone
        features = self.backbone(x)  # (B, feature_dim)

        # Classification
        logits = self.classifier(features)
        return logits

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector for fusion."""
        return self.backbone(x)

    def get_recommended_lr(self) -> float:
        """
        Get recommended learning rate based on variant.
        
        V2 variants require lower learning rates due to post-norm architecture
        and improved training stability mechanisms.
        
        Returns:
            Recommended learning rate.
        """
        if 'v2' in self.variant.lower():
            return 1e-5  # Lower LR for V2 stability
        return 2e-5  # Standard LR for V1 variants

    def clip_gradients(self, max_norm: float = 1.0) -> float:
        """
        Clip gradients to prevent exploding gradients during training.
        
        V2 variants are particularly sensitive to gradient explosions
        due to their deeper architecture and post-norm design.
        
        Args:
            max_norm: Maximum gradient norm (default: 1.0).
            
        Returns:
            Total norm of gradients before clipping.
        """
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None and p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        if total_norm > max_norm:
            for p in self.parameters():
                if p.grad is not None and p.requires_grad:
                    p.grad.data.mul_(max_norm / total_norm)
        
        return total_norm

    def check_layer_norm_health(self, x: torch.Tensor, eps: float = 1e-5) -> Dict[str, float]:
        """
        Check layer normalization health for debugging training instability.
        
        V2 variants use post-norm architecture which can be sensitive to
        input distribution shifts. This method helps diagnose issues.
        
        Args:
            x: Sample input tensor.
            eps: Small constant for numerical stability.
            
        Returns:
            Dictionary with layer norm statistics.
        """
        self.eval()
        stats = {}
        
        with torch.no_grad():
            # Check input statistics
            stats['input_mean'] = x.mean().item()
            stats['input_std'] = x.std().item()
            
            # Check classifier layer norm
            if isinstance(self.classifier[0], nn.LayerNorm):
                ln = self.classifier[0]
                stats['ln_weight_mean'] = ln.weight.mean().item()
                stats['ln_weight_std'] = ln.weight.std().item()
                stats['ln_bias_mean'] = ln.bias.mean().item() if ln.bias is not None else 0.0
                
                # Test forward through layer norm
                features = self.backbone(x)
                ln_out = ln(features)
                stats['ln_output_mean'] = ln_out.mean().item()
                stats['ln_output_std'] = ln_out.std().item()
        
        return stats
    
    def get_attention_weights(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention weights from specified layer for visualization.
        
        Note: This requires the model to be in eval mode and may not work
        with all timm implementations. Use Grad-CAM for more reliable visualization.
        """
        self.eval()
        with torch.no_grad():
            # Forward pass to get attention weights
            # Note: timm's Swin implementation may not expose attention directly
            # Use captum or Grad-CAM for better interpretability
            pass
        return None


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_swin_tiny(
    num_classes: int = 2,
    img_size: int = 224,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> SwinTransformerWrapper:
    """
    Factory for Swin Transformer Tiny.
    
    Args:
        num_classes: Number of output classes.
        img_size: Input image size.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
    
    Returns:
        Swin-Tiny model (29M params, target acc: 83-85%).
    """
    return SwinTransformerWrapper(
        num_classes=num_classes,
        variant='tiny',
        img_size=img_size,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )


def get_swin_small(
    num_classes: int = 2,
    img_size: int = 224,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> SwinTransformerWrapper:
    """
    Factory for Swin Transformer Small.
    
    Args:
        num_classes: Number of output classes.
        img_size: Input image size.
        pretrained: Use ImageNet pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
    
    Returns:
        Swin-Small model (50M params, target acc: 85-87%).
    """
    return SwinTransformerWrapper(
        num_classes=num_classes,
        variant='small',
        img_size=img_size,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )


def get_swin_v2_small(
    num_classes: int = 2,
    img_size: int = 256,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> SwinTransformerWrapper:
    """
    Factory for Swin Transformer V2 Small.
    
    Args:
        num_classes: Number of output classes.
        img_size: Input image size (256 recommended for V2).
        pretrained: Use ImageNet-21K pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
    
    Returns:
        Swin-V2-Small model (50M params, target acc: 86-88%).
    """
    return SwinTransformerWrapper(
        num_classes=num_classes,
        variant='v2_small',
        img_size=img_size,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )


def get_swin_base(
    num_classes: int = 2,
    img_size: int = 224,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> SwinTransformerWrapper:
    """
    Factory for Swin Transformer Base.
    
    Args:
        num_classes: Number of output classes.
        img_size: Input image size.
        pretrained: Use ImageNet-21K pretrained weights.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
    
    Returns:
        Swin-Base model (88M params, target acc: 87-89%).
    """
    return SwinTransformerWrapper(
        num_classes=num_classes,
        variant='base',
        img_size=img_size,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
