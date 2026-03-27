"""
Hybrid CNN + Vision Transformer and Pure ViT-Tiny for Breast Cancer Classification.

Two models:
1. HybridCNNViT: EfficientNet-B3 feature extractor → Transformer Encoder Head
2. PureViTTiny: Standard ViT-Tiny (patch 16×16, embed 384, depth 6)

Reference: Adapted from EEG project TransformerEncoderBlock and plan.md specs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Optional


# ─────────────────────────────────────────────
# Transformer Encoder Block (PyTorch)
# ─────────────────────────────────────────────

class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer Encoder block with multi-head self-attention.

    Args:
        dim: Embedding/feature dimension.
        num_heads: Number of attention heads.
        mlp_dim: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
    """
    def __init__(self, dim: int, num_heads: int = 4, mlp_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # Pre-norm MLP
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# Variant 1: CNN + ViT Head
# ─────────────────────────────────────────────

class HybridCNNViT(nn.Module):
    """
    Hybrid CNN + Vision Transformer.

    Architecture:
        EfficientNet-B3 (feature extractor, no classifier head)
            → Spatial features (B, 1536, 7, 7)
            → Flatten to patches: (B, 49, 1536)
            → Linear projection to d_model
            → Positional Embedding
            → N Transformer Encoder layers
            → CLS token → Classification Head

    Args:
        num_classes: Number of output classes.
        d_model: Transformer embedding dimension (default: 256).
        num_heads: Number of attention heads (default: 4).
        num_layers: Number of transformer encoder layers (default: 2).
        dropout: Dropout probability (default: 0.1).
        freeze_backbone: Whether to freeze EfficientNet weights (default: False).
    """
    def __init__(
        self,
        num_classes: int = 2,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.d_model = d_model

        # ── CNN Backbone (EfficientNet-B3) ──
        backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        # Remove classifier and avgpool — keep feature maps
        self.features = backbone.features  # Outputs (B, 1536, 7, 7) for 224×224 input

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        # ── Projection to d_model ──
        self.proj = nn.Linear(1536, d_model)

        # ── CLS token ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # ── Positional Embedding ──
        # 7×7 = 49 spatial patches + 1 CLS token = 50 positions
        self.pos_embed = nn.Parameter(torch.randn(1, 50, d_model) * 0.02)

        # ── Transformer Encoder ──
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(
                dim=d_model,
                num_heads=num_heads,
                mlp_dim=d_model * 4,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # ── Classification Head ──
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def extract_token_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract transformer token features before classification.

        Returns:
            Token sequence including CLS token: (B, 50, d_model)
        """
        B = x.size(0)

        feat = self.features(x)
        feat = feat.flatten(2).transpose(1, 2)
        feat = self.proj(feat)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        feat = torch.cat([cls_tokens, feat], dim=1)
        feat = feat + self.pos_embed
        feat = self.transformer(feat)
        feat = self.norm(feat)
        return feat

    def extract_cls_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CLS embedding before the classification head."""
        tokens = self.extract_token_features(x)
        return tokens[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Logits (B, num_classes).
        """
        cls_out = self.extract_cls_features(x)
        return self.head(cls_out)


# ─────────────────────────────────────────────
# Variant 2: Pure ViT-Tiny
# ─────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """Splits image into patches and embeds them."""
    def __init__(self, img_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, embed_dim: int = 384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) → (B, embed_dim, H/P, W/P) → (B, num_patches, embed_dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class PureViTTiny(nn.Module):
    """
    Pure Vision Transformer (ViT-Tiny) for image classification.

    Architecture:
        Input (B, 3, 224, 224)
            → Patch Embedding (16×16 patches → 196 patches)
            → CLS Token + Positional Embedding
            → 6 Transformer Encoder layers (embed=384, heads=6)
            → CLS token → Classification Head

    Args:
        num_classes: Number of output classes.
        img_size: Input image size (default: 224).
        patch_size: Patch size (default: 16).
        embed_dim: Embedding dimension (default: 384).
        depth: Number of transformer layers (default: 6).
        num_heads: Number of attention heads (default: 6).
        mlp_ratio: MLP hidden dim ratio (default: 4.0).
        dropout: Dropout probability (default: 0.1).
    """
    def __init__(
        self,
        num_classes: int = 2,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        num_patches = (img_size // patch_size) ** 2

        # ── Patch Embedding ──
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        # ── CLS Token ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # ── Positional Embedding ──
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )

        # ── Transformer Encoder ──
        mlp_dim = int(embed_dim * mlp_ratio)
        self.transformer = nn.Sequential(*[
            TransformerEncoderBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ── Classification Head ──
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            Logits (B, num_classes).
        """
        B = x.size(0)

        # Patch embedding: (B, 196, embed_dim)
        x = self.patch_embed(x)

        # Prepend CLS token: (B, 197, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer encoder
        x = self.transformer(x)
        x = self.norm(x)

        # CLS token → classification
        cls_out = x[:, 0]
        return self.head(cls_out)


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_hybrid_vit(num_classes: int = 2, d_model: int = 256,
                   num_heads: int = 4, num_layers: int = 2,
                   dropout: float = 0.1,
                   freeze_backbone: bool = False) -> HybridCNNViT:
    """
    Factory for HybridCNNViT (EfficientNet-B3 + Transformer Head).
    """
    return HybridCNNViT(
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )


def get_vit_tiny(num_classes: int = 2, dropout: float = 0.1) -> PureViTTiny:
    """
    Factory for Pure ViT-Tiny (patch=16, embed=384, depth=6, heads=6).
    """
    return PureViTTiny(
        num_classes=num_classes,
        img_size=224,
        patch_size=16,
        embed_dim=384,
        depth=6,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=dropout,
    )
