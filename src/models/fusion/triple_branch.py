"""
Triple-Branch Cross-Attention Fusion (TBCA-Fusion) for Breast Cancer Classification.

Architecture:
    Input (B, 3, 224, 224) → ImageNet normalization

    Branch 1 (Global Context): Swin-Small Transformer → F_swin
    Branch 2 (Local Texture): ConvNeXt-Small → F_convnext
    Branch 3 (Multi-scale): EfficientNet-B3 → F_effnet

    Cross-Attention Enhancement:
        - Swin attends to ConvNeXt features
        - Swin attends to EfficientNet features
        - ConvNeXt attends to Swin features
        - EfficientNet attends to Swin features

    Weighted Fusion:
        - Learnable branch weights (softmax normalized)
        - Self-attention refinement after fusion

    Classification:
        - Global Average Pool
        - FC(768→256) → GELU → Dropout
        - FC(256→num_classes)

    Training:
        - AdamW(lr=2e-5, weight_decay=1e-4)
        - CosineAnnealingLR with warmup
        - Loss: CrossEntropy + weight_regularization

Reference: PROPOSED_ARCHITECTURES.md - Architecture 1 (TBCA-Fusion)
Expected Accuracy: 87-89%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from .gating import entropy_regularization


class CrossAttention(nn.Module):
    """
    Cross-Attention module for feature enhancement.
    
    Query attends to Key/Value from another branch.
    
    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Attention dropout rate.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for cross-attention.
        
        Args:
            query: Query features (B, dim).
            key: Key features (B, dim).
            value: Value features (B, dim).
            attention_mask: Optional attention mask.
            
        Returns:
            Attended features (B, dim).
        """
        # Project to Q, K, V
        q = self.q_proj(query).unsqueeze(1)  # (B, 1, dim)
        k = self.k_proj(key).unsqueeze(1)    # (B, 1, dim)
        v = self.v_proj(value).unsqueeze(1)  # (B, 1, dim)
        
        batch_size = q.shape[0]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)
        k = k.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)
        v = v.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, 1, 1)
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # (B, H, 1, D)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        
        # Output projection
        attn_out = self.out_proj(attn_out.squeeze(1))  # (B, dim)
        
        # Residual connection + layer norm
        out = self.layer_norm(query + attn_out)
        
        return out


class SelfAttention(nn.Module):
    """
    Self-Attention module for feature refinement.
    
    Args:
        dim: Feature dimension.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.qkv_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input features (B, dim).
            
        Returns:
            Refined features (B, dim).
        """
        batch_size = x.shape[0]
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x).view(batch_size, 1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, 1, D)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_out = torch.matmul(attn_weights, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, self.dim)
        
        # Output projection
        out = self.out_proj(attn_out)
        
        # Residual + layer norm
        out = self.layer_norm(x + out)
        
        return out


class TripleBranchCrossAttention(nn.Module):
    """
    Triple-Branch Cross-Attention Fusion (TBCA-Fusion).
    
    Combines three complementary backbones with bidirectional
    cross-attention and learnable weighted fusion.
    
    Args:
        num_classes: Number of output classes (default: 2).
        swin_variant: Swin variant - 'tiny', 'small', 'v2_small'.
        convnext_variant: ConvNeXt variant - 'tiny', 'small', 'base'.
        efficientnet_variant: EfficientNet variant - 'b0' to 'b7'.
        dropout: Dropout rate (default: 0.3).
        fusion_dim: Fusion feature dimension (default: 768).
        num_heads: Number of attention heads (default: 8).
        entropy_weight: Entropy regularization weight for branch weights.
        freeze_backbones: Freeze all backbone weights.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        swin_variant: str = 'small',
        convnext_variant: str = 'small',
        efficientnet_variant: str = 'b3',
        dropout: float = 0.3,
        fusion_dim: int = 768,
        num_heads: int = 8,
        entropy_weight: float = 0.01,
        freeze_backbones: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.swin_variant = swin_variant
        self.convnext_variant = convnext_variant
        self.efficientnet_variant = efficientnet_variant
        self.entropy_weight = entropy_weight
        self.fusion_dim = fusion_dim
        
        # ── Branch 1: Swin Transformer (Global Context) ─────────────
        from ..transformer import get_swin_tiny, get_swin_small, get_swin_v2_small
        
        swin_factory = {
            'tiny': get_swin_tiny,
            'small': get_swin_small,
            'v2_small': get_swin_v2_small,
        }
        
        self.swin_branch = swin_factory[swin_variant](
            num_classes=num_classes,
            pretrained=True,
            dropout=dropout,
            freeze_backbone=freeze_backbones,
        )
        
        # Remove classifier
        if hasattr(self.swin_branch, 'classifier'):
            self.swin_branch.classifier = nn.Identity()
        
        swin_dim = self.swin_branch.backbone.num_features
        
        # ── Branch 2: ConvNeXt (Local Texture) ──────────────────────
        from ..transformer import get_convnext_tiny, get_convnext_small, get_convnext_base
        
        convnext_factory = {
            'tiny': get_convnext_tiny,
            'small': get_convnext_small,
            'base': get_convnext_base,
        }
        
        self.convnext_branch = convnext_factory[convnext_variant](
            num_classes=num_classes,
            pretrained=True,
            dropout=dropout,
            freeze_backbone=freeze_backbones,
        )
        
        if hasattr(self.convnext_branch, 'classifier'):
            self.convnext_branch.classifier = nn.Identity()
        
        convnext_dim = self.convnext_branch.backbone.num_features

        # ── Branch 3: EfficientNet (Multi-scale Features) ───────────
        from ..efficientnet import get_efficientnet_b3, get_efficientnet_b5

        # Support both B3 and B5 variants
        if efficientnet_variant == 'b5':
            self.efficientnet_branch = get_efficientnet_b5(num_classes=num_classes)
            effnet_dim = 2048  # EfficientNet-B5 feature dimension
        else:
            self.efficientnet_branch = get_efficientnet_b3(num_classes=num_classes)
            effnet_dim = 1536  # EfficientNet-B3 feature dimension

        # Freeze if requested
        if freeze_backbones:
            for param in self.efficientnet_branch.features.parameters():
                param.requires_grad = False

        # EfficientNet uses .features for backbone
        # Store backbone reference
        self.efficientnet_backbone = self.efficientnet_branch.features
        
        # ── Feature Projection (match dimensions) ───────────────────
        self.swin_proj = nn.Linear(swin_dim, fusion_dim)
        self.convnext_proj = nn.Linear(convnext_dim, fusion_dim)
        self.effnet_proj = nn.Linear(effnet_dim, fusion_dim)
        
        # ── Cross-Attention Modules ─────────────────────────────────
        # Swin attends to ConvNeXt and EfficientNet
        self.swin_to_convnext = CrossAttention(dim=fusion_dim, num_heads=num_heads)
        self.swin_to_effnet = CrossAttention(dim=fusion_dim, num_heads=num_heads)
        
        # ConvNeXt attends to Swin
        self.convnext_to_swin = CrossAttention(dim=fusion_dim, num_heads=num_heads)
        
        # EfficientNet attends to Swin
        self.effnet_to_swin = CrossAttention(dim=fusion_dim, num_heads=num_heads)
        
        # ── Learnable Branch Weights ────────────────────────────────
        self.branch_weights = nn.Parameter(torch.ones(3))  # [w_swin, w_convnext, w_effnet]
        
        # ── Self-Attention Refinement ───────────────────────────────
        self.fusion_attention = SelfAttention(dim=fusion_dim, num_heads=num_heads)
        
        # ── Classification Head ─────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
        # Store feature dims
        self.swin_dim = swin_dim
        self.convnext_dim = convnext_dim
        self.effnet_dim = effnet_dim
    
    def extract_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract features from all three branches.
        
        Args:
            x: Input images (B, 3, 224, 224).
            
        Returns:
            (swin_feat, convnext_feat, effnet_feat)
        """
        swin_feat = self.swin_branch.backbone(x)  # (B, swin_dim)
        convnext_feat = self.convnext_branch.backbone(x)  # (B, convnext_dim)
        effnet_feat = self.efficientnet_backbone(x)  # (B, effnet_dim)
        
        # Apply global average pooling to EfficientNet features
        effnet_feat = nn.functional.adaptive_avg_pool2d(effnet_feat, (1, 1)).squeeze(-1).squeeze(-1)
        
        return swin_feat, convnext_feat, effnet_feat
    
    def project_features(
        self,
        swin_feat: torch.Tensor,
        convnext_feat: torch.Tensor,
        effnet_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project features to common dimension."""
        swin_proj = self.swin_proj(swin_feat)
        convnext_proj = self.convnext_proj(convnext_feat)
        effnet_proj = self.effnet_proj(effnet_feat)
        
        return swin_proj, convnext_proj, effnet_proj
    
    def apply_cross_attention(
        self,
        swin_feat: torch.Tensor,
        convnext_feat: torch.Tensor,
        effnet_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply bidirectional cross-attention.
        
        Args:
            swin_feat: Projected Swin features.
            convnext_feat: Projected ConvNeXt features.
            effnet_feat: Projected EfficientNet features.
            
        Returns:
            Enhanced features from all branches.
        """
        # Swin enhanced by attending to both ConvNeXt and EfficientNet
        swin_enhanced = swin_feat + \
                       self.swin_to_convnext(swin_feat, convnext_feat, convnext_feat) + \
                       self.swin_to_effnet(swin_feat, effnet_feat, effnet_feat)
        
        # ConvNeXt enhanced by attending to Swin
        convnext_enhanced = convnext_feat + \
                           self.convnext_to_swin(convnext_feat, swin_feat, swin_feat)
        
        # EfficientNet enhanced by attending to Swin
        effnet_enhanced = effnet_feat + \
                         self.effnet_to_swin(effnet_feat, swin_feat, swin_feat)
        
        return swin_enhanced, convnext_enhanced, effnet_enhanced
    
    def fuse_features(
        self,
        swin_feat: torch.Tensor,
        convnext_feat: torch.Tensor,
        effnet_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse features using learnable branch weights.
        
        Args:
            swin_feat: Enhanced Swin features.
            convnext_feat: Enhanced ConvNeXt features.
            effnet_feat: Enhanced EfficientNet features.
            
        Returns:
            Fused features (B, fusion_dim).
        """
        # Softmax-normalized weights
        weights = F.softmax(self.branch_weights, dim=0)  # (3,)
        
        # Weighted fusion
        fused = weights[0] * swin_feat + \
                weights[1] * convnext_feat + \
                weights[2] * effnet_feat
        
        return fused
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, 224, 224).
            
        Returns:
            Logits (B, num_classes).
        """
        # Extract features from all branches
        swin_feat, convnext_feat, effnet_feat = self.extract_features(x)
        
        # Project to common dimension
        swin_proj, convnext_proj, effnet_proj = self.project_features(
            swin_feat, convnext_feat, effnet_feat
        )
        
        # Apply cross-attention enhancement
        swin_enhanced, convnext_enhanced, effnet_enhanced = self.apply_cross_attention(
            swin_proj, convnext_proj, effnet_proj
        )
        
        # Weighted fusion
        fused = self.fuse_features(swin_enhanced, convnext_enhanced, effnet_enhanced)
        
        # Self-attention refinement
        fused_refined = self.fusion_attention(fused)
        
        # Classification
        logits = self.classifier(fused_refined)
        
        return logits
    
    def forward_with_weights(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with branch weights for analysis.
        
        Args:
            x: Input images.
            
        Returns:
            (logits, branch_weights)
        """
        swin_feat, convnext_feat, effnet_feat = self.extract_features(x)
        swin_proj, convnext_proj, effnet_proj = self.project_features(
            swin_feat, convnext_feat, effnet_feat
        )
        swin_enhanced, convnext_enhanced, effnet_enhanced = self.apply_cross_attention(
            swin_proj, convnext_proj, effnet_proj
        )
        fused = self.fuse_features(swin_enhanced, convnext_enhanced, effnet_enhanced)
        fused_refined = self.fusion_attention(fused)
        logits = self.classifier(fused_refined)
        
        weights = F.softmax(self.branch_weights, dim=0)
        
        return logits, weights
    
    def compute_weight_regularization(self) -> torch.Tensor:
        """
        Compute regularization loss for branch weights.
        
        Encourages balanced contribution from all branches.
        
        Returns:
            Regularization loss.
        """
        weights = F.softmax(self.branch_weights, dim=0)
        # Penalize extreme weights (encourage balance)
        return entropy_regularization(weights.unsqueeze(0).unsqueeze(0), weight=self.entropy_weight)
    
    def get_branch_weights(self) -> Dict[str, float]:
        """
        Get normalized branch weights.
        
        Returns:
            Dictionary with branch weight values.
        """
        weights = F.softmax(self.branch_weights, dim=0)
        return {
            'swin': weights[0].item(),
            'convnext': weights[1].item(),
            'efficientnet': weights[2].item(),
        }


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_triple_branch_fusion(
    num_classes: int = 2,
    swin_variant: str = 'small',
    convnext_variant: str = 'small',
    efficientnet_variant: str = 'b3',
    dropout: float = 0.3,
    fusion_dim: int = 768,
    num_heads: int = 8,
    entropy_weight: float = 0.01,
    freeze_backbones: bool = False,
) -> TripleBranchCrossAttention:
    """
    Factory for Triple-Branch Cross-Attention Fusion (TBCA-Fusion).
    
    Args:
        num_classes: Number of output classes.
        swin_variant: 'tiny', 'small', or 'v2_small'.
        convnext_variant: 'tiny', 'small', or 'base'.
        efficientnet_variant: 'b0' to 'b7' (b3 recommended).
        dropout: Dropout rate.
        fusion_dim: Fusion feature dimension.
        num_heads: Number of attention heads.
        entropy_weight: Entropy regularization weight.
        freeze_backbones: Freeze backbone weights.
    
    Returns:
        TripleBranchCrossAttention model.
    
    Expected Performance:
        - Accuracy: 87-89%
        - Sensitivity: 88-92%
        - Specificity: 80-85%
    """
    return TripleBranchCrossAttention(
        num_classes=num_classes,
        swin_variant=swin_variant,
        convnext_variant=convnext_variant,
        efficientnet_variant=efficientnet_variant,
        dropout=dropout,
        fusion_dim=fusion_dim,
        num_heads=num_heads,
        entropy_weight=entropy_weight,
        freeze_backbones=freeze_backbones,
    )
