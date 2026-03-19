"""
Gating Network with Entropy Regularization for Dynamic Fusion.

Implements learned gating mechanism for dynamic feature fusion
between dual-branch architectures (e.g., Swin + ConvNeXt).

Reference: implementation_plan.md §3 - Dual-Branch Dynamic Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class GatingNetwork(nn.Module):
    """
    Gating network for dynamic feature fusion.
    
    Produces spatially-aware gating weights (alpha) for combining
    features from two branches.
    
    Architecture:
        Concat(F_branch1, F_branch2) → Conv3x3 → BN → ReLU
                                      → Conv1x1 → Sigmoid → alpha
    
    Args:
        in_channels: Total input channels (channels_branch1 + channels_branch2).
        hidden_channels: Hidden layer channels (default: 64).
        temperature: Temperature scaling for gate smoothing (default: 1.0).
        use_spatial: Use spatial gating (per-pixel) vs global gating.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        temperature: float = 1.0,
        use_spatial: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.use_spatial = use_spatial
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gating network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights.
        
        Args:
            feat1: Features from branch 1 (B, C1, H, W).
            feat2: Features from branch 2 (B, C2, H, W).
        
        Returns:
            alpha: Gating weights (B, 1, H, W) or (B, 1, 1, 1) if not spatial.
        """
        # Concatenate features
        concat = torch.cat([feat1, feat2], dim=1)  # (B, C1+C2, H, W)
        
        # Compute gate
        alpha = self.gate(concat)
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            alpha = torch.sigmoid(torch.log(alpha / (1 - alpha + 1e-7)) / self.temperature)
        
        return alpha


class EntropyRegularization(nn.Module):
    """
    Entropy regularization for gating network.
    
    Encourages balanced gating (alpha ≈ 0.5) to prevent gate collapse
    where one branch dominates completely.
    
    Binary entropy: H(α) = -[α log(α) + (1-α) log(1-α)]
    
    Args:
        weight: Regularization weight (default: 0.01).
        eps: Numerical stability constant (default: 1e-7).
    """
    
    def __init__(self, weight: float = 0.01, eps: float = 1e-7):
        super().__init__()
        self.weight = weight
        self.eps = eps
    
    def forward(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss.
        
        Args:
            alpha: Gating weights (B, 1, H, W) or (B, 1).
        
        Returns:
            Entropy loss (scalar, negative because we want to maximize entropy).
        """
        # Clamp alpha to avoid log(0)
        alpha = torch.clamp(alpha, self.eps, 1 - self.eps)
        
        # Binary entropy
        entropy = -(alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha))
        
        # Return negative entropy (to maximize entropy, we minimize negative entropy)
        return -self.weight * entropy.mean()


def entropy_regularization(
    alpha: torch.Tensor,
    weight: float = 0.01,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Functional interface for entropy regularization.
    
    Args:
        alpha: Gating weights.
        weight: Regularization weight.
        eps: Numerical stability constant.
    
    Returns:
        Entropy regularization loss.
    """
    alpha = torch.clamp(alpha, eps, 1 - eps)
    entropy = -(alpha * torch.log(alpha) + (1 - alpha) * torch.log(1 - alpha))
    return -weight * entropy.mean()


# ─────────────────────────────────────────────
# Feature Alignment Module
# ─────────────────────────────────────────────

class FeatureAlignment(nn.Module):
    """
    Feature alignment module for fusion.
    
    Aligns features from two branches to have matching:
    - Spatial dimensions (H, W)
    - Channel dimensions (C)
    
    Args:
        channels1: Channels in branch 1.
        channels2: Channels in branch 2.
        target_channels: Target channel dimension after alignment (default: 512).
        alignment_method: 'bilinear' or 'conv'.
    """
    
    def __init__(
        self,
        channels1: int,
        channels2: int,
        target_channels: int = 512,
        alignment_method: str = 'bilinear',
    ):
        super().__init__()
        self.channels1 = channels1
        self.channels2 = channels2
        self.target_channels = target_channels
        self.alignment_method = alignment_method
        
        # Channel projection for branch 1
        self.proj1 = nn.Sequential(
            nn.Conv2d(channels1, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_channels),
        )
        
        # Channel projection for branch 2
        self.proj2 = nn.Sequential(
            nn.Conv2d(channels2, target_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(target_channels),
        )
        
        if alignment_method == 'conv':
            # Additional convolution for better alignment
            self.refine = nn.Sequential(
                nn.Conv2d(target_channels * 2, target_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(target_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.refine = None
    
    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align features from two branches.
        
        Args:
            feat1: Features from branch 1 (B, C1, H1, W1).
            feat2: Features from branch 2 (B, C2, H2, W2).
        
        Returns:
            (aligned_feat1, aligned_feat2) with matching shapes.
        """
        # Project to common channel dimension
        feat1 = self.proj1(feat1)  # (B, target_channels, H1, W1)
        feat2 = self.proj2(feat2)  # (B, target_channels, H2, W2)
        
        # Match spatial dimensions using bilinear interpolation
        if feat1.shape[-2:] != feat2.shape[-2:]:
            # Use larger spatial size as target
            target_h = max(feat1.shape[-2], feat2.shape[-2])
            target_w = max(feat1.shape[-1], feat2.shape[-1])
            
            feat1 = F.interpolate(feat1, size=(target_h, target_w), mode='bilinear', align_corners=False)
            feat2 = F.interpolate(feat2, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        # Optional refinement
        if self.refine is not None:
            concat = torch.cat([feat1, feat2], dim=1)
            refined = self.refine(concat)
            # Split back
            feat1 = refined[:, :self.target_channels]
            feat2 = refined[:, self.target_channels:]
        
        return feat1, feat2
