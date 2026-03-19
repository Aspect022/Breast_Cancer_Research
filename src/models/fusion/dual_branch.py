"""
Dual-Branch Dynamic Fusion Model for Breast Cancer Classification.

Architecture:
    Input (B, 3, 224, 224) → ImageNet normalization
    
    Branch A (Global): Swin Transformer → F_swin
    Branch B (Local): ConvNeXt → F_convnext
    
    Feature Alignment:
        - Match spatial dimensions via bilinear interpolation
        - Match channel dimensions via 1×1 convolution
    
    Gating Network:
        - Conv3×3 → BN → ReLU
        - Conv1×1 → Sigmoid → alpha (B, 1, H, W)
    
    Dynamic Fusion:
        - F_fused = alpha * F_convnext + (1 - alpha) * F_swin
    
    Refinement:
        - Conv3×3 → BN → ReLU → Dropout
    
    Classification:
        - Global Average Pool
        - FC(512) → ReLU → Dropout
        - FC(256) → ReLU → Dropout
        - FC(num_classes)
    
    Training:
        - AdamW(lr=2e-5, weight_decay=1e-4)
        - CosineAnnealingWarmRestarts(T_0=10, T_mult=2)
        - Loss: CrossEntropy + entropy_weight * entropy_regularization(alpha)

Reference: implementation_plan.md §3 - Dual-Branch Dynamic Fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from .gating import GatingNetwork, FeatureAlignment, entropy_regularization


class DualBranchDynamicFusion(nn.Module):
    """
    Dual-Branch Dynamic Fusion Model.
    
    Combines Swin Transformer (global attention) and ConvNeXt (local texture)
    using learned dynamic gating.
    
    Args:
        num_classes: Number of output classes (2 for binary, 3 for multi-class).
        swin_variant: Swin variant - 'tiny', 'small', 'v2_small'.
        convnext_variant: ConvNeXt variant - 'tiny', 'small', 'base'.
        swin_pretrained: Use pretrained Swin weights.
        convnext_pretrained: Use pretrained ConvNeXt weights.
        fusion_channels: Feature channels for fusion (default: 512).
        dropout: Dropout rate (default: 0.3).
        entropy_weight: Entropy regularization weight (default: 0.01).
        gate_temperature: Temperature scaling for gate (default: 1.0).
        freeze_backbones: Freeze both backbones (default: False).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        swin_variant: str = 'tiny',
        convnext_variant: str = 'tiny',
        swin_pretrained: bool = True,
        convnext_pretrained: bool = True,
        fusion_channels: int = 512,
        dropout: float = 0.3,
        entropy_weight: float = 0.01,
        gate_temperature: float = 1.0,
        freeze_backbones: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.swin_variant = swin_variant
        self.convnext_variant = convnext_variant
        self.entropy_weight = entropy_weight
        
        # ── Branch A: Swin Transformer (Global Attention) ─────────────
        from ..transformer import (
            get_swin_tiny, get_swin_small, get_swin_v2_small,
            get_convnext_tiny, get_convnext_small, get_convnext_base,
        )
        
        # Swin branch
        swin_factory = {
            'tiny': get_swin_tiny,
            'small': get_swin_small,
            'v2_small': get_swin_v2_small,
        }
        
        self.swin_branch = swin_factory[swin_variant](
            num_classes=num_classes,
            pretrained=swin_pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbones,
        )
        
        # Remove Swin's classifier - we'll use our own
        if hasattr(self.swin_branch, 'classifier'):
            self.swin_classifier = self.swin_branch.classifier
            self.swin_branch.classifier = nn.Identity()
        
        # Get Swin feature dimension
        swin_feature_dim = self.swin_branch.backbone.num_features
        
        # ── Branch B: ConvNeXt (Local Texture) ────────────────────────
        convnext_factory = {
            'tiny': get_convnext_tiny,
            'small': get_convnext_small,
            'base': get_convnext_base,
        }
        
        self.convnext_branch = convnext_factory[convnext_variant](
            num_classes=num_classes,
            pretrained=convnext_pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbones,
        )
        
        # Remove ConvNeXt's classifier
        if hasattr(self.convnext_branch, 'classifier'):
            self.convnext_classifier = self.convnext_branch.classifier
            self.convnext_branch.classifier = nn.Identity()
        
        # Get ConvNeXt feature dimension
        convnext_feature_dim = self.convnext_branch.backbone.num_features
        
        # ── Feature Alignment ─────────────────────────────────────────
        # Note: Swin and ConvNeXt both output (B, feature_dim) after global pool
        # For spatial fusion, we need to extract features before pooling
        
        # For now, use feature-level fusion (after global pooling)
        # Future: Implement spatial fusion with intermediate features
        
        self.fusion_proj = nn.Sequential(
            nn.Linear(swin_feature_dim + convnext_feature_dim, fusion_channels),
            nn.LayerNorm(fusion_channels),
            nn.ReLU(inplace=True),
        )
        
        # ── Gating Network (feature-level) ───────────────────────────
        # For feature-level fusion, use simpler gating
        self.gate = nn.Sequential(
            nn.Linear(swin_feature_dim + convnext_feature_dim, fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_channels // 2, 1),
            nn.Sigmoid(),
        )
        
        # ── Classification Head ──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fusion_channels, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        
        # Store feature dims for reference
        self.swin_feature_dim = swin_feature_dim
        self.convnext_feature_dim = convnext_feature_dim
    
    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from both branches.
        
        Args:
            x: Input images (B, 3, 224, 224).
        
        Returns:
            (swin_features, convnext_features)
        """
        swin_feat = self.swin_branch.backbone(x)  # (B, swin_dim)
        convnext_feat = self.convnext_branch.backbone(x)  # (B, convnext_dim)
        
        return swin_feat, convnext_feat
    
    def compute_gate(self, swin_feat: torch.Tensor, convnext_feat: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights.
        
        Args:
            swin_feat: Swin features (B, swin_dim).
            convnext_feat: ConvNeXt features (B, convnext_dim).
        
        Returns:
            alpha: Gating weight (B, 1).
        """
        concat = torch.cat([swin_feat, convnext_feat], dim=1)
        alpha = self.gate(concat)
        return alpha
    
    def fuse_features(
        self,
        swin_feat: torch.Tensor,
        convnext_feat: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse features using dynamic gating.
        
        Args:
            swin_feat: Swin features (B, swin_dim).
            convnext_feat: ConvNeXt features (B, convnext_dim).
            alpha: Gating weight (B, 1).
        
        Returns:
            Fused features (B, fusion_channels).
        """
        # Dynamic fusion: weighted combination
        fused = alpha * convnext_feat + (1 - alpha) * swin_feat
        
        # Fusion projection
        concat = torch.cat([swin_feat, convnext_feat], dim=1)
        fused = self.fusion_proj(concat)
        
        return fused
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, 224, 224).
        
        Returns:
            Logits (B, num_classes).
        """
        # Extract features from both branches
        swin_feat, convnext_feat = self.extract_features(x)
        
        # Compute gating weight
        alpha = self.compute_gate(swin_feat, convnext_feat)
        
        # Fuse features
        fused_feat = self.fuse_features(swin_feat, convnext_feat, alpha)
        
        # Classification
        logits = self.classifier(fused_feat)
        
        return logits
    
    def forward_with_gate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with gating weights for analysis.
        
        Args:
            x: Input images.
        
        Returns:
            (logits, alpha) - predictions and gating weights.
        """
        swin_feat, convnext_feat = self.extract_features(x)
        alpha = self.compute_gate(swin_feat, convnext_feat)
        fused_feat = self.fuse_features(swin_feat, convnext_feat, alpha)
        logits = self.classifier(fused_feat)
        return logits, alpha
    
    def compute_entropy_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy regularization loss for gating.
        
        Args:
            alpha: Gating weights.
        
        Returns:
            Entropy loss.
        """
        return entropy_regularization(alpha, weight=self.entropy_weight)
    
    def get_gate_statistics(self, alpha: torch.Tensor) -> Dict[str, float]:
        """
        Compute statistics for gating weights.
        
        Args:
            alpha: Gating weights.
        
        Returns:
            Dictionary with gate statistics.
        """
        return {
            'mean': alpha.mean().item(),
            'std': alpha.std().item(),
            'min': alpha.min().item(),
            'max': alpha.max().item(),
            'convnext_bias': (alpha > 0.5).float().mean().item(),
        }


# ─────────────────────────────────────────────
# Quantum-Enhanced Fusion Model
# ─────────────────────────────────────────────

class QuantumEnhancedFusion(nn.Module):
    """
    Quantum-Enhanced Dual-Branch Fusion Model.
    
    Extends DualBranchDynamicFusion with quantum circuit for
    enhanced feature transformation.
    
    Architecture:
        Swin + ConvNeXt → Fused Features
            → Classical Compression (fusion_dim → n_qubits)
            → Quantum Circuit (n_qubits, n_layers)
            → Classical Expansion (n_qubits → 256)
            → Classification
    
    Args:
        num_classes: Number of output classes.
        swin_variant: Swin variant.
        convnext_variant: ConvNeXt variant.
        n_qubits: Number of qubits (default: 8).
        n_layers: Number of quantum layers (default: 2).
        rotation_config: Quantum rotation configuration.
        entanglement: Quantum entanglement strategy.
        dropout: Dropout rate.
        entropy_weight: Entropy regularization weight.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        swin_variant: str = 'tiny',
        convnext_variant: str = 'tiny',
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation_config: str = 'ry_only',
        entanglement: str = 'cyclic',
        dropout: float = 0.3,
        entropy_weight: float = 0.01,
        freeze_backbones: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.rotation_config = rotation_config
        self.entanglement = entanglement
        self.entropy_weight = entropy_weight
        
        from ..transformer import (
            get_swin_tiny, get_swin_small, get_swin_v2_small,
            get_convnext_tiny, get_convnext_small, get_convnext_base,
        )
        from ..quantum import get_vectorized_quantum_circuit
        
        # Swin branch
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
        
        if hasattr(self.swin_branch, 'classifier'):
            self.swin_branch.classifier = nn.Identity()
        
        swin_feature_dim = self.swin_branch.backbone.num_features
        
        # ConvNeXt branch
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
        
        convnext_feature_dim = self.convnext_branch.backbone.num_features
        
        # Feature fusion (classical)
        self.fusion_proj = nn.Sequential(
            nn.Linear(swin_feature_dim + convnext_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(swin_feature_dim + convnext_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
        # Quantum circuit
        self.quantum_circuit = get_vectorized_quantum_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=rotation_config,
            entanglement=entanglement,
        )
        
        # Classical compression to n_qubits
        self.quantum_compression = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_qubits),
        )
        
        # Classification after quantum
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        swin_feat = self.swin_branch.backbone(x)
        convnext_feat = self.convnext_branch.backbone(x)
        
        # Compute gate
        concat = torch.cat([swin_feat, convnext_feat], dim=1)
        alpha = self.gate(concat)
        
        # Fuse features
        fused = alpha * convnext_feat + (1 - alpha) * swin_feat
        fused = self.fusion_proj(concat)
        
        # Compress for quantum
        compressed = self.quantum_compression(fused)
        
        # Quantum circuit
        quantum_out = self.quantum_circuit(compressed)
        
        # Classification
        logits = self.classifier(quantum_out)
        
        return logits
    
    def forward_with_gate(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with gating weights."""
        swin_feat = self.swin_branch.backbone(x)
        convnext_feat = self.convnext_branch.backbone(x)
        
        concat = torch.cat([swin_feat, convnext_feat], dim=1)
        alpha = self.gate(concat)
        
        fused = self.fusion_proj(concat)
        compressed = self.quantum_compression(fused)
        quantum_out = self.quantum_circuit(compressed)
        logits = self.classifier(quantum_out)
        
        return logits, alpha
    
    def compute_entropy_loss(self, alpha: torch.Tensor) -> torch.Tensor:
        """Compute entropy regularization loss."""
        return entropy_regularization(alpha, weight=self.entropy_weight)


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_dual_branch_fusion(
    num_classes: int = 2,
    swin_variant: str = 'tiny',
    convnext_variant: str = 'tiny',
    dropout: float = 0.3,
    entropy_weight: float = 0.01,
    freeze_backbones: bool = False,
) -> DualBranchDynamicFusion:
    """
    Factory for Dual-Branch Dynamic Fusion model.
    
    Args:
        num_classes: Number of output classes.
        swin_variant: 'tiny', 'small', or 'v2_small'.
        convnext_variant: 'tiny', 'small', or 'base'.
        dropout: Dropout rate.
        entropy_weight: Entropy regularization weight.
        freeze_backbones: Freeze backbone weights.
    
    Returns:
        DualBranchDynamicFusion model.
    """
    return DualBranchDynamicFusion(
        num_classes=num_classes,
        swin_variant=swin_variant,
        convnext_variant=convnext_variant,
        dropout=dropout,
        entropy_weight=entropy_weight,
        freeze_backbones=freeze_backbones,
    )


def get_quantum_enhanced_fusion(
    num_classes: int = 2,
    swin_variant: str = 'tiny',
    convnext_variant: str = 'tiny',
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = 'ry_only',
    entanglement: str = 'cyclic',
    dropout: float = 0.3,
    entropy_weight: float = 0.01,
    freeze_backbones: bool = False,
) -> QuantumEnhancedFusion:
    """
    Factory for Quantum-Enhanced Fusion model.
    
    Args:
        num_classes: Number of output classes.
        swin_variant: 'tiny', 'small', or 'v2_small'.
        convnext_variant: 'tiny', 'small', or 'base'.
        n_qubits: Number of qubits.
        n_layers: Number of quantum layers.
        rotation_config: 'ry_only', 'ry_rz', 'u3', or 'rx_ry_rz'.
        entanglement: 'cyclic', 'full', 'tree', or 'none'.
        dropout: Dropout rate.
        entropy_weight: Entropy regularization weight.
        freeze_backbones: Freeze backbone weights.
    
    Returns:
        QuantumEnhancedFusion model.
    """
    return QuantumEnhancedFusion(
        num_classes=num_classes,
        swin_variant=swin_variant,
        convnext_variant=convnext_variant,
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
        dropout=dropout,
        entropy_weight=entropy_weight,
        freeze_backbones=freeze_backbones,
    )
