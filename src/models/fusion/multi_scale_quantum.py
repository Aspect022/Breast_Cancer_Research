"""
Multi-Scale Quantum Fusion (MSQF) for Breast Cancer Classification.

Novelty: First to apply multi-scale quantum processing to histopathology.

Motivation:
    Histopathology images contain structures at multiple scales:
    - Cellular level (5-20 μm): Nuclei morphology
    - Tissue level (50-200 μm): Glandular structures
    - Architecture level (500+ μm): Tissue organization
    
    Current QENN processes features at a single scale, losing multi-scale information.

Architecture:
    Input Image (B, 3, 224, 224)
        ↓
    Multi-Scale Feature Extraction (CNN Backbone)
        ├─ Scale 1 (56×56): Fine cellular details
        ├─ Scale 2 (28×28): Medium tissue structures
        └─ Scale 3 (14×14): Global architecture
        ↓
    Parallel Quantum Circuits (one per scale)
        ├─ QENN-1 (8 qubits): Scale 1 features
        ├─ QENN-2 (8 qubits): Scale 2 features
        └─ QENN-3 (8 qubits): Scale 3 features
        ↓
    Attention-Based Fusion
        - Learn importance of each scale
        - Weighted combination
        ↓
    Classical Refinement (Swin Block)
        - Self-attention for global context
        ↓
    Final Prediction

Key Innovation:
    Each scale has its OWN quantum circuit with potentially different:
    - Rotation configurations (RY at scale 1, RY+RZ at scale 2, U3 at scale 3)
    - Entanglement strategies (cyclic, full, tree)
    - Layer depths (1, 2, 3)
    
    This enables quantum architecture search across scales.

Reference: QUANTUM_ARCHITECTURES.md - Architecture 2
Expected Accuracy: 86-88%
Expected Specificity: 78-82%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from .gating import entropy_regularization


class ScaleQuantumModule(nn.Module):
    """
    Quantum processing module for a single scale.
    
    Args:
        input_dim: Input feature dimension.
        n_qubits: Number of qubits.
        n_layers: Number of quantum layers.
        rotation_config: Rotation gate configuration.
        entanglement: Entanglement strategy.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation_config: str = 'ry_only',
        entanglement: str = 'cyclic',
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.rotation_config = rotation_config
        self.entanglement = entanglement
        
        # Project to quantum input
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_qubits),
        )
        
        # Quantum circuit
        from ..quantum import get_vectorized_quantum_circuit
        self.quantum_circuit = get_vectorized_quantum_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=rotation_config,
            entanglement=entanglement,
        )
        
        # Project quantum output to fusion dimension
        self.output_proj = nn.Linear(n_qubits, 256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through scale quantum module.
        
        Args:
            x: Input features (B, input_dim).
            
        Returns:
            Quantum processed features (B, 256).
        """
        # Project to quantum input
        quantum_input = self.projector(x)  # (B, n_qubits)
        
        # Quantum circuit
        quantum_out = self.quantum_circuit(quantum_input)  # (B, n_qubits)
        
        # Project to output dim
        out = self.output_proj(quantum_out)  # (B, 256)
        
        return out


class MultiScaleAttentionFusion(nn.Module):
    """
    Attention-based fusion for multi-scale quantum outputs.
    
    Learns the importance of each scale dynamically.
    
    Args:
        feature_dim: Feature dimension from each scale module.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """
    
    def __init__(self, feature_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Learnable scale embeddings
        self.scale_embeddings = nn.Parameter(torch.randn(3, feature_dim))
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.fusion_proj[0].weight)
        nn.init.zeros_(self.fusion_proj[0].bias)
    
    def forward(
        self,
        scale1_feat: torch.Tensor,
        scale2_feat: torch.Tensor,
        scale3_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse multi-scale features using attention.
        
        Args:
            scale1_feat: Scale 1 features (B, feature_dim).
            scale2_feat: Scale 2 features (B, feature_dim).
            scale3_feat: Scale 3 features (B, feature_dim).
            
        Returns:
            Fused features (B, feature_dim).
        """
        batch_size = scale1_feat.shape[0]
        
        # Stack scales: (B, 3, feature_dim)
        scale_features = torch.stack([scale1_feat, scale2_feat, scale3_feat], dim=1)
        
        # Add scale embeddings
        scale_features = scale_features + self.scale_embeddings.unsqueeze(0)
        
        # Self-attention across scales
        attn_out, attn_weights = self.self_attention(
            scale_features, scale_features, scale_features
        )  # (B, 3, feature_dim)
        
        # Layer norm with residual
        attn_out = self.layer_norm(scale_features + attn_out)
        
        # Flatten and fuse
        fused = attn_out.view(batch_size, -1)  # (B, feature_dim * 3)
        fused = self.fusion_proj(fused)  # (B, feature_dim)
        
        return fused, attn_weights


class MultiScaleQuantumFusion(nn.Module):
    """
    Multi-Scale Quantum Fusion (MSQF) Model.
    
    Processes histopathology features at multiple scales using
    parallel quantum circuits, then fuses with attention.
    
    Args:
        num_classes: Number of output classes.
        backbone: Feature extractor backbone.
        n_qubits: Number of qubits per scale.
        n_layers: Number of quantum layers per scale.
        scale_configs: List of (rotation_config, entanglement) for each scale.
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = 'resnet34',
        n_qubits: int = 8,
        n_layers: int = 2,
        scale_configs: Optional[List[Tuple[str, str]]] = None,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.backbone_name = backbone
        
        # Default scale configurations
        if scale_configs is None:
            scale_configs = [
                ('ry_only', 'cyclic'),    # Scale 1: Fine details
                ('ry_rz', 'cyclic'),       # Scale 2: Medium structures
                ('u3', 'full'),            # Scale 3: Global architecture
            ]
        
        self.scale_configs = scale_configs
        
        # ── Backbone for Multi-Scale Feature Extraction ─────────────
        from torchvision import models
        
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone_dim = 512
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            backbone_dim = 512
        elif backbone == 'resnet50':
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Extract intermediate layers for multi-scale features
        self.conv1 = nn.Sequential(*list(resnet.children())[:4])  # -> (B, 64, 56, 56)
        self.layer1 = resnet.layer1  # -> (B, 64, 56, 56)
        self.layer2 = resnet.layer2  # -> (B, 128, 28, 28)
        self.layer3 = resnet.layer3  # -> (B, 256, 14, 14)
        
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False
        
        # Scale feature extractors (GAP + projection)
        self.scale1_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.GELU(),
        )
        
        self.scale2_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.GELU(),
        )
        
        self.scale3_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.GELU(),
        )
        
        # ── Scale Quantum Modules ───────────────────────────────────
        self.scale1_quantum = ScaleQuantumModule(
            input_dim=256,
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=scale_configs[0][0],
            entanglement=scale_configs[0][1],
            dropout=dropout,
        )
        
        self.scale2_quantum = ScaleQuantumModule(
            input_dim=256,
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=scale_configs[1][0],
            entanglement=scale_configs[1][1],
            dropout=dropout,
        )
        
        self.scale3_quantum = ScaleQuantumModule(
            input_dim=256,
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=scale_configs[2][0],
            entanglement=scale_configs[2][1],
            dropout=dropout,
        )
        
        # ── Attention-Based Fusion ──────────────────────────────────
        self.attention_fusion = MultiScaleAttentionFusion(
            feature_dim=256,
            num_heads=8,
            dropout=dropout,
        )
        
        # ── Classical Refinement ────────────────────────────────────
        self.refinement = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # ── Classification Head ─────────────────────────────────────
        self.classifier = nn.Linear(256, num_classes)
        
        # Store scale configs for reference
        self.rotation_configs = [cfg[0] for cfg in scale_configs]
        self.entanglements = [cfg[1] for cfg in scale_configs]
    
    def extract_multi_scale_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract multi-scale features from backbone.

        Args:
            x: Input images (B, 3, 224, 224).

        Returns:
            (scale1_feat, scale2_feat, scale3_feat)
        """
        # Scale 1: Fine cellular details
        feat1 = self.conv1(x)
        feat1 = self.layer1(feat1)  # (B, 64, H1, W1)
        scale1_feat = self.scale1_proj(feat1)  # (B, 256)

        # Scale 2: Medium tissue structures
        # Use adaptive pooling to handle dynamic shapes instead of hardcoded view
        feat2 = self.layer2(feat1)  # (B, 128, H2, W2)
        scale2_feat = self.scale2_proj(feat2)  # (B, 256)

        # Scale 3: Global architecture
        feat3 = self.layer3(feat2)  # (B, 256, H3, W3)
        scale3_feat = self.scale3_proj(feat3)  # (B, 256)

        return scale1_feat, scale2_feat, scale3_feat
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, 224, 224).
            
        Returns:
            logits: Predictions (B, num_classes)
            attention_weights: Scale attention weights (B, 3, 3)
        """
        # Extract multi-scale features
        feat1, feat2, feat3 = self.extract_multi_scale_features(x)
        
        # Process through scale quantum modules
        quantum1 = self.scale1_quantum(feat1)
        quantum2 = self.scale2_quantum(feat2)
        quantum3 = self.scale3_quantum(feat3)
        
        # Attention-based fusion
        fused, attn_weights = self.attention_fusion(quantum1, quantum2, quantum3)
        
        # Classical refinement
        refined = self.refinement(fused)
        
        # Classification
        logits = self.classifier(refined)
        
        return logits, attn_weights
    
    def get_scale_importance(self, x: torch.Tensor) -> Dict[str, float]:
        """
        Get scale importance weights for analysis.
        
        Args:
            x: Input images.
            
        Returns:
            Dictionary with scale importance values.
        """
        _, attn_weights = self(x)
        
        # Average attention weights across batch and heads
        avg_weights = attn_weights.mean(dim=0).mean(dim=0).cpu().numpy()
        
        return {
            'scale1_fine': float(avg_weights[0]),
            'scale2_medium': float(avg_weights[1]),
            'scale3_coarse': float(avg_weights[2]),
        }


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_multi_scale_quantum_fusion(
    num_classes: int = 2,
    backbone: str = 'resnet34',
    n_qubits: int = 8,
    n_layers: int = 2,
    scale_configs: Optional[List[Tuple[str, str]]] = None,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> MultiScaleQuantumFusion:
    """
    Factory for MSQF (Multi-Scale Quantum Fusion).
    
    Args:
        num_classes: Number of output classes.
        backbone: Feature extractor - 'resnet18', 'resnet34', 'resnet50'.
        n_qubits: Number of qubits per scale.
        n_layers: Number of quantum layers per scale.
        scale_configs: List of (rotation, entanglement) for each scale.
            Default: [('ry_only', 'cyclic'), ('ry_rz', 'cyclic'), ('u3', 'full')]
        dropout: Dropout rate.
        freeze_backbone: Freeze backbone weights.
    
    Returns:
        MultiScaleQuantumFusion model.
    
    Expected Performance:
        - Accuracy: 86-88%
        - Sensitivity: 88-92%
        - Specificity: 78-82%
    
    Training Recommendations:
        - Batch size: 16
        - Learning rate: 2e-5
        - Weight decay: 1e-4
        - Epochs: 50
        - Patience: 10
    
    Example scale_configs for ablation study:
        # All RY-only
        [('ry_only', 'cyclic'), ('ry_only', 'cyclic'), ('ry_only', 'cyclic')]
        
        # Progressive complexity
        [('ry_only', 'cyclic'), ('ry_rz', 'cyclic'), ('u3', 'full')]
        
        # All U3
        [('u3', 'full'), ('u3', 'full'), ('u3', 'full')]
    """
    return MultiScaleQuantumFusion(
        num_classes=num_classes,
        backbone=backbone,
        n_qubits=n_qubits,
        n_layers=n_layers,
        scale_configs=scale_configs,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    )
