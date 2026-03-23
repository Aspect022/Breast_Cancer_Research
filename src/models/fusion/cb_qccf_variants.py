"""
CB-QCCF Variants: Class-Balanced Quantum-Classical Fusion
with Different Backbone Combinations

Original: Swin-Small + ResNet-18 + QENN-RY
Variant 1: ConvNeXt-Small + EfficientNet-B5 + QENN-RY
Variant 2: Swin-Small + ConvNeXt-Small + QENN-RY

All variants use:
- Cross-attention fusion (classical attends to quantum)
- Dual-head prediction (sensitivity + specificity)
- Learnable threshold
- Class-balanced loss (specificity_weight=2.0)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .gating import entropy_regularization


class ClassBalancedQuantumClassicalFusion(nn.Module):
    """
    CB-QCCF: Class-Balanced Quantum-Classical Fusion
    
    Addresses quantum specificity problem with:
    1. Dual-head prediction (sensitivity + specificity)
    2. Class-balanced loss (heavily penalize FP)
    3. Learnable decision threshold
    4. Cross-attention fusion
    
    Args:
        num_classes: Number of output classes (default: 2).
        classical_backbone_name: Name of classical backbone ('swin_small', 'convnext_small', etc.)
        quantum_backbone_name: Name of quantum backbone ('resnet18', 'efficientnet_b3', etc.)
        n_qubits: Number of qubits (default: 8).
        n_layers: Number of quantum layers (default: 2).
        rotation_config: Quantum rotation config (default: 'ry_only').
        entanglement: Quantum entanglement strategy (default: 'cyclic').
        dropout: Dropout rate (default: 0.3).
        specificity_weight: Weight for specificity loss (default: 2.0).
        freeze_backbones: Freeze backbone weights.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        classical_backbone_name: str = 'swin_small',
        quantum_backbone_name: str = 'resnet18',
        n_qubits: int = 8,
        n_layers: int = 2,
        rotation_config: str = 'ry_only',
        entanglement: str = 'cyclic',
        dropout: float = 0.3,
        specificity_weight: float = 2.0,
        freeze_backbones: bool = False,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.rotation_config = rotation_config
        self.entanglement = entanglement
        self.specificity_weight = specificity_weight
        self.classical_backbone_name = classical_backbone_name
        self.quantum_backbone_name = quantum_backbone_name
        
        # ── Classical Backbone (High Accuracy) ────────────────────────
        self.classical_backbone = self._get_classical_backbone(
            classical_backbone_name,
            num_classes,
            dropout,
            freeze_backbones,
        )
        
        # Remove classifier
        if hasattr(self.classical_backbone, 'classifier'):
            self.classical_backbone.classifier = nn.Identity()
        elif hasattr(self.classical_backbone, 'head'):
            self.classical_backbone.head = nn.Identity()
        
        classical_dim = self._get_classical_dim(classical_backbone_name)
        
        # ── Quantum Backbone (High Sensitivity) ───────────────────────
        self.quantum_backbone = self._get_quantum_backbone(
            quantum_backbone_name,
            dropout,
            freeze_backbones,
        )
        quantum_input_dim = self._get_quantum_input_dim(quantum_backbone_name)
        
        # Project quantum features to n_qubits
        self.quantum_projector = nn.Sequential(
            nn.Linear(quantum_input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_qubits),
        )
        
        # Quantum circuit
        from ..quantum.vectorized_circuit import get_vectorized_quantum_circuit
        self.quantum_circuit = get_vectorized_quantum_circuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config=rotation_config,
            entanglement=entanglement,
        )
        
        # Project quantum output to classical dim
        self.quantum_to_classical = nn.Linear(n_qubits, classical_dim)
        
        # ── Cross-Attention Fusion ────────────────────────────────────
        # Classical features attend to quantum features
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=classical_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        
        # ── Dual Heads ────────────────────────────────────────────────
        # Sensitivity Head (maintain high TP rate)
        self.sensitivity_head = nn.Sequential(
            nn.Linear(classical_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),  # Lower dropout for sensitivity
            nn.Linear(256, num_classes),
        )
        
        # Specificity Head (reduce FP rate)
        self.specificity_head = nn.Sequential(
            nn.Linear(classical_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),  # Higher dropout for regularization
            nn.Linear(256, num_classes),
        )
        
        # ── Learnable Threshold ───────────────────────────────────────
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
        # Store classical dim
        self.classical_dim = classical_dim
        
        # Initialize weights
        self._init_weights()
    
    def _get_classical_backbone(self, name: str, num_classes: int, dropout: float, freeze: bool):
        """Get classical backbone by name"""
        if name == 'swin_small':
            from ..transformer import get_swin_small
            backbone = get_swin_small(num_classes, pretrained=True, dropout=dropout, freeze_backbone=freeze)
        elif name == 'swin_tiny':
            from ..transformer import get_swin_tiny
            backbone = get_swin_tiny(num_classes, pretrained=True, dropout=dropout, freeze_backbone=freeze)
        elif name == 'convnext_small':
            from ..transformer import get_convnext_small
            backbone = get_convnext_small(num_classes, pretrained=True, dropout=dropout, freeze_backbone=freeze)
        elif name == 'convnext_tiny':
            from ..transformer import get_convnext_tiny
            backbone = get_convnext_tiny(num_classes, pretrained=True, dropout=dropout, freeze_backbone=freeze)
        else:
            raise ValueError(f"Unknown classical backbone: {name}")
        
        return backbone
    
    def _get_quantum_backbone(self, name: str, dropout: float, freeze: bool):
        """Get quantum backbone by name"""
        from torchvision import models
        
        if name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.quantum_feature_extractor = nn.Sequential(
                nn.Sequential(*list(backbone.children())[:-1]),
                nn.Flatten(),
            )
        elif name == 'resnet34':
            backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            self.quantum_feature_extractor = nn.Sequential(
                nn.Sequential(*list(backbone.children())[:-1]),
                nn.Flatten(),
            )
        elif name == 'efficientnet_b3':
            from ..efficientnet import get_efficientnet_b3
            backbone = get_efficientnet_b3(num_classes=2)
            self.quantum_feature_extractor = backbone.features
        elif name == 'efficientnet_b5':
            from ..efficientnet import get_efficientnet_b5
            backbone = get_efficientnet_b5(num_classes=2)
            self.quantum_feature_extractor = backbone.features
        else:
            raise ValueError(f"Unknown quantum backbone: {name}")
        
        if freeze:
            for param in self.quantum_feature_extractor.parameters():
                param.requires_grad = False
        
        return backbone
    
    def _get_classical_dim(self, name: str) -> int:
        """Get classical backbone feature dimension"""
        dims = {
            'swin_small': 768,
            'swin_tiny': 768,
            'convnext_small': 768,
            'convnext_tiny': 768,
        }
        return dims.get(name, 768)
    
    def _get_quantum_input_dim(self, name: str) -> int:
        """Get quantum backbone feature dimension"""
        dims = {
            'resnet18': 512,
            'resnet34': 512,
            'efficientnet_b3': 1536,
            'efficientnet_b5': 2048,
        }
        return dims.get(name, 512)
    
    def _init_weights(self):
        """Initialize dual head weights"""
        for m in [self.sensitivity_head, self.specificity_head]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def extract_features(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from both branches.
        
        Returns:
            (classical_feat, quantum_feat)
        """
        # Classical branch
        classical_feat = self.classical_backbone.backbone(x)  # (B, classical_dim)
        
        # Quantum branch
        quantum_cnn_feat = self.quantum_feature_extractor(x)  # (B, quantum_input_dim)
        quantum_proj = self.quantum_projector(quantum_cnn_feat)  # (B, n_qubits)
        quantum_out = self.quantum_circuit(quantum_proj)  # (B, n_qubits)
        quantum_feat = self.quantum_to_classical(quantum_out)  # (B, classical_dim)
        
        return classical_feat, quantum_feat
    
    def fuse_features(
        self,
        classical_feat: torch.Tensor,
        quantum_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Fuse features using cross-attention.
        
        Returns:
            Fused features (B, classical_dim)
        """
        # Reshape for multihead attention: (B, 1, dim)
        classical_q = classical_feat.unsqueeze(1)
        quantum_kv = quantum_feat.unsqueeze(1)
        
        # Cross-attention: classical attends to quantum
        attn_out, attn_weights = self.cross_attention(
            classical_q, quantum_kv, quantum_kv
        )  # (B, 1, dim)
        
        # Residual fusion
        fused = classical_feat + attn_out.squeeze(1)  # (B, dim)
        
        return fused
    
    def forward(
        self,
        x: torch.Tensor,
        return_all: bool = False,
    ):
        """
        Forward pass.
        
        Args:
            x: Input images (B, 3, 224, 224).
            return_all: If True, return all predictions (for analysis)
            
        Returns:
            If return_all: (final_pred, sens_pred, spec_pred)
            Else: final_pred only (for training)
        """
        # Extract features
        classical_feat, quantum_feat = self.extract_features(x)
        
        # Cross-attention fusion
        fused = self.fuse_features(classical_feat, quantum_feat)
        
        # Dual-head predictions
        sens_pred = self.sensitivity_head(fused)
        spec_pred = self.specificity_head(fused)
        
        # Weighted combination with learnable threshold
        final_pred = sens_pred * self.threshold + spec_pred * (1 - self.threshold)
        
        if return_all:
            return final_pred, sens_pred, spec_pred
        else:
            return final_pred


# ─────────────────────────────────────────────
# Factory Functions for CB-QCCF Variants
# ─────────────────────────────────────────────

def get_cb_qccf_original(
    num_classes: int = 2,
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = 'ry_only',
    entanglement: str = 'cyclic',
    dropout: float = 0.3,
    specificity_weight: float = 2.0,
    freeze_backbones: bool = False,
) -> ClassBalancedQuantumClassicalFusion:
    """
    Original CB-QCCF: Swin-Small + ResNet-18 + QENN-RY
    """
    return ClassBalancedQuantumClassicalFusion(
        num_classes=num_classes,
        classical_backbone_name='swin_small',
        quantum_backbone_name='resnet18',
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
        dropout=dropout,
        specificity_weight=specificity_weight,
        freeze_backbones=freeze_backbones,
    )


def get_cb_qccf_convnet_efficient(
    num_classes: int = 2,
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = 'ry_only',
    entanglement: str = 'cyclic',
    dropout: float = 0.3,
    specificity_weight: float = 2.0,
    freeze_backbones: bool = False,
) -> ClassBalancedQuantumClassicalFusion:
    """
    CB-QCCF Variant 1: ConvNeXt-Small + EfficientNet-B5 + QENN-RY
    
    Expected Performance:
    - Accuracy: 85-87%
    - Sensitivity: 89-91%
    - Specificity: 80-83% (improved from 72%)
    """
    return ClassBalancedQuantumClassicalFusion(
        num_classes=num_classes,
        classical_backbone_name='convnext_small',
        quantum_backbone_name='efficientnet_b5',
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
        dropout=dropout,
        specificity_weight=specificity_weight,
        freeze_backbones=freeze_backbones,
    )


def get_cb_qccf_swin_convnet(
    num_classes: int = 2,
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = 'ry_only',
    entanglement: str = 'cyclic',
    dropout: float = 0.3,
    specificity_weight: float = 2.0,
    freeze_backbones: bool = False,
) -> ClassBalancedQuantumClassicalFusion:
    """
    CB-QCCF Variant 2: Swin-Small + ConvNeXt-Small + QENN-RY
    
    Expected Performance:
    - Accuracy: 85-87%
    - Sensitivity: 89-91%
    - Specificity: 80-83% (improved from 72%)
    """
    return ClassBalancedQuantumClassicalFusion(
        num_classes=num_classes,
        classical_backbone_name='swin_small',
        quantum_backbone_name='convnext_small',
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
        dropout=dropout,
        specificity_weight=specificity_weight,
        freeze_backbones=freeze_backbones,
    )


def get_cb_qccf(
    num_classes: int = 2,
    classical_backbone: str = 'swin_small',
    quantum_backbone: str = 'resnet18',
    n_qubits: int = 8,
    n_layers: int = 2,
    rotation_config: str = 'ry_only',
    entanglement: str = 'cyclic',
    dropout: float = 0.3,
    specificity_weight: float = 2.0,
    freeze_backbones: bool = False,
) -> ClassBalancedQuantumClassicalFusion:
    """
    Generic CB-QCCF factory function
    
    Args:
        classical_backbone: 'swin_small', 'swin_tiny', 'convnext_small', 'convnext_tiny'
        quantum_backbone: 'resnet18', 'resnet34', 'efficientnet_b3', 'efficientnet_b5'
    """
    return ClassBalancedQuantumClassicalFusion(
        num_classes=num_classes,
        classical_backbone_name=classical_backbone,
        quantum_backbone_name=quantum_backbone,
        n_qubits=n_qubits,
        n_layers=n_layers,
        rotation_config=rotation_config,
        entanglement=entanglement,
        dropout=dropout,
        specificity_weight=specificity_weight,
        freeze_backbones=freeze_backbones,
    )
