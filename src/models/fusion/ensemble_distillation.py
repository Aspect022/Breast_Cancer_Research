"""
Teacher-Student Distillation with Ensemble (TSD-Ensemble) for Breast Cancer Classification.

Novelty: Knowledge distillation from ensemble of best models to achieve
accuracy beyond individual model plateau.

Motivation:
    Single models plateau at ~85% accuracy. Knowledge distillation from
    an ensemble often beats individual models by leveraging complementary
    knowledge from multiple architectures.

Architecture:
    Teachers (Frozen, Pre-trained):
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Swin-Small  │  │ CNN+ViT-Hyb  │  │ EfficientNet │
    │   (85.67%)   │  │   (84.67%)   │  │   (83.37%)   │
    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
           │                 │                 │
           └─────────────────┴─────────────────┘
                             ↓
              [Ensemble Logits → Soft Targets]
                             ↓
              ┌──────────────────────┐
              │  Student: TBCA-Fusion│
              │  or CB-QCCF          │
              │  (Trainable)         │
              └──────────────────────┘

    Loss: KL(student, ensemble) + α * CE(student, labels)

Key Components:
    1. Multiple frozen teacher models (best performers)
    2. Ensemble soft targets (weighted average of teacher logits)
    3. Student network (any architecture)
    4. Temperature-scaled soft targets for smoother gradients
    5. Combined distillation + hard label loss

Reference: PROPOSED_ARCHITECTURES.md - Architecture 5
Expected Accuracy: 87-89%
Expected Specificity: 78-82%
Implementation Effort: 1-2 days (easiest!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from .gating import entropy_regularization


class EnsembleDistillation(nn.Module):
    """
    Teacher-Student Distillation with Ensemble Teachers (TSD-Ensemble).
    
    Distills knowledge from an ensemble of pre-trained teachers into
    a student model.
    
    Args:
        num_classes: Number of output classes.
        student_model: Student model architecture.
        teacher_models: List of teacher model names.
        teacher_weights: Weights for each teacher (default: uniform).
        temperature: Temperature for soft targets (default: 4.0).
        distillation_alpha: Weight for hard label loss (default: 0.7).
        freeze_teachers: Freeze teacher weights (default: True).
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        student_model: str = 'triple_branch_fusion',
        teacher_models: Optional[List[str]] = None,
        teacher_weights: Optional[List[float]] = None,
        temperature: float = 4.0,
        distillation_alpha: float = 0.7,
        freeze_teachers: bool = True,
        # Student model parameters
        student_kwargs: Optional[Dict] = None,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.student_model_name = student_model
        self.teacher_models = teacher_models or ['swin_small', 'cnn_vit_hybrid', 'efficientnet_b3']
        self.temperature = temperature
        self.distillation_alpha = distillation_alpha
        
        # Teacher weights (uniform if not specified)
        if teacher_weights is None:
            # Weight by reported accuracy
            default_weights = {
                'swin_small': 0.8567,
                'cnn_vit_hybrid': 0.8467,
                'efficientnet_b3': 0.8337,
                'swin_tiny': 0.8212,
                'convnext_small': 0.8343,
            }
            teacher_weights = [
                default_weights.get(name, 1.0) for name in self.teacher_models
            ]
            # Normalize
            total = sum(teacher_weights)
            teacher_weights = [w / total for w in teacher_weights]
        
        self.teacher_weights = teacher_weights
        
        # ── Student Model ───────────────────────────────────────────
        self.student = self._build_student(student_model, num_classes, student_kwargs)
        
        # ── Teacher Models (Frozen) ─────────────────────────────────
        self.teachers = nn.ModuleList()
        
        for teacher_name in self.teacher_models:
            teacher = self._build_teacher(teacher_name, num_classes)
            
            if freeze_teachers:
                for param in teacher.parameters():
                    param.requires_grad = False
            
            self.teachers.append(teacher)
        
        # Store teacher names
        self.teacher_names = self.teacher_models
    
    def _build_student(self, student_name: str, num_classes: int, kwargs: Optional[Dict]) -> nn.Module:
        """Build student model."""
        from .triple_branch import get_triple_branch_fusion
        from .class_balanced_quantum import get_cb_qccf
        from .multi_scale_quantum import get_multi_scale_quantum_fusion
        from ..transformer import get_swin_small, get_convnext_small
        from ..fusion import get_dual_branch_fusion
        
        kwargs = kwargs or {}
        
        student_factory = {
            'triple_branch_fusion': lambda: get_triple_branch_fusion(
                num_classes=num_classes, **kwargs
            ),
            'cb_qccf': lambda: get_cb_qccf(
                num_classes=num_classes, **kwargs
            ),
            'multi_scale_quantum': lambda: get_multi_scale_quantum_fusion(
                num_classes=num_classes, **kwargs
            ),
            'dual_branch_fusion': lambda: get_dual_branch_fusion(
                num_classes=num_classes, **kwargs
            ),
            'swin_small': lambda: get_swin_small(
                num_classes=num_classes, pretrained=True, **kwargs
            ),
            'convnext_small': lambda: get_convnext_small(
                num_classes=num_classes, pretrained=True, **kwargs
            ),
        }
        
        if student_name not in student_factory:
            raise ValueError(f"Unknown student_model: {student_name}. "
                           f"Choose from: {list(student_factory.keys())}")
        
        return student_factory[student_name]()
    
    def _build_teacher(self, teacher_name: str, num_classes: int) -> nn.Module:
        """Build teacher model."""
        from ..transformer import get_swin_small, get_swin_tiny, get_convnext_small
        from ..transformer.hybrid_vit import get_hybrid_vit
        from ..efficientnet import get_efficientnet
        from ..fusion import get_dual_branch_fusion
        
        teacher_factory = {
            'swin_small': lambda: get_swin_small(
                num_classes=num_classes, pretrained=True, dropout=0.3
            ),
            'swin_tiny': lambda: get_swin_tiny(
                num_classes=num_classes, pretrained=True, dropout=0.3
            ),
            'cnn_vit_hybrid': lambda: get_hybrid_vit(
                num_classes=num_classes, d_model=256, num_heads=4,
                num_layers=2, dropout=0.1, freeze_backbone=False
            ),
            'efficientnet_b3': lambda: get_efficientnet(
                num_classes=num_classes, variant='b3', pretrained=True, dropout=0.3
            ),
            'dual_branch_fusion': lambda: get_dual_branch_fusion(
                num_classes=num_classes, swin_variant='small',
                convnext_variant='small', dropout=0.3
            ),
        }
        
        if teacher_name not in teacher_factory:
            raise ValueError(f"Unknown teacher: {teacher_name}. "
                           f"Choose from: {list(teacher_factory.keys())}")
        
        return teacher_factory[teacher_name]()
    
    def get_ensemble_logits(
        self,
        x: torch.Tensor,
        return_individual: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Get ensemble logits from all teachers.
        
        Args:
            x: Input images.
            return_individual: Also return individual teacher logits.
            
        Returns:
            ensemble_logits: Weighted average of teacher logits.
            individual_logits: List of individual teacher logits (if requested).
        """
        individual_logits = []
        
        with torch.no_grad():
            for teacher in self.teachers:
                teacher.eval()
                logits = teacher(x)
                individual_logits.append(logits)
        
        # Weighted average
        ensemble_logits = sum(
            w * logits for w, logits in zip(self.teacher_weights, individual_logits)
        )
        
        if return_individual:
            return ensemble_logits, individual_logits
        
        return ensemble_logits
    
    def forward(
        self,
        x: torch.Tensor,
        return_ensemble: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input images.
            return_ensemble: Also return ensemble logits.
            
        Returns:
            student_logits: Student predictions.
            ensemble_logits: Teacher ensemble predictions (if requested).
        """
        # Student prediction
        student_logits = self.student(x)
        
        if return_ensemble:
            with torch.no_grad():
                ensemble_logits = self.get_ensemble_logits(x)
            return student_logits, ensemble_logits
        
        return student_logits, None
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        ensemble_logits: torch.Tensor,
        hard_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student predictions (B, num_classes).
            ensemble_logits: Teacher ensemble predictions (B, num_classes).
            hard_labels: Ground truth labels (B,).
            
        Returns:
            total_loss: Combined distillation + hard label loss.
            loss_dict: Dictionary with individual loss components.
        """
        T = self.temperature
        
        # Soft loss (KL divergence with temperature scaling)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(ensemble_logits / T, dim=1),
            reduction='batchmean',
        ) * (T ** 2)
        
        # Hard loss (cross-entropy with ground truth)
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        
        # Combined loss
        total_loss = self.distillation_alpha * hard_loss + \
                    (1 - self.distillation_alpha) * soft_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
            'temperature': T,
        }
        
        return total_loss, loss_dict


class EnsembleDistillationLoss(nn.Module):
    """
    Combined distillation + hard label loss.
    
    Args:
        temperature: Temperature for soft targets.
        alpha: Weight for hard label loss (0.7 recommended).
    """
    
    def __init__(self, temperature: float = 4.0, alpha: float = 0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(
        self,
        student_logits: torch.Tensor,
        ensemble_logits: torch.Tensor,
        hard_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student predictions.
            ensemble_logits: Teacher ensemble predictions.
            hard_labels: Ground truth labels.
            
        Returns:
            total_loss: Combined loss.
            loss_dict: Individual loss components.
        """
        T = self.temperature
        
        # Soft loss (KL divergence)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(ensemble_logits / T, dim=1),
            reduction='batchmean',
        ) * (T ** 2)
        
        # Hard loss (cross-entropy)
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'hard_loss': hard_loss.item(),
            'soft_loss': soft_loss.item(),
        }
        
        return total_loss, loss_dict


# ─────────────────────────────────────────────
# Factory Functions
# ─────────────────────────────────────────────

def get_ensemble_distillation(
    num_classes: int = 2,
    student_model: str = 'triple_branch_fusion',
    teacher_models: Optional[List[str]] = None,
    teacher_weights: Optional[List[float]] = None,
    temperature: float = 4.0,
    distillation_alpha: float = 0.7,
    freeze_teachers: bool = True,
    student_kwargs: Optional[Dict] = None,
) -> EnsembleDistillation:
    """
    Factory for TSD-Ensemble (Teacher-Student Distillation with Ensemble).
    
    Args:
        num_classes: Number of output classes.
        student_model: Student architecture - 'triple_branch_fusion', 'cb_qccf', etc.
        teacher_models: List of teacher names.
            Options: 'swin_small', 'swin_tiny', 'cnn_vit_hybrid',
                    'efficientnet_b3', 'dual_branch_fusion'
        teacher_weights: Weights for each teacher (default: accuracy-based).
        temperature: Temperature for soft targets (4.0 recommended).
        distillation_alpha: Weight for hard label loss (0.7 recommended).
        freeze_teachers: Freeze teacher weights.
        student_kwargs: Additional kwargs for student model.
    
    Returns:
        EnsembleDistillation model.
    
    Expected Performance:
        - Accuracy: 87-89% (beats individual models)
        - Sensitivity: 88-92%
        - Specificity: 78-82%
    
    Training Recommendations:
        - Batch size: 16-32 (depending on student)
        - Learning rate: 1e-4 (higher than normal, teachers are frozen)
        - Weight decay: 1e-4
        - Epochs: 50
        - Patience: 10
    
    Example Usage:
        # Best ensemble configuration
        model = get_ensemble_distillation(
            student_model='triple_branch_fusion',
            teacher_models=['swin_small', 'cnn_vit_hybrid', 'efficientnet_b3'],
            temperature=4.0,
            distillation_alpha=0.7,
        )
    """
    return EnsembleDistillation(
        num_classes=num_classes,
        student_model=student_model,
        teacher_models=teacher_models,
        teacher_weights=teacher_weights,
        temperature=temperature,
        distillation_alpha=distillation_alpha,
        freeze_teachers=freeze_teachers,
        student_kwargs=student_kwargs,
    )
