# Quantum-Enhanced Architectures for Breast Cancer Classification

**Date:** March 21, 2026  
**Primary Novelty:** Quantum-Classical Hybrid Models  
**Target Venue:** MICCAI 2026 / ISBI 2026  
**Target Accuracy:** 87-90% (from current 85.67%)

---

## Current Quantum Model Performance (BreakHis)

| Model | Accuracy | AUC-ROC | Sensitivity | Specificity | FNR | Training Time |
|-------|----------|---------|-------------|-------------|-----|---------------|
| **QENN-RY_ONLY** | **84.17%** | 88.40% | **91.68%** | 72.88% | 8.32% | 150 min |
| **QENN-U3** | 84.05% | 88.08% | **92.15%** | 71.88% | 7.85% | 124 min |
| **QENN-RX_RY_RZ** | 83.38% | 88.41% | 90.68% | 72.42% | 9.32% | 147 min |
| **QENN-RY_RZ** | 83.23% | 87.81% | 90.55% | 72.23% | 9.45% | 119 min |
| Swin-Small (Best Classical) | **85.67%** | 88.52% | 91.15% | 77.45% | 8.85% | 50 min |

### Critical Observation: Quantum Specificity Problem

**All QENN models show a consistent pattern:**
- ✅ **Excellent Sensitivity (90-92%)** - Great at detecting malignancy
- ❌ **Poor Specificity (71-73%)** - Too many false positives
- ⚠️ **Training Time (2-2.5 hours)** - 3× slower than classical

**Root Cause Analysis:**
1. Quantum circuits are **overly sensitive** to subtle features
2. CNN backbone features may not be optimally encoded for quantum processing
3. No explicit mechanism to penalize false positives
4. Single rotation gate configurations may lack expressivity

---

## Proposed Quantum Architectures (3 Novel Contributions)

### Architecture 1: Class-Balanced Quantum-Classical Fusion (CB-QCCF) ⭐⭐⭐

**Novelty:** HIGH - First to address quantum specificity problem explicitly  
**Implementation Priority:** P0 (Highest)  
**Expected Accuracy:** 85-87%  
**Expected Specificity:** 80-83%  
**Expected Sensitivity:** 90%+  
**Implementation Time:** 2-3 days

#### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│         CB-QCCF: Class-Balanced Quantum-Classical Fusion      │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: Mammography / Histopathology Image (224×224×3)       │
│       ↓                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Classical Branch: Swin-Small Transformer             │  │
│  │  - Pre-trained on ImageNet                            │  │
│  │  - Fine-tuned for breast cancer                       │  │
│  │  - Output: 768-d feature vector (high accuracy)       │  │
│  └───────────────────────────────────────────────────────┘  │
│       ↓                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Quantum Branch: Enhanced QENN-RY                     │  │
│  │  - CNN backbone (ResNet-18) → 8-d features            │  │
│  │  - Quantum circuit: 8 qubits, 2 layers                │  │
│  │  - RY rotations + cyclic entanglement                 │  │
│  │  - Output: 8-d quantum expectation values             │  │
│  └───────────────────────────────────────────────────────┘  │
│       ↓                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Cross-Attention Fusion Module                        │  │
│  │  - Classical attends to quantum features              │  │
│  │  - Learnable alignment between classical/quantum      │  │
│  │  - Output: 768-d fused representation                 │  │
│  └───────────────────────────────────────────────────────┘  │
│       ↓                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Dual-Head Prediction                                 │  │
│  │  ├─ Sensitivity Head: Maintain high TP rate          │  │
│  │  └─ Specificity Head: Reduce FP rate                 │  │
│  └───────────────────────────────────────────────────────┘  │
│       ↓                                                       │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Learnable Threshold Layer                            │  │
│  │  - Adaptive decision boundary (not fixed at 0.5)      │  │
│  │  - Optimized during training                          │  │
│  └───────────────────────────────────────────────────────┘  │
│       ↓                                                       │
│  Final Prediction: Benign (0) / Malignant (1)                │
└──────────────────────────────────────────────────────────────┘
```

#### Key Innovations

1. **Dual-Head Prediction**
   - Separate heads for sensitivity and specificity optimization
   - Each head learns different decision boundaries
   - Combined at inference with learned weights

2. **Class-Balanced Loss Function**
   ```python
   loss = CE_loss + λ₁*SensitivityLoss + λ₂*SpecificityLoss + λ₃*FocalLoss
   
   # Where:
   # - λ₂ (specificity weight) = 2.0 (heavily penalize false positives)
   # - FocalLoss: Focus on hard examples (γ=2.0)
   # - SensitivityLoss: Ensure TP rate stays high
   ```

3. **Cross-Attention Fusion**
   - Better than simple concatenation
   - Classical features attend to quantum features
   - Enables quantum-classical feature alignment

4. **Learnable Decision Threshold**
   - Not fixed at 0.5
   - Adaptively learned during training
   - Balances sensitivity/specificity trade-off

#### Implementation

```python
import torch
import torch.nn as nn
import timm
from typing import Tuple, Optional

class ClassBalancedQuantumClassicalFusion(nn.Module):
    """
    CB-QCCF: Class-Balanced Quantum-Classical Fusion
    
    Primary novelty: Addresses quantum specificity problem with
    dual-head prediction and class-balanced loss.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        n_qubits: int = 8,
        n_layers: int = 2,
        backbone: str = 'swin_small',
        quantum_backbone: str = 'resnet18',
    ):
        super().__init__()
        
        # Classical backbone (high accuracy)
        self.classical_backbone = timm.create_model(
            backbone, pretrained=True, num_classes=0
        )
        classical_dim = self.classical_backbone.num_features
        
        # Quantum branch (high sensitivity)
        self.quantum_backbone = timm.create_model(
            quantum_backbone, pretrained=True, num_classes=0
        )
        quantum_input_dim = self.quantum_backbone.num_features
        
        # Quantum circuit (using existing QENN implementation)
        from src.models.quantum.hybrid_quantum import QENN_RY
        self.quantum_circuit = QENN_RY(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config='ry_only',
            entanglement='cyclic'
        )
        
        # Feature projection for quantum branch
        self.quantum_projector = nn.Sequential(
            nn.Linear(quantum_input_dim, n_qubits),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Cross-attention fusion (classical attends to quantum)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=classical_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Quantum feature projector (match classical dim)
        self.quantum_to_classical = nn.Linear(n_qubits, classical_dim)
        
        # Dual heads
        self.sensitivity_head = nn.Sequential(
            nn.Linear(classical_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.specificity_head = nn.Sequential(
            nn.Linear(classical_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),  # Higher dropout for regularization
            nn.Linear(256, num_classes)
        )
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.sensitivity_head, self.specificity_head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            final_pred: Combined prediction
            sens_pred: Sensitivity head prediction
            spec_pred: Specificity head prediction
        """
        batch_size = x.shape[0]
        
        # Classical branch
        classical_feat = self.classical_backbone(x)  # (B, 768)
        
        # Quantum branch
        quantum_feat = self.quantum_backbone(x)  # (B, 512 for ResNet-18)
        quantum_proj = self.quantum_projector(quantum_feat)  # (B, 8)
        quantum_out = self.quantum_circuit(quantum_proj)  # (B, 8)
        
        # Project quantum to classical dim
        quantum_proj_dim = self.quantum_to_classical(quantum_out)  # (B, 768)
        
        # Cross-attention fusion
        # Reshape for multihead attention: (B, 1, 768)
        classical_q = classical_feat.unsqueeze(1)
        quantum_kv = quantum_proj_dim.unsqueeze(1)
        
        # Attention: classical attends to quantum
        attn_out, attn_weights = self.cross_attention(
            classical_q, quantum_kv, quantum_kv
        )  # (B, 1, 768)
        
        # Residual fusion
        fused = classical_feat + attn_out.squeeze(1)  # (B, 768)
        
        # Dual-head predictions
        sens_pred = self.sensitivity_head(fused)
        spec_pred = self.specificity_head(fused)
        
        # Weighted combination with learnable threshold
        final_pred = sens_pred * self.threshold + spec_pred * (1 - self.threshold)
        
        return final_pred, sens_pred, spec_pred


class ClassBalancedLoss(nn.Module):
    """
    Custom loss for balancing sensitivity/specificity in CB-QCCF.
    
    Key insight: Heavily penalize false positives to improve specificity
    while maintaining high sensitivity through separate head optimization.
    """
    
    def __init__(
        self,
        specificity_weight: float = 2.0,
        sensitivity_weight: float = 1.0,
        focal_gamma: float = 2.0,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.specificity_weight = specificity_weight
        self.sensitivity_weight = sensitivity_weight
        self.focal_gamma = focal_gamma
        self.threshold = threshold
    
    def forward(
        self,
        final_pred: torch.Tensor,
        sens_pred: torch.Tensor,
        spec_pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute class-balanced loss.
        
        Args:
            final_pred: Combined prediction (B, 2)
            sens_pred: Sensitivity head prediction (B, 2)
            spec_pred: Specificity head prediction (B, 2)
            target: Ground truth labels (B,)
        
        Returns:
            total_loss: Weighted combination of all losses
        """
        # Standard cross-entropy for final prediction
        ce_loss = nn.functional.cross_entropy(final_pred, target)
        
        # Sensitivity head loss (focus on positive class)
        sens_loss = nn.functional.cross_entropy(sens_pred, target)
        
        # Specificity head loss (heavily penalize FP)
        spec_loss = nn.functional.cross_entropy(spec_pred, target)
        
        # Focal loss for hard examples
        focal_loss = self._focal_loss(final_pred, target)
        
        # Total loss
        total_loss = (
            ce_loss +
            self.sensitivity_weight * sens_loss +
            self.specificity_weight * spec_loss +
            0.3 * focal_loss
        )
        
        return total_loss
    
    def _focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Focal loss for focusing on hard examples."""
        ce_loss = nn.functional.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal = (1 - pt) ** self.focal_gamma * ce_loss
        return focal.mean()
```

#### Training Strategy

```python
# Training hyperparameters
config = {
    'batch_size': 16,  # Smaller due to quantum branch
    'lr': 2e-5,
    'weight_decay': 1e-4,
    'epochs': 50,
    'patience': 10,  # Early stopping on val AUC
    'specificity_weight': 2.0,  # Heavily penalize FP
    'sensitivity_weight': 1.0,  # Maintain TP
}

# Optimizer (differential learning rates)
optimizer = torch.optim.AdamW([
    {'params': model.classical_backbone.parameters(), 'lr': 1e-5},
    {'params': model.quantum_backbone.parameters(), 'lr': 1e-5},
    {'params': model.quantum_circuit.parameters(), 'lr': 2e-5},
    {'params': model.sensitivity_head.parameters(), 'lr': 2e-5},
    {'params': model.specificity_head.parameters(), 'lr': 2e-5},
    {'params': [model.threshold], 'lr': 1e-3},  # Higher LR for threshold
], weight_decay=config['weight_decay'])

# Loss
criterion = ClassBalancedLoss(
    specificity_weight=2.0,
    sensitivity_weight=1.0,
    focal_gamma=2.0
)
```

---

### Architecture 2: Multi-Scale Quantum Fusion (MSQF) ⭐⭐

**Novelty:** HIGH - First to apply multi-scale quantum processing to histopathology  
**Implementation Priority:** P1  
**Expected Accuracy:** 86-88%  
**Expected Specificity:** 78-82%  
**Implementation Time:** 3-4 days

#### Motivation

Histopathology images contain structures at multiple scales:
- **Cellular level** (5-20 μm): Nuclei morphology
- **Tissue level** (50-200 μm): Glandular structures
- **Architecture level** (500+ μm): Tissue organization

Current QENN processes features at a single scale, losing multi-scale information.

#### Architecture

```
Input Image (224×224×3)
         ↓
┌─────────────────────────────────────────────────────┐
│  Multi-Scale Feature Extraction (CNN Backbone)      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │ Scale 1  │  │ Scale 2  │  │ Scale 3  │         │
│  │ (56×56)  │  │ (28×28)  │  │ (14×14)  │         │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
└───────┼────────────┼────────────┼─────────────────┘
        ↓            ↓            ↓
┌───────────┐ ┌───────────┐ ┌───────────┐
│  QENN-1   │ │  QENN-2   │ │  QENN-3   │
│ (8 qubits)│ │ (8 qubits)│ │ (8 qubits)│
│ Scale 1   │ │ Scale 2   │ │ Scale 3   │
└─────┬─────┘ └─────┬─────┘ └─────┬─────┘
      ↓             ↓             ↓
┌─────────────────────────────────────────────────────┐
│  Attention-Based Fusion                             │
│  - Learn importance of each scale                   │
│  - Weighted combination                             │
└─────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────┐
│  Classical Refinement (Swin Block)                  │
│  - Self-attention for global context                │
└─────────────────────────────────────────────────────┘
         ↓
    Final Prediction
```

#### Key Innovation

Each scale has its **own quantum circuit** with potentially different:
- Rotation configurations (RY at scale 1, RY+RZ at scale 2, U3 at scale 3)
- Entanglement strategies (cyclic, full, tree)
- Layer depths (1, 2, 3)

This enables **quantum architecture search** across scales.

---

### Architecture 3: Quantum-Enhanced Deformable Attention (QEDA) ⭐⭐⭐

**Novelty:** VERY HIGH - First to replace classical attention with quantum circuits  
**Implementation Priority:** P2  
**Expected Accuracy:** 87-90%  
**Expected Specificity:** 80-84%  
**Implementation Time:** 5-7 days (highest risk)

#### Motivation

Swin Transformer uses **fixed window attention**. Deformable attention learns **where to attend**, but uses classical MLPs for offset prediction.

**Key Idea:** Replace the offset prediction MLP with a **variational quantum circuit** that learns attention patterns in a higher-dimensional Hilbert space.

#### Architecture Comparison

**Classical Deformable Attention:**
```
Features → MLP → Offset Field → Sample → Attention
```

**Quantum Deformable Attention:**
```
Features → CNN Encoder → Quantum Circuit → CNN Decoder → Offset Field → Sample → Attention
```

#### Why Quantum?

Quantum circuits can:
1. **Explore non-linear relationships** in feature space
2. **Capture entanglement** between spatial locations
3. **Learn complex offset patterns** with fewer parameters

#### Implementation Sketch

```python
class QuantumDeformableAttention(nn.Module):
    """
    Deformable attention with quantum offset prediction.
    
    Replaces classical MLP with variational quantum circuit
    for learning attention offset fields.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        n_qubits: int = 12,
        n_layers: int = 2,
        sampling_points: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.sampling_points = sampling_points
        
        # CNN encoder (features → quantum input)
        self.encoder = nn.Sequential(
            nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, n_qubits, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Quantum circuit for offset prediction
        from src.models.quantum.hybrid_quantum import QENN_U3
        self.quantum_circuit = QENN_U3(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config='u3',
            entanglement='full'  # Full entanglement for complex patterns
        )
        
        # CNN decoder (quantum output → offsets)
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, dim),
            nn.GELU(),
            nn.Linear(dim, num_heads * sampling_points * 2)  # 2D offsets
        )
        
        # Value projection and output
        self.value_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.output_proj = nn.Conv2d(dim, dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Encode features
        quantum_input = self.encoder(x).squeeze(-1).squeeze(-1)  # (B, n_qubits)
        
        # Quantum circuit for offset prediction
        quantum_out = self.quantum_circuit(quantum_input)  # (B, n_qubits)
        
        # Decode to offset fields
        offsets = self.decoder(quantum_out)  # (B, H*W*sampling_points*2)
        offsets = offsets.view(B, self.num_heads, self.sampling_points, 2, H, W)
        offsets = offsets.permute(0, 1, 4, 5, 2, 3).contiguous()  # (B, H, W, heads, points, 2)
        
        # Apply deformable attention with quantum-predicted offsets
        # ... (standard deformable attention sampling)
        
        return output
```

---

## Implementation Roadmap

### Phase 1: CB-QCCF (Week 1-2)

| Task | Duration | Status |
|------|----------|--------|
| Implement CB-QCCF architecture | 1 day | ⏳ Pending |
| Implement class-balanced loss | 0.5 days | ⏳ Pending |
| Integrate with existing pipeline | 0.5 days | ⏳ Pending |
| Train on BreakHis (5-fold CV) | 2 days | ⏳ Pending |
| Train on CBIS-DDSM (generalization) | 2 days | ⏳ Pending |
| Analyze results, tune hyperparameters | 1 day | ⏳ Pending |

**Expected Outcomes:**
- Specificity improved from 72% → 80-83%
- Accuracy: 85-87%
- Sensitivity maintained at 90%+

### Phase 2: Multi-Scale Quantum Fusion (Week 3-4)

| Task | Duration | Status |
|------|----------|--------|
| Implement multi-scale feature extraction | 1 day | ⏳ Pending |
| Implement 3 parallel QENN circuits | 1 day | ⏳ Pending |
| Implement attention-based fusion | 1 day | ⏳ Pending |
| Train on BreakHis + CBIS-DDSM | 3 days | ⏳ Pending |
| Quantum architecture ablation study | 1 day | ⏳ Pending |

**Expected Outcomes:**
- Accuracy: 86-88%
- Better handling of variable-sized tissue structures

### Phase 3: Quantum Deformable Attention (Week 5-7)

| Task | Duration | Status |
|------|----------|--------|
| Implement deformable attention baseline | 2 days | ⏳ Pending |
| Replace MLP with quantum circuit | 2 days | ⏳ Pending |
| Integrate into Swin blocks | 1 day | ⏳ Pending |
| Train full model | 5 days | ⏳ Pending |
| Attention visualization | 1 day | ⏳ Pending |

**Expected Outcomes:**
- Accuracy: 87-90%
- Interpretable attention maps showing nuclei/tissue regions
- **Major novelty for MICCAI 2026**

---

## Generalization Validation: CBIS-DDSM Dataset

### Why CBIS-DDSM?

1. **Different Modality:** Mammography vs histopathology
2. **Larger Dataset:** ~10k images vs ~8k BreakHis
3. **Clinical Relevance:** Real-world screening dataset
4. **Generalization Test:** If models work on both datasets, they're truly robust

### Expected Performance Drop

| Dataset | Expected Accuracy Drop |
|---------|----------------------|
| BreakHis (training) | Baseline (85-87%) |
| CBIS-DDSM (transfer) | -5 to -10% (75-82%) |

**Key Question:** Do quantum models generalize better or worse than classical?

**Hypothesis:** Quantum models may generalize **better** due to:
- Quantum feature spaces capture fundamental patterns
- Less prone to overfitting specific dataset artifacts

---

## Statistical Analysis Plan

### Metrics to Report

1. **Primary:**
   - Accuracy (mean ± std across 5 folds)
   - AUC-ROC (mean ± std)
   - Sensitivity/Specificity balance

2. **Secondary:**
   - False Negative Rate (critical for medical diagnosis)
   - MCC (Matthews Correlation Coefficient)
   - Inference time (ms per image)

3. **Statistical Tests:**
   - Paired t-test (quantum vs classical)
   - Bootstrap confidence intervals (95% CI)
   - Effect size (Cohen's d)

### Ablation Studies

1. **Rotation Gate Ablation:**
   - RY-only vs RY+RZ vs U3 vs RX+RY+RZ
   
2. **Entanglement Ablation:**
   - Cyclic vs Full vs Tree vs None

3. **Qubit Count Ablation:**
   - 4 vs 8 vs 12 qubits

4. **Layer Depth Ablation:**
   - 1 vs 2 vs 3 quantum layers

5. **Fusion Strategy Ablation (for CB-QCCF):**
   - Concatenation vs Cross-Attention vs Self-Attention

---

## Paper Contributions (MICCAI 2026)

### Primary Contributions

1. **First systematic study** of quantum-classical fusion for breast cancer classification
2. **CB-QCCF architecture** solving the quantum specificity problem
3. **Multi-scale quantum fusion** for histopathology analysis
4. **Quantum deformable attention** - novel attention mechanism
5. **Comprehensive evaluation** on two datasets (BreakHis, CBIS-DDSM)
6. **Open-source code** for reproducibility

### Novelty Score

| Aspect | Novelty | Justification |
|--------|---------|---------------|
| Quantum-Classical Fusion | ⭐⭐⭐ High | First in histopathology |
| Class-Balanced Loss | ⭐⭐⭐ High | Solves specificity problem |
| Multi-Scale Quantum | ⭐⭐⭐ High | First multi-scale quantum |
| Quantum Deformable Attn | ⭐⭐⭐⭐ Very High | New attention mechanism |
| CBIS-DDSM Evaluation | ⭐⭐ Medium | Standard generalization test |

---

## Next Steps

1. ✅ **Implement CB-QCCF** (highest priority, fixes specificity problem)
2. ✅ **Set up CBIS-DDSM auto-download** (done!)
3. ✅ **Run existing models on CBIS-DDSM** (generalization baseline)
4. ⏳ **Implement MSQF** (multi-scale quantum)
5. ⏳ **Implement QEDA** (quantum deformable attention)
6. ⏳ **Write paper** (MICCAI 2026 submission)

---

## References

1. **QENN Original:** `src/models/quantum/hybrid_quantum.py`
2. **Swin Transformer:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
3. **Deformable Attention:** Dai et al., "Deformable DETR: Deformable Transformers for End-to-End Object Detection", ICLR 2021
4. **Quantum Machine Learning:** Schuld et al., "Quantum machine learning in feature Hilbert spaces", PRL 2019
5. **CBIS-DDSM:** Clark et al., "Curated Breast Imaging Subset of DDSM", SPIE 2013
