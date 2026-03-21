# Proposed Novel Architectures for Breast Cancer Classification

**Date:** March 21, 2026  
**Goal:** Push accuracy from 85.67% → 88-90% while maintaining novelty

---

## Current Best Results (BreakHis Dataset)

| Rank | Model | Accuracy | AUC-ROC | Sensitivity | Specificity | FNR |
|------|-------|----------|---------|-------------|-------------|-----|
| 1 | **Swin-Small** | **85.67%** | 88.52% | 91.15% | 77.45% | 8.85% |
| 2 | **CNN+ViT-Hybrid** | **84.67%** | **89.25%** | 88.49% | 78.92% | 11.51% |
| 3 | **EfficientNet-B3** | 83.37% | **89.97%** | 86.10% | 79.26% | 13.90% |
| 4 | **QENN-RY_ONLY** | 84.17% | 88.40% | **91.68%** | 72.88% | 8.32% |

### Key Problems to Address

1. **Specificity Gap:** Top models have specificity ~77-79% (need >85%)
2. **Quantum Imbalance:** QENN models have 91%+ sensitivity but only 72% specificity
3. **Fusion Underperformance:** DualBranch-Fusion (83.68%) < Swin-Small (85.67%)

---

## Architecture 1: Triple-Branch Cross-Attention Fusion (TBCA-Fusion) ⭐

**Novelty Level:** Medium  
**Implementation Effort:** 2-3 days  
**Expected Accuracy:** 87-89%

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    TBCA-Fusion Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Branch 1: Swin-Small (Global Context)                      │
│         ↓                                                    │
│         ├─→ [Cross-Attention] ←─ Branch 2 (ConvNeXt)        │
│         ├─→ [Cross-Attention] ←─ Branch 3 (EfficientNet)    │
│                                                            │
│  Branch 2: ConvNeXt-Small (Local Texture)                   │
│         ↓                                                    │
│         ├─→ [Cross-Attention] ←─ Branch 1 (Swin)            │
│                                                            │
│  Branch 3: EfficientNet-B3 (Multi-scale Features)           │
│         ↓                                                    │
│         ├─→ [Cross-Attention] ←─ Branch 1 (Swin)            │
│                                                              │
│  Fusion: [Weighted Sum + Self-Attention] → Classifier       │
└─────────────────────────────────────────────────────────────┘
```

### Key Innovations

- **Bidirectional cross-attention** between all three branches
- **Learnable branch weights** (not just simple gating)
- **EfficientNet adds multi-scale features** missing in current fusion
- **Self-attention refinement** after fusion

### Implementation Details

```python
class TripleBranchCrossAttention(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Three backbones
        self.swin_branch = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.convnext_branch = timm.create_model('convnext_small.in12k', pretrained=True)
        self.efficientnet_branch = timm.create_model('efficientnet_b3', pretrained=True)
        
        # Cross-attention modules (bidirectional)
        self.swin_to_convnext = CrossAttention(dim=768, num_heads=8)
        self.swin_to_effnet = CrossAttention(dim=768, num_heads=8)
        self.convnext_to_swin = CrossAttention(dim=768, num_heads=8)
        self.effnet_to_swin = CrossAttention(dim=768, num_heads=8)
        
        # Fusion layer
        self.fusion_attention = SelfAttention(dim=768, num_heads=8)
        self.branch_weights = nn.Parameter(torch.ones(3))  # Learnable weights
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features from all branches
        feat_swin = self.swin_branch.forward_features(x)
        feat_conv = self.convnext_branch.forward_features(x)
        feat_eff = self.efficientnet_branch.forward_features(x)
        
        # Cross-attention enhancement
        swin_enhanced = self.swin_to_convnext(feat_swin, feat_conv) + \
                       self.swin_to_effnet(feat_swin, feat_eff)
        conv_enhanced = self.convnext_to_swin(feat_conv, feat_swin)
        eff_enhanced = self.effnet_to_swin(feat_eff, feat_swin)
        
        # Weighted fusion
        weights = F.softmax(self.branch_weights, dim=0)
        fused = weights[0] * swin_enhanced + \
                weights[1] * conv_enhanced + \
                weights[2] * eff_enhanced
        
        # Self-attention refinement
        fused_refined = self.fusion_attention(fused)
        
        # Global average pooling + classification
        pooled = fused_refined.mean(dim=1)
        return self.classifier(pooled)
```

---

## Architecture 2: Swin-ConvNeXt Hybrid with Attention Pooling (SCHAP)

**Novelty Level:** Medium-High  
**Implementation Effort:** 3-4 days  
**Expected Accuracy:** 86-88%  
**Expected Specificity:** 80-82%

### Concept

Merge Swin and ConvNeXt at the **block level**, not just output fusion.

### Architecture

```python
class HybridBlock(nn.Module):
    """Single block combining Swin + ConvNeXt with attention pooling"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # Swin path: Window attention
        self.swin_block = SwinBlock(
            dim=dim,
            num_heads=num_heads,
            window_size=7
        )
        # ConvNeXt path: Large kernel convolutions
        self.convnext_block = ConvNeXtBlock(
            dim=dim,
            kernel_size=7
        )
        # Cross-attention for feature exchange
        self.cross_attn = CrossAttention(dim, num_heads)
        # Attention pooling
        self.attention_pool = AttentionPool(dim)
    
    def forward(self, x):
        # Parallel processing
        swin_out = self.swin_block(x)
        conv_out = self.convnext_block(x)
        
        # Cross-attention exchange
        swin_enhanced = self.cross_attn(swin_out, conv_out)
        conv_enhanced = self.cross_attn(conv_out, swin_out)
        
        # Attention-weighted fusion
        return self.attention_pool(swin_enhanced, conv_enhanced)


class SCHAP(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 4 stages with hybrid blocks
        self.stage1 = nn.Sequential(*[HybridBlock(96) for _ in range(2)])
        self.stage2 = nn.Sequential(*[HybridBlock(192) for _ in range(2)])
        self.stage3 = nn.Sequential(*[HybridBlock(384) for _ in range(6)])
        self.stage4 = nn.Sequential(*[HybridBlock(768) for _ in range(2)])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, num_classes)
        )
```

---

## Architecture 3: Class-Balanced Quantum-Classical Fusion (CB-QCCF) ⭐⭐

**Novelty Level:** **HIGH** (Quantum-focused)  
**Implementation Effort:** 2-3 days  
**Expected Accuracy:** 85-87%  
**Expected Specificity:** 80-83%  
**Expected Sensitivity:** 90%+

### Motivation

Current QENN models have **92% sensitivity but only 72% specificity** - they over-predict malignancy. This architecture fixes that imbalance.

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────┐
│              CB-QCCF Architecture                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Input Image                                                  │
│       ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Classical Branch: Swin-Small (High Accuracy)        │    │
│  └──────────────────────────────────────────────────────┘    │
│       ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Quantum Branch: QENN-RY (High Sensitivity)          │    │
│  └──────────────────────────────────────────────────────┘    │
│       ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Cross-Attention Fusion Module                        │    │
│  └──────────────────────────────────────────────────────┘    │
│       ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Dual-Head Prediction                                 │    │
│  │  ├─ Sensitivity Head (Maintain high TP)              │    │
│  │  └─ Specificity Head (Reduce FP)                     │    │
│  └──────────────────────────────────────────────────────┘    │
│       ↓                                                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Learnable Threshold Layer (Balance predictions)     │    │
│  └──────────────────────────────────────────────────────┘    │
│       ↓                                                       │
│  Final Prediction                                              │
└──────────────────────────────────────────────────────────────┘
```

### Key Innovations

1. **Dual-Head Prediction:** Separate heads for sensitivity/specificity optimization
2. **Class-Balanced Loss:** Explicitly penalize false positives
3. **Learnable Decision Threshold:** Not fixed at 0.5
4. **Cross-Attention Fusion:** Better than concatenation

### Implementation

```python
class ClassBalancedQuantumClassicalFusion(nn.Module):
    def __init__(self, num_classes=2, n_qubits=8, n_layers=2):
        super().__init__()
        # Classical backbone (high accuracy)
        self.classical_backbone = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        
        # Quantum branch (high sensitivity)
        self.quantum_branch = QENN_RY(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config='ry_only',
            entanglement='cyclic'
        )
        
        # Cross-attention fusion
        self.cross_attn = CrossAttention(dim=768, num_heads=8)
        
        # Dual heads
        self.sensitivity_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        self.specificity_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),  # Higher dropout for regularization
            nn.Linear(256, num_classes)
        )
        
        # Learnable threshold
        self.threshold = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        # Extract features
        classical_feat = self.classical_backbone.forward_features(x)
        quantum_feat = self.quantum_branch(x)
        
        # Cross-attention fusion (classical attends to quantum)
        fused = self.cross_attn(classical_feat, quantum_feat)
        pooled = fused.mean(dim=1)
        
        # Dual-head predictions
        sens_pred = self.sensitivity_head(pooled)
        spec_pred = self.specificity_head(pooled)
        
        # Weighted combination with learnable threshold
        final_pred = sens_pred * self.threshold + spec_pred * (1 - self.threshold)
        
        return final_pred, sens_pred, spec_pred


class ClassBalancedLoss(nn.Module):
    """Custom loss for balancing sensitivity/specificity"""
    def __init__(self, specificity_weight=2.0):
        super().__init__()
        self.specificity_weight = specificity_weight
    
    def forward(self, pred, sens_pred, spec_pred, target):
        # Standard cross-entropy
        ce_loss = F.cross_entropy(pred, target)
        
        # Sensitivity head loss (focus on positive class)
        sens_loss = F.cross_entropy(sens_pred, target)
        
        # Specificity head loss (heavily penalize FP)
        spec_loss = F.cross_entropy(spec_pred, target)
        
        # Focal loss for hard examples
        focal_loss = self.focal_loss(pred, target)
        
        # Total loss
        total_loss = ce_loss + 0.5 * sens_loss + \
                    self.specificity_weight * spec_loss + 0.3 * focal_loss
        
        return total_loss
    
    def focal_loss(self, pred, target, gamma=2.0):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        return (1 - pt) ** gamma * ce_loss
```

### Loss Function

```python
loss = CE_loss + λ₁*SensitivityLoss + λ₂*SpecificityLoss + λ₃*FocalLoss + λ₄*BalanceLoss

# Where:
# - λ₂ (specificity weight) = 2.0 (heavily penalize false positives)
# - FocalLoss: Focus on hard examples
# - BalanceLoss: Penalize threshold deviation from optimal
```

---

## Architecture 4: Multi-Scale Swin with Deformable Attention (MSS-DA)

**Novelty Level:** High  
**Implementation Effort:** 4-5 days  
**Expected Accuracy:** 86-88%  
**Expected Specificity:** 80-82%

### Motivation

Standard Swin uses **fixed window attention**. Histopathology images have **variable-sized nuclei and tissue structures**. Deformable attention can adapt to irregular shapes.

### Architecture

```python
class DeformableSwinBlock(nn.Module):
    """Swin block with deformable attention for irregular tissue structures"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        # Multi-scale pyramid pooling
        self.pyramid_pooling = PyramidPooling(dim, scales=[1, 2, 4, 8])
        
        # Deformable attention (learn where to attend)
        self.deformable_attn = DeformableAttention(
            dim=dim,
            num_heads=num_heads,
            offset_groups=4,
            sampling_points=16
        )
        
        # Offset predictor (learn offset fields)
        self.offset_predictor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 2 * offset_groups, kernel_size=1)
        )
    
    def forward(self, x):
        # Extract multi-scale features
        scales = self.pyramid_pooling(x)
        
        # Predict offset fields
        offsets = self.offset_predictor(x)
        
        # Apply deformable attention
        out = self.deformable_attn(x, offsets, scales)
        
        return out + x  # Residual connection


class MultiScaleSwinDeformable(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # 4 stages with deformable Swin blocks
        self.stage1 = nn.Sequential(*[DeformableSwinBlock(96) for _ in range(2)])
        self.stage2 = nn.Sequential(*[DeformableSwinBlock(192) for _ in range(2)])
        self.stage3 = nn.Sequential(*[DeformableSwinBlock(384) for _ in range(6)])
        self.stage4 = nn.Sequential(*[DeformableSwinBlock(768) for _ in range(2)])
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
```

### Why It Works

- **Deformable attention** can focus on irregular nuclei boundaries
- **Multi-scale pooling** captures both cellular and tissue-level features
- **Interpretable:** Attention maps show which regions (nuclei) were important

---

## Architecture 5: Teacher-Student Distillation with Ensemble Teachers (TSD-Ensemble)

**Novelty Level:** Medium  
**Implementation Effort:** 1-2 days (easiest!)  
**Expected Accuracy:** 87-89%  
**Expected Specificity:** 78-82%

### Motivation

Single models plateau at ~85%. **Knowledge distillation** from an ensemble often beats individual models.

### Architecture Diagram

```
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
              │  Student: Swin-Small │
              │  + ConvNeXt fusion   │
              │  (Trainable)         │
              └──────────────────────┘

Loss: KL(student, ensemble) + α*CE(student, labels)
```

### Implementation

```python
class EnsembleDistillation(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Student network (trainable)
        self.student = SCHAP(num_classes)  # Or TBCA-Fusion
        
        # Teacher networks (frozen)
        self.teacher_swin = timm.create_model('swin_small_patch4_window7_224', pretrained=True)
        self.teacher_hybrid = CNNViTHybrid(num_classes)
        self.teacher_effnet = timm.create_model('efficientnet_b3', pretrained=True)
        
        # Freeze teachers
        for teacher in [self.teacher_swin, self.teacher_hybrid, self.teacher_effnet]:
            for param in teacher.parameters():
                param.requires_grad = False
        
        # Temperature for soft targets
        self.temperature = 4.0
    
    def forward(self, x):
        # Student prediction
        student_logits = self.student(x)
        
        # Teacher predictions (frozen)
        with torch.no_grad():
            teacher_swin_logits = self.teacher_swin(x)
            teacher_hybrid_logits = self.teacher_hybrid(x)
            teacher_effnet_logits = self.teacher_effnet(x)
            
            # Ensemble average
            ensemble_logits = (teacher_swin_logits + 
                              teacher_hybrid_logits + 
                              teacher_effnet_logits) / 3
        
        return student_logits, ensemble_logits


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, alpha=0.7):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for hard label loss
    
    def forward(self, student_logits, teacher_logits, hard_labels):
        # KL divergence for soft targets (distillation)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Cross-entropy for hard labels (accuracy)
        hard_loss = F.cross_entropy(student_logits, hard_labels)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss
```

---

## Implementation Priority

| Priority | Architecture | Novelty | Effort | Expected Gain |
|----------|-------------|---------|--------|---------------|
| **P0** | **CB-QCCF** (Class-Balanced Quantum) | ⭐⭐⭐ High | 2-3 days | Fix specificity, 85-87% |
| **P1** | **TSD-Ensemble** (Distillation) | ⭐⭐ Medium | 1-2 days | Quickest win, 87-89% |
| **P2** | **TBCA-Fusion** (Triple-Branch) | ⭐⭐ Medium | 2-3 days | Strong baseline, 87-89% |
| **P3** | **MSS-DA** (Deformable Swin) | ⭐⭐⭐ High | 4-5 days | Novel, 86-88% |
| **P4** | **SCHAP** (Hybrid Blocks) | ⭐⭐ Medium-High | 3-4 days | Good specificity, 86-88% |

---

## Next Steps

1. ✅ **Implement CB-QCCF** (quantum-focused, highest novelty)
2. ✅ **Implement TSD-Ensemble** (quickest accuracy gain)
3. ✅ **Validate on CBIS-DDSM** (generalization check)
4. 📊 **Ablation studies** (which components matter most?)
5. 📝 **Write paper** (MICCAI 2026 submission)

---

## Generalization Validation: CBIS-DDSM Dataset

**Dataset:** https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset

**Why CBIS-DDSM?**
- Different modality (mammography vs histopathology)
- Larger dataset (~10k images)
- Both image + CSV metadata
- Tests model generalization across imaging modalities

**Plan:**
1. Add CBIS-DDSM dataloader
2. Auto-download on server
3. Run existing models (Swin, QENN, etc.)
4. Compare performance (expect 5-10% drop, but relative ranking should hold)

---

## References

1. Swin Transformer: https://arxiv.org/abs/2103.14030
2. ConvNeXt: https://arxiv.org/abs/2201.03545
3. Deformable Attention: https://arxiv.org/abs/2010.04159
4. Knowledge Distillation: https://arxiv.org/abs/1503.02531
5. QENN (Our implementation): `src/models/quantum/hybrid_quantum.py`
