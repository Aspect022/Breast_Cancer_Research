# Research Prompt: Transformer & Quantum Architecture Exploration

**For:** Breast Cancer Histopathology Classification (BreakHis Dataset)  
**Date:** 2026-03-19  
**Use Case:** Guide architecture exploration and experimental design for minor project

---

## 📋 Context

You are designing novel deep learning architectures for breast cancer classification from histopathology images (BreakHis dataset, 224×224, binary classification: benign vs malignant).

### Current Baselines
| Model | Accuracy | Training Time (5-fold CV) | Status |
|-------|----------|---------------------------|--------|
| EfficientNet-B3 | 85% | ~10 min | CNN baseline |
| CNN+ViT-Hybrid | 85% | ~9 min | Hybrid baseline |
| ViT-Tiny | 77% | ~8 min | Pure transformer |
| CNN+PQC (PennyLane) | 84% | **13.4 hours** | Quantum hybrid (too slow) |
| SpikingCNN-LIF | 57% | ~23 min | ❌ Clinically unusable (62.8% FNR) |

### Target Goals
- **Accuracy:** ≥88% (3% improvement over baseline)
- **Training Time:** <2× baseline overhead (<20 min for transformer, <2 hours for quantum)
- **Research Output:** Publication-worthy comparison + ablation studies

---

## 🎯 Research Tasks

### Task 1: Transformer Architecture Exploration

Explore and compare these transformer variants for histopathology classification:

#### A) Swin Transformer Variants
**Research Questions:**
- What are the key differences between Swin-Tiny (29M), Swin-Small (50M), and Swin-V2-Small?
- How does **shifted window attention** benefit histopathology image analysis?
- What window size (4, 7, 14) is optimal for 224×224 breast cancer images?
- Should we use ImageNet-1K or ImageNet-21K pretrained weights?

**Literature to Review:**
- "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021)
- "Swin Transformer V2: Scaling Up Capacity and Resolution" (CVPR 2022)
- Recent BreakHis classification papers using Swin (search: "Swin Transformer BreakHis")

**Implementation Plan:**
```python
from timm import create_model

# Variants to test
swin_tiny = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=2)
swin_small = create_model('swin_small_patch4_window7_224', pretrained=True, num_classes=2)
swin_v2 = create_model('swinv2_small_window16_256', pretrained=True, num_classes=2)
```

**Expected Outcomes:**
- Swin-Tiny: ~83-85% accuracy, ~30 min training
- Swin-Small: ~85-87% accuracy, ~45 min training
- Swin-V2: ~87-89% accuracy, ~50 min training

---

#### B) ConvNeXt Variants
**Research Questions:**
- How does ConvNeXt's **convolutional approach** to transformer design compare to attention-based transformers?
- Is ConvNeXt more data-efficient for medical imaging (limited datasets like BreakHis)?
- What stem configuration (patch_size=4 vs 16) works best for histopathology texture features?

**Literature to Review:**
- "A ConvNet for the 2020s" (CVPR 2022) - ConvNeXt original paper
- Medical imaging applications of ConvNeXt

**Implementation Plan:**
```python
convnext_tiny = create_model('convnext_tiny', pretrained=True, num_classes=2)
convnext_small = create_model('convnext_small', pretrained=True, num_classes=2)
```

**Expected Outcomes:**
- ConvNeXt-Tiny: ~83-85% accuracy, ~25 min training (faster than Swin)
- ConvNeXt-Small: ~84-86% accuracy, ~40 min training

---

#### C) DeiT (Data-efficient Image Transformers)
**Research Questions:**
- Does **distillation token** improve performance on BreakHis (limited data scenario)?
- How does DeiT compare to Swin for medical image classification?
- What augmentation strategy works best for DeiT training (RandAugment, Mixup, CutMix)?

**Literature to Review:**
- "Training data-efficient image transformers & distillation through attention" (ICML 2021)
- DeiT applications in medical imaging

**Implementation Plan:**
```python
deit_tiny = create_model('deit_tiny_distilled_patch16_224', pretrained=True, num_classes=2)
deit_small = create_model('deit_small_distilled_patch16_224', pretrained=True, num_classes=2)
deit_base = create_model('deit_base_distilled_patch16_224', pretrained=True, num_classes=2)
```

**Expected Outcomes:**
- DeiT-Tiny: ~78-80% accuracy (data-efficient but limited capacity)
- DeiT-Small: ~82-84% accuracy
- DeiT-Base: ~84-86% accuracy (but slower training)

---

#### D) Medical-Specific Architectures
**Research Questions:**
- What architectures are designed specifically for histopathology/medical imaging?
- Can we adapt their innovations to our dual-branch fusion design?

**Architectures to Search:**
1. **Med-Former** - Local-Global Transformer for medical imaging
2. **CrossViT** - Multi-scale patch sizes with cross-attention fusion
3. **TransPath** - Transformer-based pathology foundation model
4. **HIPT** - Hierarchical Image Pyramid Transformer for histopathology
5. **PVT (Pyramid Vision Transformer)** - Multi-scale feature pyramid

**Search Queries:**
- "Med-Former MICCAI 2024"
- "CrossViT histopathology classification"
- "TransPath pathology transformer"
- "HIPT breast cancer classification"

**Expected Outcomes:**
- Medical-specific architectures may achieve 86-88% accuracy
- Novel fusion strategies applicable to our dual-branch design

---

#### E) Other Transformer Variants (Time Permitting)
- **CvT (Convolutional Vision Transformer)** - Convolutional token embedding
- **PVT V2** - Improved pyramid vision transformer
- **PoolFormer** - Pooling-based architecture (not attention)
- **VAN (Visual Attention Network)** - Large kernel attention

---

### Task 2: Attention Mechanism Analysis

#### A) Self-Attention Types
**Research Questions:**
- **Global attention (ViT)** vs **local attention (Swin windows)** - which is better for histopathology?
- **Multi-scale attention** (feature pyramid + attention) - does it capture both cellular and tissue-level features?
- **Cross-attention** between CNN and transformer features - can this improve fusion?

**Implementation Options:**
```python
# Global attention (standard MHSA)
# Complexity: O(n²·d), captures long-range dependencies
# Best for: Global context, but memory-intensive for high-res

# Windowed attention (Swin)
# Complexity: O(m²·n) where m<<n (window size)
# Best for: High-resolution images, memory efficiency

# Axial attention
# Complexity: O(n·√n·d) - separates height/width attention
# Best for: 2D medical images (CT, MRI, histopathology)

# Linear attention (Performer, Linformer)
# Complexity: O(n·d) - linear scaling
# Best for: Memory-constrained scenarios
```

---

#### B) Attention Visualization
**Research Questions:**
- How to visualize what the model attends to in histopathology images?
- Do attention maps align with pathologist-identified regions of interest (nuclei, mitotic figures)?
- Can attention weights explain model predictions to clinicians?

**Implementation Plan:**
```python
# Grad-CAM for transformer attention
# Rollout attention weights from all layers
# Visualize attention on original image

def visualize_attention(model, image, layer_idx=-1):
    # Extract attention weights from specified layer
    # Average across heads
    # Upsample to image resolution
    # Overlay on original image
    pass
```

**Expected Output:**
- Attention heatmap gallery for all transformer models
- Comparison: Do Swin windows focus on different regions than ViT global attention?
- Clinical validation: Do attention maps highlight malignant features (irregular nuclei, high N/C ratio)?

---

#### C) Efficient Attention
**Research Questions:**
- **Linear attention, Performer, FNet** - do these reduce computation without accuracy loss?
- Is **sparse attention** beneficial for histopathology (focus on tissue regions, ignore background)?

**Tradeoff Analysis:**
| Method | Complexity | Accuracy Drop | Speed Gain |
|--------|------------|---------------|------------|
| Standard MHSA | O(n²·d) | - | 1× |
| Performer | O(n·d) | ~2-3% | 2-3× |
| Linformer | O(n·d) | ~3-5% | 2-3× |
| FNet | O(n log n) | ~5-7% | 3-4× |

---

### Task 3: Fusion Strategy Exploration

#### A) Fusion Timing
**Research Questions:**
- **Early fusion** (concatenate raw features) vs **late fusion** (concatenate predictions)?
- **Multi-stage fusion** (fuse at multiple feature levels) - does this capture complementary information better?

**Fusion Points to Test:**
```
Early Fusion:
  Image → [Swin + ConvNeXt] → Concat features → Classifier

Mid Fusion (Recommended):
  Image → [Swin Stage 1-3] → Concat → [Swin Stage 4 + ConvNeXt] → Classifier

Late Fusion:
  Image → [Swin → Pred1] + [ConvNeXt → Pred2] → Average/Weighted → Final

Multi-Stage Fusion:
  Image → [Swin S1 + ConvNeXt S1] → Fuse → [Swin S2 + ConvNeXt S2] → Fuse → ... → Classifier
```

**Expected Outcomes:**
- Mid-fusion: Best balance of accuracy and complexity
- Late fusion: Simpler but may miss feature-level interactions
- Multi-stage: Most accurate but computationally expensive

---

#### B) Fusion Mechanisms
**Research Questions:**
- **Learned gating** (current design) vs **attention-based weighting** - which is more effective?
- **Transformer-based fusion** (use cross-attention to combine branches) - overkill or optimal?
- **Dynamic convolution** (fusion weights depend on input image) - can this adapt to different tissue patterns?

**Fusion Mechanisms to Implement:**

**1. Gated Fusion (Current Design):**
```python
alpha = sigmoid(Conv(gate_network(concat(F_sw, F_cn))))
F_fused = alpha * F_cn + (1 - alpha) * F_sw
```
- Pros: Simple, interpretable (alpha shows branch preference)
- Cons: Limited expressivity (linear combination only)

**2. Cross-Attention Fusion:**
```python
# Q from Swin, K,V from ConvNeXt
F_fused = CrossAttention(Q=F_sw, K=F_cn, V=F_cn)
```
- Pros: Captures complex interactions
- Cons: More parameters, harder to train

**3. Transformer Fusion Block:**
```python
# Concat features + positional encoding → Transformer encoder
F_cat = concat(F_sw, F_cn)
F_fused = TransformerBlock(F_cat)
```
- Pros: Most expressive
- Cons: Computationally expensive

**4. Dynamic Convolution Fusion:**
```python
# Fusion weights depend on input
weights = ConvNet(concat(F_sw, F_cn))
F_fused = DynamicConv(F_sw, F_cn, weights)
```
- Pros: Input-adaptive fusion
- Cons: Additional complexity

---

#### C) Feature Alignment
**Research Questions:**
- How to handle different spatial resolutions from Swin vs ConvNeXt?
- Should we use 1×1 convolution, bilinear interpolation, or learned alignment?

**Alignment Strategies:**
```python
# Option 1: 1x1 convolution (channel projection only)
F_sw_aligned = Conv1x1(F_sw)  # (B, C_sw, H, W) → (B, C_common, H, W)

# Option 2: Bilinear interpolation (spatial resizing)
F_sw_aligned = interpolate(F_sw, size=F_cn.shape[-2:], mode='bilinear')

# Option 3: Learned alignment (convolutional)
F_sw_aligned = Conv3x3(Conv1x1(F_sw))  # More expressive

# Option 4: Cross-attention alignment
F_sw_aligned = CrossAttention(Q=F_cn, K=F_sw, V=F_sw)
```

**Recommendation:** Start with Option 1 (1×1 conv) + Option 2 (bilinear) combination

---

#### D) Entropy Regularization
**Research Questions:**
- Does encouraging balanced gating (alpha ≈ 0.5) improve generalization?
- What entropy weight (0.001, 0.01, 0.1) is optimal?

**Implementation:**
```python
def entropy_regularization(alpha, weight=0.01):
    # Binary entropy: -[α log(α) + (1-α) log(1-α)]
    eps = 1e-7
    entropy = -(alpha * torch.log(alpha + eps) + 
                (1 - alpha) * torch.log(1 - alpha + eps))
    return weight * entropy.mean()

# Total loss
loss = cross_entropy(logits, targets) + entropy_regularization(alpha)
```

**Hypothesis:** Entropy regularization prevents gate collapse (alpha → 0 or 1), encouraging both branches to contribute

**Ablation Study:**
- No entropy regularization (weight=0)
- Light regularization (weight=0.001)
- Medium regularization (weight=0.01)
- Strong regularization (weight=0.1)

**Expected Outcome:** Medium regularization (0.01) optimal - prevents collapse without forcing artificial balance

---

### Task 4: Quantum Architecture Exploration

#### A) Quantum Encoding Strategies
**Research Questions:**
- **Angle encoding** (current) vs **amplitude encoding** vs **basis encoding** - which is best for histopathology?
- How to best map 224×224×3 image to 8-16 qubits?
- Should quantum layer process **global features** (after CNN compression) or **local patches**?

**Encoding Strategies:**

**1. Angle Encoding (Recommended):**
```python
# Map each feature to rotation angle of a qubit
# N features → N qubits
for i in range(n_qubits):
    qdev.ry(wires=i, angle=x[:, i])
```
- Pros: Simple, shallow circuit, good accuracy
- Cons: Requires many qubits for high-dimensional features
- **Use case:** After CNN compression (1536 → 8 qubits)

**2. Amplitude Encoding:**
```python
# Encode N features in amplitude of log₂(N) qubits
# 256 features → 8 qubits (2^8 = 256)
state = amplitude_encoding(x)  # Mottonen state preparation
```
- Pros: Exponentially fewer qubits
- Cons: Deep circuit (Mottonen state prep), harder to train
- **Use case:** Qubit-constrained scenarios

**3. Basis Encoding:**
```python
# Binary features → computational basis states
# Rarely used for continuous image features
```

**Recommendation:** Use angle encoding (proven in existing CNN+PQC implementation)

---

#### B) Variational Circuit Design
**Research Questions:**
- Compare rotation gates: **RY-only**, **RY+RZ**, **U3**, **RX+RY+RZ** - which achieves best accuracy?
- How many layers (1, 2, 3, 4) balance expressivity vs trainability?
- What entanglement strategy (**ring**, **full**, **tree**) works best for image features?

**Rotation Gate Variants to Test:**

| Variant | Gates per Qubit | Parameters/Layer | Expressivity | Trainability |
|---------|----------------|------------------|--------------|--------------|
| **RY-only** | RY | 1 | Low | High (easy to train) |
| **RY+RZ** | RY, RZ | 2 | Medium | Medium |
| **U3** | U3(θ, φ, λ) | 3 | High | Medium |
| **RX+RY+RZ** | RX, RY, RZ | 3 | High | Low (harder to train) |

**Implementation:**
```python
ROTATION_CONFIGS = {
    'ry_only': {'axes': ['y'], 'n_params': 1},
    'ry_rz': {'axes': ['y', 'z'], 'n_params': 2},
    'u3': {'axes': ['x', 'y', 'z'], 'n_params': 3, 'use_u3': True},
    'rx_ry_rz': {'axes': ['x', 'y', 'z'], 'n_params': 3},
}

class VariationalQuantumCircuit(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2, rotation_config='ry_only'):
        super().__init__()
        config = ROTATION_CONFIGS[rotation_config]
        
        # Trainable parameters
        self.params = nn.ParameterDict()
        for layer in range(n_layers):
            for axis_idx, axis in enumerate(config['axes']):
                self.params[f'theta_{layer}_{axis}'] = nn.Parameter(
                    torch.randn(n_qubits) * 0.1
                )
    
    def forward(self, x):
        # x: (batch, n_qubits) - angle encoded features
        # Apply encoding + variational layers
        pass
```

**Circuit Depth Ablation:**
- 1 layer: Fastest, may underfit
- 2 layers: Recommended starting point
- 3 layers: More expressive, risk of barren plateaus
- 4+ layers: Likely too deep for current quantum hardware simulation

**Entanglement Strategies:**

**1. Cyclic CNOT Ring (Recommended):**
```
q0 → q1 → q2 → q3 → q0
```
- Pros: Simple, proven in HQCNN paper
- Cons: Limited connectivity

**2. Full Entanglement (All-to-All):**
```
Every qubit connected to every other qubit
```
- Pros: Maximum connectivity
- Cons: O(n²) CNOT gates, deep circuit

**3. Tree Entanglement:**
```
q0 → q1, q0 → q2, q0 → q3, ...
```
- Pros: Shallow depth (log n)
- Cons: Central qubit bottleneck

**Recommendation:** Start with cyclic CNOT ring (proven effective, shallow circuit)

---

#### C) Quantum Layer Placement
**Research Questions:**
- **Quantum as bottleneck** (after CNN compression) - current approach
- **Quantum as feature enhancer** (parallel to classical path) - does this help?
- **Quantum in fusion layer** (combine Swin+ConvNeXt features quantumly) - novel approach?

**Placement Options:**

**1. Quantum as Bottleneck (Current):**
```
Image → CNN → Compress to 8-dim → Quantum Circuit (8 qubits) → Classifier
```
- Pros: Minimal qubit requirement, proven approach
- Cons: Quantum layer may be too restrictive

**2. Quantum as Feature Enhancer:**
```
Image → CNN → Features
            ↓
    Classical Path: FC → Features_classical
    Quantum Path:   Quantum → Features_quantum
            ↓
    Concat → Classifier
```
- Pros: Quantum adds complementary information
- Cons: More complex, requires careful balancing

**3. Quantum in Fusion Layer:**
```
Image → [Swin → F_sw] + [ConvNeXt → F_cn]
                    ↓
            Concat → Quantum Circuit → Fused Features
                    ↓
            Classifier
```
- Pros: Novel contribution, quantum learns optimal fusion
- Cons: Most complex, untested approach

**Recommendation:** Implement Option 1 first (bottleneck), then explore Option 3 (fusion) if time permits

---

#### D) Hybrid Training Strategies
**Research Questions:**
- **Gradient-based training** (backprop through quantum circuit) vs **gradient-free** (SPSA) - which is better?
- Should quantum parameters be **shared** across samples or **sample-specific**?

**Training Approaches:**

**1. Gradient-Based (Recommended):**
```python
# TorchQuantum / custom vectorized: full backprop support
loss.backward()  # Gradients flow through quantum circuit
optimizer.step()
```
- Pros: Efficient, standard optimization
- Cons: Requires differentiable quantum simulation

**2. Gradient-Free (SPSA):**
```python
# Simultaneous Perturbation Stochastic Approximation
# Estimate gradient from function evaluations
```
- Pros: Works with non-differentiable circuits
- Cons: Slower convergence, noisy gradients

**Recommendation:** Use gradient-based training (TorchQuantum or custom vectorized implementation)

---

### Task 5: Research Positioning

#### A) Novel Contribution Definition

**Option 1: Dual-Branch Dynamic Fusion (Primary)**
- **Contribution:** First application of Swin + ConvNeXt dynamic fusion to breast cancer histopathology
- **Novelty:** Learned gating mechanism with entropy regularization
- **Claim:** "We propose a novel dual-branch architecture that dynamically balances global attention (Swin) and local texture features (ConvNeXt) for improved breast cancer classification"

**Option 2: GPU-Accelerated Quantum Layer (Secondary)**
- **Contribution:** First GPU-accelerated quantum-classical hybrid for histopathology (100× speedup over PennyLane)
- **Novelty:** Comprehensive quantum ablation study (rotation gates, entanglement strategies)
- **Claim:** "We demonstrate that GPU-accelerated quantum simulation enables practical training of quantum-classical hybrids, reducing training time from 13.4 hours to <2 hours"

**Option 3: Comprehensive Transformer Comparison (Tertiary)**
- **Contribution:** Systematic comparison of 10+ transformer architectures on BreakHis
- **Novelty:** First benchmark of SwinV2, ConvNeXt, DeiT, Med-Former on breast cancer histopathology
- **Claim:** "We provide the most comprehensive transformer benchmark for breast cancer classification, establishing new state-of-the-art baselines"

**Recommended Positioning:**
- **Primary:** Option 1 (dual-branch fusion) - strongest contribution, most practical
- **Secondary:** Option 2 (quantum speedup + ablation) - adds novelty
- **Tertiary:** Option 3 (comparison table) - useful for community

---

#### B) Essential Experiments for Publication

**Minimum Viable Experiments:**
1. ✅ 5-fold CV results for all standalone models (EfficientNet, Swin-T/S, ConvNeXt-T/S, DeiT-S)
2. ✅ 5-fold CV results for dual-branch fusion model
3. ✅ Comparison table: Accuracy, AUC, Sensitivity, Specificity, F1, Training Time
4. ✅ Confusion matrices + ROC curves for key models
5. ✅ Grad-CAM visualizations for fusion model

**Stronger Paper (Add These):**
6. ✅ Quantum ablation study (rotation gates, entanglement strategies)
7. ✅ Fusion ablation (early vs mid vs late, gating vs cross-attention)
8. ✅ Statistical significance testing (paired t-tests between models)
9. ✅ Attention map visualizations (Swin windows, DeiT attention)

**Journal-Quality (Add These Too):**
10. ⚠️ External validation on Camelyon16 or BACH dataset
11. ⚠️ Multi-class classification (IDC, ILC, Fibroadenoma)
12. ⚠️ Comprehensive interpretability analysis (pathologist evaluation of attention maps)

---

#### C) Target Venue Recommendation

**Conference Options:**
| Venue | Deadline | Acceptance Rate | Fit |
|-------|----------|-----------------|-----|
| **MICCAI** | March | ~30% | ⭐⭐⭐⭐⭐ Perfect fit (medical image computing) |
| **CVPR Workshop on AI for Healthcare** | Varies | ~40% | ⭐⭐⭐⭐ Good fit |
| **IPMI (Information Processing in Medical Imaging)** | Biennial | ~35% | ⭐⭐⭐⭐ Strong fit |
| **ISBI (IEEE Symposium on Biomedical Imaging)** | October | ~50% | ⭐⭐⭐⭐ Good fit |

**Journal Options:**
| Journal | Impact Factor | Review Time | Fit |
|---------|---------------|-------------|-----|
| **Medical Image Analysis (MedIA)** | 11.2 | 3-4 months | ⭐⭐⭐⭐⭐ Top venue |
| **IEEE TMI (Transactions on Medical Imaging)** | 10.6 | 3-4 months | ⭐⭐⭐⭐⭐ Top venue |
| **IEEE JBHI (Journal of Biomedical and Health Informatics)** | 7.7 | 2-3 months | ⭐⭐⭐⭐ Good fit |
| **Nature Scientific Reports** | 4.6 | 2-3 months | ⭐⭐⭐ Open access option |

**Recommendation:**
- **Timeline <10 weeks:** MICCAI or ISBI (conference paper)
- **Timeline 4-6 months:** Medical Image Analysis or IEEE TMI (journal)
- **Fallback:** Scientific Reports (faster review, open access)

---

## 📊 Deliverable Format

For each research task, provide:

### 1. Literature Review
- Key papers (title, venue, year, key findings)
- State-of-the-art accuracy on BreakHis (if available)
- Gaps in existing literature

### 2. Implementation Recommendations
- Specific architectures (timm model names)
- Hyperparameters (learning rate, batch size, augmentation)
- Pretraining strategy (ImageNet-1K vs ImageNet-21K)

### 3. Expected Outcomes
- Accuracy predictions (based on literature)
- Training time estimates
- Potential failure modes

### 4. Experimental Design
- What to compare (model variants, hyperparameters)
- Metrics (accuracy, AUC, sensitivity, specificity, F1, training time)
- Statistical tests (paired t-test, ANOVA, Wilcoxon signed-rank)

### 5. Risk Assessment
- What might fail (e.g., quantum too slow, fusion doesn't improve accuracy)
- Fallback strategies (e.g., ensemble instead of fusion, classical-only model)
- Mitigation plans

---

## 🎯 Final Output

**Comprehensive Research Plan** including:
1. **Comparison Table:** All transformer variants × all metrics
2. **Ablation Studies:**
   - Quantum rotation gates (4 variants) × entanglement strategies (3 variants)
   - Fusion mechanisms (4 variants) × fusion timing (3 variants)
   - Transformer backbones (Swin vs ConvNeXt vs DeiT)
3. **Statistical Analysis:** P-values for all pairwise model comparisons
4. **Interpretability Gallery:** Grad-CAM, attention maps, gate distributions
5. **Publication-Ready Results:** Comparison tables, ROC curves, training curves
6. **Code Repository:** Modular implementation of all architectures

**Timeline:** 10 weeks (see `implementation_plan.md` for detailed schedule)

---

## 🔗 Useful Resources

### Code Libraries
- **timm:** `pip install timm` - Swin, ConvNeXt, DeiT, ViT implementations
- **TorchQuantum:** `git clone https://github.com/mit-han-lab/torchquantum` - GPU-accelerated quantum
- **monai:** `pip install monai` - Medical image augmentation

### Datasets
- **BreakHis:** https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
- **Camelyon16:** https://camelyon16.grand-challenge.org/
- **BACH:** https://bach.grand-challenge.org/

### Pretrained Weights
- ImageNet-1K: Available in timm by default
- ImageNet-21K: Some models available (Swin-V2, DeiT)
- Medical pretraining: Search "histopathology foundation models"

### Visualization Tools
- **Grad-CAM:** `pip install grad-cam`
- **Captum:** `pip install captum` - Model interpretability
- **Weights & Biases:** `pip install wandb` - Experiment tracking

---

**Good luck with your research! 🚀**
