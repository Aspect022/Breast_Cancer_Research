# 📄 Paper Context Document — Breast Cancer Histopathology Binary Classification
## For: Claude (Paper Writing Assistant)
## Author Context: Graduate/Undergraduate Minor Project Student
## Goal: Write sections of an academic paper on quantum-classical fusion architectures for breast cancer histopathology classification

---

## 🎯 What This Paper Is About

This paper presents a **progressive architectural study** for breast cancer binary classification (Benign vs. Malignant) on the **BreakHis dataset**. The core narrative is:

> Starting from a pure Vision Transformer baseline (ViT-Tiny), we progressively build toward a novel Triple-Branch Cross-Attention (TBCA) fusion architecture, and further augment it with a quantum-classical hybrid module. The final proposed model — TBCA-CNN-FeatureMap-Quantum — achieves state-of-the-art results on BreakHis binary classification.

**The paper only covers 4 architectures** (per mentor guidance):
1. **ViT-Tiny** — Pure transformer baseline (weakest, establishes lower bound)
2. **TripleBranch-Fusion / TBCA** — Our core proposed fusion (classical)
3. **TBCA-CNN-FeatureMap-Quantum** — Best performing model (🏆)
4. **TBCA-ViT-FeatureMap-Quantum** — Second quantum variant (for ablation comparison)

---

## 📊 Dataset: BreakHis

| Property | Value |
|----------|-------|
| Full Name | Breast Cancer Histopathological Image Database |
| Total Images | 7,909 |
| Classes | Binary: Benign vs. Malignant |
| Malignant | 5,429 images (68.6%) |
| Benign | 2,480 images (31.4%) |
| Magnifications | 40×, 100×, 200×, 400× |
| Image Size | 700×460 pixels (RGB, H&E stained) |
| Class Imbalance | Yes — significant (68.6% malignant) |
| Source | Universidade Federal do Paraná (UFPR), Brazil |

**Cross-Validation Protocol:**
- **5-Fold Patient-Level Cross-Validation** (stratified)
- Patient-level split ensures **zero patient leakage** across folds
- All results reported as **mean ± std across 5 folds**
- Primary metric: **Validation Accuracy**

---

## 🏗️ Architecture 1: ViT-Tiny (Baseline)

### Role in Paper
Establishes the **lower performance bound** for pure transformer-based approaches on small medical imaging datasets. Demonstrates that vanilla ViTs are insufficient for this task without architectural augmentation.

### Technical Details
| Property | Value |
|----------|-------|
| Architecture | Pure Vision Transformer (ViT) |
| Variant | Tiny (smallest ViT variant) |
| Parameters | 11.0M |
| Patch Size | 16×16 |
| Embedding Dim | 192 |
| Heads | 3 |
| Layers | 12 |
| Input Resolution | 224×224 |
| Pre-training | ImageNet-21k |

### Mechanism
- Image is divided into fixed 16×16 patches
- Each patch linearly projected to embedding dim (192)
- [CLS] token prepended; positional embeddings added
- 12 layers of standard multi-head self-attention (global attention — every token attends to every other token)
- [CLS] token features → Linear classifier head

### Why It Struggles on BreakHis
1. **Insufficient data**: ViT's global attention requires large datasets (hundreds of thousands of images) to learn meaningful visual representations. BreakHis has only ~7,909 images.
2. **No inductive bias**: Unlike CNNs, ViT has no built-in locality or translation invariance — these must be learned from data, which is insufficient.
3. **Fine-grained texture features**: H&E-stained histopathology relies on subtle nuclear morphology and cellular arrangement. Pure patch attention without convolutional feature hierarchies misses these.

### Performance Results
| Metric | Value |
|--------|-------|
| **Val Accuracy** | **78.42% ± 1.18%** |
| Val AUC-ROC | 83.27% ± 2.42% |
| Sensitivity | 78.42% |
| Specificity | 75.24% |
| F1 (macro) | 78.26% |
| MCC | 57.33% |
| FNR (False Negative Rate) | 21.58% |
| Inference Time | 4.8 ms |
| Training Time | ~35 min |

### Clinical Significance of ViT-Tiny FNR
An FNR of **21.58%** means 1 in 5 malignant cases would be missed in a clinical screening context. This is **clinically unacceptable** and motivates the need for better architectures.

---

## 🏗️ Architecture 2: TripleBranch Cross-Attention Fusion (TBCA) — Classical

### Role in Paper
The **core proposed architecture** and primary contribution. Demonstrates that multi-backbone fusion with bidirectional cross-attention dramatically outperforms single-backbone transformers, even without quantum components.

### Technical Details
| Property | Value |
|----------|-------|
| Architecture | Triple-Branch Architecture with Cross-Attention |
| Short Name | TBCA |
| Parameters | 141.4M |
| Backbone 1 | Swin-Small (v2) |
| Backbone 2 | ConvNeXt-Small |
| Backbone 3 | EfficientNet-B5 |
| Feature Dim per Branch | 768 |
| Cross-Attention Heads | 8 |
| Self-Attention Heads | 8 |
| Fusion Strategy | Bidirectional cross-attention → self-attention → weighted fusion |

### How TBCA Works — Step by Step

**Step 1 — Feature Extraction (Three Parallel Branches)**
```
Input Image (224×224 RGB)
    ├── Branch 1: Swin-Small-v2  → Feature vector F₁ (768-dim)
    │   [Hierarchical shifted-window attention, 4 stages]
    ├── Branch 2: ConvNeXt-Small → Feature vector F₂ (768-dim)
    │   [Modern CNN with transformer design principles]
    └── Branch 3: EfficientNet-B5 → Feature vector F₃ (768-dim)
        [MBConv + Squeeze-Excitation, compound scaled]
```

**Step 2 — Bidirectional Cross-Attention Fusion**
```
Cross-Attention pairs (bidirectional):
  CA(F₁, F₂): Swin attends to ConvNeXt → F₁₂
  CA(F₂, F₁): ConvNeXt attends to Swin → F₂₁
  CA(F₁, F₃): Swin attends to EfficientNet → F₁₃
  ...and all 6 directional pairs

Each cross-attention:
  Q = Fᵢ (query branch)
  K, V = Fⱼ (key-value branch)
  Output = softmax(QKᵀ/√d) · V
```

**Step 3 — Self-Attention Aggregation**
```
Concatenated cross-attended features → 8-head self-attention
This allows features to further interact and find global relationships
across all three backbone representations.
```

**Step 4 — Weighted Fusion & Classification**
```
Learned per-sample attention weights α₁, α₂, α₃ (sum = 1)
Fused = α₁·R₁ + α₂·R₂ + α₃·R₃
→ Linear classification head → Sigmoid → P(malignant)
```

### Why Three Specific Backbones?
- **Swin-Small**: Captures **long-range spatial relationships** via hierarchical windows. Excellent at understanding global tissue architecture.
- **ConvNeXt-Small**: Captures **local texture patterns** using large 7×7 depthwise convolutions. H&E staining creates rich texture that ConvNeXt extracts well.
- **EfficientNet-B5**: Captures **multi-scale morphological features** via compound scaling and Squeeze-Excitation attention on channel features. Excellent for nuclear morphology.

The three backbones are **complementary** — they extract fundamentally different feature types (global structure, local texture, multi-scale morphology), and cross-attention allows them to share complementary information.

### Performance Results
| Metric | Value |
|--------|-------|
| **Val Accuracy** | **93.01% ± 5.05%** |
| Val AUC-ROC | 98.61% ± 2.36% |
| Sensitivity | 98.95% |
| Specificity | 80.74% |
| F1 (macro) | 91.65% |
| MCC | 87.76% |
| FNR (False Negative Rate) | 1.05% |
| Inference Time | 73.2 ms |
| Training Time | ~374 min |

### Improvement Over Baseline
| Metric | ViT-Tiny | TBCA | Δ Improvement |
|--------|----------|------|--------------|
| Val Accuracy | 78.42% | 93.01% | **+14.59%** |
| Val AUC-ROC | 83.27% | 98.61% | **+15.34%** |
| Sensitivity | 78.42% | 98.95% | **+20.53%** |
| FNR | 21.58% | 1.05% | **-20.53%** |

---

## 🏗️ Architecture 3: TBCA-CNN-FeatureMap-Quantum (Best Model 🏆)

### Role in Paper
The **best-performing proposed model** — demonstrates that augmenting the TBCA triple-backbone fusion with a **CNN feature-map quantum branch** achieves state-of-the-art performance on BreakHis binary classification.

### Technical Details
| Property | Value |
|----------|-------|
| Architecture | TBCA + CNN Feature-Map Level Quantum Branch |
| Parameters | 142.5M |
| Base Architecture | TripleBranch-Fusion (TBCA) |
| Quantum Branch | CNN + Variational Quantum Circuit (VQC) |
| Quantum Framework | PennyLane |
| Circuit Type | Variational Quantum Circuit (VQC) |
| Qubits | 8 |
| Quantum Layers | 2 |
| Gate Set | U3 rotations + CNOT entanglement |
| Entanglement | Cyclic CNOT topology |
| Feature Reduction | CNN 768 → 8 (for qubit encoding) |

### How the Quantum Branch Works — Step by Step

**The quantum branch operates at the FEATURE-MAP LEVEL — before the cross-attention fusion, not after.**

```
Input Image (224×224 RGB)
    ├── Branch 1: Swin-Small      → F₁ (768-dim)  ]
    ├── Branch 2: ConvNeXt-Small  → F₂ (768-dim)  ] → Cross-Attention Fusion (same as TBCA)
    ├── Branch 3: EfficientNet-B5 → F₃ (768-dim)  ]
    └── Branch 4: CNN Quantum Branch (NEW)
        ├── Step A: Small CNN extracts feature maps from raw image
        ├── Step B: Feature reduction layer (CNN → 8 values)
        ├── Step C: Amplitude encoding into quantum state |ψ⟩
        ├── Step D: Variational Quantum Circuit (VQC)
        │   Layer 1: U3(θ₁, φ₁, λ₁) on each qubit
        │   Layer 1: Cyclic CNOT (0→1→2→...→7→0)
        │   Layer 2: U3(θ₂, φ₂, λ₂) on each qubit
        │   Layer 2: Cyclic CNOT
        ├── Step E: Measurement: ⟨Z⟩ expectation values (8 values)
        └── Step F: Classical post-processing (8 → 768-dim)
        
Quantum features (768-dim) + Classical TBCA features (768-dim)
→ Concatenation → Final Fusion → Classification Head
```

### Variational Quantum Circuit (VQC) Detail
```
State Encoding:
|ψ⟩ = encode(x) where x ∈ R⁸ (amplitude encoding)

Circuit Structure (2 layers):
Layer l:
  - Apply U3(θ_l,i, φ_l,i, λ_l,i) to qubit i (parametric rotation)
  - Apply CNOT(i, i+1 mod 8) for i = 0..7 (entanglement ring)

Measurement:
  ⟨Z_i⟩ = ⟨ψ|Z_i|ψ⟩ for i = 0..7

Trainable Parameters:
  θ, φ, λ for each qubit × each layer
  = 8 qubits × 3 angles × 2 layers = 48 quantum parameters
  (trained jointly via gradient-based optimization using parameter-shift rule)
```

### Why CNN Feature-Map Level Instead of Classifier Level?
- **Feature-map quantum**: The quantum circuit processes **raw intermediate CNN features** — it sees the image at different levels of abstraction before the final classifier. This allows quantum superposition to explore **all possible feature combinations simultaneously**.
- **Classifier-level quantum**: Would only process the final 768-dim global feature vector — less expressive, loses spatial/structural information.
- The CNN branch extracts spatial features the Swin/ConvNeXt/EfficientNet backbones might process differently, providing a **fourth complementary view** of the image.

### Why This Outperforms Pure Classical TBCA
Quantum circuits provide:
1. **Exponential state space**: 8 qubits → 2⁸ = 256-dimensional Hilbert space. The quantum branch can represent complex, non-linear feature correlations that classical networks would need many more parameters to capture.
2. **Quantum superposition**: The VQC processes all feature combinations simultaneously during forward pass.
3. **Complementary feature extraction**: CNN feature-map quantum extracts spatial patterns at a different level of abstraction than the three classical backbones.

### Performance Results
| Metric | Value |
|--------|-------|
| **Val Accuracy** | **95.48% ± 1.53%** |
| Val AUC-ROC | 98.92% ± 1.85% |
| Sensitivity | 99.06% |
| Specificity | 88.10% |
| F1 (macro) | 94.72% |
| MCC | 89.70% |
| FNR (False Negative Rate) | 0.94% |
| Inference Time | 78.1 ms |
| Training Time | ~337 min |

### Improvement Over TBCA (Classical)
| Metric | TBCA | TBCA-CNN-FM-Quantum | Δ Improvement |
|--------|------|---------------------|--------------|
| Val Accuracy | 93.01% | 95.48% | **+2.47%** |
| Val AUC-ROC | 98.61% | 98.92% | **+0.31%** |
| Sensitivity | 98.95% | 99.06% | **+0.11%** |
| Specificity | 80.74% | 88.10% | **+7.36%** |
| MCC | 87.76% | 89.70% | **+1.94%** |
| FNR | 1.05% | 0.94% | **-0.11%** |

**Key highlight**: The most remarkable improvement is in **Specificity (+7.36%)** — meaning the quantum branch significantly reduces false positives (incorrectly classifying benign as malignant), a critical clinical requirement for avoiding unnecessary biopsies.

---

## 🏗️ Architecture 4: TBCA-ViT-FeatureMap-Quantum (Second Quantum Variant)

### Role in Paper
**Ablation/comparison model** — same TBCA framework but uses a **ViT feature-map level quantum branch** instead of CNN. Comparing with TBCA-CNN-FM-Quantum reveals the importance of the feature extraction method for the quantum branch.

### Technical Details
| Property | Value |
|----------|-------|
| Architecture | TBCA + ViT Feature-Map Level Quantum Branch |
| Parameters | 124.5M |
| Quantum Branch | ViT patch embeddings → Quantum Circuit |
| Qubits | 8 |
| Quantum Layers | 2 |
| Gate Set | U3 rotations + CNOT |

### How It Differs from TBCA-CNN-FM-Quantum
- **Same**: TBCA triple-backbone fusion, same quantum circuit (VQC with 8 qubits, 2 layers, U3+CNOT)
- **Different**: Instead of a CNN extracting spatial feature maps, a **ViT patch encoder** processes the image and produces patch embeddings → reduced to 8 values → quantum encoding
- ViT patch attention processes **global patch relationships** for quantum encoding, while CNN processes **local spatial hierarchies**

### Performance Results
| Metric | Value |
|--------|-------|
| **Val Accuracy** | **91.88% ± 1.45%** |
| Val AUC-ROC | 91.08% ± 3.26% |
| Sensitivity | 92.42% |
| Specificity | 77.17% |
| F1 (macro) | 90.09% |
| MCC | 71.28% |
| FNR (False Negative Rate) | 7.58% |
| Inference Time | 89.5 ms |
| Training Time | ~413 min |

### Why CNN Branch Outperforms ViT Branch for Quantum Input
| Metric | TBCA-ViT-FM-Q | TBCA-CNN-FM-Q | CNN Advantage |
|--------|--------------|--------------|--------------|
| Val Accuracy | 91.88% | 95.48% | **+3.60%** |
| Val AUC | 91.08% | 98.92% | **+7.84%** |
| Sensitivity | 92.42% | 99.06% | **+6.64%** |
| MCC | 71.28% | 89.70% | **+18.42%** |

**Interpretation**: For quantum feature encoding, **CNN spatial hierarchies are more informative** than ViT global patch attention. CNN feature maps preserve local texture and structural gradients that encode critical histopathological patterns (nuclear pleomorphism, gland morphology), whereas ViT patch embeddings aggregate global context that, when reduced to 8 values, loses discriminative local information.

---

## 📈 Complete Comparison: All 4 Models

| Metric | ViT-Tiny (Baseline) | TBCA (Classical) | TBCA-ViT-FM-Q | TBCA-CNN-FM-Q (Best) |
|--------|--------------------:|----------------:|---------------:|---------------------:|
| **Val Accuracy** | 78.42% | 93.01% | 91.88% | **95.48%** |
| Val Acc Std | ±1.18% | ±5.05% | ±1.45% | ±1.53% |
| **Val AUC-ROC** | 83.27% | 98.61% | 91.08% | **98.92%** |
| **Sensitivity** | 78.42% | 98.95% | 92.42% | **99.06%** |
| **Specificity** | 75.24% | 80.74% | 77.17% | **88.10%** |
| **F1 (macro)** | 78.26% | 91.65% | 90.09% | **94.72%** |
| **MCC** | 57.33% | 87.76% | 71.28% | **89.70%** |
| **FNR** | 21.58% | 1.05% | 7.58% | **0.94%** |
| Params | 11.0M | 141.4M | 124.5M | 142.5M |
| Train Time | ~35 min | ~374 min | ~413 min | ~337 min |

**Total progression**: ViT-Tiny → TBCA-CNN-FM-Quantum = **+17.06% absolute val accuracy improvement**

---

## 🧠 Key Narrative Points for the Paper

### 1. Motivation
- Breast cancer is the most common cancer in women globally
- Histopathological analysis is the gold standard for diagnosis
- Manual analysis is time-consuming, error-prone, and subject to inter-pathologist variability
- Automated deep learning-based classification can assist pathologists

### 2. Research Gap
- Most prior work uses single-backbone CNNs or Transformers
- Pure ViTs fail on small medical datasets (our ViT-Tiny: 78.42% with 21.58% FNR)
- Multi-backbone fusion has been explored but without cross-attention at the feature level
- Quantum-classical hybrid approaches for histopathology are largely unexplored

### 3. Our Contributions
1. **TBCA Architecture**: Novel triple-branch bidirectional cross-attention fusion combining Swin, ConvNeXt, and EfficientNet for complementary feature extraction
2. **Quantum Augmentation**: First application of VQC-based feature-map quantum branch to triple-backbone histopathology classification
3. **Systematic Ablation**: CNN vs. ViT feature-map encoding for quantum circuits, demonstrating CNN superiority for quantum input
4. **Clinical Safety**: Achieving FNR of 0.94% (vs. 21.58% baseline) with 99.06% sensitivity

### 4. Clinical Significance
- **FNR reduction**: From 21.58% (ViT baseline) to 0.94% (best model) — a **22.9× reduction** in missed cancer cases
- **Specificity improvement**: 88.10% specificity minimizes unnecessary biopsies from false positives
- **MCC = 89.70%**: Strong balanced performance on imbalanced data (68.6% malignant class)

---

## ⚙️ Training Configuration (for Methods Section)

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-5 |
| LR Schedule | Cosine Annealing |
| Weight Decay | 1e-4 |
| Batch Size | 8–32 (model dependent) |
| Max Epochs | 50 |
| Early Stopping Patience | 10 epochs |
| Gradient Clipping | Max norm = 1.0 |
| Mixed Precision | FP16 (AMP) |
| Cross-Validation | 5-Fold, Patient-Level Stratified |
| Hardware | NVIDIA RTX 5050 Laptop GPU |
| Framework | PyTorch + PennyLane (quantum) |
| Experiment Tracking | Weights & Biases (W&B) |

---

## 📋 Instructions for Claude (Paper Writing)

When using this document to write paper sections, please follow these guidelines:

### Tone & Style
- Academic, formal IEEE/Springer style
- Third person throughout ("We propose...", "The model achieves...")
- Quantitative — always cite specific numbers with ± std where available
- Concise but technically rigorous

### Sections to Help Write
1. **Abstract** — 250 words, covering: motivation, method (TBCA + quantum), dataset, key results
2. **Introduction** — Problem statement, motivation, research gap, contributions (4 bullet points)
3. **Related Work** — CNN approaches, Vision Transformers for medical imaging, Quantum ML (need references)
4. **Methodology** — Dataset, preprocessing, TBCA architecture, quantum branch, training setup
5. **Results** — Table of 4 models, progression narrative, clinical significance
6. **Discussion** — Why quantum helps, CNN vs. ViT for quantum input, limitations
7. **Conclusion** — Summary, clinical implications, future work (multiclass extension)

### Critical Things NOT to Fabricate
- Do not invent specific citations (use placeholder [CITE] — references will be provided separately)
- Do not add models not mentioned in this document
- Do not change the numbers — use exactly the values provided
- The dataset is BreakHis only (not CBIS-DDSM — that was a different experiment)

### Paper Title Suggestions
- "Triple-Branch Cross-Attention Fusion with Quantum Enhancement for Breast Cancer Histopathology Classification"
- "TBCA-Quantum: A Novel Quantum-Classical Hybrid Architecture for Breast Cancer Binary Classification"
- "From Vision Transformers to Quantum-Classical Fusion: A Progressive Architectural Study on BreakHis"
