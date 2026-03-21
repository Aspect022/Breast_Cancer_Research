# Complete MoE-MCGA Architecture Specification
## Attention-Enhanced Mixture of Experts for Multimodal Breast Cancer Detection

**Document Version:** 1.0  
**Date:** March 22, 2026  
**Based on:** "Attention-Enhanced Mixture of Experts for Multimodal Breast Cancer Detection"  
**Target Venue:** MICCAI 2026 / ISBI 2026  
**Project:** Breast Cancer Histopathology Classification System

---

# Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Current System Analysis](#2-current-system-analysis)
3. [Research Paper Integration](#3-research-paper-integration)
4. [Gap Analysis](#4-gap-analysis)
5. [Proposed Architecture](#5-proposed-architecture)
6. [MoE Integration Plan](#6-moe-integration-plan)
7. [MCGA Integration Plan](#7-mcga-integration-plan)
8. [Training Strategy](#8-training-strategy)
9. [Data Alignment Strategy](#9-data-alignment-strategy)
10. [Implementation Files](#10-implementation-files)
11. [Implementation Plan](#11-implementation-plan)
12. [Advanced Extensions](#12-advanced-extensions)
13. [Expected Outcomes](#13-expected-outcomes)
14. [References](#14-references)

---

# 1. Executive Summary

## 1.1 Purpose

This document specifies a **complete architectural redesign** of the breast cancer classification system, transforming it from a single-modality, static architecture into a **state-of-the-art multimodal AI system** with:

- **Mixture of Experts (MoE)** for patient-adaptive computation
- **Multi-Co-Guided Attention (MCGA)** for cross-modal fusion
- **Emergent expert specialization** through diversity loss
- **Robust multimodal data alignment**

## 1.2 Key Innovations

| Innovation | Description | Impact |
|------------|-------------|--------|
| **MoE Fusion** | Dynamic routing to 8 specialized experts (top-2 sparse) | +2-4% accuracy, 4× efficiency |
| **Bidirectional MCGA** | Image↔Clinical cross-attention | +3-5% specificity |
| **Emergent Experts** | Data-driven specialization via diversity loss | No manual design needed |
| **Multimodal Alignment** | Patient-level histopath+mammo+clinical fusion | Real-world applicability |
| **Phased Training** | Warmup→Gating→Full training schedule | Stable convergence |

## 1.3 Performance Targets

| Metric | Current Best | MoE-MCGA Target | Improvement |
|--------|-------------|-----------------|-------------|
| **Accuracy (BreakHis)** | 85.67% | 88-90% | **+2.3-4.3%** |
| **AUC-ROC** | 88.52% | 91-93% | **+2.5-4.5%** |
| **Sensitivity** | 91.15% | 92-94% | **+0.8-2.8%** |
| **Specificity** | 77.45% | 82-85% | **+4.5-7.5%** |
| **FNR** | 8.85% | 6-8% | **-0.85-2.85%** |
| **Inference Speed** | 30ms | 25ms (sparse) | **+17% faster** |

---

# 2. Current System Analysis

## 2.1 Existing Architecture Overview

### 2.1.1 Data Modalities (Current)

| Modality | Format | Preprocessing | Feature Extraction | Status |
|----------|--------|---------------|-------------------|--------|
| **Histopathology Images** | PNG, 224×224 RGB | ImageNet normalization, augmentation | CNN/Transformer backbones | ✅ Primary |
| **Mammography Images** | PNG/DCM, variable → 224×224 | Resize, normalization | CNN backbones | ✅ CBIS-DDSM |
| **Tabular Clinical (WBCD)** | CSV, 10 features | Standardization | FC layers | ✅ Supported |
| **Tabular Clinical (SEER)** | CSV, demographic/tumor | Imputation, normalization | FC layers | ✅ Supported |

### 2.1.2 Image Data Pipeline

```
Raw Image (PNG) → PIL Load → Resize(224×224) → Augmentation → 
ToTensor → ImageNet Normalize → DataLoader
```

**Train Transforms:**
```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Critical Gap:** No patient-level metadata integration beyond patient ID for grouping. No multi-modal fusion between image + clinical data exists.

### 2.1.3 Tabular Data Handling

```python
# WBCD/SEER: 10-D or N-D feature vectors
sample = sample.reshape(1, -1, 1)  # Pseudo-image format
sample = torch.from_numpy(sample).float().view(-1)  # Flatten for FC
```

**Critical Gap:** Tabular data is processed independently from images. No cross-modal attention or fusion exists.

## 2.2 Current Model Inventory (23 Models)

### 2.2.1 Model Categories

| Category | Models | Count |
|----------|--------|-------|
| **Baseline CNN** | EfficientNet-B3, SpikingCNN-LIF | 2 |
| **Transformers** | Swin-T/S/V2, ConvNeXt-T/S, DeiT-T/S/B, ViT-Tiny | 9 |
| **Hybrid** | CNN+ViT-Hybrid | 1 |
| **Fusion** | DualBranch-Fusion, TBCA-Fusion, Quantum-Enhanced-Fusion | 3 |
| **Quantum** | QENN-RY/RZ/U3/RX_RY_RZ, CB-QCCF, MSQF | 6 |
| **Ensemble** | TSD-Ensemble | 1 |
| **Legacy** | CNN+PQC (PennyLane) | 1 |

### 2.2.2 Complete Model Performance Table

| Model | Accuracy | AUC-ROC | Sensitivity | Specificity | FNR | Params |
|-------|----------|---------|-------------|-------------|-----|--------|
| **Swin-Small** | **85.67%** | 88.52% | 91.15% | 77.45% | 8.85% | 50M |
| **CNN+ViT-Hybrid** | 84.67% | **89.25%** | 88.49% | 78.92% | 11.51% | 15M |
| **QENN-RY_ONLY** | 84.17% | 88.40% | **91.68%** | 72.88% | 8.32% | 12M |
| **QENN-U3** | 84.05% | 88.08% | **92.15%** | 71.88% | 7.85% | 12M |
| **ConvNeXt-Tiny** | 83.85% | 88.37% | 85.61% | 81.20% | 14.39% | 28M |
| **DualBranch-Fusion** | 83.68% | 87.19% | 86.30% | 79.73% | 13.70% | 78M |
| **ConvNeXt-Small** | 83.43% | 87.62% | 87.03% | 78.03% | 12.97% | 50M |
| **QENN-RX_RY_RZ** | 83.38% | 88.41% | 90.68% | 72.42% | 9.32% | 12M |
| **EfficientNet-B3** | 83.37% | **89.97%** | 86.10% | 79.26% | 13.90% | 12M |
| **QENN-RY_RZ** | 83.23% | 87.81% | 90.55% | 72.23% | 9.45% | 12M |
| **Swin-Tiny** | 82.12% | 86.83% | 84.61% | 78.38% | 15.39% | 29M |
| **ViT-Tiny** | 79.14% | 83.27% | 81.72% | 75.24% | 18.28% | 5.7M |

### 2.2.3 Recently Implemented Fusion Models

| Model | File | Status | Expected |
|-------|------|--------|----------|
| **TBCA-Fusion** | `src/models/fusion/triple_branch.py` | ✅ Implemented | 87-89% |
| **CB-QCCF** | `src/models/fusion/class_balanced_quantum.py` | ✅ Implemented | 85-87% |
| **MSQF** | `src/models/fusion/multi_scale_quantum.py` | ✅ Implemented | 86-88% |
| **TSD-Ensemble** | `src/models/fusion/ensemble_distillation.py` | ✅ Implemented | 87-89% |

## 2.3 Complete Data Flow Analysis

### 2.3.1 Current Flow: Raw Data → Prediction

```
┌─────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │  Raw Images  │    │  Tabular CSV │    │  Metadata    │          │
│  │  (PNG/DCM)   │    │  (WBCD/SEER) │    │  (Patient ID)│          │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘          │
│         │                   │                   │                   │
│         ▼                   ▼                   │                   │
│  ┌──────────────┐    ┌──────────────┐          │                   │
│  │  PIL Load    │    │  Pandas Load │          │                   │
│  │  Resize 224  │    │  Preprocess  │          │                   │
│  └──────┬───────┘    └──────┬───────┘          │                   │
│         │                   │                   │                   │
│         ▼                   ▼                   │                   │
│  ┌──────────────┐    ┌──────────────┐          │                   │
│  │  Augmentation│    │  Normalize   │          │                   │
│  │  ToTensor    │    │  ToTensor    │          │                   │
│  │  Normalize   │    │              │          │                   │
│  └──────┬───────┘    └──────┬───────┘          │                   │
│         │                   │                   │                   │
│         └────────┬──────────┘                   │                   │
│                  │                              │                   │
│                  ▼                              │                   │
│         ┌────────────────┐                      │                   │
│         │  DataLoader    │◄─────────────────────┘ (grouping only)   │
│         │  (Batch, Shuffle)│                                     │
│         └────────┬───────┘                      │                   │
│                  │                              │                   │
│                  ▼                              │                   │
│         ┌────────────────┐                      │                   │
│         │  Model Backbone │                      │                   │
│         │  (CNN/Transformer)│                    │                   │
│         └────────┬───────┘                      │                   │
│                  │                              │                   │
│                  ▼                              │                   │
│         ┌────────────────┐                      │                   │
│         │  Feature Vector │                      │                   │
│         │  (B, feature_dim)│                     │                   │
│         └────────┬───────┘                      │                   │
│                  │                              │                   │
│                  ▼                              │                   │
│         ┌────────────────┐                      │                   │
│         │  Fusion Module │ (if applicable)      │                   │
│         │  (Gating/Attention)│                  │                   │
│         └────────┬───────┘                      │                   │
│                  │                              │                   │
│                  ▼                              │                   │
│         ┌────────────────┐                      │                   │
│         │  Classification │                      │                   │
│         │  Head (FC)     │                      │                   │
│         └────────┬───────┘                      │                   │
│                  │                              │                   │
│                  ▼                              │                   │
│         ┌────────────────┐                      │                   │
│         │  Softmax       │                      │                   │
│         │  (Probabilities)│                     │                   │
│         └────────┬───────┘                      │                   │
│                  │                              │                   │
│                  ▼                              │                   │
│         ┌────────────────┐                      │                   │
│         │  Prediction    │                      │                   │
│         │  (Benign/Malignant)│                  │                   │
│         └────────────────┘                      │                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3.2 Feature Extraction Points

```
┌─────────────────────────────────────────────────────────────────┐
│                    FEATURE EXTRACTION POINTS                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  CNN Models (EfficientNet, ConvNeXt):                           │
│    Image → .features → (B, C, H, W) → GAP → (B, C)             │
│                                                                  │
│  Transformer Models (Swin, DeiT):                               │
│    Image → .backbone → (B, num_tokens, dim) → CLS → (B, dim)   │
│                                                                  │
│  Hybrid Models (CNN+ViT):                                        │
│    Image → CNN.features → (B, C, H, W) → Flatten →             │
│    (B, H*W, C) → Project → Transformer → CLS → (B, dim)        │
│                                                                  │
│  Quantum Models (QENN):                                          │
│    Image → CNN → GAP → (B, 1536) → Compress → (B, n_qubits) →  │
│    Quantum Circuit → (B, n_qubits) → Expand → Logits           │
│                                                                  │
│  Fusion Models:                                                  │
│    Image → [Branch1, Branch2, ...] → [Features] →              │
│    Fusion Module → (B, fusion_dim) → Logits                    │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3.3 Classification Heads

**Standard Pattern:**
```python
nn.Sequential(
    nn.LayerNorm(feature_dim),
    nn.Dropout(0.3),
    nn.Linear(feature_dim, 512),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(512, 256),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
```

## 2.4 Existing Fusion Mechanisms

### 2.4.1 Dual-Branch Dynamic Fusion

```python
# src/models/fusion/dual_branch.py

Image → [Swin Branch] ──┐
        [ConvNeXt Branch]─┤
                          ▼
                    Concat(F_swin, F_convnext)
                          ▼
                    Gate Network → α (B, 1)
                          ▼
                    F_fused = α·F_convnext + (1-α)·F_swin
                          ▼
                    FC Head → Logits
```

**Characteristics:**
- Fusion Type: Feature-level, Dynamic, Static Weighting
- Gating: MLP-based sigmoid
- Entropy Regularization: Yes (encourages balanced gating)

### 2.4.2 Triple-Branch Cross-Attention Fusion

```python
# src/models/fusion/triple_branch.py

Image → [Swin] ──┬──→ CrossAttn(Swin→ConvNext)
        [ConvNeXt]─┤──→ CrossAttn(Swin→EffNet)
        [EffNet] ──┴──→ CrossAttn(ConvNext→Swin)
                          ▼
                    Enhanced Features
                          ▼
                    Learnable Weights (softmax)
                          ▼
                    F_fused = Σ w_i·F_enhanced_i
                          ▼
                    Self-Attention Refinement
                          ▼
                    FC Head → Logits
```

**Characteristics:**
- Fusion Type: Feature-level, Cross-Attention, Learnable Weights
- Attention: Bidirectional cross-attention between branches
- Weighting: Softmax-normalized learnable parameters

### 2.4.3 Quantum-Enhanced Fusion

```python
# src/models/fusion/dual_branch.py

Image → [Swin + ConvNeXt] → Fused → Compress → Quantum → Expand
                          ▼
                    Quantum Circuit (8 qubits, 2 layers)
                          ▼
                    FC Head → Logits
```

**Characteristics:**
- Fusion Type: Quantum-Classical Hybrid
- Quantum: Vectorized GPU-accelerated simulation
- Rotation: RY-only, RY+RZ, U3, or RX+RY+RZ

### 2.4.4 Class-Balanced Quantum-Classical Fusion

```python
# src/models/fusion/class_balanced_quantum.py

Image → [Swin (Classical)] ──┬──→ CrossAttn
        [QENN (Quantum)] ────┤    (Classical attends Quantum)
                              ▼
                        Fused Features
                              ▼
                        ┌──────────────┐
                        │ Sensitivity  │
                        │ Head         │
                        └──────────────┘
                        ┌──────────────┐
                        │ Specificity  │
                        │ Head         │
                        └──────────────┘
                              ▼
                        Learnable Threshold
                              ▼
                        Final Prediction
```

**Characteristics:**
- Fusion Type: Cross-Attention, Dual-Head, Learnable Threshold
- Loss: Class-balanced (penalizes false positives)

### 2.4.5 Multi-Scale Quantum Fusion

```python
# src/models/fusion/multi_scale_quantum.py

Image → CNN → [Scale1(56×56), Scale2(28×28), Scale3(14×14)]
                  ↓              ↓              ↓
              QENN-1         QENN-2         QENN-3
              (RY-only)     (RY+RZ)        (U3)
                  ↓             ↓              ↓
                  └──────┬──────┴──────────────┘
                         ▼
                 Multi-Scale Attention Fusion
                         ▼
                 Classical Refinement → Logits
```

**Characteristics:**
- Fusion Type: Multi-Scale, Parallel Quantum, Attention Fusion
- Innovation: Different quantum configs per scale

## 2.5 Existing Gating/Routing Mechanisms

### 2.5.1 DualBranch Gating Network

```python
# src/models/fusion/gating.py

class GatingNetwork(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),  # α ∈ (0, 1)
        )
    
    def forward(self, feat1, feat2):
        concat = torch.cat([feat1, feat2], dim=1)
        return self.gate(concat)  # Spatial gating weights
```

**Characteristics:**
- ❌ **Binary gating** (only 2 branches)
- ❌ **No sparsity** (both branches always active)
- ❌ **No expert specialization** (branches are fixed architectures)
- ✅ **Entropy regularization** (encourages balanced usage)

### 2.5.2 TBCA-Fusion Branch Weights

```python
# src/models/fusion/triple_branch.py

self.branch_weights = nn.Parameter(torch.ones(3))  # Learnable, not dynamic
# Forward:
weights = F.softmax(self.branch_weights, dim=0)  # Static across samples
fused = weights[0]*swin + weights[1]*convnext + weights[2]*effnet
```

**Characteristics:**
- ❌ **Static weights** (same for all samples)
- ❌ **No conditional computation** (all branches always run)

### 2.5.3 CB-QCCF Dual-Head

```python
# src/models/fusion/class_balanced_quantum.py

self.threshold = nn.Parameter(torch.tensor(0.5))
final_pred = sens_pred * threshold + spec_pred * (1 - threshold)
```

**Characteristics:**
- ✅ **Learnable threshold** (adaptively balances heads)
- ❌ **Not sample-specific** (single threshold for all)

## 2.6 Existing Attention Mechanisms

### 2.6.1 Current Attention Types

| Model | Attention Type | Implementation | Purpose |
|-------|---------------|----------------|---------|
| **Swin** | Shifted Window Self-Attention | timm | Local + global context |
| **ViT/DeiT** | Global Self-Attention | timm/custom | Global context |
| **Hybrid CNN+ViT** | Self-Attention (Transformer Encoder) | custom | Refine CNN features |
| **TBCA-Fusion** | Cross-Attention + Self-Attention | custom | Inter-branch fusion |
| **CB-QCCF** | Cross-Attention (MultiheadAttention) | PyTorch | Classical↔Quantum fusion |
| **MSQF** | Multi-Scale Self-Attention | PyTorch | Scale fusion |

### 2.6.2 TBCA-Fusion Cross-Attention Implementation

```python
# src/models/fusion/triple_branch.py

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(dim)
    
    def forward(self, query, key, value):
        q = self.q_proj(query).unsqueeze(1)  # (B, 1, dim)
        k = self.k_proj(key).unsqueeze(1)
        v = self.v_proj(value).unsqueeze(1)
        
        # Multi-head attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, v)
        
        out = self.out_proj(attn_out.squeeze(1))
        return self.layer_norm(query + out)  # Residual + LN
```

---

# 3. Research Paper Integration

## 3.1 Paper Overview

**Title:** "Attention-Enhanced Mixture of Experts for Multimodal Breast Cancer Detection"

**Key Concepts:**

### 3.1.1 Core Problem

- **Multimodal medical data** (images + clinical/tabular data)
- **Static fusion is suboptimal** - same weights for all patients
- **Need dynamic, patient-specific modality weighting**

### 3.1.2 Architecture Components

#### A. Mixture of Experts (MoE)

```
Separate experts per modality:
  - Image Expert (CNN → feature vector)
  - Tabular Expert (MLP → feature vector)

Gating Network:
  - Learns weights per sample
  - Produces: α_img, α_tab
  - Fusion: h_fused = α_img * h_img + α_tab * h_tab
```

#### B. Multi-Co-Guided Attention (MCGA)

```
Bidirectional cross-attention:
  - Image attends to tabular
  - Tabular attends to image
Enables feature-level interaction BEFORE fusion
```

#### C. Variants to Understand

1. **Baseline MoE** (no cross-modal interaction)
2. **MCGA** (bidirectional attention)
3. **MCGA + Intra-modality self-attention**
4. **MCGA + Cosine similarity attention** (BEST)
5. **Transformer-style unidirectional attention**

### 3.1.3 Key Insights from Paper

- ✅ **Adaptive fusion > static fusion**
- ✅ **Bidirectional cross-modal interaction is critical**
- ✅ **Cosine similarity improves cross-modal alignment**
- ✅ **Attention improves interpretability and performance**

---

# 4. Gap Analysis

## 4.1 Critical Limitations

### ❌ Gap 1: No True Multimodal Fusion

**Current State:**
- Images and clinical data processed independently
- No cross-modal attention between image features and patient metadata
- Lost opportunity: Age, tumor size, grade could guide image interpretation

**Required:**
- Explicit image↔clinical cross-attention
- Patient metadata as guidance signals
- Modality importance weighting per sample

### ❌ Gap 2: Static Architecture Per Sample

**Current State:**
- All samples use identical model architecture
- No patient-specific adaptation
- Simple cases use same compute as complex cases

**Required:**
- Dynamic expert routing (top-k of N experts)
- Sample-adaptive computation
- Conditional computation based on case complexity

### ❌ Gap 3: No Sparse Expert Routing

**Current State:**
- Current fusion: All branches always active (dense)
- MoE would enable: Top-2 of 8 experts (sparse, 4× efficiency gain)

**Required:**
- Sparse gating (top-k selection)
- Load balancing auxiliary loss
- Expert capacity constraints

### ❌ Gap 4: Limited Guidance Signals

**Current State:**
- Attention is self-guided (features attend to features)
- No external guidance (clinical, pathology reports, radiologist notes)

**Required:**
- Clinical metadata as attention guidance
- Multi-source guidance fusion
- Guidance-modulated attention computation

### ❌ Gap 5: No Missing Modality Handling

**Current State:**
- If clinical data unavailable, system fails
- MoE should route around missing modalities gracefully

**Required:**
- Modality availability masking
- Fallback routing strategies
- Robust fusion with partial modalities

### ❌ Gap 6: Interpretability Gaps

**Current State:**
- Can't explain WHY a prediction was made
- No expert attribution (which expert contributed most?)
- No modality importance (was image or clinical more decisive?)

**Required:**
- Expert attribution scores
- Modality importance weights
- Attention visualization

### ❌ Gap 7: Scalability Limits

**Current State:**
- Adding new modalities requires new model
- Can't incrementally add experts
- No multi-task learning (benign/malignant + subtype + grading)

**Required:**
- Modular expert addition
- Multi-task heads
- Incremental architecture growth

---

# 5. Proposed Architecture

## 5.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│              NEXT-GENERATION MoE-MCGA ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  INPUT MODALITIES (Patient-Specific):                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │ Histopath    │  │ Mammography  │  │ Clinical     │              │
│  │ Image        │  │ Image        │  │ Metadata     │              │
│  │ (224×224×3)  │  │ (224×224×3)  │  │ (10-D vector)│              │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘              │
│         │                 │                 │                       │
│         ▼                 ▼                 ▼                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │           MODALITY-SPECIFIC EXPERTS (Backbones)         │       │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │       │
│  │  │ Image Expert│ │ Mammo Expert│ │ Clinical    │       │       │
│  │  │ Swin-Small  │ │ ConvNeXt    │ │ MLP Encoder │       │       │
│  │  │ (B, 768)    │ │ (B, 768)    │ │ (B, 128)    │       │       │
│  │  └─────────────┘ └─────────────┘ └─────────────┘       │       │
│  └─────────────────────────────────────────────────────────┘       │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │         MULTI-CO-GUIDED ATTENTION (MCGA)                │       │
│  │  - Image attends to Clinical (clinical guides vision)   │       │
│  │  - Clinical attends to Image (image informs metadata)   │       │
│  │  - Mammo attends to Histopath (cross-modality align)    │       │
│  │  - Self-attention within each modality                  │       │
│  └─────────────────────────────────────────────────────────┘       │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │         MOE FUSION LAYER (Dynamic Routing)              │       │
│  │  ┌─────────────────────────────────────────────────┐   │       │
│  │  │  Gating Network (Patient-Adaptive)              │   │       │
│  │  │  Input: Concat(Image, Mammo, Clinical)          │   │       │
│  │  │  Output: Top-2 expert weights (sparse softmax)  │   │       │
│  └─────────────────────────────────────────────────┘   │       │
│  │  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐   │       │
│  │  │Expert1│ │Expert2│ │Expert3│ │Expert4│ │Expert5│   │       │
│  │  │Global │ │Local  │ │Multi- │ │Quantum│ │Clinical│   │       │
│  │  │Context│ │Texture│ │scale  │ │Enhanced│ │Focused │   │       │
│  │  └───────┘ └───────┘ └───────┘ └───────┘ └───────┘   │       │
│  └─────────────────────────────────────────────────────────┘       │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │         TASK-SPECIFIC HEADS (Multi-Task Learning)       │       │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐       │       │
│  │  │ Binary      │ │ Multi-Class │ │ Subtype     │       │       │
│  │  │ Classifier  │ │ Classifier  │ │ Classifier  │       │       │
│  │  │ (Benign/    │ │ (Grade 1-4) │ │ (Ductal/    │       │       │
│  │  │  Malignant) │ │             │ │  Lobular)   │       │       │
│  │  └─────────────┘ └─────────────┘ └─────────────┘       │       │
│  └─────────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

## 5.2 Data Flow (Step-by-Step)

### Step 1: Multi-Modal Input

```
Patient #12345:
  - Histopath image: 224×224×3 RGB
  - Mammography image: 224×224×3 grayscale→RGB
  - Clinical metadata: [age=52, tumor_size=2.3cm, grade=2, ...]
```

### Step 2: Modality Encoding

```python
F_histopath = SwinSmall(histopath_image)      # (B, 768)
F_mammo = ConvNeXt(mammo_image)               # (B, 768)
F_clinical = MLP(clinical_metadata)           # (B, 128)
```

### Step 3: MCGA Enhancement

```python
# Clinical guides image interpretation
F_histopath_enhanced = CrossAttn(
    Q=F_histopath,
    K=F_histopath + Project(F_clinical),
    V=F_histopath
)

# Image informs clinical relevance
F_clinical_enhanced = CrossAttn(
    Q=F_clinical,
    K=F_clinical + Project(F_histopath),
    V=F_clinical
)
```

### Step 4: MoE Routing

```python
# Concatenate all enhanced features
F_concat = Concat(F_histopath_enhanced, F_mammo_enhanced, F_clinical_enhanced)

# Gating network computes expert weights
gate_logits = GateNetwork(F_concat)           # (B, num_experts)
gate_probs = SparseTopK(gate_logits, k=2)     # (B, num_experts), sparse

# Route to experts
expert_outputs = Stack([expert(F_concat) for expert in experts])
F_moe = Sum(gate_probs * expert_outputs)      # (B, fusion_dim)
```

### Step 5: Multi-Task Prediction

```python
pred_binary = BinaryHead(F_moe)               # Benign/Malignant
pred_grade = GradeHead(F_moe)                 # Grade 1-4
pred_subtype = SubtypeHead(F_moe)             # Ductal/Lobular
```

### Step 6: Interpretability Output

```
- Expert attribution: [Expert1: 45%, Expert3: 55%]
- Modality importance: [Histopath: 60%, Mammo: 25%, Clinical: 15%]
- Attention maps: Grad-CAM on histopath regions
```

---

# 6. MoE Integration Plan

## 6.1 Experts Definition

### 6.1.1 Expert Specifications

| Expert ID | Name | Specialization | Architecture | When Activated |
|-----------|------|---------------|--------------|----------------|
| **Expert 1** | Global-Context | Tissue architecture | Swin-Small FFN | High-grade tumors |
| **Expert 2** | Local-Texture | Cellular morphology | ConvNeXt FFN | Early-stage cancer |
| **Expert 3** | Multi-Scale | Nuclei + tissue | Multi-resolution FFN | Ambiguous cases |
| **Expert 4** | Quantum-Enhanced | Subtle patterns | QENN-based FFN | Complex boundaries |
| **Expert 5** | Clinical-Focused | Metadata-driven | Clinical MLP | Strong indicators |
| **Expert 6** | Cross-Modal | Image+clinical fusion | Fusion FFN | Multi-modal cases |
| **Expert 7** | Uncertainty | Low-confidence cases | Bayesian FFN | Edge cases |
| **Expert 8** | Rare-Pattern | Unusual presentations | Specialized FFN | Rare subtypes |

### 6.1.2 Expert Architecture (Identical Initialization)

```python
# All experts start identical, specialize through training
expert = nn.Sequential(
    nn.Linear(fusion_dim, fusion_dim * 2),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(fusion_dim * 2, fusion_dim),
    nn.LayerNorm(fusion_dim),
)
```

## 6.2 Gating Network Design

### 6.2.1 Complete Implementation

```python
# src/models/moe/gating.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class MoEGatingNetwork(nn.Module):
    """
    Patient-Adaptive Sparse Gating Network
    
    Routes each patient to the most relevant experts
    based on their multi-modal feature representation.
    """
    
    def __init__(
        self,
        input_dim: int = 1664,  # 768 + 768 + 128
        num_experts: int = 8,
        top_k: int = 2,
        hidden_dim: int = 512,
        noise_variance: float = 0.1,  # For exploration
        load_balance_weight: float = 0.01,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.noise_variance = noise_variance
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_experts),
        )
        
        # Load balancing parameters
        self.load_balance_weight = load_balance_weight
    
    def forward(
        self,
        x: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute sparse expert routing.
        
        Returns:
            gate_probs: Sparse softmax weights (B, num_experts)
            expert_indices: Top-k expert indices (B, top_k)
            aux_loss: Load balancing auxiliary loss
        """
        # Add noise for exploration during training
        if training and self.noise_variance > 0:
            x = x + torch.randn_like(x) * self.noise_variance
        
        # Compute gating logits
        gate_logits = self.gate(x)  # (B, num_experts)
        
        # Sparse top-k routing
        gate_probs = F.softmax(gate_logits, dim=-1)
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Create sparse gate (zero out non-top-k)
        sparse_gate = torch.zeros_like(gate_probs)
        sparse_gate.scatter_(1, top_k_indices, top_k_probs)
        
        # Compute load balancing auxiliary loss
        aux_loss = self._compute_load_balance_loss(gate_probs)
        
        # Compute expert usage statistics
        expert_usage = (gate_probs > 0.01).float().mean(dim=0)
        
        info = {
            'aux_loss': aux_loss,
            'expert_usage': expert_usage,
            'gate_entropy': self._compute_entropy(gate_probs),
        }
        
        return sparse_gate, top_k_indices, aux_loss, info
    
    def _compute_load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Encourage balanced expert usage across batch.
        
        Loss = CV(expert_usage)²  (coefficient of variation)
        """
        # Average usage per expert
        usage = gate_probs.mean(dim=0)  # (num_experts,)
        
        # Coefficient of variation
        cv = usage.std() / (usage.mean() + 1e-8)
        
        return self.load_balance_weight * (cv ** 2)
    
    def _compute_entropy(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """Compute gating entropy (measure of routing confidence)."""
        eps = 1e-8
        entropy = -(gate_probs * torch.log(gate_probs + eps)).sum(dim=-1).mean()
        return entropy
```

## 6.3 Fusion Logic

### 6.3.1 MoE Fusion Layer Implementation

```python
# src/models/fusion/moe_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class MoEFusionLayer(nn.Module):
    """
    Mixture of Experts Fusion Layer
    
    Dynamically routes samples to specialized experts
    and combines their outputs.
    """
    
    def __init__(
        self,
        input_dim: int = 1664,
        fusion_dim: int = 1024,
        num_experts: int = 8,
        top_k: int = 2,
        expert_dropout: float = 0.3,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(expert_dropout),
        )
        
        # Expert FFNs (each expert is a specialized MLP)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Dropout(expert_dropout),
                nn.Linear(fusion_dim * 2, fusion_dim),
                nn.LayerNorm(fusion_dim),
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating = MoEGatingNetwork(
            input_dim=input_dim,
            num_experts=num_experts,
            top_k=top_k,
        )
        
        # Output projection
        self.output_proj = nn.Linear(fusion_dim, fusion_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_info: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass through MoE fusion layer.
        
        Args:
            x: Input features (B, input_dim)
            return_info: Return gating statistics
        
        Returns:
            output: Fused features (B, fusion_dim)
            info: Gating statistics (if requested)
        """
        # Project input
        x_proj = self.input_proj(x)  # (B, fusion_dim)
        
        # Compute gating
        gate_probs, expert_indices, aux_loss, gate_info = self.gating(x_proj)
        
        # Route to experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Get samples routed to this expert
            expert_mask = (expert_indices == i).any(dim=-1)  # (B,)
            
            if expert_mask.any():
                # Process through expert
                expert_out = expert(x_proj[expert_mask])
                
                # Weight by gate probability
                expert_weight = gate_probs[expert_mask, i:i+1]
                expert_outputs.append(expert_out * expert_weight)
        
        # Sum expert contributions
        if expert_outputs:
            output = torch.cat(expert_outputs, dim=0)
            output = output.view(x.shape[0], -1, x_proj.shape[1]).sum(dim=1)
        else:
            output = torch.zeros_like(x_proj)
        
        # Output projection with residual
        output = self.output_proj(output)
        output = output + x_proj  # Residual connection
        
        info = {
            'aux_loss': aux_loss,
            'gate_info': gate_info,
            'expert_indices': expert_indices,
        } if return_info else {}
        
        return output, info
```

---

# 7. MCGA Integration Plan

## 7.1 True Bidirectional MCGA Architecture

### 7.1.1 Complete Implementation

```python
# src/models/attention/mcga.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict


class TrueMultiCoGuidedAttention(nn.Module):
    """
    TRUE Multi-Co-Guided Attention (MCGA)
    
    Bidirectional cross-modal attention:
    - Image attends to Clinical (clinical guides vision)
    - Clinical attends to Image (image informs metadata)
    - BOTH directions are computed explicitly
    
    This is DIFFERENT from guided self-attention!
    """
    
    def __init__(
        self,
        image_dim: int = 768,
        clinical_dim: int = 128,
        mammo_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_type: str = 'cosine',  # 'dot_product' or 'cosine'
    ):
        super().__init__()
        self.image_dim = image_dim
        self.clinical_dim = clinical_dim
        self.mammo_dim = mammo_dim
        self.num_heads = num_heads
        self.attention_type = attention_type
        
        # ── Modality Projections (to common dim) ─────────────────
        self.clinical_to_common = nn.Sequential(
            nn.Linear(clinical_dim, 256),
            nn.GELU(),
            nn.Linear(256, image_dim),
            nn.LayerNorm(image_dim),
        )
        
        self.mammo_to_common = nn.Sequential(
            nn.Linear(mammo_dim, image_dim),
            nn.LayerNorm(image_dim),
        )
        
        # ── Bidirectional Cross-Attention Blocks ─────────────────
        
        # BLOCK 1: Image ← Clinical (Clinical guides Image)
        self.image_from_clinical_attn = CrossModalAttention(
            query_dim=image_dim,
            key_value_dim=clinical_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.norm_img_clin = nn.LayerNorm(image_dim)
        
        # BLOCK 2: Clinical ← Image (Image informs Clinical)
        self.clinical_from_image_attn = CrossModalAttention(
            query_dim=clinical_dim,
            key_value_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.norm_clin_img = nn.LayerNorm(clinical_dim)
        
        # BLOCK 3: Image ← Mammo (Mammo guides Image)
        self.image_from_mammo_attn = CrossModalAttention(
            query_dim=image_dim,
            key_value_dim=mammo_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.norm_img_mammo = nn.LayerNorm(image_dim)
        
        # BLOCK 4: Mammo ← Image (Image informs Mammo)
        self.mammo_from_image_attn = CrossModalAttention(
            query_dim=mammo_dim,
            key_value_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_type=attention_type,
        )
        self.norm_mammo_img = nn.LayerNorm(mammo_dim)
        
        # ── Self-Attention Refinement ────────────────────────────
        self.image_self_attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_image_self = nn.LayerNorm(image_dim)
        
        self.clinical_self_attn = nn.MultiheadAttention(
            embed_dim=clinical_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_clinical_self = nn.LayerNorm(clinical_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        image_features: torch.Tensor,      # (B, N_img, image_dim)
        clinical_features: torch.Tensor,    # (B, clinical_dim)
        mammo_features: Optional[torch.Tensor] = None,  # (B, mammo_dim) or None
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Dict]:
        """
        TRUE Bidirectional MCGA Forward Pass
        
        Step 1: Image attends to Clinical
        Step 2: Clinical attends to Image
        Step 3: Image attends to Mammo (if available)
        Step 4: Mammo attends to Image (if available)
        Step 5: Self-attention refinement
        """
        batch_size = image_features.shape[0]
        attention_info = {}
        
        # ── STEP 1: Image ← Clinical (Clinical guides Image) ─────
        # Query: Image, Key/Value: Clinical
        image_enhanced = self.image_from_clinical_attn(
            query=image_features,
            key=clinical_features.unsqueeze(1),  # (B, 1, clin_dim)
            value=clinical_features.unsqueeze(1),
        )
        image_enhanced = self.norm_img_clin(image_features + self.dropout(image_enhanced))
        attention_info['image_clinical_attn'] = True
        
        # ── STEP 2: Clinical ← Image (Image informs Clinical) ───
        # Query: Clinical, Key/Value: Image
        clinical_enhanced = self.clinical_from_image_attn(
            query=clinical_features.unsqueeze(1),
            key=image_features,
            value=image_features,
        )
        clinical_enhanced = self.norm_clin_img(
            clinical_features.unsqueeze(1) + self.dropout(clinical_enhanced)
        ).squeeze(1)  # (B, clinical_dim)
        attention_info['clinical_image_attn'] = True
        
        # ── STEP 3 & 4: Image ↔ Mammo (if mammo available) ──────
        if mammo_features is not None:
            # Image ← Mammo
            image_from_mammo = self.image_from_mammo_attn(
                query=image_enhanced,
                key=mammo_features.unsqueeze(1),
                value=mammo_features.unsqueeze(1),
            )
            image_enhanced = self.norm_img_mammo(
                image_enhanced + self.dropout(image_from_mammo)
            )
            
            # Mammo ← Image
            mammo_enhanced = self.mammo_from_image_attn(
                query=mammo_features.unsqueeze(1),
                key=image_features,
                value=image_features,
            )
            mammo_enhanced = self.norm_mammo_img(
                mammo_features.unsqueeze(1) + self.dropout(mammo_enhanced)
            ).squeeze(1)
            
            attention_info['mammo_available'] = True
        else:
            mammo_enhanced = None
            attention_info['mammo_available'] = False
        
        # ── STEP 5: Self-Attention Refinement ────────────────────
        # Image self-attention
        image_self, img_self_weights = self.image_self_attn(
            image_enhanced, image_enhanced, image_enhanced
        )
        image_final = self.norm_image_self(image_enhanced + self.dropout(image_self))
        
        # Clinical self-attention
        clinical_self, clin_self_weights = self.clinical_self_attn(
            clinical_enhanced.unsqueeze(1),
            clinical_enhanced.unsqueeze(1),
            clinical_enhanced.unsqueeze(1),
        )
        clinical_final = self.norm_clinical_self(
            clinical_enhanced.unsqueeze(1) + self.dropout(clinical_self)
        ).squeeze(1)
        
        if return_attention:
            attention_info['image_self_weights'] = img_self_weights
            attention_info['clinical_self_weights'] = clin_self_weights
        
        return image_final, clinical_final, mammo_enhanced, attention_info


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Block
    
    Generic cross-attention where query and key/value can have
    DIFFERENT dimensions (for cross-modal attention).
    """
    
    def __init__(
        self,
        query_dim: int,
        key_value_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_type: str = 'cosine',
    ):
        super().__init__()
        self.query_dim = query_dim
        self.key_value_dim = key_value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.attention_type = attention_type
        
        # Projections
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_value_dim, query_dim)
        self.v_proj = nn.Linear(key_value_dim, query_dim)
        self.out_proj = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for cosine attention
        if attention_type == 'cosine':
            self.scale = nn.Parameter(torch.ones(1) * 10.0)
        else:
            self.scale = query_dim ** -0.5
        
        self._init_weights()
    
    def _init_weights(self):
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
    ) -> torch.Tensor:
        """
        Cross-modal attention forward pass.
        
        Args:
            query: (B, N_q, query_dim)
            key: (B, N_kv, key_value_dim)
            value: (B, N_kv, key_value_dim)
        
        Returns:
            output: (B, N_q, query_dim)
        """
        batch_size, seq_len_q = query.shape[:2]
        seq_len_kv = key.shape[1]
        
        # Project Q, K, V
        Q = self.q_proj(query)  # (B, N_q, dim)
        K = self.k_proj(key)    # (B, N_kv, dim)
        V = self.v_proj(value)  # (B, N_kv, dim)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        if self.attention_type == 'cosine':
            # Cosine similarity attention
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        else:
            # Dot-product attention
            attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply to values
        attn_out = torch.matmul(attn_weights, V)
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.query_dim)
        
        # Output projection
        output = self.out_proj(attn_out)
        
        return output
```

## 7.2 MCGA Insertion Points

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MCGA INSERTION POINTS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Option 1: After Modality Encoders (RECOMMENDED)                    │
│  ──────────────────────────────────────                             │
│  Image → Encoder → [MCGA] → MoE Fusion → Head                       │
│  Clinical → Encoder → ─┘                                            │
│                                                                      │
│  Option 2: Inside MoE Experts                                       │
│  ───────────────────────────                                        │
│  Image → Encoder → MoE → [MCGA inside each expert] → Head           │
│                                                                      │
│  Option 3: Before Classification Head                               │
│  ────────────────────────────────                                   │
│  Image → Encoder → MoE → [MCGA] → Head                              │
│  Clinical → ────────────────→ ─┘                                    │
│                                                                      │
│  RECOMMENDED: Option 1 + Option 3 (dual MCGA layers)                │
└─────────────────────────────────────────────────────────────────────┘
```

---

# 8. Training Strategy

## 8.1 Complete MoE Loss Function

```python
# src/utils/moe_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class MoELoss(nn.Module):
    """
    Complete MoE Loss with Multiple Components
    
    Total Loss = λ₁·L_cls + λ₂·L_aux + λ₃·L_div + λ₄·L_entropy
    
    Where:
        L_cls: Classification loss (cross-entropy)
        L_aux: Load balancing auxiliary loss
        L_div: Expert diversity loss (prevent collapse)
        L_entropy: Routing entropy regularization
    """
    
    def __init__(
        self,
        cls_weight: float = 1.0,
        aux_weight: float = 0.01,
        div_weight: float = 0.1,
        entropy_weight: float = 0.001,
        num_classes: int = 2,
    ):
        super().__init__()
        self.cls_weight = cls_weight
        self.aux_weight = aux_weight
        self.div_weight = div_weight
        self.entropy_weight = entropy_weight
        self.num_classes = num_classes
        
        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        gate_probs: torch.Tensor,
        expert_outputs: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute complete MoE loss.
        
        Args:
            logits: Final predictions (B, num_classes)
            targets: Ground truth labels (B,)
            gate_probs: Gating probabilities (B, num_experts)
            expert_outputs: Individual expert outputs (B, num_experts, dim)
            expert_indices: Top-k expert indices (B, top_k)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Individual loss components
        """
        # 1. Classification Loss (Cross-Entropy)
        L_cls = self.ce_loss(logits, targets)
        
        # 2. Load Balancing Auxiliary Loss
        L_aux = self._load_balance_loss(gate_probs)
        
        # 3. Expert Diversity Loss (prevent collapse)
        L_div = self._diversity_loss(expert_outputs, expert_indices)
        
        # 4. Routing Entropy Regularization
        L_entropy = self._entropy_loss(gate_probs)
        
        # Total loss
        total_loss = (
            self.cls_weight * L_cls +
            self.aux_weight * L_aux +
            self.div_weight * L_div +
            self.entropy_weight * L_entropy
        )
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'cls_loss': L_cls.item(),
            'aux_loss': L_aux.item(),
            'div_loss': L_div.item(),
            'entropy_loss': L_entropy.item(),
        }
        
        return total_loss, loss_dict
    
    def _load_balance_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Load Balancing Loss (from Switch Transformer)
        
        Encourages equal expert usage across batch.
        
        Loss = num_experts * CV(expert_usage)²
        
        Where CV = coefficient of variation = std/mean
        """
        num_experts = gate_probs.shape[1]
        
        # Fraction of total capacity per expert
        # (how much each expert is used)
        expert_usage = gate_probs.mean(dim=0)  # (num_experts,)
        
        # Coefficient of variation
        cv = expert_usage.std() / (expert_usage.mean() + 1e-8)
        
        # Loss: minimize CV (encourage uniform usage)
        return num_experts * (cv ** 2)
    
    def _diversity_loss(self, expert_outputs: torch.Tensor, 
                       expert_indices: torch.Tensor) -> torch.Tensor:
        """
        Expert Diversity Loss
        
        Prevents expert collapse (all experts learning same thing).
        
        Encourages experts to produce DIFFERENT outputs.
        """
        batch_size, num_experts, dim = expert_outputs.shape
        
        # Normalize expert outputs
        expert_norm = F.normalize(expert_outputs, dim=-1)
        
        # Compute pairwise cosine similarity between experts
        # similarity[i,j] = cos(expert_i, expert_j)
        similarity = torch.bmm(expert_norm, expert_norm.transpose(-2, -1))
        
        # Mask diagonal (self-similarity = 1 is OK)
        mask = torch.eye(num_experts, device=expert_outputs.device).unsqueeze(0)
        similarity = similarity * (1 - mask)
        
        # Loss: minimize average pairwise similarity
        # (encourage experts to be different)
        return (similarity.sum() / (batch_size * num_experts * (num_experts - 1)))
    
    def _entropy_loss(self, gate_probs: torch.Tensor) -> torch.Tensor:
        """
        Routing Entropy Loss
        
        Two competing objectives:
        1. Low entropy per sample (confident routing)
        2. High entropy across batch (balanced usage)
        
        This loss encourages CONFIDENT routing (low per-sample entropy)
        """
        eps = 1e-8
        
        # Per-sample entropy
        entropy = -(gate_probs * torch.log(gate_probs + eps)).sum(dim=-1)
        
        # Mean across batch
        return entropy.mean()
```

## 8.2 Phased Training Schedule

### 8.2.1 Training Phases

```python
# src/utils/moe_trainer.py

class MoETrainer:
    """
    Complete MoE Training Strategy with Phases
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        moe_loss: MoELoss,
        device: torch.device,
        use_amp: bool = True,
        grad_clip: float = 1.0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.moe_loss = moe_loss
        self.device = device
        self.use_amp = use_amp
        self.grad_clip = grad_clip
        self.scaler = torch.amp.GradScaler('cuda' if device.type == 'cuda' else 'cpu')
        
        # Training phase tracking
        self.current_phase = 'warmup'
        self.warmup_epochs = 5
    
    def train_one_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """
        Training with MoE-specific considerations
        """
        # Update phase
        if epoch >= self.warmup_epochs:
            self.current_phase = 'full_moe'
        
        self.model.train()
        running_loss = 0.0
        loss_components = {
            'total': 0.0,
            'cls': 0.0,
            'aux': 0.0,
            'div': 0.0,
            'entropy': 0.0,
        }
        
        for batch_idx, (images, clinical, targets) in enumerate(loader):
            images = images.to(self.device)
            clinical = clinical.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.amp.autocast('cuda' if self.device.type == 'cuda' else 'cpu', 
                                   enabled=self.use_amp):
                # Forward pass
                outputs = self.model(images, clinical, return_moe_info=True)
                logits = outputs['logits']
                
                # Get MoE-specific outputs
                gate_probs = outputs['gate_probs']
                expert_outputs = outputs['expert_outputs']
                expert_indices = outputs['expert_indices']
                
                # Compute loss
                loss, loss_dict = self.moe_loss(
                    logits=logits,
                    targets=targets,
                    gate_probs=gate_probs,
                    expert_outputs=expert_outputs,
                    expert_indices=expert_indices,
                )
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (critical for MoE stability)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update running loss
            running_loss += loss.item()
            for key in loss_components:
                if key in loss_dict:
                    loss_components[key] += loss_dict[key]
        
        # Compute averages
        num_batches = len(loader)
        avg_loss = {
            key: value / num_batches for key, value in loss_components.items()
        }
        
        # Update scheduler
        self.scheduler.step()
        
        return avg_loss
    
    def get_training_schedule(self, total_epochs: int) -> Dict:
        """
        MoE Training Schedule
        
        Phase 1 (Epochs 0-5): Warmup without MoE
        Phase 2 (Epochs 6-20): Enable gating with high aux_weight
        Phase 3 (Epochs 21+): Full training with balanced losses
        """
        return {
            'warmup': {
                'epochs': list(range(0, 5)),
                'moe_enabled': False,
                'aux_weight': 0.0,
                'div_weight': 0.0,
                'lr_multiplier': 0.5,
                'description': 'Backbone pre-training without MoE',
            },
            'gating_warmup': {
                'epochs': list(range(5, 15)),
                'moe_enabled': True,
                'aux_weight': 0.1,  # High to encourage exploration
                'div_weight': 0.2,
                'lr_multiplier': 1.0,
                'description': 'Enable gating with high aux loss',
            },
            'full_training': {
                'epochs': list(range(15, total_epochs)),
                'moe_enabled': True,
                'aux_weight': 0.01,
                'div_weight': 0.1,
                'lr_multiplier': 1.0,
                'description': 'Full MoE training',
            },
        }
```

### 8.2.2 Training Configuration (YAML)

```yaml
# config.yaml - Add to training section

training:
  # MoE-specific training strategy
  moe:
    # Phase 1: Warmup (epochs 0-5)
    warmup_epochs: 5
    warmup_lr_multiplier: 0.5
    warmup_moe_enabled: false
    
    # Phase 2: Gating warmup (epochs 6-15)
    gating_warmup_epochs: 10
    gating_warmup_aux_weight: 0.1
    gating_warmup_div_weight: 0.2
    
    # Phase 3: Full training (epochs 16+)
    full_aux_weight: 0.01
    full_div_weight: 0.1
    entropy_weight: 0.001
    
    # Gradient clipping (critical for MoE)
    grad_clip: 1.0
    
    # Expert dropout
    expert_dropout: 0.3
    
    # Gating noise (for exploration)
    gating_noise_variance: 0.1
  
  # Standard training params
  epochs: 50
  patience: 10
  lr: 2.0e-5
  weight_decay: 1.0e-4
```

---

# 9. Data Alignment Strategy

## 9.1 Multimodal Dataset Implementation

```python
# src/data/multimodal_dataset.py

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class AlignedMultimodalDataset(Dataset):
    """
    CRITICAL: Aligns multiple modalities per patient
    
    Handles:
    - Histopathology images
    - Mammography images (optional)
    - Clinical metadata
    - Patient-level alignment
    """
    
    def __init__(
        self,
        histopath_dir: str,
        clinical_csv: str,
        mammo_dir: Optional[str] = None,
        transform_histopath = None,
        transform_mammo = None,
        modality: str = 'all',  # 'all', 'image_only', 'clinical_only'
        missing_modality_strategy: str = 'zero',  # 'zero', 'mean', 'drop'
    ):
        self.histopath_dir = histopath_dir
        self.mammo_dir = mammo_dir
        self.transform_histopath = transform_histopath
        self.transform_mammo = transform_mammo
        self.modality = modality
        self.missing_strategy = missing_modality_strategy
        
        # Load clinical data
        self.clinical_df = pd.read_csv(clinical_csv)
        
        # Create patient_id → index mapping
        self.patient_to_idx = {
            pid: idx for idx, pid in enumerate(self.clinical_df['patient_id'].values)
        }
        
        # Align modalities
        self.aligned_samples = self._align_modalities()
        
        # Compute clinical feature statistics (for imputation)
        self.clinical_mean = self.clinical_df.drop(columns=['patient_id', 'label']).mean()
        self.clinical_std = self.clinical_df.drop(columns=['patient_id', 'label']).std()
    
    def _align_modalities(self) -> List[Dict]:
        """
        CRITICAL: Align all modalities per patient
        
        Returns list of:
        {
            'patient_id': str,
            'histopath_path': str or None,
            'mammo_path': str or None,
            'clinical_features': np.array,
            'label': int,
        }
        """
        aligned = []
        
        for idx, row in self.clinical_df.iterrows():
            patient_id = row['patient_id']
            label = row['label']
            
            # Clinical features (normalize)
            clinical_features = row.drop(['patient_id', 'label']).values.astype(np.float32)
            clinical_features = (clinical_features - self.clinical_mean) / self.clinical_std
            
            # Find histopath image
            histopath_path = self._find_histopath_image(patient_id)
            
            # Find mammography image
            mammo_path = self._find_mammo_image(patient_id) if self.mammo_dir else None
            
            aligned.append({
                'patient_id': patient_id,
                'histopath_path': histopath_path,
                'mammo_path': mammo_path,
                'clinical_features': clinical_features,
                'label': label,
            })
        
        return aligned
    
    def _find_histopath_image(self, patient_id: str) -> Optional[str]:
        """Find histopathology image for patient."""
        # Search in histopath directory
        for ext in ['png', 'jpg', 'jpeg']:
            path = os.path.join(self.histopath_dir, f"{patient_id}.{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def _find_mammo_image(self, patient_id: str) -> Optional[str]:
        """Find mammography image for patient."""
        for ext in ['png', 'jpg', 'jpeg', 'dcm']:
            path = os.path.join(self.mammo_dir, f"{patient_id}.{ext}")
            if os.path.exists(path):
                return path
        return None
    
    def __len__(self) -> int:
        return len(self.aligned_samples)
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Get aligned multimodal sample.
        
        Returns:
            modalities: Dict with available modalities
            label: Ground truth label
        """
        sample = self.aligned_samples[idx]
        modalities = {}
        
        # Load histopath image
        if sample['histopath_path'] is not None:
            histopath_img = Image.open(sample['histopath_path']).convert('RGB')
            if self.transform_histopath:
                histopath_img = self.transform_histopath(histopath_img)
            modalities['histopath'] = histopath_img
        else:
            # Missing modality handling
            if self.missing_strategy == 'zero':
                modalities['histopath'] = torch.zeros(3, 224, 224)
            elif self.missing_strategy == 'mean':
                modalities['histopath'] = torch.randn(3, 224, 224) * 0.01
            # else: 'drop' - handled by collate_fn
        
        # Load mammography image (if available)
        if sample['mammo_path'] is not None:
            mammo_img = Image.open(sample['mammo_path']).convert('RGB')
            if self.transform_mammo:
                mammo_img = self.transform_mammo(mammo_img)
            modalities['mammo'] = mammo_img
        else:
            if self.missing_strategy == 'zero':
                modalities['mammo'] = torch.zeros(3, 224, 224)
            elif self.missing_strategy == 'mean':
                modalities['mammo'] = torch.randn(3, 224, 224) * 0.01
        
        # Clinical features
        modalities['clinical'] = torch.from_numpy(sample['clinical_features']).float()
        
        # Patient ID (for tracking)
        modalities['patient_id'] = sample['patient_id']
        
        label = sample['label']
        
        return modalities, label


def multimodal_collate_fn(batch, missing_strategy: str = 'zero'):
    """
    Custom collate function for multimodal batches.
    
    Handles:
    - Variable modalities per sample
    - Missing modality masking
    """
    modalities_batch = {
        'histopath': [],
        'mammo': [],
        'clinical': [],
        'patient_id': [],
    }
    labels = []
    
    # Track which samples have which modalities
    has_histopath = []
    has_mammo = []
    
    for modalities, label in batch:
        labels.append(label)
        
        # Histopath
        if 'histopath' in modalities:
            modalities_batch['histopath'].append(modalities['histopath'])
            has_histopath.append(True)
        else:
            has_histopath.append(False)
        
        # Mammo
        if 'mammo' in modalities:
            modalities_batch['mammo'].append(modalities['mammo'])
            has_mammo.append(True)
        else:
            has_mammo.append(False)
        
        # Clinical
        modalities_batch['clinical'].append(modalities['clinical'])
        modalities_batch['patient_id'].append(modalities['patient_id'])
    
    # Stack tensors
    result = {
        'histopath': torch.stack(modalities_batch['histopath']) if has_histopath else None,
        'mammo': torch.stack(modalities_batch['mammo']) if has_mammo else None,
        'clinical': torch.stack(modalities_batch['clinical']),
        'patient_id': modalities_batch['patient_id'],
        'has_histopath': torch.tensor(has_histopath),
        'has_mammo': torch.tensor(has_mammo),
    }
    
    labels = torch.tensor(labels, dtype=torch.long)
    
    return result, labels


# Example usage in config
"""
data:
  multimodal:
    histopath_dir: "data/BreaKHis_v1/images"
    clinical_csv: "data/clinical_metadata.csv"
    mammo_dir: "data/CBIS-DDSM/images"  # Optional
    missing_modality_strategy: "zero"  # or "mean", "drop"
"""
```

---

# 10. Implementation Files

## 10.1 File Structure

```
src/
├── models/
│   ├── moe/
│   │   ├── __init__.py
│   │   ├── gating.py              # MoE gating network
│   │   ├── moe_fusion.py          # MoE fusion layer
│   │   └── emergent_experts.py    # Emergent specialization
│   │
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── mcga.py                # Multi-Co-Guided Attention
│   │   └── cross_modal.py         # Cross-modal attention primitives
│   │
│   └── fusion/
│       ├── __init__.py            # Updated exports
│       ├── moe_head.py            # MoE classification head
│       └── ... (existing files)
│
├── data/
│   ├── multimodal_dataset.py      # Aligned multimodal dataset
│   └── ... (existing files)
│
├── utils/
│   ├── moe_loss.py                # Complete MoE loss
│   ├── moe_trainer.py             # Phased training strategy
│   └── ... (existing files)
│
└── ... (existing structure)
```

## 10.2 Module Exports

```python
# src/models/moe/__init__.py

from .gating import MoEGatingNetwork
from .moe_fusion import MoEFusionLayer
from .emergent_experts import EmergentExpertLayer

__all__ = [
    'MoEGatingNetwork',
    'MoEFusionLayer',
    'EmergentExpertLayer',
]

# src/models/attention/__init__.py

from .mcga import TrueMultiCoGuidedAttention, CrossModalAttention

__all__ = [
    'TrueMultiCoGuidedAttention',
    'CrossModalAttention',
]

# src/models/fusion/__init__.py (Updated)

from .moe_head import MoEHead, get_moe_head

__all__ = [
    # Existing exports
    'DualBranchDynamicFusion',
    'QuantumEnhancedFusion',
    'get_dual_branch_fusion',
    'get_quantum_enhanced_fusion',
    'TripleBranchCrossAttention',
    'get_triple_branch_fusion',
    'ClassBalancedQuantumClassicalFusion',
    'get_cb_qccf',
    'MultiScaleQuantumFusion',
    'get_multi_scale_quantum_fusion',
    'EnsembleDistillation',
    'get_ensemble_distillation',
    
    # New MoE exports
    'MoEHead',
    'get_moe_head',
]
```

## 10.3 Pipeline Integration

```python
# run_pipeline.py (Add to build_model function)

# New MoE Models
elif model_name == 'moe_head':
    model = get_moe_head(
        num_classes=num_classes,
        backbone=model_cfg.get('backbone', 'swin_small'),
        num_experts=model_cfg.get('num_experts', 8),
        top_k=model_cfg.get('top_k', 2),
        dropout=model_cfg.get('dropout', 0.3),
    )
    return model, 'MoE-Head', 'moe'

elif model_name == 'mcga_fusion':
    model = get_mcga_fusion(
        num_classes=num_classes,
        image_backbone=model_cfg.get('image_backbone', 'swin_small'),
        clinical_dim=model_cfg.get('clinical_dim', 10),
        num_heads=model_cfg.get('num_heads', 8),
        attention_type=model_cfg.get('attention_type', 'cosine'),
        dropout=model_cfg.get('dropout', 0.1),
    )
    return model, 'MCGA-Fusion', 'mcga'

elif model_name == 'moe_mcga':
    model = get_moe_mcga(
        num_classes=num_classes,
        num_experts=model_cfg.get('num_experts', 8),
        top_k=model_cfg.get('top_k', 2),
        mcga_type=model_cfg.get('mcga_type', 'bidirectional'),
    )
    return model, 'MoE-MCGA', 'moe_mcga'
```

## 10.4 Configuration (config.yaml)

```yaml
# config.yaml - Add new model sections

models:
  # MoE Models
  moe_head:
    enabled: false  # Enable for testing
    backbone: "swin_small"
    num_experts: 8
    top_k: 2
    dropout: 0.3
    batch_size: 16
    lr: 2.0e-5
    weight_decay: 1.0e-4
  
  mcga_fusion:
    enabled: false
    image_backbone: "swin_small"
    clinical_dim: 10
    num_heads: 8
    attention_type: "cosine"  # 'dot_product' or 'cosine'
    dropout: 0.1
    batch_size: 16
    lr: 2.0e-5
  
  moe_mcga:
    enabled: false
    num_experts: 8
    top_k: 2
    mcga_type: "bidirectional"
    batch_size: 16
    lr: 2.0e-5
```

---

# 11. Implementation Plan

## 11.1 Phase 1: Foundation (Week 1-2)

| Task | Files | Priority | Effort | Status |
|------|-------|----------|--------|--------|
| **P1.1** Create MoE Head module | `src/models/moe/moe_head.py` | P0 | 2 days | ⏳ Pending |
| **P1.2** Create MoE Gating network | `src/models/moe/gating.py` | P0 | 1 day | ⏳ Pending |
| **P1.3** Create MCGA module | `src/models/attention/mcga.py` | P0 | 2 days | ⏳ Pending |
| **P1.4** Update `src/models/fusion/__init__.py` | Export new modules | P0 | 0.5 day | ⏳ Pending |
| **P1.5** Add clinical data loader | `src/data/multimodal_dataset.py` | P1 | 2 days | ⏳ Pending |

## 11.2 Phase 2: Integration (Week 3-4)

| Task | Files | Priority | Effort | Status |
|------|-------|----------|--------|--------|
| **P2.1** Integrate MoE into pipeline | `run_pipeline.py` | P0 | 1 day | ⏳ Pending |
| **P2.2** Add MoE config to `config.yaml` | `config.yaml` | P0 | 0.5 day | ⏳ Pending |
| **P2.3** Implement MoE loss (aux + main) | `src/utils/moe_loss.py` | P0 | 1 day | ⏳ Pending |
| **P2.4** Add MCGA to fusion models | Modify existing fusion | P1 | 2 days | ⏳ Pending |
| **P2.5** Test on BreakHis | Run pipeline | P0 | 1 day | ⏳ Pending |

## 11.3 Phase 3: Advanced Features (Week 5-6)

| Task | Files | Priority | Effort | Status |
|------|-------|----------|--------|--------|
| **P3.1** Implement clinical metadata fusion | Extend data loader | P1 | 2 days | ⏳ Pending |
| **P3.2** Add interpretability (expert attribution) | `src/utils/interpretability.py` | P2 | 2 days | ⏳ Pending |
| **P3.3** Implement missing modality handling | MoE routing logic | P2 | 2 days | ⏳ Pending |
| **P3.4** Multi-task learning heads | Add grade/subtype classifiers | P2 | 2 days | ⏳ Pending |
| **P3.5** Ablation studies | Run experiments | P1 | 3 days | ⏳ Pending |

## 11.4 Phase 4: Optimization (Week 7-8)

| Task | Files | Priority | Effort | Status |
|------|-------|----------|--------|--------|
| **P4.1** Expert pruning (remove unused) | MoE analysis | P3 | 1 day | ⏳ Pending |
| **P4.2** Gating temperature tuning | Hyperparameter search | P2 | 1 day | ⏳ Pending |
| **P4.3** Quantization for deployment | Model optimization | P3 | 2 days | ⏳ Pending |
| **P4.4** Write paper | MICCAI 2026 submission | P0 | 5 days | ⏳ Pending |

---

# 12. Advanced Extensions

## 12.1 Next-Level Innovations (Beyond the Paper)

### 12.1.1 Patient-Adaptive Expert Specialization

```
Instead of fixed experts, let experts SPECIALIZE during training:

- Initialize 8 identical experts
- Add diversity loss: maximize expert output divergence
- Experts EMERGENTLY specialize (no manual design)
- Result: Data-driven expert specialization
```

### 12.1.2 Hierarchical MoE Routing

```
Two-level routing for fine-grained adaptation:

Level 1: Modality routing (which modalities to use?)
Level 2: Expert routing (which experts within modality?)

Benefit: Can skip entire modalities if not informative
```

### 12.1.3 Temporal MoE for Treatment Response

```
Add time dimension for longitudinal patient data:

Patient at t0: Biopsy → MoE prediction
Patient at t1: Post-treatment → Different experts activate
Patient at t2: Follow-up → Track expert usage over time

Novelty: First temporal MoE for cancer progression
```

### 12.1.4 Federated MoE Learning

```
Train across hospitals without sharing data:

Hospital A: Trains Experts 1-4
Hospital B: Trains Experts 5-8
Shared: Gating network (aggregated)

Benefit: Privacy-preserving, diverse expert training
```

### 12.1.5 Neural Architecture Search for Experts

```
AutoML for expert design:

- Search space: [FFN depth, width, activation, dropout]
- Search algorithm: DARTS or ENAS
- Result: Optimal expert architecture per specialization
```

### 12.1.6 Uncertainty-Aware Gating

```
Bayesian gating for confidence estimation:

- Gating network outputs: (mean, variance) per expert
- High variance → Low confidence → Route to uncertainty expert
- Provides: Prediction confidence intervals
```

### 12.1.7 Cross-Modal Contrastive Pre-Training

```
Before MoE training, pre-align modalities:

- Contrastive loss: Pull (image, clinical) pairs together
- Push (image, wrong_clinical) apart
- Benefit: Better MCGA initialization
```

### 12.1.8 Dynamic Expert Expansion

```
Grow experts as needed during training:

- Start with 4 experts
- Monitor gating entropy
- If entropy high → Add new expert
- Result: Architecture grows with dataset complexity
```

---

# 13. Expected Outcomes

## 13.1 Performance Targets

| Metric | Current Best | MoE-MCGA Target | Improvement |
|--------|-------------|-----------------|-------------|
| **Accuracy (BreakHis)** | 85.67% | 88-90% | **+2.3-4.3%** |
| **AUC-ROC** | 88.52% | 91-93% | **+2.5-4.5%** |
| **Sensitivity** | 91.15% | 92-94% | **+0.8-2.8%** |
| **Specificity** | 77.45% | 82-85% | **+4.5-7.5%** |
| **FNR** | 8.85% | 6-8% | **-0.85-2.85%** |
| **Inference Speed** | 30ms | 25ms (sparse) | **+17% faster** |

## 13.2 Research Contributions

1. ✅ **First MoE for breast cancer histopathology**
2. ✅ **Multi-Co-Guided Attention with clinical metadata**
3. ✅ **Patient-adaptive expert routing**
4. ✅ **Interpretable expert attribution**
5. ✅ **Missing modality robustness**

## 13.3 Publication Venues

- **MICCAI 2026** (Medical Image Computing) - Primary target
- **ISBI 2026** (Biomedical Imaging) - Secondary target
- **Nature Scientific Reports** (if clinical validation strong) - High impact

## 13.4 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| **MoE converges** | Aux loss < 0.1 | Monitor training logs |
| **Experts specialize** | Different usage patterns | Expert attribution analysis |
| **Accuracy improves** | >1% over baseline | 5-fold CV comparison |
| **Inference speed** | <30ms per sample | Benchmark on test set |
| **Clinical integration** | Works with partial modalities | Missing modality tests |

---

# 14. References

## 14.1 Research Papers

1. **Attention-Enhanced Mixture of Experts for Multimodal Breast Cancer Detection** (Primary reference)
2. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (MoE routing)
3. **Swin Transformer: Hierarchical Vision Transformer using Shifted Windows** (Backbone)
4. **Deformable Attention: Learning Where to Attend** (Attention mechanism)
5. **Knowledge Distillation: A Survey** (Ensemble distillation)

## 14.2 Code References

- **Existing Models:** `src/models/` directory
- **Fusion Models:** `src/models/fusion/` directory
- **Quantum Models:** `src/models/quantum/` directory
- **Training Pipeline:** `run_pipeline.py`
- **Configuration:** `config.yaml`

## 14.3 Documentation

- **PROPOSED_ARCHITECTURES.md** - Initial fusion architecture proposals
- **QUANTUM_ARCHITECTURES.md** - Quantum model specifications
- **IMPLEMENTATION_COMPLETE.md** - Recently implemented models
- **MASTER_RESEARCH_DOCUMENTATION.md** - Overall project documentation

---

# Appendix A: Complete MoE-MCGA Model Specification

## A.1 Model Configuration

```yaml
# Complete MoE-MCGA configuration for config.yaml

models:
  moe_mcga_full:
    enabled: true
    display_name: "MoE-MCGA-Full"
    
    # Backbone configuration
    image_backbone: "swin_small"
    mammo_backbone: "convnext_small"
    clinical_encoder:
      input_dim: 10
      hidden_dim: 64
      output_dim: 128
    
    # MCGA configuration
    mcga:
      num_heads: 8
      dropout: 0.1
      attention_type: "cosine"
      bidirectional: true
    
    # MoE configuration
    moe:
      num_experts: 8
      top_k: 2
      fusion_dim: 1024
      expert_dropout: 0.3
      capacity_factor: 1.25
    
    # Training configuration
    training:
      batch_size: 16
      lr: 2.0e-5
      weight_decay: 1.0e-4
      epochs: 50
      patience: 10
      grad_clip: 1.0
      use_amp: true
    
    # Loss configuration
    loss:
      cls_weight: 1.0
      aux_weight: 0.01
      div_weight: 0.1
      entropy_weight: 0.001
```

## A.2 Expected Expert Usage Distribution

```
Ideal expert usage (after convergence):
  Expert 1: 15-20% (Global context)
  Expert 2: 15-20% (Local texture)
  Expert 3: 10-15% (Multi-scale)
  Expert 4: 10-15% (Quantum-enhanced)
  Expert 5: 10-15% (Clinical-focused)
  Expert 6: 10-15% (Cross-modal)
  Expert 7: 5-10%  (Uncertainty)
  Expert 8: 5-10%  (Rare patterns)

Load balancing loss ensures no expert is:
  - Overused (>30%)
  - Underused (<5%)
```

---

# Appendix B: Troubleshooting Guide

## B.1 Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| **Expert collapse** | All experts same output | Increase div_weight, check diversity loss |
| **Gating instability** | Routing oscillates | Increase aux_weight, reduce lr |
| **NaN loss** | Training diverges | Reduce grad_clip, check for overflow |
| **Slow convergence** | Accuracy plateaus | Increase warmup_epochs, check data loading |
| **Memory overflow** | OOM errors | Reduce batch_size, enable gradient checkpointing |

## B.2 Debugging Commands

```bash
# Check expert usage
python analyze_experts.py --model moe_mcga_full --checkpoint best.pth

# Visualize attention
python visualize_attention.py --model moe_mcga_full --sample_id 12345

# Profile memory
python profile_memory.py --model moe_mcga_full --batch_size 16
```

---

**Document End**

---

**Last Updated:** March 22, 2026  
**Author:** AI Research Engineering Team  
**Status:** Ready for Implementation  
**Next Review:** After Phase 1 completion
