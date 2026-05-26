# Genomic Model Zoo — All Architectures
## 12 Architectures from Baseline to Quantum-Enhanced

---

## Design Philosophy

Mirror the exact same progression used in the histopathology paper:

| Histopathology Paper | Genomic Extension |
|---|---|
| ViT-Tiny (baseline) | G-Baseline-MLP (baseline) |
| EfficientNet-B3 (CNN baseline) | G-PASNet (pathway baseline) |
| CNN+ViT Hybrid | G-CrossOmics-Attention |
| Swin-Small, ConvNeXt-Small | G-GCN-PPI, G-PathFormer-Lite |
| DualBranch-Fusion | G-Dual-Omics-Fusion |
| TBCA (TripleBranch) | G-TripleOmics-Fusion |
| TBCA-Quantum-Fusion | G-Pathway-Quantum |
| TBCA-CNN-FM-Quantum (best) | G-FM-Quantum (best, to find) |

**Primary task:** pCR binary classification on GEO TNBC (325 patients)  
**Secondary task:** PAM50 subtype on TCGA-BRCA (~1,093 patients)  
**Evaluation:** 5-fold stratified CV, AUC-ROC, Sensitivity, Specificity, FNR, Brier score  

---

## Architecture 1: G-Baseline-MLP

**Role:** Weakest baseline (analogous to ViT-Tiny)

```
Input: top-500 genes (z-score normalized)  [B, 500]
→ Linear(500, 256) → BatchNorm → ReLU → Dropout(0.3)
→ Linear(256, 128) → BatchNorm → ReLU → Dropout(0.3)
→ Linear(128, 64)  → ReLU
→ Linear(64, 1)    → Sigmoid
```

**Config:**
```yaml
g_baseline_mlp:
  input_genes: 500        # top-500 by CV
  hidden: [256, 128, 64]
  dropout: 0.3
  lr: 1e-3
  batch_size: 32
  epochs: 100
  optimizer: Adam
  loss: BCEWithLogitsLoss
  class_weight: 2.1       # 220/105
```

**Expected performance:** AUC ~0.72–0.78 (overfitting likely on 325 samples)

---

## Architecture 2: G-Baseline-Trees (XGBoost + RF Stacking)

**Role:** Strong classical baseline (from TNBC-DT and personalized treatment papers)

```
Input: top-100 genes (3-stage filtered)  [N, 100]

Base learners (OOF predictions):
  ├── Random Forest (500 trees, balanced weight)
  ├── XGBoost (η=0.05, max_depth=3, scale_pos_weight=2.1)
  └── SVM (RBF, Platt scaling)

Meta-learner:
  → Elastic Net on (N, 3) OOF matrix
  → Isotonic regression calibration
  → Final pCR probability
```

**This is the TNBC-DT Stream 1 — already achieving AUC 0.883 in your paper.**

```python
# sklearn pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

base_learners = [
    ('rf', RandomForestClassifier(n_estimators=500, class_weight='balanced')),
    ('xgb', XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, scale_pos_weight=2.1)),
    ('svm', CalibratedClassifierCV(SVC(kernel='rbf', probability=True))),
]
meta = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5)
stack = StackingClassifier(estimators=base_learners, final_estimator=meta, cv=5, passthrough=False)
```

---

## Architecture 3: G-PASNet (Pathway-Aware Sparse Network)

**Role:** Biologically grounded baseline (best single-modality expected)

**Concept:** Sparse connectivity enforced by KEGG pathway membership. Genes within the same pathway are connected; cross-pathway connections are zero.

```
Input: all pathway genes (~3000–6000)  [B, G]
↓
Pathway Mask M: binary matrix [G × P]  (G genes, P=231 pathways)
↓
Pathway Layer: sparse linear, only M_{g,p}=1 connections active
  output: [B, P, γ]  (P pathways, γ learnable filters each)
↓
1D Conv per pathway: kernel [n_genes_in_pathway × 1]
  output: [B, P, γ]
↓
2D Conv (pathway-pathway interactions): [P × γ → P × γ]
↓
Flatten → FC(P×γ, 256) → ReLU → Dropout → FC(256, 1) → Sigmoid
```

**Implementation:**
```python
class PASNetLayer(nn.Module):
    """Pathway-associated sparse layer"""
    def __init__(self, pathway_membership, n_filters=32):
        super().__init__()
        self.pathway_membership = pathway_membership  # dict: pathway -> gene indices
        self.pathways = list(pathway_membership.keys())
        self.filters = nn.ModuleList([
            nn.Conv1d(1, n_filters, kernel_size=len(pathway_membership[p]))
            for p in self.pathways
        ])
    
    def forward(self, x):  # x: [B, G]
        pathway_outs = []
        for i, (p, conv) in enumerate(zip(self.pathways, self.filters)):
            gene_idx = self.pathway_membership[p]
            pathway_genes = x[:, gene_idx].unsqueeze(1)  # [B, 1, n_genes]
            pathway_feat = conv(pathway_genes).squeeze(-1)  # [B, n_filters]
            pathway_outs.append(pathway_feat)
        return torch.stack(pathway_outs, dim=1)  # [B, P, n_filters]
```

**KEGG pathway setup:**
```python
# Download 231 KEGG human pathways (filtered: 15–300 genes)
import gseapy
pathways = gseapy.get_library('KEGG_2021_Human')
# Filter to pathways with 15–300 genes, then intersect with your gene set
```

**Config:**
```yaml
g_pasnet:
  n_filters: 32              # γ filters per pathway
  dropout: 0.3
  pathway_conv_dropout: 0.2
  pathway_pathway_conv: true # 2D conv for pathway-pathway interactions
  kernel_2d: [3, 3]
  lr: 5e-4
  batch_size: 32
  epochs: 150
  optimizer: AdamW
  weight_decay: 1e-4
```

---

## Architecture 4: G-TabTransformer

**Role:** Attention-based baseline treating genes as tokens

```
Input: top-500 genes  [B, 500]
↓
Linear embedding: [B, 500] → [B, 500, 64]  (embed each gene to 64d)
+ Learnable positional encoding [500, 64]
↓
Transformer Encoder (4 layers, 8 heads, FFN dim=256)
  → [B, 500, 64]
↓
CLS token aggregation OR mean pooling → [B, 64]
↓
FC(64, 32) → ReLU → Dropout → FC(32, 1) → Sigmoid
```

**Config:**
```yaml
g_tabtransformer:
  input_genes: 500
  embed_dim: 64
  n_heads: 8
  n_layers: 4
  ffn_dim: 256
  dropout: 0.2
  lr: 1e-4
  batch_size: 16     # transformers need smaller batch on 325 samples
  epochs: 100
  optimizer: AdamW
  weight_decay: 1e-4
  warmup_steps: 100
```

**Note:** Likely to overfit on 325 samples; use heavy regularization. Expected to underperform trees but test for learning signal.

---

## Architecture 5: G-MultiScale-1D-CNN

**Inspiration:** EEG-MFTNet multi-scale temporal convolutions applied to genomic data

**Concept:** Multiple 1D conv branches with different kernel sizes, each capturing different "genomic scales" (local co-expression, pathway-scale, global patterns).

```
Input: top-1000 genes (sorted by pathway order)  [B, 1000, 1]
↓
Parallel 1D conv branches (6 scales):
  ├── Conv1D(kernel=5)  → [B, 1000, 16]   # very local
  ├── Conv1D(kernel=15) → [B, 1000, 16]   # small cluster
  ├── Conv1D(kernel=31) → [B, 1000, 16]   # pathway-scale
  ├── Conv1D(kernel=63) → [B, 1000, 16]   # super-pathway
  ├── Conv1D(kernel=127)→ [B, 1000, 16]   # large module
  └── Conv1D(kernel=255)→ [B, 1000, 16]   # global trend
Each: BatchNorm → ReLU → SpatialDropout(0.3)
↓
Concatenate → [B, 1000, 96] (6 × 16)
↓
Learnable scalar weights (6-way softmax) → weighted sum → [B, 1000, 16]
↓
Global Average Pool → [B, 16]
↓
+ Transformer encoder branch (parallel, same input):
  Input [B, 1000, 1] reshape→ [B, 1000, 1]
  ProjectUp → [B, 1000, 32]
  TransformerEncoder(1 layer, 4 heads) → [B, 1000, 32]
  Mean Pool → [B, 32]
↓
Fuse [B, 16] + [B, 32] → [B, 48]
↓
FC(48, 32) → ReLU → FC(32, 1) → Sigmoid
```

**Gene ordering:** Sort by KEGG pathway membership (group co-pathway genes together) before feeding as sequence. This makes local conv kernels biologically meaningful.

**Config:**
```yaml
g_multiscale_1dcnn:
  input_genes: 1000
  n_scales: 6
  filters_per_scale: 16
  kernel_sizes: [5, 15, 31, 63, 127, 255]
  transformer_dim: 32
  transformer_heads: 4
  dropout: 0.3
  lr: 5e-4
  batch_size: 32
  epochs: 150
```

---

## Architecture 6: G-BiLSTM

**Role:** Sequential processing baseline

```
Input: top-500 genes (pathway-ordered)  [B, 500, 1]
↓
Embedding: Linear(1, 32) → [B, 500, 32]
↓
BiLSTM (2 layers, hidden=64, bidirectional)
  → [B, 500, 128]
↓
Self-attention pooling (learn which positions matter)
  → [B, 128]
↓
FC(128, 64) → ReLU → Dropout(0.3) → FC(64, 1) → Sigmoid
```

**Config:**
```yaml
g_bilstm:
  input_genes: 500
  embed_dim: 32
  lstm_hidden: 64
  lstm_layers: 2
  dropout: 0.3
  lr: 5e-4
  batch_size: 32
  epochs: 100
```

---

## Architecture 7: G-GCN-PPI (Graph Convolutional on PPI Network)

**Concept:** Genes as nodes on the STRING PPI network, RNA expression as node features, GCN learns topology-informed embeddings.

```
Graph G = (V, E)
V: genes with expression values as node features  [N_genes, 1]
E: PPI edges from STRING (score > 0.7 threshold)

2-layer GCN:
  Layer 1: GCN_conv(1 → 64) → ReLU → Dropout(0.3)
  Layer 2: GCN_conv(64 → 128) → ReLU
↓
Global mean pooling over all gene nodes → [B, 128]
↓
FC(128, 64) → ReLU → Dropout → FC(64, 1) → Sigmoid
```

**STRING PPI download:**
```python
import pandas as pd
# STRING v12 human PPI
url = "https://stringdb-downloads.org/download/protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
ppi = pd.read_csv(url, sep=' ')
ppi_filtered = ppi[ppi['combined_score'] > 700]  # high confidence
```

**PyTorch Geometric implementation:**
```python
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader

class GenomicGCN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=128):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, 64), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(64, 1)
        )
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return self.classifier(x)
```

**Config:**
```yaml
g_gcn_ppi:
  ppi_score_threshold: 700      # STRING confidence
  hidden_dim: 64
  output_dim: 128
  dropout: 0.3
  lr: 5e-4
  batch_size: 16                # graphs are larger
  epochs: 150
  optimizer: AdamW
```

---

## Architecture 8: G-CrossOmics-Attention (PIMO-style)

**Requires:** RNA-seq + DNA methylation + CNA (TCGA-BRCA only, not GEO)  
**Task:** PAM50 subtype or survival on TCGA-BRCA

```
For each gene i:
  g_i = expression value    (scalar)
  d_i = methylation value   (scalar)
  c_i = CNA value           (scalar)

Cross-attention interactions (PIMO Eq. 1-2):
  d_g_i = (w_kd · g_i) ⊙ (w_qd · d_i)    # methylation-expression interaction
  c_g_i = (w_kc · g_i) ⊙ (w_qc · c_i)    # CNA-expression interaction

Gene representation:
  φ_i = [g_i, d_g_i, c_g_i]               # [B, 3] per gene

Pathway grouping (KEGG, 231 pathways):
  Φ_j = stack(φ_i for i in pathway j)     # [B, 3 × n_genes_j]

Pathway representation:
  P = Φ * H(x)                            # learnable pathway filters
  P: [B, 231, γ]

2D Conv (pathway-pathway):
  → [B, 231, γ'] 

FC → Output
```

**Implementation:**
```python
class CrossOmicsGene(nn.Module):
    def __init__(self, n_genes):
        super().__init__()
        self.w_kd = nn.Parameter(torch.ones(n_genes))  # gene-specific scalars
        self.w_qd = nn.Parameter(torch.ones(n_genes))
        self.w_kc = nn.Parameter(torch.ones(n_genes))
        self.w_qc = nn.Parameter(torch.ones(n_genes))
    
    def forward(self, g, d, c):
        d_g = (self.w_kd * g) * (self.w_qd * d)  # element-wise
        c_g = (self.w_kc * g) * (self.w_qc * c)
        phi = torch.stack([g, d_g, c_g], dim=-1)  # [B, n_genes, 3]
        return phi
```

**Config:**
```yaml
g_crossomics:
  dataset: tcga_brca          # needs 3 modalities
  task: pam50_5class
  n_filters: 32               # γ
  pathway_source: kegg
  dropout: 0.3
  lr: 1e-4
  batch_size: 32
  epochs: 200
```

---

## Architecture 9: G-PathFormer-Lite (Pathway Transformer)

**Concept:** Each pathway is a TOKEN. Transformer models pathway-pathway interactions (like Pathformer 2024).

```
Input: top-N genes mapped to P=231 pathways
↓
Pathway mean expression: [B, 231]   (mean of gene expr within each pathway)
↓
Pathway embedding: Linear(1, 128) → [B, 231, 128]
+ Learnable pathway position encoding [231, 128]
↓
Transformer Encoder (3 layers, 4 heads, FFN=256):
  - Row attention: cross-pathway interactions
  [B, 231, 128]
↓
CLS aggregation → [B, 128]
↓
FC(128, 64) → GELU → Dropout(0.2) → FC(64, 1)
```

**Biological advantage:** Each attention head can learn different pathway crosstalk patterns (e.g., cell cycle ↔ DNA repair).

**Config:**
```yaml
g_pathformer_lite:
  n_pathways: 231
  pathway_embed_dim: 128
  n_heads: 4
  n_layers: 3
  ffn_dim: 256
  dropout: 0.2
  lr: 1e-4
  batch_size: 16
  epochs: 150
  optimizer: AdamW
  weight_decay: 1e-4
```

---

## Architecture 10: G-Quantum-MLP

**Concept:** MLP backbone + VQC branch. Same 8-qubit design as TBCA paper.

```
Input: top-100 genes (biologically filtered)  [B, 100]
↓
Classical branch:
  Linear(100, 64) → ReLU → Linear(64, 32) → f_pool [B, 32]

Quantum branch (parallel):
  Linear(100, 8) → tanh → z' [B, 8]     (amplitude encoding input)
  ↓
  Amplitude encode: |ψ⟩ = ⊗ RY(z'_i)|0⟩   (8 qubits)
  ↓
  VQC Layer 1: U3(θ,φ,λ) on all 8 qubits + cyclic CNOT ring
  VQC Layer 2: U3(θ,φ,λ) on all 8 qubits + cyclic CNOT ring
  ↓
  Measure ⟨Z_i⟩ for all 8 qubits → 8 expectation values
  Linear(8, 32) → f_q [B, 32]

Residual fusion:
  f* = f_pool + f_q    [B, 32]
↓
FC(32, 16) → ReLU → Dropout(0.3) → FC(16, 1) → Sigmoid
```

**This directly mirrors TBCA-CNN-FM-Quantum — VQC on intermediate representation before final prediction.**

**Config:**
```yaml
g_quantum_mlp:
  input_genes: 100          # biologically filtered 100-gene panel
  hidden_dim: 32
  n_qubits: 8
  n_layers: 2
  rotation: u3              # rx+ry+rz per qubit
  entanglement: cyclic_cnot
  backend: pennylane         # same as TBCA paper
  lr: 1e-4
  batch_size: 16            # quantum circuits are slow
  epochs: 100
  optimizer: AdamW
```

---

## Architecture 11: G-Pathway-Quantum (BEST EXPECTED)

**Concept:** PASNet pathway encoder + VQC on PATHWAY-LEVEL features (biologically meaningful quantum encoding)

This is the most important architecture. The quantum circuit encodes PATHWAY ACTIVATION LEVELS (not individual genes), which is:
1. Biologically meaningful (pathway = functional unit)
2. Dimensionally compact (231 pathways → 8 qubits is natural compression)
3. Consistent with TBCA: VQC on intermediate representation before final fusion

```
Input: all genes  [B, G]
↓
PASNet Pathway Layer:
  Group genes by KEGG pathways
  1D conv per pathway → pathway activation scores
  → [B, 231]   (231 pathway scores)
↓
Pathway selection: top-32 most variable pathways → [B, 32]

Classical branch (parallel):
  Linear(32, 32) → f_pathway [B, 32]

Quantum branch (parallel):
  Linear(32, 8) → tanh → z' [B, 8]
  ↓
  8-qubit VQC (same as G-Quantum-MLP)
  ↓
  Linear(8, 32) → f_q [B, 32]

Residual:
  f* = f_pathway + f_q    [B, 32]
↓
FC(32, 16) → GELU → Dropout → FC(16, 1)
```

**Why this is the "CNN-FM-Quantum equivalent" for genomics:**
- TBCA paper: VQC on EfficientNet's **intermediate spatial feature map** (before GAP) = best model
- This: VQC on PASNet's **intermediate pathway activations** (before global pooling) = analogous design
- Both apply quantum on the **richest intermediate representation** before dimensionality collapse

**Config:**
```yaml
g_pathway_quantum:
  pathway_source: kegg
  n_pathways_selected: 32     # top-32 variable
  pathway_filters: 16         # γ in PASNet
  n_qubits: 8
  n_layers: 2
  rotation: u3
  entanglement: cyclic_cnot
  backend: pennylane
  lr: 5e-5
  batch_size: 16
  epochs: 150
  optimizer: AdamW
  weight_decay: 1e-4
```

---

## Architecture 12: G-TNBC-DT-Neural (Digital Twin Hybrid)

**Concept:** From your TNBC-DT paper: Genomic ML + Gompertz ODE + Lehmann priors. Extend with neural components.

```
Stream 1 — Neural Genomic:
  G-Pathway-Quantum output → p_gen [B, 1]

Stream 2 — Mechanistic ODE:
  κ_0 = κ_min + (κ_max - κ_min) × p_gen   (Gompertz kill rate)
  Integrate ODE for 168 days → V(T_final), AUC_V
  → mechanistic_features [B, 2]

Stream 3 — Lehmann Subtype Prior:
  Assign BL1/BL2/M/LAR from 101-gene algorithm
  → one_hot [B, 4]

Meta-learner:
  f = [p_gen, V(T_final), AUC_V, one_hot]  [B, 7]
  Logistic Regression (calibrated) → final pCR prob
```

This is essentially TNBC-DT but with G-Pathway-Quantum replacing the classical RF/XGBoost/SVM stacking ensemble in Stream 1.

**Expected improvement:** TNBC-DT AUC was 0.890. Replacing the classical ensemble with G-Pathway-Quantum should push this to ~0.90–0.92.

---

## Summary Table

| # | Architecture | Paradigm | Params | Biological Prior | Expected AUC |
|---|---|---|---|---|---|
| G1 | Baseline-MLP | DL-Simple | ~140K | None | 0.72–0.78 |
| G2 | Baseline-Trees | Classical ML | ~50K | Curated genes | 0.883 (proven) |
| G3 | PASNet | Pathway-NN | ~500K | KEGG pathways | 0.85–0.89 |
| G4 | TabTransformer | Transformer | ~1M | None | 0.76–0.82 |
| G5 | MultiScale-1DCNN | Multi-scale CNN | ~200K | Pathway ordering | 0.82–0.87 |
| G6 | BiLSTM | Recurrent | ~300K | Pathway ordering | 0.79–0.84 |
| G7 | GCN-PPI | Graph-based | ~600K | PPI network | 0.83–0.88 |
| G8 | CrossOmics-Attn | Multi-omics | ~2M | Pathway + methyl/CNA | 0.87–0.91 |
| G9 | PathFormer-Lite | Pathway-Transformer | ~1.5M | KEGG pathways | 0.85–0.90 |
| G10 | Quantum-MLP | Quantum-Classical | ~50K | None | 0.84–0.89 |
| G11 | Pathway-Quantum | **Pathway+VQC** | ~500K | KEGG + quantum | **0.89–0.93** |
| G12 | TNBC-DT-Neural | **Hybrid DT** | ~500K | KEGG + ODE + Lehmann | **0.90–0.93** |

---

## Training Configuration (All Genomic Models)

```yaml
# Common config for all genomic models
genomic_training:
  cv_strategy: stratified_5fold
  patient_level_split: true       # no leakage
  test_holdout: 0.15              # 15% held out before CV
  smote_training: true            # SMOTE only on training folds
  metrics:
    - auroc
    - sensitivity
    - specificity
    - f1
    - brier_score
    - ece                         # expected calibration error
  loss: BCEWithLogitsLoss
  class_weight_auto: true
  optimizer: AdamW
  gradient_clipping: 1.0
  early_stopping_patience: 15
  seed: 42
  wandb_project: breast-cancer-genomic
```

---

## Running Order (A100 Server)

```bash
# Run all in parallel (separate jobs per architecture)

# Phase 1: Baselines (fast)
python run_genomic.py --model g_baseline_mlp --dataset geo_tnbc
python run_genomic.py --model g_baseline_trees --dataset geo_tnbc

# Phase 2: Pathway models
python run_genomic.py --model g_pasnet --dataset geo_tnbc
python run_genomic.py --model g_pathformer_lite --dataset geo_tnbc

# Phase 3: Attention + sequence
python run_genomic.py --model g_tabtransformer --dataset geo_tnbc
python run_genomic.py --model g_multiscale_1dcnn --dataset geo_tnbc
python run_genomic.py --model g_bilstm --dataset geo_tnbc

# Phase 4: Graph
python run_genomic.py --model g_gcn_ppi --dataset geo_tnbc

# Phase 5: Multi-omics (TCGA-BRCA only)
python run_genomic.py --model g_crossomics --dataset tcga_brca

# Phase 6: Quantum (GPU-intensive)
python run_genomic.py --model g_quantum_mlp --dataset geo_tnbc
python run_genomic.py --model g_pathway_quantum --dataset geo_tnbc
python run_genomic.py --model g_tnbc_dt_neural --dataset geo_tnbc
```
