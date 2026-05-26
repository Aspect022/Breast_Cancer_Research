# Genomics Implementation And Compute Plan

This file turns the genomics planning notes into an implementation roadmap for this repository. It assumes the existing histopathology pipeline stays intact, and genomics is added as a parallel patient-level pipeline with later multimodal fusion.

## 1. Recommended Strategy

Do not begin with all 12 genomic models at once.

The safest implementation order is:

1. Build the genomic data layer and one simple baseline.
2. Add classical genomic baselines that can be trusted on small patient counts.
3. Add pathway-aware neural models.
4. Add quantum genomic models using the existing vectorized quantum circuit style.
5. Add TCGA multi-omics and paired late-fusion only after single-modality genomics is stable.
6. Add new TBCA quantum variants separately from genomic modeling, because they are image-heavy and compute-heavy.

This gives an executable research pipeline early, instead of a large architecture zoo that is hard to debug.

## 2. What We Are Implementing

The genomics folder proposes three workstreams:

```text
Workstream A: Genomic model zoo
Workstream B: New TBCA quantum variants
Workstream C: Late-fusion multimodal validation
```

The main first target should be:

```text
GEO TNBC pCR prediction
Binary task: pCR vs residual disease
Patient-level samples: about 325
```

The secondary target should be:

```text
TCGA-BRCA PAM50 subtype
Multi-class task: LumA, LumB, HER2, Basal, Normal-like
Patient-level samples: about 1,093
```

The final multimodal target should be:

```text
TCGA-BRCA paired WSI + RNA-seq late fusion
Patient-level intersection: roughly 900-950 patients
```

## 3. Proposed Repository Layout

Add these files and folders:

```text
config_genomics.yaml
run_genomic.py

src/data/genomics/
  __init__.py
  geo_tnbc.py
  tcga_brca.py
  preprocessing.py
  pathway.py
  splits.py

src/models/genomics/
  __init__.py
  baseline_mlp.py
  baseline_trees.py
  pasnet.py
  tab_transformer.py
  multiscale_1dcnn.py
  bilstm.py
  gcn_ppi.py
  crossomics.py
  pathformer_lite.py
  quantum_mlp.py
  pathway_quantum.py
  tnbc_dt_neural.py

src/utils/genomics_metrics.py

scripts/genomics/
  download_geo.py
  preprocess_geo.py
  download_pathways.py
  download_tcga_notes.md
```

Why separate `run_genomic.py` from `run_pipeline.py`:

```text
The current run_pipeline.py is image-centered.
Genomics has different data shapes, losses, models, preprocessing, and metrics.
A separate entrypoint is cleaner and avoids breaking the histopathology experiments.
```

## 4. Implementation Phase Plan

### Phase 0: Dependency And Config Setup

Add a genomics-specific config:

```yaml
data:
  dataset: geo_tnbc
  task: pcr_binary
  data_dir: data/geo_tnbc
  seed: 42

training:
  epochs: 150
  patience: 20
  grad_clip: 1.0
  n_folds: 5
  test_holdout: 0.15

models:
  g_baseline_mlp:
    enabled: true
    input_genes: 500
    hidden: [256, 128, 64]
    dropout: 0.3
    lr: 1.0e-3
    weight_decay: 1.0e-4
    batch_size: 32

output:
  output_dir: outputs_genomics
  wandb_enabled: true
  wandb_project: breast-cancer-genomics
```

Add or document extra packages:

```text
GEOparse
combat or pycombat
gseapy
xgboost
imbalanced-learn
lifelines
```

Optional later:

```text
torch-geometric
gdc-client
CLAM
openslide
```

### Phase 1: GEO TNBC Data Pipeline

Implement:

```text
scripts/genomics/download_geo.py
scripts/genomics/preprocess_geo.py
src/data/genomics/geo_tnbc.py
```

Expected processed files:

```text
data/geo_tnbc/processed/combined_325_genes_100.csv
data/geo_tnbc/processed/combined_325_genes_500.csv
data/geo_tnbc/processed/combined_325_genes_1000.csv
data/geo_tnbc/processed/labels_pcr_rd.csv
data/geo_tnbc/processed/batches.csv
data/geo_tnbc/processed/lehmann_subtypes.csv
```

Minimum viable preprocessing:

```text
load expression matrices
align sample IDs
extract pCR/RD labels
log2 transform when needed
quantile normalize
batch correction if multiple GEO cohorts are pooled
probe-to-gene collapse
z-score per gene
select top 100, 500, 1000 genes
save final CSVs
```

Important rule:

```text
Any feature selection that uses labels must happen inside each training fold, not once globally before splitting.
Otherwise there is leakage.
```

For the first working version, use unsupervised top-variance/top-CV genes globally. Then add fold-local supervised limma-style selection later.

### Phase 2: Genomic Dataset And Splits

Implement a tabular patient-level dataset:

```python
class GenomicExpressionDataset(Dataset):
    def __init__(self, expression_df, labels_df):
        self.x = expression_df.values.astype("float32")
        self.y = labels_df["label"].values.astype("float32")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), torch.tensor(self.y[idx])
```

Splitting:

```text
15% held-out test set
5-fold stratified CV on the remaining 85%
SMOTE only inside training folds
validation/test never receive SMOTE
```

Because samples are patients, patient leakage is easier than with patches, but still enforce unique patient IDs.

### Phase 3: Baseline Models

Implement first:

```text
G1: G-Baseline-MLP
G2: G-Baseline-Trees
```

Reason:

```text
They validate the data pipeline quickly.
They establish whether labels, splits, and metrics are correct.
They are small enough to run on CPU or laptop GPU.
```

Outputs should match the image pipeline style:

```text
outputs_genomics/G-Baseline-MLP_geo_tnbc/
  G-Baseline-MLP_fold_results.csv
  G-Baseline-MLP_training_curves.png
  G-Baseline-MLP_calibration_curve.png
```

Required metrics:

```text
Accuracy
Balanced accuracy
AUROC
AUPRC
Sensitivity
Specificity
F1
MCC
FNR
Brier score
ECE
Calibration curve
```

### Phase 4: Pathway Models

Implement:

```text
G3: G-PASNet
G9: G-PathFormer-Lite
```

Add pathway utilities:

```text
src/data/genomics/pathway.py
```

Responsibilities:

```text
download KEGG pathway sets through gseapy
filter pathways to reasonable sizes, for example 15-300 genes
intersect pathway genes with current expression feature set
build gene-to-pathway membership
save pathway masks and pathway ordering
```

For PASNet, start with a simple pathway aggregation implementation before adding more complex sparse/convolutional variants:

```text
gene expression -> pathway mean/max/attention pooling -> MLP classifier
```

Then upgrade to the planned sparse pathway layer.

### Phase 5: Sequence, Attention, And Graph Models

Implement:

```text
G4: G-TabTransformer
G5: G-MultiScale-1D-CNN
G6: G-BiLSTM
G7: G-GCN-PPI
```

Priority:

```text
G-MultiScale-1D-CNN first
G-TabTransformer second
G-BiLSTM third
G-GCN-PPI last
```

Reason:

```text
GCN needs extra PPI graph setup and torch-geometric.
The CNN/Transformer/LSTM models can use the same expression CSV and are simpler to debug.
```

### Phase 6: Quantum Genomic Models

Implement:

```text
G10: G-Quantum-MLP
G11: G-Pathway-Quantum
G12: G-TNBC-DT-Neural
```

Recommendation:

```text
Use the existing src/models/quantum/vectorized_circuit.py approach first.
Avoid PennyLane in the main training loop unless a paper-specific comparison requires it.
```

Why:

```text
The project already moved TBCA quantum layers toward native PyTorch vectorized circuits.
That is faster, batched, GPU-friendly, and easier to run on a laptop.
```

Suggested reusable genomic quantum block:

```python
class GenomicQuantumBlock(nn.Module):
    def __init__(self, input_dim, output_dim, n_qubits=8, n_layers=2):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, n_qubits),
        )
        self.quantum = VectorizedQuantumCircuit(
            n_qubits=n_qubits,
            n_layers=n_layers,
            rotation_config="u3",
            entanglement="cyclic",
        )
        self.expand = nn.Sequential(
            nn.LayerNorm(n_qubits),
            nn.Linear(n_qubits, output_dim),
        )

    def forward(self, x):
        z = torch.tanh(self.compress(x)) * math.pi
        q = self.quantum(z).to(x.dtype)
        return self.expand(q)
```

For `G-Pathway-Quantum`, the key flow should be:

```text
genes -> pathway activations -> select/learn compact pathway vector -> quantum block -> residual fusion -> classifier
```

### Phase 7: TCGA-BRCA Genomic Pipeline

Implement after GEO is stable:

```text
src/data/genomics/tcga_brca.py
```

Start with RNA-seq only:

```text
PAM50 subtype classification
ER/PR/HER2 binary tasks
tumor vs normal sanity-check task
```

Then add:

```text
methylation
CNV
mutation features
clinical covariates
```

Only then implement `G-CrossOmics-Attention`, because it needs multiple omics modalities.

### Phase 8: Multimodal Late Fusion

Implement only after both unimodal arms are stable:

```text
src/models/multimodal/
  late_fusion.py
  abmil.py
```

Minimum viable design:

```text
TBCA image model -> frozen 768-d image/slide embedding
best genomic model -> genomic embedding or p_genomic
late fusion MLP/XGBoost -> final patient-level prediction
```

Recommended first fusion:

```text
Input: [p_image, p_genomic, image_embedding_128, genomic_embedding_128]
Model: calibrated logistic regression or small MLP
```

Avoid training full WSI patch models end-to-end at first. Extract features once, cache them, and train slide/patient-level heads.

## 5. New TBCA Quantum Variant Plan

The genomics folder also proposes new TBCA image-side variants:

```text
E: PostCrossAttn-Quantum
F: DualQuantum
G: QuantumCrossAttention
H: AllBranch-FM-Quantum
```

Implementation priority:

1. `PostCrossAttn-Quantum`: easiest and most promising.
2. `DualQuantum`: combines current best CNN-FM quantum with post-cross-attn quantum.
3. `AllBranch-FM-Quantum`: useful ablation.
4. `QuantumCrossAttention`: last, highest risk.

Suggested code approach:

```text
Do not rewrite the TBCA class from scratch.
Add a new optional mode, for example quantum_mode.
Keep existing model names working exactly as they do now.
```

Suggested additions:

```python
quantum_mode: Optional[str] = None

# allowed values:
# None
# "post_cross_attn"
# "dual_quantum"
# "all_branch_fm"
# "quantum_cross_attn"
```

For `post_cross_attn`, add:

```text
concat enhanced features: [swin_enh, convnext_enh, effnet_enh] -> 2304 dims
compress 2304 -> 8 qubits
vectorized quantum circuit
expand 8 -> 768
residual add to classical fused vector
```

This is a small, controlled change and can reuse existing quantum code.

## 6. Concrete Task Checklist

### Milestone 1: Genomic MVP

```text
[ ] Add config_genomics.yaml
[ ] Add run_genomic.py skeleton
[ ] Add src/data/genomics package
[ ] Add GEO processed CSV loader
[ ] Add stratified holdout + 5-fold splitter
[ ] Add genomic metrics helper
[ ] Add G-Baseline-MLP
[ ] Add G-Baseline-Trees
[ ] Save fold_results.csv and comparison_genomics.csv
[ ] Run smoke test on synthetic data
[ ] Run real GEO top-500 baseline
```

Exit criteria:

```text
One command trains G-Baseline-MLP and G-Baseline-Trees on GEO TNBC and writes fold metrics.
```

### Milestone 2: Biological Pathway Layer

```text
[ ] Add KEGG pathway download/cache script
[ ] Add pathway membership builder
[ ] Add pathway-ordered gene list
[ ] Add G-PASNet simple version
[ ] Add G-PathFormer-Lite
[ ] Add pathway ablations: top-100, top-500, pathway genes
```

Exit criteria:

```text
Pathway models run with the same CV protocol and outperform or match simple MLP on AUC/calibration.
```

### Milestone 3: Genomic Deep Model Zoo

```text
[ ] Add G-MultiScale-1D-CNN
[ ] Add G-TabTransformer
[ ] Add G-BiLSTM
[ ] Add G-GCN-PPI only after graph dependencies are stable
[ ] Add model registry to run_genomic.py
[ ] Add W&B logging parity with image pipeline
```

Exit criteria:

```text
At least 6 genomic models produce comparable metrics in one summary CSV.
```

### Milestone 4: Quantum Genomics

```text
[ ] Add GenomicQuantumBlock using VectorizedQuantumCircuit
[ ] Add G-Quantum-MLP
[ ] Add G-Pathway-Quantum
[ ] Add ablations: rotation, entanglement, qubits, layers
[ ] Compare vectorized quantum vs PennyLane only if required
```

Exit criteria:

```text
Quantum models run batched on GPU, and results can be compared against G-PASNet and G-Baseline-Trees.
```

### Milestone 5: TCGA And Multimodal

```text
[ ] Add TCGA RNA-seq loader
[ ] Add PAM50 label loader
[ ] Add TCGA RNA-seq preprocessing
[ ] Add TCGA genomic model runs
[ ] Add WSI feature extraction plan/scripts
[ ] Add ABMIL slide aggregator
[ ] Add late-fusion MLP/logistic/XGBoost
```

Exit criteria:

```text
Paired patient-level fusion runs without leakage and reports both unimodal and fused performance.
```

### Milestone 6: New TBCA Quantum Variants

```text
[ ] Add PostCrossAttn quantum block to triple_branch.py
[ ] Add model registry entries in run_pipeline.py
[ ] Add config entries
[ ] Run 1-fold smoke test
[ ] Run 5-fold full training for promising variants
[ ] Compare against TBCA-CNN-FeatureMap-Quantum
```

Exit criteria:

```text
At least Variant E has full 5-fold results and can be compared to existing TBCA variants.
```

## 7. Compute Requirement Summary

Short answer:

```text
An A100 is not mandatory for most genomics work.
An RTX 5050 can run the genomic models and smoke-test the new TBCA variants.
An A100 is strongly recommended for full 5-fold image-side TBCA quantum runs, WSI feature extraction, and large parallel sweeps.
```

Assumption:

```text
RTX 5050 laptop GPU: likely 6-8 GB VRAM.
A100: usually 40-80 GB VRAM.
```

If your RTX 5050 has 8 GB VRAM, the following is realistic.

## 8. Compute Matrix

| Workload | RTX 5050 laptop | A100 needed? | Notes |
|---|---:|---:|---|
| GEO preprocessing | Yes | No | Mostly CPU/RAM. |
| G-Baseline-Trees | Yes | No | CPU is enough; XGBoost can use GPU if available. |
| G-Baseline-MLP | Yes | No | Tiny model. |
| G-PASNet | Yes | No | Small model; batch 16-32. |
| G-TabTransformer | Yes | No | Use batch 8-16 if memory tight. |
| G-MultiScale-1D-CNN | Yes | No | Lightweight. |
| G-BiLSTM | Yes | No | Lightweight. |
| G-GCN-PPI | Maybe | No, but helpful | Depends on graph size and PyG install. CPU may be fine for 325 patients. |
| G-CrossOmics TCGA | Yes | No | More RAM/data complexity than GPU complexity. |
| G-Quantum-MLP vectorized | Yes | No | Use vectorized PyTorch quantum circuit. |
| G-Pathway-Quantum vectorized | Yes | No | Batch 8-16. |
| PennyLane quantum variants | Maybe slow | Helpful | Not a VRAM issue; speed is the issue. |
| Classical TBCA image model | Maybe | Recommended | Batch 1-2 on 8 GB; slow. |
| TBCA-CNN-FM-Quantum | Maybe | Recommended | Existing runs used around 5.7 GB, but runtime is long. |
| New PostCrossAttn TBCA quantum | Maybe | Recommended | Batch 1-2 on RTX 5050; full 5-fold better on A100. |
| WSI patch feature extraction | Possible but slow | Strongly recommended | TCGA WSI scale is huge. |
| End-to-end WSI training | No | Yes | A100 or multi-GPU preferred. |
| Full parallel model zoo | No | Yes | Laptop can run sequentially, not in parallel. |

## 9. Expected Runtime By Stage

These are planning estimates, not benchmarked numbers.

### On RTX 5050 laptop

```text
GEO preprocessing:             minutes to 1 hour
G-Baseline-Trees:              minutes
G-Baseline-MLP 5-fold:         5-20 minutes
G-PASNet / PathFormer 5-fold:  15-90 minutes each
Genomic quantum 5-fold:        30 minutes to a few hours each
TCGA RNA models:               30 minutes to a few hours
TBCA image 1-fold smoke test:  possible, slow
TBCA image full 5-fold:        possible only with patience; likely many hours to days
WSI feature extraction:        possible but impractically slow for large TCGA runs
```

### On A100

```text
GEO genomics models:           very fast; often CPU/data overhead dominates
Genomic model zoo:             practical to run many jobs
TBCA image full 5-fold:        practical
New TBCA quantum variants:     practical
TCGA WSI feature extraction:   practical
Multimodal experiments:        practical
```

## 10. Practical Recommendation

Use RTX 5050 for:

```text
code implementation
data preprocessing debugging
small/subset tests
all classical genomic models
vectorized quantum genomic models
single-fold sanity checks
```

Use A100 for:

```text
final 5-fold TBCA image quantum variants
large WSI feature extraction
full TCGA paired multimodal experiments
parallel sweeps
paper-grade final runs
```

So the answer is:

```text
A100 is not mandatory to start.
RTX 5050 is enough for the genomics implementation and most genomic experiments.
A100 becomes important for final image-side TBCA quantum experiments and WSI-scale multimodal validation.
```

## 11. First Commands After Implementation

Smoke test:

```bash
python run_genomic.py --config config_genomics.yaml --models g_baseline_mlp --synthetic
```

First real GEO run:

```bash
python scripts/genomics/download_geo.py
python scripts/genomics/preprocess_geo.py
python run_genomic.py --config config_genomics.yaml --models g_baseline_mlp g_baseline_trees
```

Pathway run:

```bash
python scripts/genomics/download_pathways.py
python run_genomic.py --config config_genomics.yaml --models g_pasnet g_pathformer_lite
```

Quantum genomic run:

```bash
python run_genomic.py --config config_genomics.yaml --models g_quantum_mlp g_pathway_quantum
```

TBCA post-cross-attention quantum run after implementation:

```bash
python run_pipeline.py --models triple_branch_fusion_post_crossattn_quantum
```

