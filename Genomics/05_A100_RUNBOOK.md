# A100 Genomics And TBCA Runbook

This runbook prepares the project for server-first execution on an NVIDIA A100. It assumes final paper-grade runs will happen on the A100, while local/laptop work is only for quick debugging.

## 1. Files Added For A100

```text
config_genomics_a100.yaml
scripts/genomics/a100_setup.sh
scripts/genomics/run_a100_genomics.sh
scripts/genomics/run_a100_tbca_quantum_variants.sh
Genomics/05_A100_RUNBOOK.md
```

Important:

```text
run_genomic.py is still the next implementation target.
The A100 config and scripts are ready for that runner once implemented.
```

## 2. Server Setup

From the project root:

```bash
chmod +x scripts/genomics/a100_setup.sh
chmod +x scripts/genomics/run_a100_genomics.sh
chmod +x scripts/genomics/run_a100_tbca_quantum_variants.sh

scripts/genomics/a100_setup.sh
```

Verify GPU:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

```text
CUDA available: True
GPU: NVIDIA A100...
```

## 3. A100 Defaults

The A100 config enables:

```text
5-fold CV
15% held-out test set
AMP
TF32
cuDNN benchmark mode
W&B project: breast-cancer-genomics-a100
larger genomic batch sizes
SMOTE on training folds only
class-weighted losses
```

The main config is:

```text
config_genomics_a100.yaml
```

## 4. Recommended Execution Order

### Step 1: Implement And Smoke-Test `run_genomic.py`

Before real server runs, the genomics runner needs to exist. Minimum supported command:

```bash
python run_genomic.py --config config_genomics_a100.yaml --models g_baseline_mlp --synthetic
```

Exit condition:

```text
The command creates outputs_genomics/.../fold_results.csv without touching real GEO data.
```

### Step 2: Download And Preprocess GEO TNBC

Once scripts are implemented:

```bash
python scripts/genomics/download_geo.py
python scripts/genomics/preprocess_geo.py --config config_genomics_a100.yaml
```

Expected processed data:

```text
data/geo_tnbc/processed/combined_325_genes_100.csv
data/geo_tnbc/processed/combined_325_genes_500.csv
data/geo_tnbc/processed/combined_325_genes_1000.csv
data/geo_tnbc/processed/labels_pcr_rd.csv
```

### Step 3: Phase 1 Baselines

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase1
```

Models:

```text
g_baseline_mlp
g_baseline_trees
```

Purpose:

```text
Validate labels, splits, metrics, and output saving.
```

### Step 4: Phase 2 Pathway Models

```bash
python scripts/genomics/download_pathways.py
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase2
```

Models:

```text
g_pasnet
g_pathformer_lite
```

Purpose:

```text
Add biological grounding through KEGG pathway structure.
```

### Step 5: Phase 3 Deep Genomic Models

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase3
```

Models:

```text
g_tabtransformer
g_multiscale_1dcnn
g_bilstm
```

### Step 6: Phase 5 Quantum Genomics

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase5
```

Models:

```text
g_quantum_mlp
g_pathway_quantum
```

Recommendation:

```text
Use vectorized_torch backend first.
Only run PennyLane ablations later if specifically required for the paper.
```

### Step 7: Optional Graph Model

Only after installing PyTorch Geometric cleanly:

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase4
```

Model:

```text
g_gcn_ppi
```

### Step 8: TCGA Multi-Omics

Only after GEO genomics is stable:

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase7
```

Model:

```text
g_crossomics
```

## 5. TBCA Image-Side A100 Runs

Existing TBCA variants can already be run with:

```bash
scripts/genomics/run_a100_tbca_quantum_variants.sh config.yaml
```

This currently runs:

```text
triple_branch_fusion
triple_branch_fusion_quantum
triple_branch_fusion_bottleneck
triple_branch_fusion_cnn_featuremap_quantum
triple_branch_fusion_vit_featuremap_quantum
```

The proposed new variants still need implementation:

```text
triple_branch_fusion_post_crossattn_quantum
triple_branch_fusion_dual_quantum
triple_branch_fusion_all_branch_fm_quantum
triple_branch_fusion_quantum_cross_attn
```

Recommended image-side implementation order:

```text
1. PostCrossAttn-Quantum
2. DualQuantum
3. AllBranch-FM-Quantum
4. QuantumCrossAttention
```

## 6. A100 Compute Settings

Use these for A100 final runs:

```bash
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export WANDB_PROJECT=breast-cancer-genomics-a100
```

PyTorch settings are handled in the scripts:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```

## 7. Output Expectations

Genomics outputs:

```text
outputs_genomics/
  comparison_geo_tnbc.csv
  G-Baseline-MLP_geo_tnbc/
  G-Baseline-Trees_geo_tnbc/
  G-PASNet_geo_tnbc/
  G-PathFormer-Lite_geo_tnbc/
  ...
```

Image outputs continue under:

```text
outputs/
```

## 8. Practical A100 Run Policy

Use A100 for:

```text
full 5-fold genomics model zoo
quantum genomic ablations
new TBCA quantum variants
WSI patch feature extraction
TCGA paired multimodal validation
parallel W&B sweeps
```

Do not waste A100 time on:

```text
syntax-only smoke tests
synthetic data tests
single-batch debugging
```

Those should be done locally first.

## 9. Immediate Next Engineering Task

The next code task is:

```text
Implement run_genomic.py and the minimal src/data/genomics + src/models/genomics baseline modules.
```

Minimum useful first deliverable:

```text
python run_genomic.py --config config_genomics_a100.yaml --models g_baseline_mlp g_baseline_trees
```

Once that works, the A100 scripts in this runbook can drive the rest of the experiment phases.

