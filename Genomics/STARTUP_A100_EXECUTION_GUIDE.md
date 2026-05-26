# A100 Startup And Execution Guide For Genomics + TBCA Experiments

This guide explains how to take this repository on a fresh production machine, prepare the environment, download/prepare the required datasets, run the genomics experiments, run the TBCA experiments, and verify that results are saved correctly.

It is written as a handoff document. A new collaborator should be able to follow it step by step.

## 1. What This Repository Contains

The project has two connected research tracks:

```text
Track 1: Histopathology image models
  Existing runner: run_pipeline.py
  Main dataset: BreakHis
  Main advanced architecture: TripleBranch-Fusion / TBCA variants

Track 2: Genomics and multimodal extension
  New runner: run_genomic.py
  Main genomic dataset: GEO TNBC pCR/RD cohorts
  Later multimodal dataset: TCGA-BRCA paired WSI + genomics
```

The current genomics implementation starts with:

```text
G-Baseline-MLP
G-Baseline-Trees
```

The planned next models are already documented/configured:

```text
G-PASNet
G-PathFormer-Lite
G-TabTransformer
G-MultiScale-1D-CNN
G-BiLSTM
G-GCN-PPI
G-Quantum-MLP
G-Pathway-Quantum
G-TNBC-DT-Neural
G-CrossOmics
```

## 2. Recommended Machine

For final experiments, use:

```text
GPU: NVIDIA A100, 40 GB or 80 GB preferred
CPU: 16+ cores
RAM: 64 GB minimum, 128 GB preferred for TCGA/WSI work
Disk: 500 GB minimum, 1-2 TB preferred if downloading TCGA slides
OS: Linux server, Ubuntu 20.04/22.04 recommended
Python: 3.10 or 3.11 recommended
CUDA: compatible with PyTorch CUDA 12.1 or CUDA 11.8
```

A100 is not required for tiny genomics smoke tests, but this guide assumes production-grade A100 execution.

## 3. Fresh Clone

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd Breast_cancer_Minor_Project
```

Check the important files:

```bash
ls
ls Genomics
ls scripts/genomics
```

You should see:

```text
run_pipeline.py
run_genomic.py
config.yaml
config_genomics_a100.yaml
requirements.txt
Genomics/
scripts/genomics/
src/
```

## 4. Create The Python Environment

Use a project-local virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

If the server uses Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Do not use Python 3.14 for this project unless all ML packages are confirmed compatible.

## 5. Install PyTorch For A100

Use one of the following depending on the server CUDA setup.

Recommended CUDA 12.1 wheel:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Alternative CUDA 11.8 wheel:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Verify:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Expected:

```text
True
NVIDIA A100...
```

## 6. Install Project Requirements

Install the base project dependencies:

```bash
pip install -r requirements.txt
```

Install genomics-specific dependencies:

```bash
pip install GEOparse gseapy xgboost imbalanced-learn lifelines pycombat
```

Optional graph-model dependency, only needed for `G-GCN-PPI`:

```bash
pip install torch-geometric
```

If `torch-geometric` fails, skip `G-GCN-PPI` first. It is not required for the baseline genomics pipeline.

## 7. A100 Setup Script Option

The repository includes a helper script:

```bash
chmod +x scripts/genomics/a100_setup.sh
scripts/genomics/a100_setup.sh
```

This installs PyTorch, project requirements, and genomics packages. If your production server has a strict CUDA/PyTorch policy, install manually using Sections 4-6 instead.

## 8. Configure Weights & Biases

The project supports W&B logging. If you want W&B:

```bash
wandb login
```

Or set an API key:

```bash
export WANDB_API_KEY=<YOUR_KEY>
```

The genomics config uses:

```text
breast-cancer-genomics-a100
```

If you want to disable W&B, edit:

```text
config_genomics_a100.yaml
```

Set:

```yaml
output:
  wandb_enabled: false
```

## 9. Directory Structure For Data

Create the data directories:

```bash
mkdir -p data/geo_tnbc/raw
mkdir -p data/geo_tnbc/processed
mkdir -p data/geo_tnbc/pathway_masks

mkdir -p data/tcga_brca/rnaseq
mkdir -p data/tcga_brca/methylation
mkdir -p data/tcga_brca/cnv
mkdir -p data/tcga_brca/clinical
mkdir -p data/tcga_brca/processed
mkdir -p data/tcga_brca/slides
mkdir -p data/tcga_brca/patches
mkdir -p data/tcga_brca/features

mkdir -p data/BreaKHis_v1
```

The final expected structure:

```text
data/
  geo_tnbc/
    raw/
    processed/
    pathway_masks/
  tcga_brca/
    rnaseq/
    methylation/
    cnv/
    clinical/
    processed/
    slides/
    patches/
    features/
  BreaKHis_v1/
```

## 10. Dataset 1: GEO TNBC pCR/RD

This is the first genomics dataset to prepare.

Required GEO cohorts:

```text
GSE25066
GSE20271
GSE20194
GSE32646
```

Task:

```text
Binary classification
0 = RD, residual disease
1 = pCR, pathologic complete response
```

Download raw GEO files:

```bash
python - <<'PY'
import GEOparse

cohorts = ["GSE25066", "GSE20271", "GSE20194", "GSE32646"]
for gse_id in cohorts:
    print("Downloading", gse_id)
    GEOparse.get_GEO(geo=gse_id, destdir="data/geo_tnbc/raw")
PY
```

After download, the raw folder should contain GEO files for the four cohorts:

```bash
ls data/geo_tnbc/raw
```

## 11. GEO Preprocessing Requirements

The genomics runner expects processed CSV files in:

```text
data/geo_tnbc/processed/
```

Required files:

```text
combined_325_genes_100.csv
combined_325_genes_500.csv
combined_325_genes_1000.csv
labels_pcr_rd.csv
batches.csv
lehmann_subtypes.csv
```

Minimum required to run current implemented models:

```text
combined_325_genes_100.csv
combined_325_genes_500.csv
labels_pcr_rd.csv
```

CSV format for expression:

```text
sample_id,GENE1,GENE2,GENE3,...
GSMxxxxx,0.12,-1.24,0.88,...
GSMyyyyy,-0.44,0.31,1.12,...
```

CSV format for labels:

```text
sample_id,label
GSMxxxxx,1
GSMyyyyy,0
```

The preprocessing should do:

```text
1. Load expression matrices from all GEO cohorts.
2. Extract pCR/RD labels from phenotype metadata.
3. Align sample IDs between expression and labels.
4. Log2 transform when needed.
5. Quantile normalize.
6. Correct batch effects across cohorts.
7. Collapse probes to genes.
8. Z-score each gene.
9. Create feature panels:
   - top 100 genes
   - top 500 genes
   - top 1000 genes
10. Save processed CSVs.
```

Important leakage rule:

```text
Unsupervised feature selection, such as top variance/CV genes, can be done globally for the first baseline.
Supervised differential-expression feature selection must be done inside each training fold.
Do not use validation/test labels to select genes.
```

## 12. Quick Synthetic Smoke Test

Before real GEO data is ready, run the synthetic test:

```bash
python run_genomic.py \
  --config config_genomics_a100.yaml \
  --models g_baseline_mlp \
  --synthetic
```

Expected output folder:

```text
outputs_genomics/G-Baseline-MLP_synthetic/
```

Expected files:

```text
G-Baseline-MLP_fold_results.csv
G-Baseline-MLP_summary.csv
fold1_epoch_log.csv
fold1_calibration_curve.png
best_model_fold1.pth
```

Also check:

```text
outputs_genomics/comparison_synthetic.csv
```

If this works, the runner, split logic, model, metrics, and saving pipeline are functioning.

## 13. Run GEO Baselines

After processed GEO files are available:

```bash
python run_genomic.py \
  --config config_genomics_a100.yaml \
  --models g_baseline_mlp g_baseline_trees
```

Or use the A100 phase script:

```bash
chmod +x scripts/genomics/run_a100_genomics.sh
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase1
```

Expected output folders:

```text
outputs_genomics/G-Baseline-MLP_geo_tnbc/
outputs_genomics/G-Baseline-Trees_geo_tnbc/
```

Expected comparison file:

```text
outputs_genomics/comparison_geo_tnbc.csv
```

## 14. Validate GEO Results Were Saved Correctly

Run:

```bash
ls outputs_genomics
ls outputs_genomics/G-Baseline-MLP_geo_tnbc
ls outputs_genomics/G-Baseline-Trees_geo_tnbc
```

Check fold results:

```bash
python - <<'PY'
import pandas as pd

paths = [
    "outputs_genomics/G-Baseline-MLP_geo_tnbc/G-Baseline-MLP_fold_results.csv",
    "outputs_genomics/G-Baseline-Trees_geo_tnbc/G-Baseline-Trees_fold_results.csv",
    "outputs_genomics/comparison_geo_tnbc.csv",
]

for path in paths:
    print("\n==", path)
    df = pd.read_csv(path)
    print(df.head())
    print("shape:", df.shape)
PY
```

For 5-fold CV, each model fold file should have:

```text
5 rows
```

The comparison file should have one row per completed model.

Core columns to verify:

```text
Val_AUC
Test_accuracy
Test_balanced_accuracy
Test_auroc
Test_auprc
Test_sensitivity
Test_specificity
Test_f1
Test_mcc
Test_fnr
Test_brier
Test_ece
```

## 15. Run Future Genomic Phases

The A100 phase script is already prepared for the full plan.

Phase 2 pathway models:

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase2
```

Phase 3 deep genomic models:

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase3
```

Phase 4 graph model:

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase4
```

Phase 5 quantum genomic models:

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase5
```

Phase 6 TNBC-DT neural:

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase6
```

Phase 7 TCGA cross-omics:

```bash
scripts/genomics/run_a100_genomics.sh config_genomics_a100.yaml phase7
```

Note:

```text
Only G-Baseline-MLP and G-Baseline-Trees are implemented in the current runner.
The later phases are already planned/configured, but their model modules must be implemented before those commands will run successfully.
```

## 16. Dataset 2: BreakHis Histopathology

BreakHis is needed for the existing image/TBCA pipeline.

Expected location:

```text
data/BreaKHis_v1/
```

Expected internal structure:

```text
data/BreaKHis_v1/histology_slides/breast/benign/...
data/BreaKHis_v1/histology_slides/breast/malignant/...
```

Run only TBCA classical and implemented quantum variants:

```bash
python run_pipeline.py \
  --config config.yaml \
  --models \
  triple_branch_fusion \
  triple_branch_fusion_quantum \
  triple_branch_fusion_bottleneck \
  triple_branch_fusion_cnn_featuremap_quantum \
  triple_branch_fusion_vit_featuremap_quantum
```

Or use:

```bash
chmod +x scripts/genomics/run_a100_tbca_quantum_variants.sh
scripts/genomics/run_a100_tbca_quantum_variants.sh config.yaml
```

Expected output folders:

```text
outputs/TripleBranch-Fusion_binary/
outputs/TBCA-Quantum-Fusion_binary/
outputs/TBCA-Quantum-Bottleneck_binary/
outputs/TBCA-CNN-FeatureMap-Quantum_binary/
outputs/TBCA-ViT-FeatureMap-Quantum_binary/
```

Expected comparison:

```text
outputs/comparison_binary.csv
```

## 17. Validate TBCA Results Were Saved Correctly

Run:

```bash
python - <<'PY'
import pandas as pd

path = "outputs/comparison_binary.csv"
df = pd.read_csv(path)
print(df[["Model", "Mean_Best_Val_Acc", "Mean_Acc", "Mean_AUC", "Mean_Sensitivity", "Mean_Specificity", "Mean_FNR"]])
PY
```

Each model output folder should contain:

```text
*_fold_results.csv
*_fold1_epoch_log.csv
*_fold1_training_curves.png
*_fold1_confusion_matrix.png
*_fold1_roc_curve.png
best_model_fold1.pth
```

For full 5-fold runs, check that there are fold 1 through fold 5 files.

## 18. Dataset 3: TCGA-BRCA Genomics

TCGA-BRCA is needed for the later genomic/multimodal extension.

Use:

```text
GDC Portal
Project: TCGA-BRCA
Data Category: Transcriptome Profiling
Data Type: Gene Expression Quantification
Workflow Type: HTSeq Counts or FPKM
```

Download with GDC client:

```bash
gdc-client download -m gdc_manifest.txt -d data/tcga_brca/rnaseq
```

Clinical/PAM50 labels should be saved to:

```text
data/tcga_brca/clinical/
data/tcga_brca/processed/pam50_labels.csv
```

Expected processed RNA file:

```text
data/tcga_brca/processed/rnaseq_fpkm_top1000.csv
```

TCGA commands in this repository should only be run after the TCGA preprocessing loader is implemented.

## 19. Dataset 4: TCGA-BRCA WSI Slides

WSI slides are needed only for late multimodal validation.

Use:

```text
GDC Portal
Project: TCGA-BRCA
Data Category: Biospecimen / Slide Images
Experimental Strategy: Diagnostic Slide
File format: SVS
```

Download:

```bash
gdc-client download -m gdc_slide_manifest.txt -d data/tcga_brca/slides
```

Expected:

```text
data/tcga_brca/slides/*.svs
```

WSI processing requires CLAM or equivalent patch extraction:

```bash
git clone https://github.com/mahmoodlab/CLAM.git external/CLAM
```

Patch extraction and feature extraction are expensive. Run them only on the A100/server.

## 20. Multimodal Fusion Plan

The final multimodal stage should run only after:

```text
1. A trained TBCA or TBCA quantum image model exists.
2. TCGA WSI patches/features are extracted.
3. TCGA RNA-seq features are processed.
4. Patient IDs are matched by first 12 TCGA barcode characters.
```

Patient barcode rule:

```python
patient_id = barcode[:12]
```

Late fusion inputs:

```text
p_image
p_genomic
image_embedding
genomic_embedding
clinical covariates if used
```

Expected later output:

```text
outputs_multimodal/
  comparison_tcga_paired.csv
  late_fusion_fold_results.csv
```

## 21. Full Production Run Order

Recommended order for a clean production run:

```bash
# 1. Environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install GEOparse gseapy xgboost imbalanced-learn lifelines pycombat

# 2. Verify A100
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 3. Download GEO
python - <<'PY'
import GEOparse
for gse_id in ["GSE25066", "GSE20271", "GSE20194", "GSE32646"]:
    GEOparse.get_GEO(geo=gse_id, destdir="data/geo_tnbc/raw")
PY

# 4. Prepare processed GEO CSVs
# Use the preprocessing pipeline described in Section 11.

# 5. Smoke test
python run_genomic.py --config config_genomics_a100.yaml --models g_baseline_mlp --synthetic

# 6. GEO baseline results
python run_genomic.py --config config_genomics_a100.yaml --models g_baseline_mlp g_baseline_trees

# 7. Existing TBCA image results
python run_pipeline.py --config config.yaml --models triple_branch_fusion triple_branch_fusion_cnn_featuremap_quantum
```

## 22. Final Result Files To Collect

For genomics:

```text
outputs_genomics/comparison_geo_tnbc.csv
outputs_genomics/G-Baseline-MLP_geo_tnbc/G-Baseline-MLP_fold_results.csv
outputs_genomics/G-Baseline-Trees_geo_tnbc/G-Baseline-Trees_fold_results.csv
```

For histopathology/TBCA:

```text
outputs/comparison_binary.csv
outputs/TripleBranch-Fusion_binary/TripleBranch-Fusion_fold_results.csv
outputs/TBCA-CNN-FeatureMap-Quantum_binary/TBCA-CNN-FeatureMap-Quantum_fold_results.csv
```

For each model, archive:

```text
fold_results.csv
summary.csv if present
epoch logs
training curves
ROC curves
confusion matrices
calibration curves for genomics
best_model_fold*.pth checkpoints
```

## 23. Common Failure Points

### CUDA unavailable

Check:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

If false, reinstall PyTorch with the correct CUDA wheel.

### GEO labels do not align with expression

The runner requires overlapping sample IDs.

Check:

```bash
python - <<'PY'
import pandas as pd
x = pd.read_csv("data/geo_tnbc/processed/combined_325_genes_500.csv")
y = pd.read_csv("data/geo_tnbc/processed/labels_pcr_rd.csv")
print(x.head())
print(y.head())
PY
```

Both files must share the same `sample_id` values.

### Too few samples in one class

Check label balance:

```bash
python - <<'PY'
import pandas as pd
y = pd.read_csv("data/geo_tnbc/processed/labels_pcr_rd.csv")
print(y["label"].value_counts())
PY
```

Both classes must be present.

### Missing processed files

The runner will raise:

```text
Expression file not found
Labels file not found
```

Create the files listed in Section 11.

### W&B not desired

Disable it in config:

```yaml
output:
  wandb_enabled: false
```

## 24. What To Send Back After Running

After the A100 run, share these files:

```text
outputs_genomics/comparison_geo_tnbc.csv
outputs_genomics/*/*_fold_results.csv
outputs/comparison_binary.csv
outputs/*/*_fold_results.csv
```

Also share the terminal log or `nohup.out` if using background execution.

Recommended background command:

```bash
nohup python run_genomic.py --config config_genomics_a100.yaml --models g_baseline_mlp g_baseline_trees > genomics_a100.out 2>&1 &
tail -f genomics_a100.out
```

For TBCA:

```bash
nohup python run_pipeline.py --config config.yaml --models triple_branch_fusion triple_branch_fusion_cnn_featuremap_quantum > tbca_a100.out 2>&1 &
tail -f tbca_a100.out
```

## 25. Current Implementation Status

Implemented now:

```text
run_genomic.py
src/data/genomics/
src/models/genomics/baseline_mlp.py
src/models/genomics/baseline_trees.py
src/utils/genomics_metrics.py
config_genomics_a100.yaml
A100 helper scripts
```

Ready to run once dependencies and processed data exist:

```text
G-Baseline-MLP
G-Baseline-Trees
Synthetic smoke test
GEO TNBC baseline run with processed CSVs
```

Still to implement:

```text
GEO raw preprocessing automation
G-PASNet
G-PathFormer-Lite
G-TabTransformer
G-MultiScale-1D-CNN
G-BiLSTM
G-GCN-PPI
G-Quantum-MLP
G-Pathway-Quantum
TCGA-BRCA loader
Multimodal late fusion
New TBCA post-cross-attention quantum variants
```

This status is intentional: the first executable target is to get trustworthy baseline genomic results saved correctly, then expand the model zoo.

