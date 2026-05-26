# RTX 5050 Startup And Execution Guide For Genomics + TBCA Experiments

This is the local RTX 5050 version of the startup guide. It is written for a Windows laptop/desktop with an NVIDIA RTX 5050 GPU and a project-local Python environment.

The goal is simple:

```text
1. Create the environment.
2. Install dependencies.
3. Prepare the required datasets.
4. Run the implemented genomics baselines.
5. Verify that result files are saved correctly.
6. Know what can and cannot realistically be run on the RTX 5050.
```

## 1. What Can Run On RTX 5050

Good fit for RTX 5050:

```text
GEO TNBC genomics baselines
G-Baseline-MLP
G-Baseline-Trees
future small pathway/genomic models
synthetic smoke tests
single-model or sequential runs
small TBCA image smoke tests
```

Not recommended on RTX 5050:

```text
full TBCA image model zoo
full 5-fold TBCA quantum sweeps
TCGA whole-slide image feature extraction at scale
parallel W&B sweeps
end-to-end WSI training
```

The genomics part is small enough for the RTX 5050. The image TBCA part is much heavier because it runs Swin + ConvNeXt + EfficientNet-B5 together.

## 2. Files To Use For RTX 5050

Use these files:

```text
config_genomics_rtx5050.yaml
run_genomic.py
scripts/genomics/rtx5050_setup.ps1
scripts/genomics/run_rtx5050_genomics.ps1
Genomics/STARTUP_RTX5050_EXECUTION_GUIDE.md
```

The A100 files can stay in the repository for later server runs, but for this laptop workflow use the RTX 5050 files above.

## 3. Fresh Clone Or Fresh Copy

Open PowerShell in the repository root:

```powershell
cd D:\Projects\AI-Projects\Breast_cancer_Minor_Project
```

If a friend clones the repository:

```powershell
git clone <YOUR_REPOSITORY_URL>
cd Breast_cancer_Minor_Project
```

Check important files:

```powershell
dir
dir Genomics
dir scripts\genomics
```

You should see:

```text
run_genomic.py
config_genomics_rtx5050.yaml
requirements.txt
src/
Genomics/
scripts/genomics/
```

## 4. Install NVIDIA Driver

Before Python setup, make sure the RTX 5050 driver works:

```powershell
nvidia-smi
```

Expected:

```text
NVIDIA RTX 5050
Driver Version ...
CUDA Version ...
```

If `nvidia-smi` is not recognized, install/update the NVIDIA driver first.

## 5. Create Python Environment

Recommended Python:

```text
Python 3.10 or Python 3.11
```

Avoid Python 3.14 for now because many ML packages may not be fully compatible.

Create the environment manually:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 6. Install PyTorch For RTX 5050

Install CUDA PyTorch:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

Expected:

```text
True
NVIDIA GeForce RTX 5050...
```

If CUDA is false, reinstall the NVIDIA driver and PyTorch CUDA wheel.

## 7. Install Project Dependencies

Install base requirements:

```powershell
pip install -r requirements.txt
```

Install genomics requirements:

```powershell
pip install GEOparse gseapy xgboost imbalanced-learn lifelines pycombat
```

Optional:

```powershell
pip install torch-geometric
```

Skip `torch-geometric` unless using the future `G-GCN-PPI` graph model.

## 8. One-Command RTX Setup Option

Instead of Sections 5-7, you can run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\genomics\rtx5050_setup.ps1 -Python "py -3.11"
```

If that command does not parse on your machine, use the manual steps above.

After setup:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 9. Data Folders

Create folders:

```powershell
mkdir data
mkdir data\geo_tnbc
mkdir data\geo_tnbc\raw
mkdir data\geo_tnbc\processed
mkdir data\geo_tnbc\pathway_masks
mkdir data\BreaKHis_v1
mkdir outputs_genomics
```

Expected:

```text
data/
  geo_tnbc/
    raw/
    processed/
    pathway_masks/
  BreaKHis_v1/
outputs_genomics/
```

## 10. Dataset 1: GEO TNBC

Main genomics task:

```text
pCR vs RD binary classification
0 = RD
1 = pCR
```

Required cohorts:

```text
GSE25066
GSE20271
GSE20194
GSE32646
```

Download raw GEO files:

```powershell
python -c "import GEOparse; [GEOparse.get_GEO(geo=g, destdir='data/geo_tnbc/raw') for g in ['GSE25066','GSE20271','GSE20194','GSE32646']]"
```

This can take time depending on internet speed.

## 11. Processed GEO Files Required

The implemented `run_genomic.py` expects processed CSV files.

Minimum files needed for current baselines:

```text
data/geo_tnbc/processed/combined_325_genes_100.csv
data/geo_tnbc/processed/combined_325_genes_500.csv
data/geo_tnbc/processed/labels_pcr_rd.csv
```

Expression CSV format:

```text
sample_id,GENE1,GENE2,GENE3
GSM000001,0.12,-0.31,1.50
GSM000002,-1.20,0.44,0.17
```

Label CSV format:

```text
sample_id,label
GSM000001,1
GSM000002,0
```

Required preprocessing:

```text
load GEO expression
extract pCR/RD labels
align sample IDs
log2 transform when needed
quantile normalize
batch correct across cohorts
collapse probes to genes
z-score each gene
save top-100 and top-500 gene CSVs
```

Important:

```text
The current implementation includes the runner and baselines.
Raw GEO preprocessing automation is still a next implementation task.
If processed CSVs are already prepared externally, place them in data/geo_tnbc/processed.
```

## 12. Synthetic Smoke Test

Run this before real data:

```powershell
python run_genomic.py --config config_genomics_rtx5050.yaml --models g_baseline_mlp --synthetic
```

Or use the helper:

```powershell
.\scripts\genomics\run_rtx5050_genomics.ps1 -Config config_genomics_rtx5050.yaml -Phase smoke -Synthetic
```

Expected output:

```text
outputs_genomics/G-Baseline-MLP_synthetic/
outputs_genomics/comparison_synthetic.csv
```

Check:

```powershell
dir outputs_genomics\G-Baseline-MLP_synthetic
type outputs_genomics\comparison_synthetic.csv
```

Expected files:

```text
G-Baseline-MLP_fold_results.csv
G-Baseline-MLP_summary.csv
fold1_epoch_log.csv
fold1_calibration_curve.png
best_model_fold1.pth
```

For full 5-fold synthetic testing, you should see fold 1 through fold 5 files.

## 13. Run GEO Baselines

Once processed GEO CSVs exist:

```powershell
python run_genomic.py --config config_genomics_rtx5050.yaml --models g_baseline_mlp g_baseline_trees
```

Or:

```powershell
.\scripts\genomics\run_rtx5050_genomics.ps1 -Config config_genomics_rtx5050.yaml -Phase phase1
```

Expected output:

```text
outputs_genomics/G-Baseline-MLP_geo_tnbc/
outputs_genomics/G-Baseline-Trees_geo_tnbc/
outputs_genomics/comparison_geo_tnbc.csv
```

## 14. Verify Results

Run:

```powershell
python -c "import pandas as pd; print(pd.read_csv('outputs_genomics/comparison_geo_tnbc.csv').head())"
```

Check fold files:

```powershell
python -c "import pandas as pd; df=pd.read_csv('outputs_genomics/G-Baseline-MLP_geo_tnbc/G-Baseline-MLP_fold_results.csv'); print(df.shape); print(df[['Fold','Val_AUC','Test_auroc','Test_accuracy','Test_fnr']])"
```

Expected:

```text
5 rows for 5 folds
non-empty Val_AUC and Test_auroc columns
comparison_geo_tnbc.csv has one row per completed model
```

Important columns:

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

## 15. W&B On RTX 5050

The RTX config disables W&B by default:

```yaml
wandb_enabled: false
```

This is intentional for laptop runs.

If you want W&B:

```powershell
wandb login
```

Then edit `config_genomics_rtx5050.yaml`:

```yaml
output:
  wandb_enabled: true
```

## 16. BreakHis / TBCA On RTX 5050

BreakHis is the histopathology dataset used by the image pipeline.

Expected folder:

```text
data/BreaKHis_v1/
```

The RTX 5050 can run small tests, but full TBCA models are heavy. For local sanity checks, use a small subset and only one model:

```powershell
python run_pipeline.py --config config.yaml --models efficientnet
```

For TBCA, reduce memory pressure before running:

```yaml
data:
  subset: 200

training:
  epochs: 2

models:
  triple_branch_fusion:
    batch_size: 1
```

Then run:

```powershell
python run_pipeline.py --config config.yaml --models triple_branch_fusion
```

Do not use the RTX 5050 for final full TBCA quantum sweeps unless you are okay with very long runtimes and possible VRAM issues.

## 17. What To Run First

Recommended first run:

```powershell
python run_genomic.py --config config_genomics_rtx5050.yaml --models g_baseline_mlp --synthetic
```

Recommended first real run:

```powershell
python run_genomic.py --config config_genomics_rtx5050.yaml --models g_baseline_mlp g_baseline_trees
```

## 18. Common Problems

### PowerShell blocks script activation

Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### CUDA false

Check:

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

Reinstall PyTorch CUDA wheel if needed.

### Missing processed GEO CSV

Error will look like:

```text
Expression file not found
Labels file not found
```

Fix by placing the files listed in Section 11.

### Sample IDs do not match

Check:

```powershell
python -c "import pandas as pd; x=pd.read_csv('data/geo_tnbc/processed/combined_325_genes_500.csv'); y=pd.read_csv('data/geo_tnbc/processed/labels_pcr_rd.csv'); print(x.iloc[:3,0]); print(y.iloc[:3,0])"
```

The first column should contain matching sample IDs.

### Laptop gets hot or slow

Use smaller runs:

```powershell
python run_genomic.py --config config_genomics_rtx5050.yaml --models g_baseline_mlp --synthetic
```

Or lower:

```yaml
training:
  epochs: 30
  patience: 5
```

## 19. Files To Share After Running

For synthetic test:

```text
outputs_genomics/comparison_synthetic.csv
outputs_genomics/G-Baseline-MLP_synthetic/G-Baseline-MLP_fold_results.csv
```

For real GEO:

```text
outputs_genomics/comparison_geo_tnbc.csv
outputs_genomics/G-Baseline-MLP_geo_tnbc/G-Baseline-MLP_fold_results.csv
outputs_genomics/G-Baseline-Trees_geo_tnbc/G-Baseline-Trees_fold_results.csv
```

Also share terminal logs if something fails.

## 20. Current Implementation Status

Implemented:

```text
run_genomic.py
config_genomics_rtx5050.yaml
src/data/genomics/
src/models/genomics/baseline_mlp.py
src/models/genomics/baseline_trees.py
src/utils/genomics_metrics.py
scripts/genomics/rtx5050_setup.ps1
scripts/genomics/run_rtx5050_genomics.ps1
```

Ready now:

```text
synthetic smoke test
G-Baseline-MLP on processed GEO CSVs
G-Baseline-Trees on processed GEO CSVs
result CSV saving
calibration plot saving
checkpoint saving for MLP
```

Next implementation tasks:

```text
automated GEO preprocessing
G-PASNet
G-PathFormer-Lite
G-Quantum-MLP
G-Pathway-Quantum
TCGA-BRCA loader
multimodal late fusion
```

