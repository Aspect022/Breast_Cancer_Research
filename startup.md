# Startup Guide: Full Breast Cancer Research Pipeline

This file is the handoff guide for running the full project on a Linux GPU
server. Follow it from top to bottom. The main goal is that you can start the
pipeline once, leave it running, and come back later to logs and saved results.

## One Command To Run Everything

From the repository root on the server, run:

```bash
mkdir -p logs
nohup bash run_server_auto_pipeline.sh > logs/server_auto_pipeline.out 2>&1 &
tail -f logs/server_auto_pipeline.out
```

That command starts the full unattended runner in the background and streams the
live log. You can close the SSH session after confirming it has started.

To check progress later:

```bash
tail -f logs/server_auto_pipeline.out
```

## What The Startup Script Does

The script `run_server_auto_pipeline.sh` performs the full setup and execution:

1. Creates or reuses `.venv`.
2. Installs Python packages from `requirements.txt`.
3. Installs PyTorch using the CUDA 12.8 wheel index by default.
4. Installs genomics dependencies.
5. Checks that CUDA/GPU is visible.
6. Checks each dataset path.
7. Downloads datasets that can be downloaded automatically on the server.
8. Runs genomics models.
9. Runs histopathology models.
10. Validates that result CSV files were created.
11. Saves logs and outputs.

The script is strict on purpose. It does not create fake or simulated research
data. If a required real dataset is missing and cannot be downloaded, it stops
with a clear error.

## Datasets Used

This project uses the following datasets.

| Dataset | Purpose | Used By | Default Path |
|---|---|---|---|
| BreakHis | Breast histopathology image classification | CNN, transformer, quantum, fusion, TBCA/triple-branch models | `data/BreaKHis_v1` |
| WBCD | Small tabular breast cancer dataset | Optional classical/clinical checks and baseline dataset availability | `data/WBCD/wbcd.csv` |
| GEO TNBC cohorts | Genomics model experiments for TNBC pCR/RD prediction | `run_genomic.py` baseline genomics models | `data/geo_tnbc/raw` and `data/geo_tnbc/processed` |

## Automatic Dataset Downloads

The startup script downloads the datasets on the server whenever possible.

| Dataset | Automatic Download Method | Server Requirement |
|---|---|---|
| BreakHis | Kaggle API dataset download | Kaggle credentials must already be configured on the server |
| WBCD | Direct URL download from UCI | `curl` or `wget` |
| GEO raw cohorts | Direct URL download from NCBI GEO FTP | `curl` or `wget` |
| GEO processed CSVs | Not generated automatically | Must already exist in `data/geo_tnbc/processed` |

## Dataset Storage Paths

Keep the paths exactly like this unless you also update the config files.

```text
data/
  BreaKHis_v1/
    histology_slides/
      breast/
        benign/
        malignant/

  WBCD/
    wbcd.csv

  geo_tnbc/
    raw/
      GSE25066_family.soft.gz
      GSE20271_family.soft.gz
      GSE20194_family.soft.gz
      GSE32646_family.soft.gz

    processed/
      combined_325_genes_100.csv
      combined_325_genes_500.csv
      labels_pcr_rd.csv
```

The histopathology path is also configured in:

```text
config.yaml
```

The genomics path is also configured in:

```text
config_genomics_rtx5050.yaml
```

If your server stores data somewhere else, either edit those config files or run
with environment variables:

```bash
BREAKHIS_ROOT=/absolute/path/to/BreaKHis_v1 \
GEO_ROOT=/absolute/path/to/geo_tnbc \
nohup bash run_server_auto_pipeline.sh > logs/server_auto_pipeline.out 2>&1 &
```

## Manual Download Links If Automatic Download Fails

Use these links only if the automatic server download fails.

### BreakHis

Primary automatic method:

```text
Kaggle dataset id: ambarish/breakhis
```

Kaggle page:

```text
https://www.kaggle.com/datasets/ambarish/breakhis
```

Official dataset page:

```text
https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/
```

Expected storage after extraction:

```text
data/BreaKHis_v1/
```

If Kaggle extracts into a different folder under `data/`, move or point
`BREAKHIS_ROOT` to the folder that contains the BreakHis image tree.

### WBCD

Direct URL:

```text
https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
```

Expected storage:

```text
data/WBCD/wbcd.csv
```

### GEO TNBC Raw Cohorts

The startup script downloads these files automatically from NCBI:

```text
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE25nnn/GSE25066/soft/GSE25066_family.soft.gz
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20271/soft/GSE20271_family.soft.gz
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE20nnn/GSE20194/soft/GSE20194_family.soft.gz
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE32nnn/GSE32646/soft/GSE32646_family.soft.gz
```

Expected storage:

```text
data/geo_tnbc/raw/
```

### GEO TNBC Processed Files

These files are required for `run_genomic.py`:

```text
data/geo_tnbc/processed/combined_325_genes_100.csv
data/geo_tnbc/processed/combined_325_genes_500.csv
data/geo_tnbc/processed/labels_pcr_rd.csv
```

The startup script checks for them before running genomics. If they are missing,
the script stops and prints the missing file names. It does not create synthetic
genomics data.

## Kaggle Setup For BreakHis

BreakHis is downloaded through Kaggle. On the server, Kaggle credentials should
already be configured. If not, place `kaggle.json` here:

```text
~/.kaggle/kaggle.json
```

Then secure it:

```bash
chmod 600 ~/.kaggle/kaggle.json
```

You can verify Kaggle access with:

```bash
kaggle datasets files ambarish/breakhis
```

The startup script also installs the Python `kaggle` package through
`requirements.txt`.

## Expected Result Files

After a successful full run, check these files:

```text
outputs/comparison_binary.csv
outputs_genomics/comparison_geo_tnbc.csv
```

Logs are saved here:

```text
logs/server_auto_pipeline.out
logs/server_auto_pipeline_<timestamp>.log
logs/histology_<timestamp>.log
logs/genomics_<timestamp>.log
```

Model checkpoints and detailed outputs are saved under:

```text
outputs/
outputs_genomics/
```

## Common Run Options

Run only histopathology models:

```bash
RUN_GENOMICS=0 nohup bash run_server_auto_pipeline.sh > logs/server_auto_pipeline.out 2>&1 &
```

Run only genomics models:

```bash
RUN_HISTOLOGY=0 nohup bash run_server_auto_pipeline.sh > logs/server_auto_pipeline.out 2>&1 &
```

Use A100 genomics config:

```bash
GENOMICS_CONFIG=config_genomics_a100.yaml nohup bash run_server_auto_pipeline.sh > logs/server_auto_pipeline.out 2>&1 &
```

Run selected histology models only:

```bash
HISTOLOGY_MODELS="triple_branch_fusion triple_branch_fusion_quantum" \
nohup bash run_server_auto_pipeline.sh > logs/server_auto_pipeline.out 2>&1 &
```

Disable the strict CUDA check for a CPU smoke test:

```bash
REQUIRE_CUDA=0 nohup bash run_server_auto_pipeline.sh > logs/server_auto_pipeline.out 2>&1 &
```

## Final Checklist Before Running

Before starting the long run, confirm:

1. You are in the repository root.
2. The server has Python 3.10 or 3.11.
3. The GPU driver is installed and `nvidia-smi` works.
4. Kaggle credentials exist at `~/.kaggle/kaggle.json`.
5. If running genomics, the processed GEO CSV files exist in `data/geo_tnbc/processed`.
6. You have enough disk space for datasets, checkpoints, and outputs.

Then run the one command from the top of this file.
