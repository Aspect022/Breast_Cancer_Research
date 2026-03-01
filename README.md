# Breast Cancer Histopathological Classification — EfficientNet-B3 Baseline

Research-grade baseline for classifying breast cancer histopathology images (BreakHis dataset) using an EfficientNet-B3 pretrained on ImageNet, with mixed-precision training, patient-level data splitting, and comprehensive metrics logging.

## Project Structure

```text
.
├── data/                  # (gitignored) Raw / processed BreakHis dataset
├── outputs/               # (gitignored) Saved models, plots, CSVs
├── notebooks/             # Jupyter notebooks for exploration
├── docs/                  # Documentation
├── src/
│   ├── data/
│   │   └── dataset.py     # BreakHis loader + patient-grouped stratified split
│   ├── models/
│   │   ├── efficientnet.py # EfficientNet-B3 transfer learning model
│   │   ├── quantum/        # (future) Quantum ML approaches
│   │   └── spiking/        # (future) Spiking Neural Network approaches
│   ├── utils/
│   │   └── metrics.py      # Metrics computation, plotting, profiling, CSV logging
│   └── train.py            # Main training script (mixed precision + early stopping)
├── requirements.txt
└── README.md
```

## Quick Start / Setup for Collaborators

We have provided automated setup scripts to get you up and running instantly.

### 1. Environment Initialization
Run the setup script for your operating system:

**Windows:**
Double-click `setup.bat` or run:
```cmd
setup.bat
```

**macOS/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

This will automatically create a `.venv` virtual environment and install all dependencies.

### 2. Activate the Environment
Before running any code, ensure your terminal is using the virtual environment:
- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 3. Dataset Setup
Download the [BreakHis dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) and extract it into the `data/` folder in this repository. 
Your path should look something like: `data/BreaKHis_v1/...`

### 4. Sanity Check First (Recommended!)
Test the pipeline to ensure your environment and dataset are wired up correctly:
```bash
python src/train.py --task binary --data_dir data/BreaKHis_v1 --subset 200 --epochs 3
```

### 5. Full Training Runs

**Train — Binary (Benign vs Malignant)**

```bash
python src/train.py --task binary --data_dir data/BreaKHis_v1
```

**Train — 3-Class (IDC, ILC, Fibroadenoma)**

```bash
python src/train.py --task multi --data_dir data/BreaKHis_v1
```

## Key Design Decisions

| Decision | Choice | Rationale |
| --- | --- | --- |
| **Data split** | GroupShuffleSplit by Patient ID | Prevents data leakage |
| **Loss** | CrossEntropyLoss (2-class for binary) | Cleaner ROC / confusion matrix |
| **Early Stopping** | On Validation AUC | More clinically relevant than loss |
| **Mixed Precision** | `torch.cuda.amp` | Faster training on RTX 5050 |
| **Optimizer** | AdamW (lr=1e-4, wd=1e-4) | Standard for fine-tuning |
| **Scheduler** | CosineAnnealingLR | Smooth LR decay |

## Outputs (saved to `outputs/`)

- `best_model.pth` — Best checkpoint (by val AUC)
- `*_training_curves.png` — Loss & accuracy per epoch
- `*_confusion_matrix.png` — Heatmap
- `*_roc_curve.png` — ROC curve(s)
- `*_results.csv` — Summary table (Accuracy, F1, AUC, Params, FLOPs, Time, etc.)
- `*_epoch_log.csv` — Per-epoch metrics log

## Reproducibility

All random seeds are fixed (`torch`, `numpy`, `random` — default seed 42).

## GPU Requirements

Tested on NVIDIA RTX 5050. Mixed precision enabled by default.
