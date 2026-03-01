# Breast Cancer Histopathological Classification

Research-grade multi-paradigm framework for classifying breast cancer histopathology images (BreakHis dataset). Implements and compares **5 model architectures** with 5-fold patient-grouped cross-validation:

| # | Model | Architecture |
|---|---|---|
| 1 | **EfficientNet-B3** | Baseline CNN (ImageNet pretrained) |
| 2 | **SpikingCNN-LIF** | Native Spiking CNN with LIF neurons + surrogate gradients |
| 3 | **CNN+ViT Hybrid** | EfficientNet-B3 backbone + 2-layer Transformer head |
| 4 | **ViT-Tiny** | Pure Vision Transformer (patch=16, embed=384, depth=6) |
| 5 | **CNN+PQC** | EfficientNet-B3 + 8-qubit PennyLane Parameterized Quantum Circuit |

## Project Structure

```text
.
├── config.yaml               # ← Central config file (edit epochs, LR, models, etc.)
├── run_pipeline.py            # ← Unified pipeline (runs all models × 5-fold CV)
├── data/                      # (gitignored) Raw BreakHis dataset
├── outputs/                   # (gitignored) Models, plots, CSVs per model
├── src/
│   ├── data/
│   │   └── dataset.py         # BreakHis loader + GroupKFold 5-fold CV splits
│   ├── models/
│   │   ├── efficientnet.py    # EfficientNet-B3 baseline
│   │   ├── spiking/
│   │   │   └── lif_snn.py     # Spiking CNN with LIF neurons
│   │   ├── transformer/
│   │   │   └── hybrid_vit.py  # CNN+ViT Hybrid and ViT-Tiny
│   │   └── quantum/
│   │       └── hybrid_quantum.py  # EfficientNet-B3 + PennyLane PQC
│   ├── utils/
│   │   └── metrics.py         # Metrics, plotting, profiling, CSV logging
│   ├── train.py               # Standalone: EfficientNet-B3 baseline
│   ├── train_spiking.py       # Standalone: Spiking CNN
│   ├── train_transformer.py   # Standalone: Transformer variants
│   └── train_quantum.py       # Standalone: Quantum hybrid
├── notebooks/                 # Jupyter notebooks for exploration
├── docs/                      # Documentation
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Setup
```bash
# Windows
setup.bat

# macOS/Linux
chmod +x setup.sh && ./setup.sh
```
Then activate: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Mac/Linux).

### 2. Dataset
Download [BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) → extract to `data/BreaKHis_v1/`.

### 3. Configuration
Edit `config.yaml` to control:
- **Dataset**: path, task (binary/multi), subset size
- **CV**: number of folds
- **Training**: epochs, patience, gradient clipping
- **Models**: enable/disable each model, set LR, batch size, model-specific hyperparams
- **Output**: directory, what to save

### 4. Run the Full Pipeline
```bash
# All enabled models × 5-fold CV (reads config.yaml)
python run_pipeline.py

# Specific models only
python run_pipeline.py --models efficientnet spiking

# Custom config
python run_pipeline.py --config my_config.yaml

# Quick sanity check: set subset: 200 and epochs: 3 in config.yaml
```

### 5. Run Individual Models (Standalone)
```bash
# Baseline
python src/train.py --task binary --data_dir data/BreaKHis_v1

# Spiking CNN
python src/train_spiking.py --task binary --data_dir data/BreaKHis_v1

# CNN+ViT or ViT-Tiny
python src/train_transformer.py --model hybrid --task binary --data_dir data/BreaKHis_v1
python src/train_transformer.py --model vit --task binary --data_dir data/BreaKHis_v1

# Quantum Hybrid
python src/train_quantum.py --task binary --data_dir data/BreaKHis_v1
```

## Outputs

For each model, the pipeline saves to `outputs/<ModelName>_<task>/`:

| File | Description |
|---|---|
| `best_model_foldN.pth` | Best checkpoint per fold |
| `*_training_curves.png` | Loss & accuracy per epoch |
| `*_confusion_matrix.png` | Test set confusion matrix |
| `*_roc_curve.png` | ROC curve(s) |
| `*_fold_results.csv` | Per-fold metrics |
| `*_epoch_log.csv` | Per-epoch training log |
| `comparison_<task>.csv` | **Cross-model comparison** (mean ± std) |

## Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Data split** | GroupKFold by Patient ID | Prevents data leakage |
| **CV** | 5-fold + 15% held-out test | Robust evaluation |
| **Loss** | CrossEntropyLoss | Unified for binary/multi |
| **Early Stopping** | On Validation AUC | Clinically relevant |
| **Mixed Precision** | AMP (except SNN/Quantum) | Faster training |
| **Config** | YAML | Easy to edit without code changes |

## Reproducibility

All random seeds fixed (default: 42). Set in `config.yaml`.

## GPU Requirements

Tested on NVIDIA RTX 5050. Quantum model runs on CPU (PennyLane requirement).
