# CBIS-DDSM Generalization Validation Setup

**Date:** March 21, 2026  
**Purpose:** Validate model generalization across imaging modalities (Histopathology → Mammography)

---

## Overview

This document explains how to run your breast cancer classification models on the **CBIS-DDSM** mammography dataset to validate generalization beyond the BreakHis histopathology dataset.

### Why CBIS-DDSM?

| Aspect | BreakHis | CBIS-DDSM |
|--------|----------|-----------|
| **Modality** | Histopathology (microscope slides) | Mammography (X-ray) |
| **Image Type** | RGB, 400× magnification | Grayscale/RGB, ROI cropped |
| **Samples** | ~8,000 images | ~10,000 images |
| **Classes** | Benign vs Malignant | Benign vs Malignant |
| **Challenge** | Cellular morphology | Tissue density, calcifications |

**Key Question:** Do models trained on histopathology generalize to mammography?

**Expected Performance:**
- BreakHis accuracy: 85-87%
- CBIS-DDSM accuracy: 75-82% (5-10% drop expected)
- **Quantum models may generalize better** (hypothesis)

---

## Quick Start

### Prerequisites

1. **Kaggle Account** (free): https://www.kaggle.com/account
2. **Kaggle API Key**: Download from account settings
3. **Kaggle CLI**: `pip install kaggle`

### Setup Environment Variables

**Linux/Mac:**
```bash
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key
```

**Windows (PowerShell):**
```powershell
$env:KAGGLE_USERNAME="your_username"
$env:KAGGLE_KEY="your_api_key"
```

**Windows (CMD):**
```cmd
set KAGGLE_USERNAME=your_username
set KAGGLE_KEY=your_api_key
```

---

## Running the Pipeline

### Option 1: Full Pipeline (Recommended)

**Linux/Mac:**
```bash
chmod +x run_cbis_ddsm.sh
./run_cbis_ddsm.sh
```

**Windows:**
```cmd
run_cbis_ddsm.bat
```

This will:
1. Auto-download CBIS-DDSM from Kaggle (~10-30 minutes)
2. Preprocess and split data (70% train, 15% val, 15% test)
3. Run all enabled models from config.yaml with 5-fold CV
4. Generate comparison report with BreakHis results

### Option 2: Quick Test (200 samples)

**Linux/Mac:**
```bash
./run_cbis_ddsm.sh --quick
```

**Windows:**
```cmd
run_cbis_ddsm.bat --quick
```

### Option 3: Specific Models Only

```bash
# Run only Swin and QENN models
./run_cbis_ddsm.sh --models swin_tiny qenn_ry

# Run all quantum models
./run_cbis_ddsm.sh --models qenn_ry qenn_ry_rz qenn_u3 qenn_rx_ry_rz
```

### Option 4: Using run_pipeline.py directly

```bash
python run_pipeline.py --dataset cbis_ddsm --config config.yaml
```

---

## Configuration

### config.yaml Settings

```yaml
data:
  dataset: "cbis_ddsm"  # Change from "breakhis"
  data_dir: "data/CBIS-DDSM"
  task: "binary"
  subset: null  # Set to 200 for quick tests
  num_workers: 8
  
  cbis_ddsm:
    data_dir: "data/CBIS-DDSM"
    auto_download: true
    kaggle_username: null  # Or set here
    kaggle_key: null  # Or set here
```

### Model Configuration

All models enabled in `config.yaml` will run on CBIS-DDSM. To disable models:

```yaml
models:
  efficientnet:
    enabled: true
  swin_tiny:
    enabled: true
  qenn_ry:
    enabled: false  # Disable quantum if needed
```

---

## Dataset Structure

After download, CBIS-DDSM will be organized as:

```
data/CBIS-DDSM/
├── images/              # Mammography images (PNG/JPG)
├── metadata.csv         # Image labels and metadata
├── train.csv           # Training split (70%)
├── val.csv             # Validation split (15%)
├── test.csv            # Test split (15%)
└── splits.json         # Split statistics
```

### Metadata Format

The loader expects `metadata.csv` with columns:
- `image_path`: Path to image file
- `label`: Benign (0) or Malignant (1)

Alternative column names are auto-detected:
- `path`, `filename` → `image_path`
- `class`, `diagnosis`, `type` → `label`

---

## Expected Results

### Performance Comparison Template

After running, you'll get:

```
outputs/
├── comparison_cbis_ddsm_binary.csv      # CBIS-DDSM results
├── comparison_binary.csv                # BreakHis results
└── cbis_ddsm_generalization_comparison.csv  # Side-by-side comparison
```

### Example Results (Hypothetical)

| Model | BreakHis Acc | CBIS-DDSM Acc | Drop | BreakHis AUC | CBIS-DDSM AUC |
|-------|-------------|---------------|------|--------------|---------------|
| Swin-Small | 85.67% | 78.2% | -7.5% | 88.52% | 82.1% |
| QENN-RY | 84.17% | 79.5% | -4.7% | 88.40% | 83.8% |
| EfficientNet-B3 | 83.37% | 76.8% | -6.6% | 89.97% | 81.2% |

**Key Metric:** Smaller drop = better generalization

---

## Analysis Scripts

### Compare Results

```bash
python scripts/analyze_results.py --save-summary
```

### Generate Report

```python
import pandas as pd

# Load results
breakhis = pd.read_csv('outputs/comparison_binary.csv')
cbis_ddsm = pd.read_csv('outputs/comparison_cbis_ddsm_binary.csv')

# Merge for comparison
comparison = pd.merge(
    breakhis[['Model', 'accuracy', 'auc_roc']],
    cbis_ddsm[['Model', 'accuracy', 'auc_roc']],
    on='Model',
    suffixes=('_breakhis', '_cbis_ddsm')
)

# Calculate drop
comparison['accuracy_drop'] = comparison['accuracy_breakhis'] - comparison['accuracy_cbis_ddsm']
print(comparison.sort_values('accuracy_drop'))
```

---

## Troubleshooting

### Issue: Kaggle Download Fails

**Error:** `403 Forbidden` or `Could not find user`

**Solution:**
```bash
# Check credentials
echo $KAGGLE_USERNAME
echo $KAGGLE_KEY

# Re-download kaggle.json
kaggle competitions download -c <any-competition>  # Test download
```

### Issue: Out of Memory

**Error:** CUDA out of memory

**Solution:**
```yaml
# Reduce batch size in config.yaml
models:
  swin_tiny:
    batch_size: 16  # Reduce from 32
  qenn_ry:
    batch_size: 8   # Reduce from 16
```

### Issue: No Images Found

**Error:** `No images found in data/CBIS-DDSM`

**Solution:**
```bash
# Check if images downloaded correctly
ls -la data/CBIS-DDSM/images/

# If empty, re-download
rm -rf data/CBIS-DDSM
./run_cbis_ddsm.sh
```

### Issue: Wrong Label Format

**Error:** `KeyError: 'label'`

**Solution:** Check metadata.csv format:
```python
import pandas as pd
df = pd.read_csv('data/CBIS-DDSM/metadata.csv')
print(df.columns)
print(df.head())
```

---

## Server Deployment (Linux)

### For Remote Servers (A100 GPUs)

1. **SSH into server:**
   ```bash
   ssh user@server.edu
   ```

2. **Clone repository:**
   ```bash
   git clone <your-repo>
   cd Breast_cancer_Minor_Project
   ```

3. **Setup environment:**
   ```bash
   conda create -n breast-cancer python=3.9
   conda activate breast-cancer
   pip install -r requirements.txt
   pip install kaggle
   ```

4. **Set credentials:**
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

5. **Run pipeline:**
   ```bash
   chmod +x run_cbis_ddsm.sh
   ./run_cbis_ddsm.sh
   ```

6. **Monitor with W&B:**
   - Login: `wandb login`
   - View dashboard: https://wandb.ai/your-username/breast-cancer-transformers

### SLURM Job Script (for HPC clusters)

```bash
#!/bin/bash
#SBATCH --job-name=cbis-ddsm
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/cbis_ddsm_%j.out

module load cuda/11.8
module load python/3.9

source activate breast-cancer

cd $SLURM_SUBMIT_DIR
./run_cbis_ddsm.sh
```

Submit with: `sbatch run_cbis_ddsm.slurm`

---

## Next Steps After Validation

1. **Compare Results:**
   - Which models generalize best?
   - Do quantum models have smaller performance drop?

2. **Implement New Architectures:**
   - CB-QCCF (Class-Balanced Quantum-Classical Fusion)
   - Multi-Scale Quantum Fusion
   - Quantum Deformable Attention

3. **Write Paper:**
   - Include both BreakHis and CBIS-DDSM results
   - Highlight generalization performance
   - Emphasize quantum novelty

4. **MICCAI 2026 Submission:**
   - Deadline: Typically April 2026
   - Format: 8-page LNCS format
   - Code: Make public on GitHub

---

## Files Created

| File | Purpose |
|------|---------|
| `src/data/dataset.py` | CBIS-DDSM loader with auto-download |
| `config.yaml` | Updated with CBIS-DDSM config |
| `run_cbis_ddsm.sh` | Linux/Mac pipeline script |
| `run_cbis_ddsm.bat` | Windows pipeline script |
| `QUANTUM_ARCHITECTURES.md` | Proposed quantum architectures |
| `PROPOSED_ARCHITECTURES.md` | All 5 proposed architectures |
| `CBIS_DDSM_SETUP.md` | This document |

---

## References

1. **CBIS-DDSM Paper:** Clark et al., "Curated Breast Imaging Subset of DDSM", SPIE Medical Imaging 2013
2. **Kaggle Dataset:** https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
3. **BreakHis Paper:** Spanhol et al., "A dataset for breast cancer histopathological image classification", IEEE TBME 2016
4. **Quantum ML:** Schuld et al., "Quantum machine learning in feature Hilbert spaces", PRL 2019

---

## Support

For issues or questions:
1. Check `issues.md` for known problems
2. Review `IMPLEMENTATION_COMPLETE.md` for implementation status
3. Check W&B dashboard for training logs
4. Inspect `outputs/` directory for results

---

**Good luck with your experiments! 🚀**
