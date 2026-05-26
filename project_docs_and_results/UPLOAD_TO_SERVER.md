# Files to Upload to Server

**Date:** March 21, 2026
**Server:** user04@ubuntu-Standard-PC-Q35-ICH9-2009:~/Projects/Cancer/Breast_Cancer_Research/

---

## Critical Files (Must Upload)

These files have been updated with CBIS-DDSM support:

### 1. src/data/dataset.py
**Status:** ✅ Updated with CBIS-DDSM loader and k-fold support
**Size:** ~1,135 lines
**Changes:**
- Added CBISDDSMDataset class
- Added download_cbis_ddsm() function
- Added get_cbis_ddsm_dataloaders() function
- Added get_multidataset_kfold_splits() function
- Updated SUPPORTED_DATASETS to include 'cbis_ddsm'

**Upload command:**
```bash
scp src/data/dataset.py user04@server:~/Projects/Cancer/Breast_Cancer_Research/src/data/
```

### 2. run_pipeline.py
**Status:** ✅ Updated to use get_multidataset_kfold_splits
**Changes:**
- Line 32: Added import for get_multidataset_kfold_splits
- Line 611-620: Changed to use get_multidataset_kfold_splits instead of get_kfold_splits
- Line 768: Updated help text to include 'cbis_ddsm'
- Line 783-785: Added cbis_ddsm data_dir handling

**Upload command:**
```bash
scp run_pipeline.py user04@server:~/Projects/Cancer/Breast_Cancer_Research/
```

### 3. run_cbis_ddsm.sh
**Status:** ✅ Fixed to use python3 and detect kaggle.json
**Changes:**
- Changed `python` to `python3`
- Changed `pip` to `pip3`
- Added kaggle.json file detection
- Added PYTHON_CMD variable

**Upload command:**
```bash
scp run_cbis_ddsm.sh user04@server:~/Projects/Cancer/Breast_Cancer_Research/
```

### 4. config.yaml
**Status:** ✅ Updated with CBIS-DDSM configuration section
**Changes:**
- Added cbis_ddsm data_dir configuration
- Added auto_download and kaggle credentials options

**Upload command:**
```bash
scp config.yaml user04@server:~/Projects/Cancer/Breast_Cancer_Research/
```

---

## Optional Files (Documentation)

These are new documentation files:

```bash
scp MASTER_RESEARCH_DOCUMENTATION.md user04@server:~/Projects/Cancer/Breast_Cancer_Research/
scp QUANTUM_ARCHITECTURES.md user04@server:~/Projects/Cancer/Breast_Cancer_Research/
scp PROPOSED_ARCHITECTURES.md user04@server:~/Projects/Cancer/Breast_Cancer_Research/
scp CBIS_DDSM_SETUP.md user04@server:~/Projects/Cancer/Breast_Cancer_Research/
scp NEXT_STEPS.md user04@server:~/Projects/Cancer/Breast_Cancer_Research/
```

---

## Quick Upload Script

Run this from your local machine (Windows):

```powershell
# PowerShell script to upload all files
$server = "user04@your-server-ip"
$dest = "~/Projects/Cancer/Breast_Cancer_Research/"

# Critical files
scp src/data/dataset.py $server:$dest/src/data/
scp run_pipeline.py $server:$dest/
scp run_cbis_ddsm.sh $server:$dest/
scp config.yaml $server:$dest/

# Documentation (optional)
scp *.md $server:$dest/

Write-Host "Upload complete!"
```

Or use WinSCP / FileZilla for GUI upload.

---

## After Upload: Verify and Run

SSH into server and run:

```bash
# Navigate to project
cd ~/Projects/Cancer/Breast_Cancer_Research

# Verify files are updated
grep -n "get_multidataset_kfold_splits" src/data/dataset.py
# Should return line numbers

grep -n "get_multidataset_kfold_splits" run_pipeline.py
# Should return line 32 and 611

# Restart the pipeline
pkill -f run_cbis_ddsm.sh
nohup ./run_cbis_ddsm.sh --quick > quick_test_2.log 2>&1 &
tail -f quick_test_2.log
```

---

## Expected Output After Fix

```
════════════════════════════════════════════════════════════════
🚀 Running Training Pipeline
════════════════════════════════════════════════════════════════

Running: python3 run_pipeline.py --config config_cbis_ddsm_temp.yaml

CBIS-DDSM: Found 200 valid images
label
1    100
0    100
Name: count, dtype: int64
Split: Train=140, Val=30, Test=30

── Fold 1/5 ──
Model: Swin-Tiny
Training...
```

---

## Troubleshooting

If you still get import errors:

```bash
# On server, check Python path
python3 -c "import sys; print('\n'.join(sys.path))"

# Check if src is in path
python3 -c "from src.data.dataset import get_multidataset_kfold_splits; print('OK')"

# If fails, add src to PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Or run with explicit path
PYTHONPATH=./src python3 run_pipeline.py --config config_cbis_ddsm_temp.yaml
```

---

## Alternative: Git Pull (If Using Git)

If you have a git repository:

```bash
# On server
cd ~/Projects/Cancer/Breast_Cancer_Research
git pull origin main
```

This will automatically get all updated files.

---

**Upload the 4 critical files and the pipeline should work!** 🚀
