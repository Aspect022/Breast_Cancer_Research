# 🐧 Linux Quick Start Guide

**For Ubuntu/Debian-based systems**

---

## 🚀 Quick Setup (5 minutes)

### 1. Clone and Enter Repository
```bash
cd ~/Projects/Cancer/Breast_Cancer_Research
```

### 2. Make Scripts Executable
```bash
chmod +x setup.sh run_full_pipeline.sh
```

### 3. Run Setup (One-time)
```bash
./setup.sh
```

This will:
- Create virtual environment
- Install PyTorch with CUDA support
- Install all requirements

### 4. Download Datasets
```bash
python3 scripts/download_datasets.py
```

### 5. Run Quick Test
```bash
./run_full_pipeline.sh --quick
```

**Expected time:** 2-3 minutes on GPU

---

## 📋 Full Usage

### Run Complete Pipeline
```bash
# Activate environment first
source .venv/bin/activate

# Run all enabled models
./run_full_pipeline.sh

# Or directly
python3 run_pipeline.py
```

### Quick Test Mode
```bash
./run_full_pipeline.sh --quick
```

### Specific Dataset
```bash
./run_full_pipeline.sh --dataset wbcd
```

### Specific Models
```bash
python3 run_pipeline.py --models swin_tiny convnext_tiny dual_branch_fusion
```

---

## 🔧 Troubleshooting

### Python Not Found
```bash
# Install Python 3.9+
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# If driver missing, install:
sudo apt install nvidia-driver-525  # Or latest version

# Reboot after installation
```

### Permission Denied
```bash
# Make scripts executable
chmod +x setup.sh run_full_pipeline.sh
```

### Out of Memory
```bash
# Reduce batch_size in config.yaml
# Or use subset mode
./run_full_pipeline.sh --quick
```

### W&B Connection Error
```bash
# Disable W&B offline mode
export WANDB_MODE=offline

# Or disable in config.yaml:
# output.wandb_enabled: false
```

---

## 📊 Monitor Training

### Weights & Biases Dashboard

When training starts, you'll see:
```
🚀 Initializing Weights & Biases...
✓ W&B initialized: https://wandb.ai/username/breast-cancer-transformers/runs/xyz123
```

**Click the link to monitor from any device!**

### Local Results

After training completes:
```bash
# View comparison table
cat outputs/comparison_binary.csv

# View model-specific results
ls outputs/Swin-Tiny_binary/

# Run analysis
python3 scripts/analyze_results.py --save-summary
```

---

## 🎯 Common Commands

### Environment
```bash
# Activate
source .venv/bin/activate

# Deactivate
deactivate

# Check Python
which python
python --version
```

### Datasets
```bash
# Download all
python3 scripts/download_datasets.py

# Download specific
python3 scripts/download_datasets.py --dataset breakhis
python3 scripts/download_datasets.py --dataset wbcd
python3 scripts/download_datasets.py --dataset seer
```

### Training
```bash
# Quick test
./run_full_pipeline.sh --quick

# Full training
./run_full_pipeline.sh

# Specific models
python3 run_pipeline.py --models efficientnet swin_tiny

# Specific dataset
python3 run_pipeline.py --dataset wbcd
```

### Analysis
```bash
# Generate statistics
python3 scripts/analyze_results.py

# Save summary
python3 scripts/analyze_results.py --save-summary

# Specific metric
python3 scripts/analyze_results.py --metric Test_AUC
```

---

## 📦 Package Management

### Install Additional Packages
```bash
source .venv/bin/activate
pip install package_name
```

### Update Requirements
```bash
pip freeze > requirements.txt
```

### Check Installed Packages
```bash
pip list
```

---

## 💾 Disk Space

### Check Usage
```bash
du -sh outputs/
du -sh data/
```

### Clean Up
```bash
# Remove quick test results
rm -rf outputs_quick_test/

# Remove old checkpoints
find outputs/ -name "*.pth" -mtime +7 -delete

# Clear W&B cache
rm -rf wandb/
```

---

## 🎓 Performance Tips

### Maximize GPU Usage
```bash
# In config.yaml:
gpu:
  use_tf32: true        # Enable TF32 on A100
  benchmark_mode: true  # cudnn.benchmark
  use_flash_attention: true  # If installed
```

### Multi-GPU (If Available)
```bash
# Install apex for multi-GPU
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-build-isolation --config-settings '--build-option=--cpp_ext --cuda_ext' ./

# Enable in config.yaml (future feature)
```

### Reduce Training Time
```bash
# Use subset for testing
subset: 200  # in config.yaml

# Reduce epochs
epochs: 10

# Use fewer folds
cv:
  n_folds: 3  # Instead of 5
```

---

## 📞 Support

### Check System Requirements
```bash
# Python version
python3 --version

# CUDA version
nvcc --version

# GPU info
nvidia-smi

# Available memory
free -h
```

### Common Issues

**Issue:** `CUDA out of memory`
**Solution:** Reduce batch_size or use `--quick` mode

**Issue:** `ModuleNotFoundError: No module named 'timm'`
**Solution:** `source .venv/bin/activate && pip install -r requirements.txt`

**Issue:** `Permission denied`
**Solution:** `chmod +x *.sh`

**Issue:** W&B not logging
**Solution:** Check internet connection or set `WANDB_MODE=offline`

---

## 🎯 Next Steps After Setup

1. **Run Quick Test:**
   ```bash
   ./run_full_pipeline.sh --quick
   ```

2. **View Results:**
   ```bash
   cat outputs_quick_test/comparison_binary.csv
   ```

3. **Check W&B:**
   - Click URL from training output
   - Monitor from phone/tablet

4. **Run Full Training:**
   ```bash
   ./run_full_pipeline.sh
   ```

5. **Analyze Results:**
   ```bash
   python3 scripts/analyze_results.py --save-summary
   ```

---

**Happy Training! 🚀**

*For detailed documentation, see:*
- `README.md` - Project overview
- `docs/WANDB_GUIDE.md` - W&B monitoring
- `IMPLEMENTATION_COMPLETE.md` - Implementation details
