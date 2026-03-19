# 🚀 One-Command Setup & Run

## **Copy-Paste This Single Command:**

```bash
cd ~/Projects/Cancer/Breast_Cancer_Research && chmod +x setup_and_run.sh && nohup bash setup_and_run.sh > training.log 2>&1 &
```

**That's it!** The script will:
1. ✅ Install all dependencies
2. ✅ Create virtual environment
3. ✅ Download datasets
4. ✅ Run complete training pipeline
5. ✅ Log everything to Weights & Biases
6. ✅ Save all results

---

## **Monitor Progress:**

### **1. Check Training Log (Real-time)**
```bash
tail -f training.log
```

### **2. View Weights & Biases Dashboard**
- Training log will show W&B URL like:
  ```
  ✓ W&B initialized: https://wandb.ai/username/breast-cancer-transformers/runs/xyz123
  ```
- **Click the link to monitor from anywhere!**

### **3. Check if Running**
```bash
ps aux | grep run_pipeline.py
```

### **4. Check GPU Usage**
```bash
watch -n 1 nvidia-smi
```

---

## **Quick Test Mode (2-3 minutes):**

```bash
cd ~/Projects/Cancer/Breast_Cancer_Research && chmod +x setup_and_run.sh && nohup bash setup_and_run.sh --quick > training.log 2>&1 &
```

---

## **After Training Completes:**

### **View Results:**
```bash
# Comparison table
cat outputs/comparison_binary.csv

# Model-specific results
ls -lh outputs/*/

# Run analysis
python3 scripts/analyze_results.py --save-summary
```

### **Check W&B:**
- Open URL from training.log
- View metrics, weights, visualizations
- Compare models
- Export results

---

## **If You Need to Stop:**

```bash
# Find process
ps aux | grep run_pipeline.py

# Kill process (replace PID with actual number)
kill PID
```

---

## **Full Command Breakdown:**

```bash
cd ~/Projects/Cancer/Breast_Cancer_Research  # Navigate to project
&& chmod +x setup_and_run.sh                 # Make script executable
&& nohup bash setup_and_run.sh               # Run script (continues after logout)
> training.log 2>&1                          # Log all output
&                                            # Run in background
```

---

## **What Gets Logged:**

### **training.log file contains:**
- All setup steps
- Package installation progress
- Dataset download status
- Training progress (epoch-by-epoch)
- Validation metrics
- Final results
- W&B dashboard URL

### **W&B Dashboard shows:**
- Real-time training curves
- Weight/bias histograms
- Gradient distributions
- Confusion matrices
- ROC curves
- GPU utilization

---

## **Expected Timeline:**

| Mode | Time | Output |
|------|------|--------|
| **Quick Test** | 2-3 min | `outputs_auto/` |
| **Full Training** | 4-8 hours | `outputs/` |

---

## **Troubleshooting:**

### **Script not found:**
```bash
ls -la setup_and_run.sh
# If missing, pull from git:
git pull origin main
```

### **Permission denied:**
```bash
chmod +x setup_and_run.sh
```

### **Python not found:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

### **Out of disk space:**
```bash
df -h
# Clean up if needed
rm -rf outputs_quick_test/
```

---

## **Server Reboot Safe:**

If server reboots during training:
```bash
# Re-activate environment
source .venv/bin/activate

# Resume from last checkpoint (manual)
python3 run_pipeline.py --config config.yaml
```

**Note:** Training will restart from beginning. Checkpoints are saved per fold.

---

## **Multiple GPU Servers:**

For multi-GPU setups, the script automatically uses first available GPU.

---

**Just copy-paste the command at the top and monitor from W&B!** 🚀
