# 🚀 NoHup Commands: Run Fusion Models in Background

**NoHup** = "No Hangup" - keeps training running even if you close terminal or SSH disconnects!

---

## ⚡ Quick Command (All 5 Models)

```bash
nohup python run_pipeline.py --models triple_branch_fusion cb_qccf multi_scale_quantum_fusion ensemble_distillation quantum_enhanced_fusion > training_output.log 2>&1 &
```

**That's it!** Training will continue in background even if you close terminal.

---

## 📋 What This Does

| Part | Meaning |
|------|---------|
| `nohup` | Keep running after terminal closes |
| `python run_pipeline.py` | Run the training pipeline |
| `--models ...` | All 5 fusion models |
| `> training_output.log` | Save all output to log file |
| `2>&1` | Also capture error messages |
| `&` | Run in background |

---

## 🔍 Check Training Progress

```bash
# View live output (like tail -f)
tail -f training_output.log

# Or view last 100 lines
tail -n 100 training_output.log

# Check if process is running
ps aux | grep run_pipeline.py

# Check GPU usage
nvidia-smi
```

---

## 🛑 Stop Training (If Needed)

```bash
# Find process ID
ps aux | grep run_pipeline.py

# Kill process (replace PID with actual number)
kill -9 PID

# Or kill all python processes (careful!)
pkill -9 python
```

---

## 📊 Full NoHup Commands (Variants)

### 1. All 5 Models (Recommended)

```bash
nohup python run_pipeline.py --models triple_branch_fusion cb_qccf multi_scale_quantum_fusion ensemble_distillation quantum_enhanced_fusion > training_all_5.log 2>&1 &
```

**Estimated Time:** 10-12 hours (5-fold CV)

---

### 2. Single Model (Fastest)

```bash
# Just Ensemble Distillation (BEST model)
nohup python run_pipeline.py --models ensemble_distillation > training_ensemble.log 2>&1 &
```

**Estimated Time:** 3 hours

---

### 3. Quick Test First

```bash
# Test with 200 samples, 5 epochs
nohup python run_pipeline.py --models triple_branch_fusion cb_qccf quantum_enhanced_fusion --subset 200 --epochs 5 > training_test.log 2>&1 &
```

**Estimated Time:** 30 minutes

---

### 4. With Custom Config

```bash
nohup python run_pipeline.py --models triple_branch_fusion cb_qccf multi_scale_quantum_fusion ensemble_distillation quantum_enhanced_fusion --config config_full.yaml > training_config.log 2>&1 &
```

---

### 5. On Specific GPU

```bash
# Run on GPU 0
CUDA_VISIBLE_DEVICES=0 nohup python run_pipeline.py --models triple_branch_fusion cb_qccf multi_scale_quantum_fusion ensemble_distillation quantum_enhanced_fusion > training_gpu0.log 2>&1 &
```

---

### 6. With W&B Disabled (Faster)

```bash
nohup python run_pipeline.py --models triple_branch_fusion cb_qccf multi_scale_quantum_fusion ensemble_distillation quantum_enhanced_fusion --no-wandb > training_no_wandb.log 2>&1 &
```

---

## 📁 Output Files Created

After running, you'll have:

```
training_output.log          # Main log file (all output)
outputs/                     # Model checkpoints, plots, CSVs
outputs/comparison_binary.csv # Final results table
```

---

## 🎯 Monitoring Script

Create `monitor.sh`:

```bash
#!/bin/bash
echo "=== Training Progress ==="
echo ""
echo "📊 Last 20 lines of output:"
tail -n 20 training_output.log
echo ""
echo "🔍 Process status:"
ps aux | grep run_pipeline.py | grep -v grep
echo ""
echo "💻 GPU Usage:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

Make executable and run:
```bash
chmod +x monitor.sh
./monitor.sh
```

---

## ⚠️ Important Notes

1. **Log file grows large** - Check disk space periodically
2. **Don't run multiple trainings** to same output dir (will overwrite)
3. **SSH disconnect is OK** - nohup keeps it running!
4. **Server restart will kill** - Use `screen` or `tmux` for multi-day training

---

## 🔄 Alternative: Screen/Tmux (Even Better)

For multi-day training, use `screen`:

```bash
# Start new screen session
screen -S training

# Run your command
python run_pipeline.py --models triple_branch_fusion cb_qccf multi_scale_quantum_fusion ensemble_distillation quantum_enhanced_fusion

# Detach (Ctrl+A, then D)
# Training keeps running!

# Reattach later
screen -r training

# List sessions
screen -ls
```

---

## 📊 Expected Log Output

```
╔══════════════════════════════════════════════════════════╗
║  Starting Training Pipeline                              ║
║  Models: triple_branch_fusion, cb_qccf, ...              ║
║  Dataset: BreakHis v1                                    ║
╠══════════════════════════════════════════════════════════╣
║  [Model 1/5] triple_branch_fusion                        ║
║  Fold 1/5: Training...                                   ║
║  Epoch 1/50: loss=0.6923, acc=0.5234                     ║
║  Epoch 2/50: loss=0.6512, acc=0.6123                     ║
║  ...                                                     ║
║  Fold 1/5 Complete: acc=0.8733, auc=0.8921               ║
║  Saving checkpoint...                                    ║
╚══════════════════════════════════════════════════════════╝
```

---

**Ready to run!** Copy the nohup command and training will continue even if you disconnect! 🚀
