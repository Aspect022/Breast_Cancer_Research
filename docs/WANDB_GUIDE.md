# 📊 Weights & Biases Integration Guide

**Monitor your breast cancer classification experiments from anywhere!**

---

## 🚀 Quick Start

### 1. W&B is Already Configured!

Your API key has been embedded in `wandb_config.yaml`. The pipeline will automatically:
- ✅ Log training/validation metrics in real-time
- ✅ Track model weights and biases
- ✅ Monitor gradient distributions
- ✅ Save confusion matrices and ROC curves
- ✅ Log model checkpoints
- ✅ Create interactive dashboards

### 2. Run Training with W&B

```bash
# Just run the pipeline - W&B is enabled by default!
python run_pipeline.py

# Or use the automated script
run_full_pipeline.bat
```

### 3. View Your Dashboard

After training starts, you'll see:
```
🚀 Initializing Weights & Biases...
✓ W&B initialized: https://wandb.ai/your-username/breast-cancer-transformers/runs/abc123
```

**Click the link to monitor your training in real-time from any device!**

---

## 📈 What Gets Logged to W&B

### Training Metrics (Every Epoch)
- `train/loss` - Training loss
- `train/accuracy` - Training accuracy
- `val/loss` - Validation loss
- `val/accuracy` - Validation accuracy
- `val/auc` - Validation AUC (Area Under Curve)
- `optimizer/lr` - Learning rate

### Model Parameters (Every 5 Epochs)
- **Weights**: Histograms of weight distributions for each layer
- **Biases**: Bias value distributions
- **Gradients**: Gradient histograms for backpropagation analysis

### Visualizations
- Confusion matrices
- ROC curves
- Training curves
- Sample predictions with images

### Model Information
- Total parameters
- Trainable parameters
- Model architecture diagram

---

## 🎛️ W&B Dashboard Features

### Real-Time Monitoring
Watch your training progress live from your phone, tablet, or another computer!

**Access your dashboard at:**
```
https://wandb.ai/your-username/breast-cancer-transformers
```

### Available Panels

1. **Metrics Panel**: Line charts for loss, accuracy, AUC
2. **Weights & Biases**: Histograms showing parameter distributions
3. **Gradients**: Monitor gradient flow and detect vanishing/exploding gradients
4. **Media**: Confusion matrices, ROC curves, sample images
5. **System**: GPU utilization, memory usage, temperature

### Comparison View
Compare multiple runs side-by-side:
- Different models
- Different hyperparameters
- Different datasets

---

## ⚙️ Configuration

### Enable/Disable W&B

Edit `config.yaml`:
```yaml
output:
  wandb_enabled: true  # Set to false to disable W&B
  wandb_project: "breast-cancer-transformers"
```

### Customize Logging

Edit `wandb_config.yaml`:
```yaml
wandb:
  api_key: "your-api-key"  # Already configured
  project: "breast-cancer-transformers"
  entity: "your-username"  # Optional: set your W&B username
  
  # Logging options
  log_model: true          # Log model checkpoints
  log_gradients: true      # Log gradient histograms
  log_weights: true        # Log weight distributions
  log_images: true         # Log sample images
  log_gradcam: true        # Log Grad-CAM heatmaps
  
  # Performance
  log_interval: 10         # Log every N batches
  val_log_interval: 1      # Log every validation epoch
```

---

## 🔬 Advanced Features

### 1. Compare Models

After running multiple experiments, compare them in the W&B dashboard:
1. Go to your project page
2. Select runs to compare (checkboxes)
3. Click "Compare" button
4. View side-by-side metrics and plots

### 2. Hyperparameter Sweeps (Optional)

Create a sweep to automatically find the best hyperparameters:

```bash
# Create sweep configuration
wandb sweep sweep_config.yaml

# Run sweep agent
wandb agent sweep_id
```

### 3. Export Results

Download your experiment data:

```python
import wandb

api = wandb.Api()
runs = api.runs("your-username/breast-cancer-transformers")

for run in runs:
    # Download metrics
    metrics = run.history()
    metrics.to_csv(f"{run.name}_metrics.csv")
    
    # Download model
    run.file("model.pth").download()
```

---

## 📱 Mobile Monitoring

Access your W&B dashboard from anywhere:

1. **Install W&B app** (iOS/Android)
2. **Login** with your W&B account
3. **View runs** in real-time
4. **Get notifications** when training completes

**QR Code**: Generate from your W&B project page for quick mobile access!

---

## 🔐 API Key Management

### Your API Key

Already configured in `wandb_config.yaml`:
```
WANDB_API_KEY="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"
```

### Security Notes

- ✅ API key is stored in `wandb_config.yaml` (gitignored)
- ✅ Key has project-scoped permissions
- ⚠️ Do not share this key publicly
- ⚠️ Regenerate if compromised from https://wandb.ai/authorize

### Update API Key

```bash
# Login with new key
wandb login

# Or update wandb_config.yaml manually
```

---

## 📊 Example Dashboard Views

### Training Progress View
```
Model: Swin-Tiny | Fold 3/5
────────────────────────────────────────
Epoch 15/50 | Loss: 0.234 | Acc: 0.892 | AUC: 0.943
GPU: 87% | Memory: 11.2/16GB | Temp: 72°C
```

### Model Comparison View
```
Model              | Acc    | F1     | AUC    | Time
───────────────────┼────────┼────────┼────────┼──────
EfficientNet-B3    | 0.850  | 0.846  | 0.923  | 10m
Swin-Tiny          | 0.863  | 0.860  | 0.939  | 30m
ConvNeXt-Tiny      | 0.858  | 0.854  | 0.934  | 25m
DualBranch-Fusion  | 0.882  | 0.879  | 0.951  | 45m  ← Best!
```

---

## 🐛 Troubleshooting

### W&B Not Logging

**Problem**: No data appearing in dashboard

**Solutions**:
1. Check `wandb_enabled: true` in `config.yaml`
2. Verify API key in `wandb_config.yaml`
3. Check internet connection
4. Run `wandb login` to re-authenticate

### API Key Invalid

**Problem**: "Invalid API key" error

**Solutions**:
1. Get new key from https://wandb.ai/authorize
2. Update `wandb_config.yaml`
3. Run `wandb login --relogin`

### Offline Mode

**Problem**: No internet connection during training

**Solution**: Enable offline mode
```bash
export WANDB_MODE=offline
python run_pipeline.py

# Sync later when online
wandb sync --sync-all
```

---

## 📚 Additional Resources

- **W&B Documentation**: https://docs.wandb.ai
- **W&B API Reference**: https://docs.wandb.ai/ref
- **Community Forum**: https://community.wandb.ai
- **Example Projects**: https://wandb.ai/examples

---

## 🎯 Best Practices

1. **Enable W&B for all serious training runs**
2. **Use descriptive run names**: `swin_tiny_lr2e-5_bs32`
3. **Tag related runs**: `transformer_comparison`, `quantum_ablation`
4. **Save important checkpoints**: Best model per fold
5. **Share dashboards** with collaborators
6. **Export results** for papers/reports

---

## 📞 Support

For W&B-specific issues:
- Check W&B status: https://status.wandb.ai
- Contact W&B support: support@wandb.ai
- Community help: https://community.wandb.ai

For project-specific issues:
- Check `issues.md`
- Review `IMPLEMENTATION_COMPLETE.md`
- Contact project maintainer

---

**Happy Training! 🚀**

*Monitor your breast cancer classification experiments from anywhere in the world!*
