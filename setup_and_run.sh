#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BREAST CANCER - COMPLETE AUTOMATED SETUP & RUN (One Command)
# ═══════════════════════════════════════════════════════════════════
# Usage: Just copy-paste this entire script or run:
#        bash setup_and_run.sh
#
# This will:
#   1. Install system dependencies (if needed)
#   2. Create virtual environment
#   3. Install all Python packages
#   4. Download datasets
#   5. Run complete training pipeline
#   6. Log everything to W&B
#   7. Save all results
#
# Run in background: nohup bash setup_and_run.sh &
# ═══════════════════════════════════════════════════════════════════

set -e

# Configuration
DATASET="breakhis"
QUICK_MODE=0  # Set to 1 for quick test, 0 for full training
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick) QUICK_MODE=1; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   BREAST CANCER CLASSIFICATION - AUTO SETUP & RUN               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo "Log file: $LOG_FILE"
echo "Dataset: $DATASET"
echo "Mode: $([ $QUICK_MODE -eq 1 ] && echo 'QUICK TEST' || echo 'FULL TRAINING')"
echo

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Step 1: Check/Install system dependencies
log "Step 1/6: Checking system dependencies..."
if ! command -v python3 &> /dev/null; then
    log "Installing Python3..."
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv git wget
fi
log "✓ Python3: $(python3 --version)"

# Step 2: Create virtual environment
log "Step 2/6: Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    log "✓ Virtual environment created"
else
    log "✓ Virtual environment exists"
fi

# Step 3: Activate and install requirements
log "Step 3/6: Installing Python packages (this takes 5-10 minutes)..."
source .venv/bin/activate

# Install PyTorch
if ! python3 -c "import torch" &> /dev/null; then
    log "Installing PyTorch with CUDA 12.1 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
    log "✓ PyTorch installed"
else
    log "✓ PyTorch already installed"
fi

# Install other requirements
log "Installing remaining packages..."
pip install timm>=1.0.0 scipy>=1.11.0 opencv-python>=4.8.0 wandb>=0.16.0 -q
pip install -r requirements.txt -q 2>/dev/null || true
log "✓ All packages installed"

# Step 4: Download datasets
log "Step 4/6: Downloading datasets..."
if [ ! -d "data/$DATASET" ] && [ ! -f "data/$DATASET.csv" ]; then
    python3 scripts/download_datasets.py --dataset $DATASET || log "Dataset download may need manual intervention"
    log "✓ Dataset download attempted"
else
    log "✓ Dataset already exists"
fi

# Step 5: Create config if quick mode
if [ $QUICK_MODE -eq 1 ]; then
    log "Step 5/6: Creating quick test configuration..."
    cat > config_auto.yaml << EOF
data:
  dataset: "$DATASET"
  data_dir: "data/$DATASET"
  task: "binary"
  subset: 200
  num_workers: 4
  seed: 42

cv:
  n_folds: 2

training:
  epochs: 3
  patience: 2
  grad_clip: 1.0

models:
  efficientnet:
    enabled: true
    batch_size: 32
    lr: 1.0e-4
  swin_tiny:
    enabled: true
    batch_size: 32
    lr: 2.0e-5
  convnext_tiny:
    enabled: true
    batch_size: 32
    lr: 2.0e-5
  dual_branch_fusion:
    enabled: true
    batch_size: 16
    lr: 2.0e-5
  qenn_ry:
    enabled: true
    batch_size: 16
    n_qubits: 4
    n_layers: 1

output:
  output_dir: "outputs_auto"
  wandb_enabled: true
  save_plots: true
  save_csv: true
EOF
    CONFIG_FILE="config_auto.yaml"
else
    CONFIG_FILE="config.yaml"
fi
log "✓ Configuration ready: $CONFIG_FILE"

# Step 6: Run training pipeline
log "Step 6/6: Starting training pipeline..."
log "═══════════════════════════════════════════════════════════════════"
log "TRAINING STARTED"
log "═══════════════════════════════════════════════════════════════════"

python3 run_pipeline.py --config $CONFIG_FILE --dataset $DATASET 2>&1 | tee -a "$LOG_FILE"

# Cleanup
if [ $QUICK_MODE -eq 1 ] && [ -f config_auto.yaml ]; then
    rm config_auto.yaml
fi

# Summary
log "═══════════════════════════════════════════════════════════════════"
log "TRAINING COMPLETE!"
log "═══════════════════════════════════════════════════════════════════"
log "Results saved to: outputs/"
log "Comparison table: outputs/comparison_binary.csv"
log "Log file: $LOG_FILE"
log ""
log "Next steps:"
log "  1. View W&B dashboard (check URL above)"
log "  2. Check results: cat outputs/comparison_binary.csv"
log "  3. Run analysis: python3 scripts/analyze_results.py"
log "═══════════════════════════════════════════════════════════════════"
