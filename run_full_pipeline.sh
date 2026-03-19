#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BREAST CANCER CLASSIFICATION - FULLY AUTOMATED PIPELINE (Linux)
# ═══════════════════════════════════════════════════════════════════
# This script will:
#   1. Create/activate virtual environment
#   2. Install all requirements (GPU optimized)
#   3. Download datasets (BreakHis, WBCD, SEER)
#   4. Run complete pipeline with all enabled models
#   5. Generate results summary and statistical analysis
#
# Usage: ./run_full_pipeline.sh [--quick] [--dataset DATASET] [--folds N]
#   --quick    : Run with subset=200 and epochs=3 for testing
#   --dataset  : Specify dataset (breakhis/wbcd/seer, default: breakhis)
#   --folds N  : Number of CV folds (1 for quick, 5 for final, default: 1)
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
QUICK_MODE=0
DATASET="breakhis"
FOLDS=5  # Default to 5 folds for research-grade results

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=1
            shift
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --folds)
            FOLDS="$2"
            # Ensure minimum 2 folds
            if [ "$FOLDS" -lt 2 ]; then
                echo -e "${YELLOW}Warning: Minimum 2 folds required. Setting to 2.${NC}"
                FOLDS=2
            fi
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Usage: ./run_full_pipeline.sh [--quick] [--dataset DATASET] [--folds N]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   BREAST CANCER CLASSIFICATION - AUTOMATED PIPELINE             ║${NC}"
echo -e "${BLUE}║   Transformers + Quantum-Enhanced Neural Networks               ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo

# Configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Dataset: $DATASET"
echo "  Folds: $FOLDS (2=quick comparison, 5=final results)"
if [ $QUICK_MODE -eq 1 ]; then
    echo -e "  Mode: ${GREEN}QUICK TEST${NC} (subset=200, epochs=3)"
else
    echo -e "  Mode: ${GREEN}FULL TRAINING${NC}"
fi
echo

# Step 1: Check Python
echo -e "${BLUE}[1/7]${NC} Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: Python3 not found. Please install Python 3.9+${NC}"
    echo "  sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓${NC} $PYTHON_VERSION"
echo

# Step 2: Create virtual environment
echo -e "${BLUE}[2/7]${NC} Setting up virtual environment..."
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi
echo

# Step 3: Activate venv
echo -e "${BLUE}[3/7]${NC} Activating virtual environment..."
source .venv/bin/activate
echo -e "${GREEN}✓${NC} Activated"
echo

# Step 4: Install requirements
echo -e "${BLUE}[4/7]${NC} Installing requirements (GPU optimized)..."
echo "This may take 5-10 minutes on first run..."
echo

# Install wheel first (required for some packages)
echo "Installing build dependencies..."
pip install wheel setuptools -q

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

echo "Installing timm and other core packages..."
pip install timm>=1.0.0 scipy>=1.11.0 opencv-python>=4.8.0 -q

echo "Installing remaining requirements..."
pip install -r requirements.txt -q

echo -e "${GREEN}✓${NC} All packages installed"
echo

# Step 5: Check GPU
echo -e "${BLUE}[5/7]${NC} Checking GPU availability..."
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo

# Step 6: Download datasets
echo -e "${BLUE}[6/7]${NC} Checking/downloading datasets..."
python3 scripts/download_datasets.py --dataset $DATASET
echo

# Step 7: Configure for quick or full run
if [ $QUICK_MODE -eq 1 ]; then
    echo -e "${BLUE}[7/7]${NC} Preparing quick test configuration..."
    
    # Create temporary quick config
    cat > config_quick.yaml << EOF
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
  output_dir: "outputs_quick_test"
  wandb_enabled: false
EOF
    
    CONFIG_FILE=config_quick.yaml
    echo -e "${GREEN}✓${NC} Quick test config created"
else
    # Update folds in config.yaml
    sed -i "s/n_folds: .*/n_folds: $FOLDS  # Set via command line/" config.yaml
    CONFIG_FILE=config.yaml
    echo -e "${GREEN}✓${NC} Using full configuration ($FOLDS folds)"
fi
echo

# Run pipeline
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}Starting full pipeline execution...${NC}"
echo "Configuration: $CONFIG_FILE"
echo "Dataset: $DATASET"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo

python3 run_pipeline.py --config $CONFIG_FILE --dataset $DATASET

if [ $? -ne 0 ]; then
    echo
    echo -e "${RED}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║   PIPELINE EXECUTION FAILED                                     ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo "Check error messages above for details."
    
    # Cleanup quick config
    if [ $QUICK_MODE -eq 1 ] && [ -f config_quick.yaml ]; then
        rm config_quick.yaml
    fi
    exit 1
fi

# Cleanup quick config
if [ $QUICK_MODE -eq 1 ] && [ -f config_quick.yaml ]; then
    rm config_quick.yaml
fi

# Summary
echo
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   PIPELINE COMPLETE!                                            ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Results saved to: outputs/                                      ║${NC}"
if [ $QUICK_MODE -eq 1 ]; then
    echo -e "${GREEN}║  Quick test results: outputs_quick_test/comparison_binary.csv    ║${NC}"
else
    echo -e "${GREEN}║  Full results: outputs/comparison_binary.csv                     ║${NC}"
fi
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo
echo "Next steps:"
echo "  1. Review results in outputs/comparison_binary.csv"
echo "  2. Check individual model results in outputs/<Model>_<task>/"
echo "  3. Run statistical analysis: python3 scripts/analyze_results.py"
echo "  4. View W&B dashboard: Check terminal output for URL"
echo
