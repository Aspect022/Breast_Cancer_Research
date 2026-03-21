#!/bin/bash
# ════════════════════════════════════════════════════════════════════════════════
# CBIS-DDSM Generalization Validation Pipeline
# ════════════════════════════════════════════════════════════════════════════════
# 
# This script:
# 1. Auto-downloads CBIS-DDSM dataset from Kaggle
# 2. Runs all enabled models from config.yaml
# 3. Compares performance with BreakHis results
#
# Usage:
#   ./run_cbis_ddsm.sh                    # Full pipeline
#   ./run_cbis_ddsm.sh --quick            # Quick test with 200 samples
#   ./run_cbis_ddsm.sh --models swin_tiny # Specific models only
#
# Prerequisites:
#   - Kaggle CLI installed: pip install kaggle
#   - Kaggle credentials set:
#     export KAGGLE_USERNAME=your_username
#     export KAGGLE_KEY=your_api_key
# ════════════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# ════════════════════════════════════════════════════════════════════════════════
# Configuration
# ════════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default parameters
DATASET="cbis_ddsm"
DATA_DIR="data/CBIS-DDSM"
CONFIG_FILE="config.yaml"
QUICK_MODE=false
MODELS=""
SUBSET_SIZE=""

# ════════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ════════════════════════════════════════════════════════════════════════════════

print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ════════════════════════════════════════════════════════════════════════════════
# Parse Command Line Arguments
# ════════════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            SUBSET_SIZE="200"
            shift
            ;;
        --models)
            MODELS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick           Run quick test with 200 samples"
            echo "  --models MODELS   Comma-separated list of models to run"
            echo "  --config FILE     Use custom config file (default: config.yaml)"
            echo "  --data-dir DIR    Custom data directory (default: data/CBIS-DDSM)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ════════════════════════════════════════════════════════════════════════════════
# Check Prerequisites
# ════════════════════════════════════════════════════════════════════════════════

print_header "🔍 Checking Prerequisites"

# Check Python
if ! command -v python &> /dev/null; then
    print_error "Python not found. Please install Python 3.9+"
    exit 1
fi
print_success "Python found: $(python --version)"

# Check Kaggle CLI
if ! command -v kaggle &> /dev/null; then
    print_warning "Kaggle CLI not installed. Installing..."
    pip install kaggle
fi
print_success "Kaggle CLI found"

# Check Kaggle credentials
if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    print_error "Kaggle credentials not set!"
    echo ""
    echo "Please set the following environment variables:"
    echo "  export KAGGLE_USERNAME=your_username"
    echo "  export KAGGLE_KEY=your_api_key"
    echo ""
    echo "Get your API key from: https://www.kaggle.com/account"
    exit 1
fi
print_success "Kaggle credentials found"

# Check requirements
if [ ! -f "requirements.txt" ]; then
    print_error "requirements.txt not found!"
    exit 1
fi

print_success "Installing/verifying requirements..."
pip install -q -r requirements.txt

# ════════════════════════════════════════════════════════════════════════════════
# Download Dataset
# ════════════════════════════════════════════════════════════════════════════════

print_header "📥 Downloading CBIS-DDSM Dataset"

if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/metadata.csv" ]; then
    print_success "Dataset already exists at $DATA_DIR"
    echo "Skipping download..."
else
    echo "Creating data directory: $DATA_DIR"
    mkdir -p "$DATA_DIR"
    
    echo "Downloading CBIS-DDSM from Kaggle..."
    echo "This may take 10-30 minutes depending on your connection speed."
    
    kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p "$DATA_DIR"
    
    echo "Extracting dataset..."
    cd "$DATA_DIR"
    unzip -q cbis-ddsm-breast-cancer-image-dataset.zip
    rm cbis-ddsm-breast-cancer-image-dataset.zip
    cd - > /dev/null
    
    print_success "Dataset downloaded and extracted to $DATA_DIR"
fi

# ════════════════════════════════════════════════════════════════════════════════
# Update Configuration
# ════════════════════════════════════════════════════════════════════════════════

print_header "⚙️  Updating Configuration"

# Create temporary config for CBIS-DDSM
TEMP_CONFIG="config_cbis_ddsm_temp.yaml"

echo "Creating temporary config: $TEMP_CONFIG"

# Read original config and modify dataset settings
cat > "$TEMP_CONFIG" << EOF
# Auto-generated config for CBIS-DDSM validation
# Generated: $(date)

data:
  dataset: "$DATASET"
  data_dir: "$DATA_DIR"
  task: "binary"
  subset: $([ "$SUBSET_SIZE" = "" ] && echo "null" || echo "$SUBSET_SIZE")
  num_workers: 8
  seed: 42
  
  cbis_ddsm:
    data_dir: "$DATA_DIR"
    auto_download: false
    kaggle_username: null
    kaggle_key: null

# Include all other settings from original config
EOF

# Append model configurations from original config
grep -A 1000 "^cv:" "$CONFIG_FILE" >> "$TEMP_CONFIG"

print_success "Configuration updated"

echo ""
echo "Configuration summary:"
echo "  Dataset: $DATASET"
echo "  Data Directory: $DATA_DIR"
echo "  Subset Size: $([ "$SUBSET_SIZE" = "" ] && echo "Full" || echo "$SUBSET_SIZE")"
echo "  Config File: $TEMP_CONFIG"

# ════════════════════════════════════════════════════════════════════════════════
# Run Training Pipeline
# ════════════════════════════════════════════════════════════════════════════════

print_header "🚀 Running Training Pipeline"

# Build command
CMD="python run_pipeline.py --config $TEMP_CONFIG"

if [ "$MODELS" != "" ]; then
    CMD="$CMD --models $MODELS"
fi

echo "Running: $CMD"
echo ""

# Run pipeline
$CMD

# ════════════════════════════════════════════════════════════════════════════════
# Generate Comparison Report
# ════════════════════════════════════════════════════════════════════════════════

print_header "📊 Generating Comparison Report"

python << 'PYTHON_SCRIPT'
import os
import pandas as pd
import glob

# Find comparison files
breakhis_files = glob.glob('outputs/comparison_binary.csv')
cbis_ddsm_files = glob.glob('outputs/comparison_cbis_ddsm*.csv')

if len(breakhis_files) > 0 and len(cbis_ddsm_files) > 0:
    # Load results
    breakhis_df = pd.read_csv(breakhis_files[0])
    cbis_ddsm_df = pd.read_csv(cbis_ddsm_files[0])
    
    # Merge for comparison
    comparison = pd.merge(
        breakhis_df[['Model', 'accuracy', 'auc_roc', 'sensitivity', 'specificity']],
        cbis_ddsm_df[['Model', 'accuracy', 'auc_roc', 'sensitivity', 'specificity']],
        on='Model',
        suffixes=('_breakhis', '_cbis_ddsm')
    )
    
    # Calculate performance drop
    comparison['accuracy_drop'] = comparison['accuracy_breakhis'] - comparison['accuracy_cbis_ddsm']
    comparison['auc_drop'] = comparison['auc_roc_breakhis'] - comparison['auc_roc_cbis_ddsm']
    
    # Save comparison
    comparison.to_csv('outputs/cbis_ddsm_generalization_comparison.csv', index=False)
    
    print("\n📊 Generalization Comparison (BreakHis → CBIS-DDSM)")
    print("=" * 80)
    print(comparison[['Model', 'accuracy_breakhis', 'accuracy_cbis_ddsm', 'accuracy_drop']].to_string(index=False))
    print("\nFull comparison saved to: outputs/cbis_ddsm_generalization_comparison.csv")
else:
    print("⚠️  Could not generate comparison - missing result files")
PYTHON_SCRIPT

# ════════════════════════════════════════════════════════════════════════════════
# Cleanup
# ════════════════════════════════════════════════════════════════════════════════

print_header "🧹 Cleanup"

if [ -f "$TEMP_CONFIG" ]; then
    rm "$TEMP_CONFIG"
    print_success "Removed temporary config file"
fi

# ════════════════════════════════════════════════════════════════════════════════
# Summary
# ════════════════════════════════════════════════════════════════════════════════

print_header "✅ Pipeline Complete!"

echo "Results saved to: outputs/"
echo ""
echo "Next steps:"
echo "  1. Review results: cat outputs/comparison_*.csv"
echo "  2. Compare with BreakHis: python scripts/analyze_results.py"
echo "  3. Check W&B for training curves: https://wandb.ai/your-username/breast-cancer-transformers"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
