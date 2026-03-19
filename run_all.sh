#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Breast Cancer Classification - One-Click Run Script (Linux)
# ═══════════════════════════════════════════════════════════════════
# 
# This script runs the complete training pipeline with automatic
# environment and dataset checks.
#
# Features:
# - Auto-creates venv if missing
# - Auto-downloads dataset if missing
# - Activates environment automatically
# - Runs full pipeline with all models
# - Logs all output to logs/training_log.txt
# - Shows estimated time remaining
# - Handles errors gracefully
#
# Usage:
#   ./run_all.sh
#   ./run_all.sh --models efficientnet spiking
#   ./run_all.sh --config my_config.yaml
#   ./run_all.sh --quick  # Quick test with subset
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on error (will be disabled for pipeline execution)

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
VENV_DIR="$PROJECT_ROOT/.venv"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/training_log_$(date +%Y%m%d_%H%M%S).txt"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"
PIPELINE_SCRIPT="$PROJECT_ROOT/run_pipeline.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Pipeline arguments
PIPELINE_ARGS=""
QUICK_MODE=false

# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1" | tee -a "$LOG_FILE"
}

log_step() {
    echo -e "\n${CYAN}════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}$1${NC}" | tee -a "$LOG_FILE"
    echo -e "${CYAN}════════════════════════════════════════════════════════${NC}\n" | tee -a "$LOG_FILE"
}

check_command() {
    command -v "$1" &> /dev/null
}

# ═══════════════════════════════════════════════════════════════════
# Setup Check Functions
# ═══════════════════════════════════════════════════════════════════

check_venv() {
    log_step "Checking Virtual Environment"
    
    if [ ! -d "$VENV_DIR" ]; then
        log_warning "Virtual environment not found at $VENV_DIR"
        log_info "Running auto_setup.sh to create environment..."
        
        if [ -f "$SCRIPT_DIR/scripts/auto_setup.sh" ]; then
            bash "$SCRIPT_DIR/scripts/auto_setup.sh" --skip-dataset
        else
            log_error "auto_setup.sh not found. Please run setup first."
            exit 1
        fi
    else
        log_success "Virtual environment found"
    fi
}

activate_venv() {
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    log_success "Virtual environment activated"
}

check_dataset() {
    log_step "Checking Dataset"
    
    local dataset_path="$DATA_DIR/BreaKHis_v1"
    
    if [ ! -d "$dataset_path" ]; then
        log_warning "Dataset not found at $dataset_path"
        log_info "Downloading dataset..."
        
        if [ -f "$SCRIPT_DIR/scripts/download_dataset.py" ]; then
            python "$SCRIPT_DIR/scripts/download_dataset.py" --output_dir "$DATA_DIR" --resume
        else
            log_error "download_dataset.py not found."
            log_info "Please download BreakHis dataset manually from:"
            log_info "https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/"
            exit 1
        fi
    else
        log_success "Dataset found at $dataset_path"
        
        # Quick validation
        local image_count=$(find "$dataset_path" -name "*.png" 2>/dev/null | wc -l)
        if [ "$image_count" -gt 0 ]; then
            log_success "Found $image_count images"
        else
            log_warning "No PNG images found in dataset directory"
        fi
    fi
}

check_gpu() {
    log_step "Checking GPU Status"
    
    # Check nvidia-smi
    if check_command nvidia-smi; then
        log_success "NVIDIA GPU detected"
        
        # Get GPU info
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        local gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | wc -l)
        
        echo -e "  ${GREEN}GPU: $gpu_name${NC}" | tee -a "$LOG_FILE"
        echo -e "  ${GREEN}Count: $gpu_count${NC}" | tee -a "$LOG_FILE"
        
        # Show memory info
        nvidia-smi --query-gpu=memory.total,memory.used,memory.free --format=csv | head -2 | tee -a "$LOG_FILE"
    else
        log_warning "No NVIDIA GPU detected (running on CPU)"
    fi
    
    # Check PyTorch CUDA
    log_info "Checking PyTorch CUDA..."
    python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
" 2>&1 | tee -a "$LOG_FILE"
}

check_config() {
    log_step "Checking Configuration"
    
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Config file not found: $CONFIG_FILE"
        exit 1
    fi
    
    log_success "Config file: $CONFIG_FILE"
    
    # Show key settings
    log_info "Key settings:"
    python -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
print(f\"  Task: {cfg['data']['task']}\")
print(f\"  Folds: {cfg['cv']['n_folds']}\")
print(f\"  Epochs: {cfg['training']['epochs']}\")
models = [k for k, v in cfg['models'].items() if v.get('enabled', False)]
print(f\"  Models: {', '.join(models)}\")
" 2>&1 | tee -a "$LOG_FILE"
}

# ═══════════════════════════════════════════════════════════════════
# Time Estimation
# ═══════════════════════════════════════════════════════════════════

estimate_time() {
    log_step "Time Estimation"
    
    python -c "
import yaml

with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)

epochs = cfg['training']['epochs']
folds = cfg['cv']['n_folds']
models = [k for k, v in cfg['models'].items() if v.get('enabled', False)]

# Rough estimates per fold (based on typical A100 performance)
# These are approximate and vary based on dataset size and model
time_per_fold = {
    'efficientnet': 300,    # ~5 min per fold
    'spiking': 400,         # ~6.5 min per fold
    'hybrid_vit': 350,      # ~6 min per fold
    'vit_tiny': 250,        # ~4 min per fold
    'quantum': 600,         # ~10 min per fold (CPU)
}

total_seconds = 0
print('Estimated time per model (all folds):')
for model in models:
    est = time_per_fold.get(model, 300) * folds
    total_seconds += est
    mins = est // 60
    print(f'  {model}: ~{mins} minutes')

total_mins = total_seconds // 60
total_hours = total_mins // 60
remaining_mins = total_mins % 60

print()
if total_hours > 0:
    print(f'TOTAL ESTIMATE: ~{total_hours}h {remaining_mins}m')
else:
    print(f'TOTAL ESTIMATE: ~{total_mins} minutes')
print()
print('Note: Actual times vary based on GPU, dataset size, and settings.')
" 2>&1 | tee -a "$LOG_FILE"
}

# ═══════════════════════════════════════════════════════════════════
# Pipeline Execution
# ═══════════════════════════════════════════════════════════════════

run_pipeline() {
    log_step "Starting Training Pipeline"
    
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}  TRAINING STARTED: $(date)${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}\n" | tee -a "$LOG_FILE"
    
    # Record start time
    local start_time=$(date +%s)
    
    # Disable exit on error for pipeline execution
    set +e
    
    # Run pipeline with tee for logging
    if [ "$QUICK_MODE" = true ]; then
        log_info "Running in QUICK mode (subset=200, epochs=3)"
        python "$PIPELINE_SCRIPT" --config "$CONFIG_FILE" $PIPELINE_ARGS 2>&1 | tee -a "$LOG_FILE"
    else
        python "$PIPELINE_SCRIPT" --config "$CONFIG_FILE" $PIPELINE_ARGS 2>&1 | tee -a "$LOG_FILE"
    fi
    
    local exit_code=${PIPESTATUS[0]}
    
    # Re-enable exit on error
    set -e
    
    # Record end time
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local hours=$((duration / 3600))
    local minutes=$(((duration % 3600) / 60))
    local seconds=$((duration % 60))
    
    echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}  TRAINING COMPLETED: $(date)${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}  Duration: ${hours}h ${minutes}m ${seconds}s${NC}" | tee -a "$LOG_FILE"
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}" | tee -a "$LOG_FILE"
    
    if [ $exit_code -eq 0 ]; then
        log_success "Pipeline completed successfully!"
    else
        log_error "Pipeline failed with exit code: $exit_code"
        log_info "Check log file for details: $LOG_FILE"
    fi
    
    return $exit_code
}

show_results() {
    log_step "Results Summary"
    
    # Find comparison CSV
    local comparison_file=$(find "$PROJECT_ROOT/outputs" -name "comparison_*.csv" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    
    if [ -n "$comparison_file" ] && [ -f "$comparison_file" ]; then
        log_success "Latest results: $comparison_file"
        echo "" | tee -a "$LOG_FILE"
        echo "Results preview:" | tee -a "$LOG_FILE"
        python -c "
import pandas as pd
df = pd.read_csv('$comparison_file')
# Show key columns
cols = [c for c in df.columns if any(x in c for x in ['Model', 'Mean_Acc', 'Mean_F1', 'Mean_AUC', 'Mean_MCC'])]
if cols:
    print(df[cols].to_string(index=False))
" 2>&1 | tee -a "$LOG_FILE"
    else
        log_warning "No comparison results found yet"
    fi
    
    echo "" | tee -a "$LOG_FILE"
    log_info "Full log saved to: $LOG_FILE"
    log_info "Outputs directory: $PROJECT_ROOT/outputs"
}

# ═══════════════════════════════════════════════════════════════════
# Parse Arguments
# ═══════════════════════════════════════════════════════════════════

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --models)
                PIPELINE_ARGS="$PIPELINE_ARGS --models $2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --quick)
                QUICK_MODE=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --models M1 M2    Run specific models only"
                echo "  --config FILE     Use custom config file"
                echo "  --quick           Quick test mode (subset, fewer epochs)"
                echo "  --help, -h        Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
}

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

main() {
    parse_args "$@"
    
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Initialize log file
    echo "════════════════════════════════════════════════════════" > "$LOG_FILE"
    echo "  BREAST CANCER CLASSIFICATION - TRAINING LOG" >> "$LOG_FILE"
    echo "  Started: $(date)" >> "$LOG_FILE"
    echo "════════════════════════════════════════════════════════" >> "$LOG_FILE"
    echo "" >> "$LOG_FILE"
    
    log_step "Breast Cancer Classification - One-Click Runner"
    log_info "Project: $PROJECT_ROOT"
    log_info "Log file: $LOG_FILE"
    
    # Pre-flight checks
    check_venv
    activate_venv
    check_dataset
    check_gpu
    check_config
    estimate_time
    
    # Confirm before starting
    echo ""
    read -p "Start training? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Training cancelled by user"
        exit 0
    fi
    
    # Run pipeline
    run_pipeline
    local pipeline_result=$?
    
    # Show results
    show_results
    
    exit $pipeline_result
}

# Run main function
main "$@"
