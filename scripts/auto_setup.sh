#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# Breast Cancer Classification - Automated Setup Script (Linux)
# ═══════════════════════════════════════════════════════════════════
# 
# This script automates the complete setup process:
# 1. Creates and activates virtual environment
# 2. Installs all dependencies
# 3. Downloads the BreakHis dataset
# 4. Verifies GPU availability
# 5. Creates output directories
# 6. Generates config if missing
#
# Usage:
#   ./scripts/auto_setup.sh
#   ./scripts/auto_setup.sh --skip-dataset
#   ./scripts/auto_setup.sh --data-dir /path/to/data
# ═══════════════════════════════════════════════════════════════════

set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
CONFIG_FILE="$PROJECT_ROOT/config.yaml"
LOG_FILE="$PROJECT_ROOT/logs/setup.log"
PYTHON_VERSION="3.10"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ═══════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
}

log_step() {
    echo -e "\n${GREEN}════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════${NC}\n"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Setup Functions
# ═══════════════════════════════════════════════════════════════════

check_prerequisites() {
    log_step "Checking Prerequisites"
    
    local missing=()
    
    # Check Python
    if check_command python3; then
        PYTHON_CMD="python3"
        PYTHON_VER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
        log_success "Python found: $PYTHON_VER"
    elif check_command python; then
        PYTHON_CMD="python"
        PYTHON_VER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
        log_success "Python found: $PYTHON_VER"
    else
        log_error "Python not found. Please install Python $PYTHON_VERSION or higher."
        missing+=("python")
    fi
    
    # Check pip
    if check_command pip3; then
        PIP_CMD="pip3"
    elif check_command pip; then
        PIP_CMD="pip"
    else
        log_error "pip not found"
        missing+=("pip")
    fi
    
    # Check git
    if check_command git; then
        log_success "Git found: $(git --version)"
    else
        log_warning "Git not found (optional)"
    fi
    
    # Check wget/curl for downloads
    if check_command wget; then
        DOWNLOAD_CMD="wget"
    elif check_command curl; then
        DOWNLOAD_CMD="curl"
    else
        log_warning "Neither wget nor curl found (downloads may fail)"
    fi
    
    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing prerequisites: ${missing[*]}"
        echo ""
        echo "Install with:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip git wget"
        echo "  Fedora: sudo dnf install python3 python3-pip git wget"
        echo "  Arch: sudo pacman -S python python-pip git wget"
        exit 1
    fi
}

create_virtual_environment() {
    log_step "Creating Virtual Environment"
    
    if [ -d "$VENV_DIR" ]; then
        log_warning "Virtual environment already exists at $VENV_DIR"
        read -p "Recreate? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Using existing virtual environment"
            return 0
        fi
        log_info "Removing old virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    log_info "Creating virtual environment in $VENV_DIR..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    
    if [ $? -eq 0 ]; then
        log_success "Virtual environment created successfully"
    else
        log_error "Failed to create virtual environment"
        exit 1
    fi
}

activate_virtual_environment() {
    log_info "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
    
    if [ $? -eq 0 ]; then
        log_success "Virtual environment activated"
        log_info "Python: $(which python)"
        log_info "Pip: $(which pip)"
    else
        log_error "Failed to activate virtual environment"
        exit 1
    fi
}

install_dependencies() {
    log_step "Installing Dependencies"
    
    local requirements_file="$PROJECT_ROOT/requirements.txt"
    
    if [ ! -f "$requirements_file" ]; then
        log_error "requirements.txt not found at $requirements_file"
        exit 1
    fi
    
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    log_info "Installing requirements from $requirements_file..."
    pip install -r "$requirements_file"
    
    if [ $? -eq 0 ]; then
        log_success "Dependencies installed successfully"
    else
        log_error "Failed to install dependencies"
        exit 1
    fi
    
    # Install optional A100 optimizations
    log_info "Checking for optional A100 optimizations..."
    
    # Check if NVIDIA GPU is available
    if check_command nvidia-smi; then
        log_success "NVIDIA GPU detected"
        
        # Install apex for mixed precision (optional, requires CUDA toolkit)
        log_info "Apex can be installed manually for better A100 performance:"
        log_info "  git clone https://github.com/NVIDIA/apex"
        log_info "  cd apex && pip install -v --no-build-isolation --config-settings '--build-option=--cpp_ext --cuda_ext' ./"
    fi
}

download_dataset() {
    log_step "Downloading BreakHis Dataset"
    
    local skip_dataset="${SKIP_DATASET:-false}"
    
    if [ "$skip_dataset" = true ]; then
        log_warning "Skipping dataset download (--skip-dataset flag set)"
        return 0
    fi
    
    # Check if dataset already exists
    if [ -d "$DATA_DIR/BreaKHis_v1" ]; then
        log_success "Dataset already exists at $DATA_DIR/BreaKHis_v1"
        return 0
    fi
    
    log_info "Running dataset downloader..."
    
    # Create data directory
    mkdir -p "$DATA_DIR"
    
    # Run the Python download script
    python "$SCRIPT_DIR/download_dataset.py" \
        --output_dir "$DATA_DIR" \
        --resume
    
    if [ $? -eq 0 ]; then
        log_success "Dataset downloaded successfully"
    else
        log_warning "Dataset download failed or was interrupted"
        log_info "You can run it manually later:"
        log_info "  python scripts/download_dataset.py"
    fi
}

verify_gpu() {
    log_step "Verifying GPU Availability"
    
    # Check nvidia-smi
    if check_command nvidia-smi; then
        log_success "NVIDIA driver detected"
        echo ""
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
        echo ""
    else
        log_warning "nvidia-smi not found - NVIDIA driver may not be installed"
    fi
    
    # Check CUDA via PyTorch
    log_info "Checking PyTorch CUDA support..."
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')
" 2>/dev/null || log_warning "PyTorch not installed yet or CUDA not available"
}

create_directories() {
    log_step "Creating Output Directories"
    
    local dirs=(
        "$PROJECT_ROOT/outputs"
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/checkpoints"
        "$DATA_DIR"
    )
    
    for dir in "${dirs[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            log_success "Created: $dir"
        else
            log_info "Exists: $dir"
        fi
    done
}

generate_config() {
    log_step "Checking Configuration"
    
    if [ -f "$CONFIG_FILE" ]; then
        log_success "Config file exists: $CONFIG_FILE"
    else
        log_warning "Config file not found. Creating default config..."
        
        cat > "$CONFIG_FILE" << 'EOF'
# BreakHis Breast Cancer Classification - Default Config
# Generated by auto_setup.sh

data:
  data_dir: "data/BreaKHis_v1"
  task: "binary"
  num_workers: 8
  seed: 42

cv:
  n_folds: 5

training:
  epochs: 30
  patience: 7
  grad_clip: 1.0

models:
  efficientnet:
    enabled: true
    batch_size: 64
    lr: 1.0e-4
    use_amp: true

  spiking:
    enabled: true
    batch_size: 32
    lr: 8.0e-4
    use_amp: false

  hybrid_vit:
    enabled: true
    batch_size: 64
    lr: 1.0e-4
    use_amp: true

  vit_tiny:
    enabled: true
    batch_size: 64
    lr: 1.0e-4
    use_amp: true

  quantum:
    enabled: true
    batch_size: 16
    lr: 1.0e-4
    use_amp: false

output:
  output_dir: "outputs"
  save_best_model: true
  save_plots: true
  save_csv: true
EOF
        
        log_success "Default config created: $CONFIG_FILE"
    fi
}

print_summary() {
    log_step "Setup Complete!"
    
    echo ""
    echo "┌─────────────────────────────────────────────────────────────┐"
    echo "│  SETUP SUMMARY                                              │"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│  Project Root:    $PROJECT_ROOT"
    echo "│  Virtual Env:     $VENV_DIR"
    echo "│  Data Directory:  $DATA_DIR"
    echo "│  Config File:     $CONFIG_FILE"
    echo "│  Logs:            $LOG_FILE"
    echo "├─────────────────────────────────────────────────────────────┤"
    echo "│  Next Steps:                                                │"
    echo "│  1. Activate venv:  source $VENV_DIR/bin/activate"
    echo "│  2. Verify data:    ls -la $DATA_DIR"
    echo "│  3. Edit config:    nano $CONFIG_FILE"
    echo "│  4. Run training:   python run_pipeline.py"
    echo "└─────────────────────────────────────────────────────────────┘"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

main() {
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-dataset)
                SKIP_DATASET=true
                shift
                ;;
            --data-dir)
                DATA_DIR="$2"
                shift 2
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --skip-dataset    Skip dataset download"
                echo "  --data-dir DIR    Set custom data directory"
                echo "  --help, -h        Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Ensure we're in project root
    cd "$PROJECT_ROOT"
    
    # Create log directory
    mkdir -p "$(dirname "$LOG_FILE")"
    
    # Run setup steps
    check_prerequisites
    create_virtual_environment
    activate_virtual_environment
    install_dependencies
    create_directories
    download_dataset
    verify_gpu
    generate_config
    print_summary
    
    log_success "All setup steps completed!"
}

# Run main function
main "$@"
