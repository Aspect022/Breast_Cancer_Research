#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# BREAST CANCER CLASSIFICATION - SETUP SCRIPT (Linux)
# ═══════════════════════════════════════════════════════════════════
# One-time setup: Creates venv and installs requirements
# ═══════════════════════════════════════════════════════════════════

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   BREAST CANCER CLASSIFICATION - SETUP                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo

# Check Python
echo "[1/4] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 not found!"
    echo "Install with: sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi
echo "✓ Python found: $(python3 --version)"

# Create venv
echo
echo "[2/4] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate
echo
echo "[3/4] Activating virtual environment..."
source .venv/bin/activate
echo "✓ Activated"

# Install requirements
echo
echo "[4/4] Installing requirements..."
echo "This may take 5-10 minutes..."

# Install PyTorch
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q

# Install other requirements
echo "Installing remaining packages..."
pip install -r requirements.txt -q

echo "✓ All packages installed"

# Summary
echo
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║   SETUP COMPLETE!                                               ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Next steps:                                                     ║"
echo "║  1. Activate venv: source .venv/bin/activate                     ║"
echo "║  2. Download datasets: python3 scripts/download_datasets.py      ║"
echo "║  3. Run pipeline: ./run_full_pipeline.sh                         ║"
echo "║     or: python3 run_pipeline.py                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
