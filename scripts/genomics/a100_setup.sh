#!/usr/bin/env bash
set -euo pipefail

echo "=== A100 Genomics Environment Setup ==="

python -m pip install --upgrade pip

echo "Installing PyTorch CUDA build..."
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "Installing project requirements..."
python -m pip install -r requirements.txt

echo "Installing genomics-specific dependencies..."
python -m pip install \
  GEOparse \
  gseapy \
  xgboost \
  imbalanced-learn \
  lifelines \
  pycombat

echo "Optional graph dependency:"
echo "  Install torch-geometric only when enabling g_gcn_ppi, matching the exact server CUDA/PyTorch build."

echo "Verifying CUDA..."
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("TF32 enabled:", torch.backends.cuda.matmul.allow_tf32)
PY

echo "Setup complete."

