#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config.yaml}"

echo "=== A100 TBCA Quantum Variant Run ==="
echo "Config: ${CONFIG}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-breast-cancer-tbca-a100}"

python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
PY

echo "Running existing implemented TBCA models first..."
python run_pipeline.py --config "${CONFIG}" --models \
  triple_branch_fusion \
  triple_branch_fusion_quantum \
  triple_branch_fusion_bottleneck \
  triple_branch_fusion_cnn_featuremap_quantum \
  triple_branch_fusion_vit_featuremap_quantum

echo "The proposed post-cross-attention and dual-quantum variants must be implemented before adding them here."

