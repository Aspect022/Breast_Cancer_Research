#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-config_genomics_a100.yaml}"
PHASE="${2:-all}"

echo "=== A100 Genomics Run ==="
echo "Config: ${CONFIG}"
echo "Phase:  ${PHASE}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-breast-cancer-genomics-a100}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-16}"

python - <<'PY'
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("TF32:", torch.backends.cuda.matmul.allow_tf32)
PY

case "${PHASE}" in
  phase1)
    python run_genomic.py --config "${CONFIG}" --models g_baseline_mlp g_baseline_trees
    ;;
  phase2)
    python run_genomic.py --config "${CONFIG}" --models g_pasnet g_pathformer_lite
    ;;
  phase3)
    python run_genomic.py --config "${CONFIG}" --models g_tabtransformer g_multiscale_1dcnn g_bilstm
    ;;
  phase4)
    python run_genomic.py --config "${CONFIG}" --models g_gcn_ppi
    ;;
  phase5)
    python run_genomic.py --config "${CONFIG}" --models g_quantum_mlp g_pathway_quantum
    ;;
  phase6)
    python run_genomic.py --config "${CONFIG}" --models g_tnbc_dt_neural
    ;;
  phase7)
    python run_genomic.py --config "${CONFIG}" --models g_crossomics --dataset tcga_brca
    ;;
  all)
    python run_genomic.py --config "${CONFIG}"
    ;;
  *)
    echo "Unknown phase: ${PHASE}"
    echo "Use one of: phase1 phase2 phase3 phase4 phase5 phase6 phase7 all"
    exit 1
    ;;
esac

echo "A100 genomics run finished."

