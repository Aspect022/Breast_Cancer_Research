#!/bin/bash
# Quick Training Script for Previously Crashed Models
# Run this to train all the models that were fixed

echo "======================================================================"
echo "  Training Previously Crashed Models"
echo "======================================================================"
echo ""
echo "Models to train:"
echo "  1. CB-QCCF (Class-Balanced Quantum-Classical Fusion)"
echo "  2. MSQF (Multi-Scale Quantum Fusion)"
echo "  3. Swin-V2-Small"
echo "  4. Ensemble Distillation"
echo ""
echo "======================================================================"
echo ""

# Run the training
python3 run_pipeline.py \
  --models cb_qccf multi_scale_quantum_fusion swin_v2_small ensemble_distillation \
  --cv-folds 5 \
  --epochs 50

echo ""
echo "======================================================================"
echo "  Training Complete!"
echo "======================================================================"
