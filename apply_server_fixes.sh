#!/bin/bash
# Run this on your server to fix all quantum bugs

echo "=== FIXING QUANTUM MODEL BUGS ==="
echo ""

# Fix 1: Remove execution_config parameter (causes TypeError)
echo "Fix 1: Removing invalid execution_config parameter..."
sed -i "/execution_config/d" src/models/quantum/quantum_fusion_layer.py

# Fix 2: Add dtype conversion after quantum output
echo "Fix 2: Adding dtype conversion in quantum_fusion_layer.py..."
sed -i '/quantum_out = torch.stack(quantum_outputs)/a\        quantum_out = quantum_out.to(x.dtype)  # Fix AMP dtype mismatch' src/models/quantum/quantum_fusion_layer.py

# Fix 3: Add dtype conversion in bottleneck layer
echo "Fix 3: Adding dtype conversion in quantum_bottleneck_layer.py..."
sed -i '/quantum_out = torch.stack(quantum_outputs)/a\        quantum_out = quantum_out.to(x.dtype)  # Fix AMP dtype mismatch' src/models/quantum/quantum_bottleneck_layer.py

# Fix 4: Add missing imports
echo "Fix 4: Adding missing CB-QCCF variant imports..."
sed -i '/get_cb_qccf,$/a\    get_cb_qccf_convnet_efficient,\n    get_cb_qccf_swin_convnet,' run_pipeline.py

echo ""
echo "✅ ALL FIXES APPLIED!"
echo ""
echo "Now restart training with:"
echo "  nohup python3 run_pipeline.py --models \\"
echo "    triple_branch_fusion_quantum \\"
echo "    triple_branch_fusion_bottleneck \\"
echo "    cb_qccf_convnet_efficient \\"
echo "    cb_qccf_swin_convnet \\"
echo "    ensemble_distillation \\"
echo "    > training_fixed.log 2>&1 &"
echo ""
