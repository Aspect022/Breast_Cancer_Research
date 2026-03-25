#!/bin/bash
# Fix all quantum layer bugs

echo "Fixing quantum layer bugs..."

# Fix 1: Remove execution_config from quantum_fusion_layer.py
sed -i "s/execution_config={'gradient_method': 'backprop'}//" src/models/quantum/quantum_fusion_layer.py

# Fix 2: Add dtype conversion after quantum output in quantum_fusion_layer.py
sed -i '/quantum_out = torch.stack(quantum_outputs)/a\        # Ensure same dtype as input (fixes AMP issues)\n        quantum_out = quantum_out.to(x.dtype)' src/models/quantum/quantum_fusion_layer.py

# Fix 3: Add dtype conversion in quantum_bottleneck_layer.py
sed -i '/quantum_out = torch.stack(quantum_outputs)/a\        # Ensure same dtype as input (fixes AMP issues)\n        quantum_out = quantum_out.to(x.dtype)' src/models/quantum/quantum_bottleneck_layer.py

# Fix 4: Add missing imports to run_pipeline.py
sed -i '/from .fusion import (/a\    get_cb_qccf_convnet_efficient,\n    get_cb_qccf_swin_convnet,' run_pipeline.py

# Fix 5: Fix ensemble teacher name (hybrid_vit -> cnn_vit_hybrid)
sed -i 's/"hybrid_vit"/"cnn_vit_hybrid"/g' config.yaml

echo "✅ All fixes applied!"
echo ""
echo "Now pull these changes on your server and restart training:"
echo ""
echo "  cd ~/Projects/Cancer/Breast_Cancer_Research"
echo "  git pull"
echo "  nohup python3 run_pipeline.py --models \\"
echo "    triple_branch_fusion \\"
echo "    triple_branch_fusion_quantum \\"
echo "    triple_branch_fusion_bottleneck \\"
echo "    cb_qccf \\"
echo "    cb_qccf_convnet_efficient \\"
echo "    cb_qccf_swin_convnet \\"
echo "    msq_fusion \\"
echo "    ensemble_distillation \\"
echo "    > training_all_complete.log 2>&1 &"
