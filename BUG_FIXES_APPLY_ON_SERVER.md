# 🔧 CRITICAL BUG FIXES FOR QUANTUM MODELS

## Issues Found:

1. ✅ **QNode execution_config error** - PennyLane API doesn't support this parameter
2. ✅ **dtype mismatch (Double vs Half)** - Quantum output needs dtype conversion for AMP
3. ✅ **Missing imports** - CB-QCCF variants not imported in run_pipeline.py
4. ✅ **Ensemble teacher name** - `hybrid_vit` should be `cnn_vit_hybrid`

## Fixes Applied (Local):

### Fix 1 & 2: Quantum Layer dtype Fixes
- Added `quantum_out = quantum_out.to(x.dtype)` after quantum circuit output
- Removed invalid `execution_config` parameter

### Fix 3: Missing Imports
Need to add to run_pipeline.py imports:
```python
from src.models.fusion import (
    get_dual_branch_fusion,
    get_triple_branch_fusion,
    get_cb_qccf,
    get_cb_qccf_convnet_efficient,  # ← ADD THIS
    get_cb_qccf_swin_convnet,        # ← ADD THIS
    get_multi_scale_quantum_fusion,
    get_ensemble_distillation,
)
```

### Fix 4: Ensemble Teacher Name
In config.yaml, change:
```yaml
teacher_models:
  - "swin_small"
  - "cnn_vit_hybrid"  # ← Was "hybrid_vit"
  - "efficientnet_b3"
```

---

## 🚀 APPLY THESE FIXES ON YOUR SERVER:

Run this script on your server:

```bash
cd ~/Projects/Cancer/Breast_Cancer_Research

# Fix 1: Add dtype conversion to quantum_fusion_layer.py
sed -i '/quantum_out = torch.stack(quantum_outputs)/a\        # Ensure same dtype as input (fixes AMP issues)\n        quantum_out = quantum_out.to(x.dtype)' src/models/quantum/quantum_fusion_layer.py

# Fix 2: Add dtype conversion to quantum_bottleneck_layer.py  
sed -i '/quantum_out = torch.stack(quantum_outputs)/a\        # Ensure same dtype as input (fixes AMP issues)\n        quantum_out = quantum_out.to(x.dtype)' src/models/quantum/quantum_bottleneck_layer.py

# Fix 3: Add missing imports to run_pipeline.py
sed -i 's/get_cb_qccf,$/get_cb_qccf,\n    get_cb_qccf_convnet_efficient,\n    get_cb_qccf_swin_convnet,/' run_pipeline.py

# Fix 4: Fix ensemble teacher name
sed -i 's/"hybrid_vit"/"cnn_vit_hybrid"/g' config.yaml

# Pull the other fixes from git
git pull

# Now run all models
nohup python3 run_pipeline.py --models \
  triple_branch_fusion \
  triple_branch_fusion_quantum \
  triple_branch_fusion_bottleneck \
  cb_qccf \
  cb_qccf_convnet_efficient \
  cb_qccf_swin_convnet \
  msq_fusion \
  ensemble_distillation \
  > training_all_complete.log 2>&1 &

# Monitor
tail -f training_all_complete.log
```

---

## ✅ Expected Result:

All 8 models should now run without errors:
1. ✅ triple_branch_fusion (classical B5)
2. ✅ triple_branch_fusion_quantum (Placement 3)
3. ✅ triple_branch_fusion_bottleneck (Placement 2)
4. ✅ cb_qccf (original)
5. ✅ cb_qccf_convnet_efficient (ConvNet+EffB5)
6. ✅ cb_qccf_swin_convnet (Swin+ConvNet)
7. ✅ msq_fusion (multi-scale quantum)
8. ✅ ensemble_distillation (teacher-student)

---

**Copy and paste the server commands above!** 🚀
