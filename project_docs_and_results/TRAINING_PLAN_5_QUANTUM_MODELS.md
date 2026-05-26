# 🚀 Complete Training Plan: 5 Quantum-Enhanced Models

**Date:** March 23, 2026  
**Goal:** Train 5 quantum-enhanced models for comprehensive comparison  
**Expected Total Time:** ~18-20 hours (5-fold CV)

---

# 📋 The 5 Models to Run

## Model 1: TBCA-Fusion with Quantum Fusion (Placement 3) ⭐

```bash
python3 run_pipeline.py --models triple_branch_fusion_quantum
```

**Architecture:**
```
Swin-Small + ConvNeXt-Small + EfficientNet-B5 →
Cross-Attention → Weighted Fusion →
⭐ QUANTUM FUSION (8 qubits, 2 layers) →
Self-Attention → Classification
```

**Expected Performance:**
- Accuracy: 88.0-89.0% (+0.5-1.5% vs classical)
- Training Time: ~360-375 min (+12-17%)
- Parameters: 141.3M (+0.3M)

---

## Model 2: TBCA-Fusion with Quantum Bottleneck (Placement 2)

```bash
python3 run_pipeline.py --models triple_branch_fusion_bottleneck
```

**Architecture:**
```
Swin-Small → ⭐ VQC (8 qubits) →
ConvNeXt-Small → ⭐ VQC (8 qubits) →
EfficientNet-B5 →
Cross-Attention → Weighted Fusion →
Self-Attention → Classification
```

**Expected Performance:**
- Accuracy: 88.0-89.0% (+0.5-1.5% vs classical)
- Training Time: ~385-415 min (+20-30%)
- Parameters: 141.5M (+0.5M)

---

## Model 3: CB-QCCF Original (Swin + ResNet-18 + Quantum)

```bash
python3 run_pipeline.py --models cb_qccf
```

**Architecture:**
```
Swin-Small (Classical) →
                    → Cross-Attention → Dual-Head → Threshold → Output
ResNet-18 + QENN-RY (Quantum) →
```

**Expected Performance:**
- Accuracy: 85-87%
- Sensitivity: 88-91%
- Specificity: 80-84% (improved from 72%!)
- Training Time: ~150 min

---

## Model 4: CB-QCCF Variant 1 (ConvNeXt + EfficientNet-B5 + Quantum)

```bash
python3 run_pipeline.py --models cb_qccf_convnet_efficient
```

**Architecture:**
```
ConvNeXt-Small (Classical) →
                           → Cross-Attention → Dual-Head → Threshold → Output
EfficientNet-B5 + QENN-RY (Quantum) →
```

**Expected Performance:**
- Accuracy: 86-88% (B5 upgrade)
- Sensitivity: 89-91%
- Specificity: 80-84%
- Training Time: ~180 min (+20% for B5)

---

## Model 5: CB-QCCF Variant 2 (Swin + ConvNeXt + Quantum)

```bash
python3 run_pipeline.py --models cb_qccf_swin_convnet
```

**Architecture:**
```
Swin-Small (Classical) →
                    → Cross-Attention → Dual-Head → Threshold → Output
ConvNeXt-Small + QENN-RY (Quantum) →
```

**Expected Performance:**
- Accuracy: 85-87%
- Sensitivity: 88-91%
- Specificity: 80-84%
- Training Time: ~160 min

---

# 🎯 Training Commands

## Option A: Run All 5 Sequentially (Recommended)

```bash
# Create run script
cat > run_all_5_quantum.sh << 'EOF'
#!/bin/bash
echo "=========================================="
echo "Training 5 Quantum-Enhanced Models"
echo "=========================================="

# Model 1: TBCA + Quantum Fusion
echo "[1/5] TBCA-Fusion with Quantum Fusion (Placement 3)"
nohup python3 run_pipeline.py --models triple_branch_fusion_quantum > training_tbca_quantum_fusion.log 2>&1 &
wait

# Model 2: TBCA + Quantum Bottleneck
echo "[2/5] TBCA-Fusion with Quantum Bottleneck (Placement 2)"
nohup python3 run_pipeline.py --models triple_branch_fusion_bottleneck > training_tbca_quantum_bottleneck.log 2>&1 &
wait

# Model 3: CB-QCCF Original
echo "[3/5] CB-QCCF Original (Swin + ResNet-18 + Quantum)"
nohup python3 run_pipeline.py --models cb_qccf > training_cbqccf_original.log 2>&1 &
wait

# Model 4: CB-QCCF Variant 1
echo "[4/5] CB-QCCF Variant 1 (ConvNeXt + EfficientNet-B5 + Quantum)"
nohup python3 run_pipeline.py --models cb_qccf_convnet_efficient > training_cbqccf_convnet_efficient.log 2>&1 &
wait

# Model 5: CB-QCCF Variant 2
echo "[5/5] CB-QCCF Variant 2 (Swin + ConvNeXt + Quantum)"
nohup python3 run_pipeline.py --models cb_qccf_swin_convnet > training_cbqccf_swin_convnet.log 2>&1 &
wait

echo "=========================================="
echo "All 5 models completed!"
echo "=========================================="
EOF

chmod +x run_all_5_quantum.sh
./run_all_5_quantum.sh
```

## Option B: Run All 5 in Parallel (Faster, More GPU Memory)

```bash
# Run all 5 in parallel (requires ~60GB GPU memory)
nohup python3 run_pipeline.py --models \
  triple_branch_fusion_quantum \
  triple_branch_fusion_bottleneck \
  cb_qccf \
  cb_qccf_convnet_efficient \
  cb_qccf_swin_convnet \
  > training_all_5_quantum.log 2>&1 &

# Monitor
tail -f training_all_5_quantum.log
```

## Option C: Run Individually (Most Control)

```bash
# Run one at a time
nohup python3 run_pipeline.py --models triple_branch_fusion_quantum > training_1.log 2>&1 &
nohup python3 run_pipeline.py --models triple_branch_fusion_bottleneck > training_2.log 2>&1 &
nohup python3 run_pipeline.py --models cb_qccf > training_3.log 2>&1 &
nohup python3 run_pipeline.py --models cb_qccf_convnet_efficient > training_4.log 2>&1 &
nohup python3 run_pipeline.py --models cb_qccf_swin_convnet > training_5.log 2>&1 &
```

---

# 📊 Expected Results Summary

| Model | Accuracy | Sensitivity | Specificity | FNR | Training Time |
|-------|----------|-------------|-------------|-----|---------------|
| **TBCA + Quantum Fusion** | 88.0-89.0% | 92-93% | 80-83% | 6-7% | ~360-375 min |
| **TBCA + Quantum Bottleneck** | 88.0-89.0% | 92-93% | 80-83% | 6-7% | ~385-415 min |
| **CB-QCCF Original** | 85-87% | 88-91% | 80-84% | 8-9% | ~150 min |
| **CB-QCCF ConvNet+EffB5** | 86-88% | 89-91% | 80-84% | 8-9% | ~180 min |
| **CB-QCCF Swin+ConvNet** | 85-87% | 88-91% | 80-84% | 8-9% | ~160 min |

---

# 🔍 Research Questions to Answer

## Primary Questions:

1. **Which quantum placement is better for TBCA-Fusion?**
   - Placement 3 (Fusion) vs Placement 2 (Bottleneck)
   - Hypothesis: Placement 3 will have better accuracy/time ratio

2. **Which backbone combination works best for CB-QCCF?**
   - Swin+ResNet vs ConvNet+EffB5 vs Swin+ConvNet
   - Hypothesis: ConvNet+EffB5 will achieve highest accuracy

3. **Does quantum improve specificity?**
   - All CB-QCCF variants target 80%+ specificity (vs 72% baseline)
   - Hypothesis: Dual-head + class-balanced loss will fix specificity problem

## Secondary Questions:

4. **Is EfficientNet-B5 worth the extra compute?**
   - Compare CB-QCCF ConvNet+EffB5 vs other variants
   - Expected: +1-2% accuracy for +20% training time

5. **Does quantum fusion outperform classical fusion?**
   - Compare TBCA-Quantum-Fusion vs classical TBCA-Fusion
   - Expected: +0.5-1.5% accuracy improvement

---

# 📈 Success Metrics

## Primary Success Criteria:

✅ **At least 1 model achieves ≥88% accuracy**  
✅ **All CB-QCCF variants achieve ≥80% specificity**  
✅ **Quantum placements show improvement over classical**  
✅ **Training completes without errors**  

## Secondary Success Criteria:

✅ **Placement 3 shows better efficiency than Placement 2**  
✅ **ConvNet+EffB5 shows best accuracy among CB-QCCF variants**  
✅ **All models converge within 50 epochs**  
✅ **No barren plateau issues**  

---

# 🎯 Post-Training Analysis Plan

After all 5 models complete training:

1. **Compare accuracy across all 5 models**
2. **Analyze specificity improvement in CB-QCCF variants**
3. **Compare training time vs accuracy trade-offs**
4. **Select best performer for paper submission**
5. **Run ablation studies on best model** (qubits, layers, entanglement)
6. **Generate comparison tables for MICCAI 2026 paper**

---

# 🚀 Ready to Train!

All 5 models are implemented, configured, and ready to run.

**Pull latest code on server:**
```bash
cd ~/Projects/Cancer/Breast_Cancer_Research
git pull
```

**Verify all models are registered:**
```bash
python3 -c "from run_pipeline import build_model; print('✅ All models registered')"
```

**Start training:**
```bash
./run_all_5_quantum.sh
```

**Expected completion:** ~18-20 hours from start

---

**Good luck with the training!** 🎯
