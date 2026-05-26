# Next Steps - Breast Cancer Classification Project

**Date:** March 21, 2026  
**Current Status:** Ready for CBIS-DDSM validation & quantum architecture implementation

---

## 📊 Current Results Summary (BreakHis Dataset)

### Top Performers

| Rank | Model | Accuracy | AUC-ROC | Sensitivity | Specificity |
|------|-------|----------|---------|-------------|-------------|
| 1 | **Swin-Small** | **85.67%** | 88.52% | 91.15% | 77.45% |
| 2 | **CNN+ViT-Hybrid** | **84.67%** | 89.25% | 88.49% | 78.92% |
| 3 | **EfficientNet-B3** | 83.37% | **89.97%** | 86.10% | 79.26% |
| 4 | **QENN-RY_ONLY** | 84.17% | 88.40% | **91.68%** | 72.88% |

### Key Findings

✅ **Swin-Small** achieves best accuracy (85.67%)  
✅ **Quantum models** have excellent sensitivity (91%+) but poor specificity (72%)  
⚠️ **Specificity gap** across all models (need >85%, currently ~77%)  
⚠️ **Fusion underperforming** - DualBranch-Fusion (83.68%) < Swin-Small (85.67%)

---

## 🎯 Immediate Actions (This Week)

### 1. Validate on CBIS-DDSM Dataset ⭐⭐⭐

**Goal:** Test generalization across imaging modalities

**Steps:**
```bash
# Set Kaggle credentials
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_api_key

# Run full pipeline
./run_cbis_ddsm.sh

# Or quick test
./run_cbis_ddsm.sh --quick
```

**Expected Timeline:** 6-8 hours for all models  
**Expected Results:** 75-82% accuracy (5-10% drop from BreakHis)

**Hypothesis:** Quantum models may generalize better due to quantum feature spaces

---

### 2. Implement CB-QCCF Architecture ⭐⭐⭐

**Priority:** P0 (Highest)  
**Novelty:** HIGH - First to address quantum specificity problem  
**Expected:** 85-87% accuracy, 80-83% specificity

**What:** Class-Balanced Quantum-Classical Fusion with:
- Dual-head prediction (sensitivity + specificity)
- Cross-attention fusion
- Class-balanced loss
- Learnable decision threshold

**Implementation Guide:** See `QUANTUM_ARCHITECTURES.md` for full code

**Timeline:** 2-3 days

---

### 3. Fix Swin-V2-Small Training ⭐⭐

**Issue:** Swin-V2-Small crashed during training (no results)  
**Cause:** Likely OOM with 256×256 images and batch_size=16

**Solution:**
```yaml
# In config.yaml
models:
  swin_v2_small:
    batch_size: 8  # Reduce from 16
    img_size: 224  # Reduce from 256
```

Then re-run:
```bash
python run_pipeline.py --models swin_v2_small
```

---

## 📅 Week-by-Week Implementation Plan

### Week 1-2: Foundation

- [x] Add CBIS-DDSM dataset loader
- [x] Create auto-download scripts
- [ ] **Run CBIS-DDSM validation** (all models)
- [ ] **Implement CB-QCCF architecture**
- [ ] Train CB-QCCF on BreakHis

**Deliverables:**
- CBIS-DDSM results table
- CB-QCCF implementation
- Initial accuracy/specificity metrics

### Week 3-4: Multi-Scale Quantum

- [ ] Implement Multi-Scale Quantum Fusion (MSQF)
- [ ] Train on BreakHis (5-fold CV)
- [ ] Train on CBIS-DDSM
- [ ] Quantum architecture ablation study

**Deliverables:**
- MSQF results
- Rotation gate ablation (RY vs RY+RZ vs U3)
- Entanglement ablation (cyclic vs full vs tree)
- Qubit count ablation (4 vs 8 vs 12)

### Week 5-7: Quantum Deformable Attention

- [ ] Implement Quantum Deformable Attention (QEDA)
- [ ] Replace Swin blocks with QEDA blocks
- [ ] Train full model
- [ ] Attention visualization

**Deliverables:**
- QEDA results
- Attention maps showing nuclei/tissue regions
- **Major novelty for MICCAI 2026**

### Week 8-9: Analysis & Writing

- [ ] Statistical significance testing
- [ ] Compare all architectures
- [ ] Write paper (MICCAI 2026 format)
- [ ] Prepare supplementary materials

**Deliverables:**
- Complete results table (18+ models × 2 datasets)
- Paper draft (8 pages, LNCS format)
- Code repository (anonymized for review)

### Week 10: Final Polish

- [ ] Paper revision
- [ ] Additional experiments (if needed)
- [ ] Submit to MICCAI 2026
- [ ] Prepare code for public release

---

## 🏗️ Architecture Implementation Priority

| Priority | Architecture | Novelty | Effort | Expected Gain | Status |
|----------|-------------|---------|--------|---------------|--------|
| **P0** | CB-QCCF | ⭐⭐⭐ High | 2-3 days | Fix specificity, 85-87% | ⏳ Pending |
| **P1** | TSD-Ensemble | ⭐⭐ Medium | 1-2 days | Quickest win, 87-89% | ⏳ Pending |
| **P2** | MSQF | ⭐⭐⭐ High | 3-4 days | Multi-scale features, 86-88% | ⏳ Pending |
| **P3** | TBCA-Fusion | ⭐⭐ Medium | 2-3 days | Strong baseline, 87-89% | ⏳ Pending |
| **P4** | QEDA | ⭐⭐⭐⭐ Very High | 5-7 days | Major novelty, 87-90% | ⏳ Pending |

**Recommendation:** Start with CB-QCCF (fixes specificity) → TSD-Ensemble (quick win) → MSQF (multi-scale) → QEDA (high risk/high reward)

---

## 📝 Paper Contributions (MICCAI 2026)

### Primary Contributions

1. **First systematic study** of quantum-classical fusion for breast cancer classification
   - 18 models across 4 paradigms (CNN, Transformer, Fusion, Quantum)
   - 5-fold cross-validation on 2 datasets (BreakHis, CBIS-DDSM)

2. **CB-QCCF architecture** solving the quantum specificity problem
   - Dual-head prediction (sensitivity + specificity)
   - Class-balanced loss function
   - Learnable decision threshold

3. **Multi-scale quantum fusion** for histopathology analysis
   - Parallel quantum circuits at different scales
   - Attention-based fusion

4. **Quantum deformable attention** (if implemented)
   - Novel attention mechanism
   - Replaces classical MLP with quantum circuit

5. **Comprehensive evaluation**
   - Statistical significance testing
   - Ablation studies (rotation, entanglement, qubits, layers)
   - Interpretability analysis (Grad-CAM, attention maps)

6. **Open-source code** for reproducibility

### Novelty Score: **HIGH**

| Aspect | Novelty | Justification |
|--------|---------|---------------|
| Quantum-Classical Fusion | ⭐⭐⭐ High | First in histopathology/mammography |
| Class-Balanced Loss | ⭐⭐⭐ High | Solves known quantum specificity problem |
| Multi-Scale Quantum | ⭐⭐⭐ High | First multi-scale quantum for medical imaging |
| Quantum Deformable Attn | ⭐⭐⭐⭐ Very High | New attention mechanism |
| CBIS-DDSM Evaluation | ⭐⭐ Medium | Standard generalization test |

---

## 🎓 Target Venues

### Primary: MICCAI 2026

- **Conference:** Medical Image Computing and Computer Assisted Intervention
- **Deadline:** April 2026 (typically)
- **Format:** 8-page LNCS format
- **Acceptance Rate:** ~30%
- **Impact:** Top venue for medical image analysis

### Backup: ISBI 2026

- **Conference:** IEEE International Symposium on Biomedical Imaging
- **Deadline:** October 2025 (typically)
- **Format:** 4-5 pages
- **Acceptance Rate:** ~50%
- **Impact:** Good venue, earlier deadline

### Journal Extension: IEEE TMI or Medical Image Analysis

- **Journal:** IEEE Transactions on Medical Imaging / Medical Image Analysis
- **Timeline:** 6-12 months review
- **Impact Factor:** 7-10
- **Extension:** Add more experiments, ablation studies, clinical validation

---

## 📊 Expected Results Table (Template)

```
Table 1: Performance comparison on BreakHis and CBIS-DDSM datasets.
Values show mean ± std across 5-fold cross-validation.
Best results in bold, second-best underlined.

┌────────────────────────────┬─────────────────┬─────────────────┬──────────────┐
│ Model                      │ BreakHis        │ CBIS-DDSM       │ Drop         │
│                            │ Acc    AUC      │ Acc    AUC      │ Acc   AUC    │
├────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ EfficientNet-B3            │ 83.4±3.6 89.0±2.2│ 76.8±4.1 81.2±3.5│ -6.6  -7.8   │
│ Swin-Small                 │ 85.7±2.3 88.5±1.7│ 78.2±3.8 82.1±2.9│ -7.5  -6.4   │
│ CNN+ViT-Hybrid             │ 84.7±3.0 89.3±2.0│ 77.5±4.2 82.8±3.2│ -7.2  -6.5   │
├────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ QENN-RY_ONLY               │ 84.2±3.2 88.4±2.4│ 79.5±3.5 83.8±2.8│ -4.7  -4.6   │
│ QENN-U3                    │ 84.1±2.7 88.1±2.1│ 79.8±3.2 84.2±2.5│ -4.3  -3.9   │
├────────────────────────────┼─────────────────┼─────────────────┼──────────────┤
│ CB-QCCF (Ours)             │ 86.5±2.1 89.8±1.8│ 81.2±3.0 85.6±2.4│ -5.3  -4.2   │
│ MSQF (Ours)                │ 87.2±1.9 90.5±1.6│ 82.1±2.8 86.3±2.1│ -5.1  -4.2   │
│ QEDA (Ours)                │ 88.5±1.7 91.2±1.5│ 83.5±2.5 87.1±1.9│ -5.0  -4.1   │
└────────────────────────────┴─────────────────┴─────────────────┴──────────────┘

Key Finding: Quantum models show smaller performance drop (-4 to -5%)
compared to classical models (-6 to -8%), suggesting better generalization.
```

---

## 🔧 Resources Needed

### Computational Resources

- **GPU:** A100 (40GB) or V100 (32GB) recommended
- **VRAM:** Minimum 16GB for fusion/quantum models
- **Time:** ~8-10 hours per dataset for full pipeline
- **Storage:** ~50GB for datasets + checkpoints

### Software

- Python 3.9+
- PyTorch 2.1+
- PennyLane (for quantum models)
- Weights & Biases (experiment tracking)
- Kaggle CLI (dataset download)

### Datasets

- **BreakHis:** Already downloaded (`data/BreaKHis_v1/`)
- **CBIS-DDSM:** Auto-downloaded by script (~10GB)

---

## 📚 Key References

### Papers

1. **Swin Transformer:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows", ICCV 2021
2. **ConvNeXt:** Liu et al., "A ConvNet for the 2020s", CVPR 2022
3. **Deformable Attention:** Dai et al., "Deformable DETR", ICLR 2021
4. **Quantum ML:** Schuld et al., "Quantum machine learning in feature Hilbert spaces", PRL 2019
5. **QENN:** Our implementation (`src/models/quantum/hybrid_quantum.py`)

### Datasets

1. **BreakHis:** Spanhol et al., "A dataset for breast cancer histopathological image classification", IEEE TBME 2016
2. **CBIS-DDSM:** Clark et al., "Curated Breast Imaging Subset of DDSM", SPIE 2013

---

## ✅ Checklist for MICCAI Submission

- [ ] All 18 models trained on BreakHis
- [ ] All models trained on CBIS-DDSM
- [ ] CB-QCCF implemented and trained
- [ ] MSQF implemented and trained
- [ ] QEDA implemented and trained (optional but recommended)
- [ ] Statistical significance testing completed
- [ ] Ablation studies completed
- [ ] Interpretability analysis (Grad-CAM, attention maps)
- [ ] Paper written (8 pages, LNCS format)
- [ ] Supplementary materials prepared
- [ ] Code anonymized and uploaded (for reviewers)
- [ ] Submit to MICCAI 2026

---

## 🚀 Quick Commands Reference

```bash
# CBIS-DDSM validation
./run_cbis_ddsm.sh                    # Full pipeline
./run_cbis_ddsm.sh --quick            # Quick test (200 samples)
./run_cbis_ddsm.sh --models swin_tiny # Specific models

# Individual model training
python run_pipeline.py --models qenn_ry
python run_pipeline.py --models cb_qccf  # After implementation

# Analysis
python scripts/analyze_results.py --save-summary

# Check results
cat outputs/comparison_cbis_ddsm_binary.csv
cat outputs/cbis_ddsm_generalization_comparison.csv
```

---

## 📞 Support & Documentation

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview |
| `IMPLEMENTATION_COMPLETE.md` | Implementation status |
| `QUANTUM_ARCHITECTURES.md` | Quantum architecture proposals |
| `PROPOSED_ARCHITECTURES.md` | All 5 proposed architectures |
| `CBIS_DDSM_SETUP.md` | CBIS-DDSM setup guide |
| `issues.md` | Known issues |
| `plan.md` | Original project plan |

---

**Ready to proceed? Start with:**
1. `./run_cbis_ddsm.sh` (validate on CBIS-DDSM)
2. Implement CB-QCCF (see `QUANTUM_ARCHITECTURES.md`)

**Good luck! 🎯**
