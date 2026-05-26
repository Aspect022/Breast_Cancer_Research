# 🎯 Project Planning Summary

**Date:** 2026-03-19  
**Decision:** Transform breast cancer classification project into focused research on Transformers + Quantum

---

## 📁 Documents Created

| Document | Purpose | Status |
|----------|---------|--------|
| **issues.md** | Track all identified issues and limitations | ✅ Created |
| **research_prompt.md** | Comprehensive research questions for architecture exploration | ✅ Created |
| **implementation_plan.md** | 10-week phased implementation schedule | ✅ Created |
| **PLANNING_SUMMARY.md** | This document - quick reference | ✅ Created |

---

## 🎯 Strategic Decisions

### What We're Removing
- ❌ **Spiking Neural Networks** - Clinically unusable (62.8% false negative rate)
- ❌ **PennyLane CPU quantum** - Too slow (13.4 hours for 5-fold CV)

### What We're Adding
- ✅ **Swin Transformer** (Tiny, Small, V2) - Hierarchical attention
- ✅ **ConvNeXt** (Tiny, Small) - Convolutional approach to transformers
- ✅ **DeiT** (Tiny, Small, Base) - Data-efficient transformers
- ✅ **Dual-Branch Dynamic Fusion** - Swin + ConvNeXt with learned gating
- ✅ **GPU-Accelerated Quantum** - Custom vectorized or TorchQuantum (100× speedup target)
- ✅ **Medical-Specific Architectures** - CrossViT, Med-Former (time permitting)

### What We're Testing
- ✅ **Quantum Rotation Gates:** RY-only, RY+RZ, U3, RX+RY+RZ
- ✅ **Entanglement Strategies:** Cyclic CNOT (primary), full, tree
- ✅ **Fusion Mechanisms:** Gating, cross-attention, dynamic convolution
- ✅ **Fusion Timing:** Early, mid, late, multi-stage

---

## 🎯 Targets

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| **Accuracy** | 85% (EfficientNet-B3) | ≥88% | +3% |
| **Quantum Training Time** | 13.4 hours | <2 hours | 6-7× faster |
| **Vectorized Quantum Forward** | ~500ms | <10ms | 50-100× faster |

---

## 📅 Timeline Overview

```
Week 1-2: Phase 1 - Quick Wins + Foundation
├── Swin-T, ConvNeXt-T standalone
├── Vectorized quantum circuit
└── Timing benchmarks

Week 3-5: Phase 2 - Core New Architectures
├── Full transformer zoo (Swin-T/S/V2, ConvNeXt-T/S, DeiT-T/S/B)
├── Dual-branch fusion implementation
└── Fusion achieves ≥87% accuracy

Week 6-8: Phase 3 - Advanced Fusion + Quantum
├── Quantum rotation gate ablation (4 variants)
├── Entanglement strategy comparison
├── Quantum-enhanced fusion
└── ≥88% accuracy, <2 hour training

Week 9-10: Phase 4 - Ablation + Polish
├── Statistical significance testing
├── Comprehensive ablation studies
├── Interpretability visualizations
└── Paper draft
```

---

## 🔧 Immediate Next Steps

### Today
1. Install dependencies: `pip install timm`
2. Clone TorchQuantum: `git clone https://github.com/mit-han-lab/torchquantum`
3. Review EEG project quantum implementation: `D:\Projects\AI-Projects\EEG`
4. Start Swin-T implementation using timm

### This Week (Week 1)
- [ ] Vectorized quantum circuit working on GPU (<10ms forward pass)
- [ ] Swin-T achieving ≥80% on single fold
- [ ] ConvNeXt-T achieving ≥80% on single fold
- [ ] Timing benchmark: PennyLane vs vectorized

### Week 2 Checkpoint
- [ ] 5-fold CV results for Swin-T and ConvNeXt-T
- [ ] Decision: TorchQuantum vs custom vectorized
- [ ] Begin fusion model implementation

---

## 📊 Model Zoo

### Standalone Transformers
| Model | Variants | Params | Target Acc | Target Time |
|-------|----------|--------|------------|-------------|
| Swin | Tiny, Small, V2 | 29M, 50M, 50M | 83%, 85%, 86% | 30, 45, 50 min |
| ConvNeXt | Tiny, Small | 29M, 50M | 83%, 84% | 25, 40 min |
| DeiT | Tiny, Small, Base | 5.7M, 22M, 87M | 78%, 82%, 84% | 20, 30, 50 min |

### Fusion Models
| Model | Backbones | Target Acc | Target Time |
|-------|-----------|------------|-------------|
| Dual-Branch Fusion | Swin-T + ConvNeXt-T | 87% | 45 min |
| Quantum-Enhanced Fusion | Swin-T + ConvNeXt-T + VQC | 88% | 60 min |

### Quantum Variants
| Variant | Rotation Gates | Params/Qubit/Layer | Target Acc |
|---------|---------------|-------------------|------------|
| RY-only | RY | 1 | 84% |
| RY+RZ | RY, RZ | 2 | 85% |
| U3 | U3(θ, φ, λ) | 3 | 86% |
| RX+RY+RZ | RX, RY, RZ | 3 | 85% |

---

## ⚠️ Key Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| TorchQuantum doesn't work on Windows | Use custom vectorized implementation (EEG project reference) |
| Fusion doesn't improve accuracy | Fallback to ensemble averaging |
| Quantum still too slow | Reduce qubits (8→4), reduce layers (2→1) |
| Gating network collapses | Add entropy regularization (weight=0.01) |

---

## 📚 Key References

### Code Libraries
- **timm:** `pip install timm` - Swin, ConvNeXt, DeiT implementations
- **TorchQuantum:** MIT Han Lab GPU-accelerated quantum
- **Custom Vectorized:** Reference EEG project implementation

### Datasets
- **BreakHis:** Primary dataset (current)
- **Camelyon16/17:** External validation (optional, Phase 4)
- **BACH:** Alternative external dataset

### Target Venues
- **MICCAI 2026:** March deadline, ~30% acceptance
- **ISBI 2026:** October deadline, ~50% acceptance
- **Medical Image Analysis:** Journal, 3-4 month review
- **IEEE TMI:** Journal, 3-4 month review

---

## 🎯 Success Criteria

### Minimum Viable Results (Workshop Paper)
- ✅ Dual-branch fusion achieves ≥87% accuracy
- ✅ Quantum ablation study complete (4 rotation variants)
- ✅ 5-fold CV results for all key models
- ✅ Grad-CAM visualizations

### Target Results (Conference Paper)
- ✅ Quantum-enhanced fusion achieves ≥88% accuracy
- ✅ Statistical significance testing (p-values)
- ✅ Comprehensive ablation studies
- ✅ External validation (optional)

### Stretch Goals (Journal Article)
- ✅ ≥90% accuracy
- ✅ Multi-class classification results
- ✅ Pathologist evaluation of attention maps
- ✅ Full external validation study

---

## 📞 Quick Reference

### File Locations
- **Issues:** `issues.md`
- **Research Questions:** `research_prompt.md`
- **Implementation Schedule:** `implementation_plan.md`
- **Original Plan:** `plan.md`
- **Code:** `src/`
- **Outputs:** `outputs/`
- **EEG Reference:** `D:\Projects\AI-Projects\EEG`

### Key Commands
```bash
# Install dependencies
pip install timm
git clone https://github.com/mit-han-lab/torchquantum.git

# Run training (example)
python run_pipeline.py --model swin_tiny --fold 0

# Run quantum benchmark
python benchmarks/quantum_timing.py
```

---

**Ready to begin! 🚀**

Next action: Install dependencies and start Swin-T implementation.
