# Issues & Limitations - Breast Cancer Classification Project

**Generated:** 2026-03-19  
**Status:** Active tracking document

---

## 🔴 Critical Issues

### 1. SpikingCNN-LIF Clinically Unusable
- **Problem:** 62.8% False Negative Rate (misses cancer in ~63% of malignant cases)
- **Metrics:** 97.5% specificity but only 30.1% sensitivity
- **Impact:** Dangerous for cancer diagnosis - high false negative rate
- **Root Cause:** 
  - LIF neurons with rate encoding not suitable for medical imaging
  - Native SNN training unstable
- **Potential Fixes:**
  - Try ALIF neurons with spike-frequency adaptation
  - Consider ANN-to-SNN conversion approach
  - Tune threshold and beta hyperparameters
  - Different surrogate gradient functions
- **Decision:** ⚠️ **Consider removing from project** (per user decision 2026-03-19)

### 2. Quantum Model Training Time Prohibitive
- **Problem:** 48,284s (~13.4 hours) for 5-fold CV vs ~10 minutes for CNN
- **Impact:** 82× slower than baseline - not scalable for research iteration
- **Root Cause:** PennyLane CPU simulation bottleneck (`lightning.qubit` device)
- **Potential Fixes:**
  - Use PennyLane's `lightning.gpu` if available
  - Migrate to Qiskit with GPU acceleration
  - Implement custom quantum layer with direct PyTorch integration
  - Reduce circuit depth or qubit count
  - Use gradient-free optimization for quantum parameters
- **Decision:** ⚠️ **Migrate to GPU-accelerated quantum implementation** (per user decision 2026-03-19)

### 3. No Statistical Significance Testing
- **Problem:** Results reported as mean ± std but no p-values or confidence intervals
- **Impact:** Cannot claim one model is statistically better than another
- **Missing:**
  - Paired t-tests between models
  - Wilcoxon signed-rank tests
  - Confidence intervals for all metrics
  - Effect size calculations (Cohen's d)
- **Priority:** High for research publication

---

## 🟡 Code Quality Issues

### 4. Missing Data File in Repository
- **Problem:** `src/data/dataset.py` is git-ignored but critical for understanding
- **Impact:** Repository incomplete for external users
- **Fix:** Include file or document as external dependency

### 5. No Unit Tests
- **Problem:** No automated testing for data loading, metrics, or model forward passes
- **Impact:** Relies on manual sanity checks (`--subset` flag)
- **Missing:**
  - Data loader tests
  - Metrics computation tests
  - Model forward pass shape validation
  - Configuration validation tests

### 6. Hardcoded Paths
- **Problem:** Data directory must be manually configured
- **Impact:** Poor portability
- **Fix:** Add environment variable support

### 7. Limited Error Handling
- **Problem:** Quantum model falls back to classical simulation silently
- **Impact:** Debugging difficult
- **Fix:** Add proper error logging and config schema validation

---

## 🟠 Research Limitations

### 8. Single Dataset (BreakHis)
- **Problem:** No external validation on other histopathology datasets
- **Impact:** BreakHis has known biases (single institution, limited diversity)
- **Missing Datasets:**
  - Camelyon16
  - Camelyon17
  - BACH dataset
  - TCGA breast cancer histology

### 9. No Ablation Studies
- **Problem:** No systematic comparison of architectural choices
- **Missing:**
  - **Quantum:** Different qubit counts (4, 8, 16), layer depths, entanglement strategies
  - **SNN:** Different neuron types (only LIF implemented, not ALIF/AdEx as planned)
  - **Transformer:** Different ViT sizes, patch sizes, attention mechanisms
  - **Fusion:** Different fusion strategies in hybrid models

### 10. Missing Energy Efficiency Analysis
- **Problem:** SNN energy estimates mentioned in plan.md but not implemented
- **Impact:** Cannot claim SNN efficiency benefits
- **Missing:**
  - MAC vs AC operation counting for spiking model
  - Energy consumption estimates
  - FLOPs comparison across all models

### 11. No External Test Set Evaluation
- **Problem:** All results are from cross-validation
- **Impact:** No hold-out dataset from different institution for generalization testing
- **Fix:** Create institutional split or use external dataset

### 12. Multi-class Classification Not Run
- **Problem:** Config supports it (`task: "multi"`) but only binary results exist
- **Impact:** Plan mentions 3-class (IDC, ILC, Fibroadenoma) but no outputs generated
- **Priority:** Medium - depends on research goals

---

## 🟢 Enhancement Opportunities

### 13. Model Interpretability Missing
- **Problem:** No explainability analysis
- **Missing:**
  - Grad-CAM for CNN attention maps
  - Quantum feature visualization
  - Transformer attention weight analysis
  - Saliency maps
  - LIME/SHAP explanations

### 14. No Hyperparameter Optimization
- **Problem:** Manual tuning only
- **Impact:** Suboptimal performance
- **Tools:**
  - Optuna
  - Ray Tune
  - Weights & Biases Sweeps
- **Priority:** Especially for SNN (beta, threshold, num_steps) and Quantum (n_qubits, n_layers)

### 15. Clinical Deployment Preparation Missing
- **Problem:** No deployment optimization
- **Missing:**
  - Model quantization for edge deployment
  - Inference optimization (TensorRT, ONNX)
  - Integration with pathology workflow systems
  - Model compression (pruning, distillation)

### 16. No Ensemble Methods
- **Problem:** Single model predictions only
- **Impact:** Missing potential accuracy gains
- **Options:**
  - Cross-validation ensembles
  - Multi-model ensembles (CNN + Transformer + Quantum)
  - Test-time augmentation

---

## 📋 Issue Priority Matrix

| Priority | Issue ID | Issue Name | Effort | Impact |
|----------|----------|-----------|--------|--------|
| P0 | 1 | SpikingCNN-LIF Clinically Unusable | High | Critical |
| P0 | 2 | Quantum Model Training Time | High | Critical |
| P1 | 3 | No Statistical Significance Testing | Low | High |
| P1 | 8 | Single Dataset | Medium | High |
| P1 | 9 | No Ablation Studies | Medium | High |
| P2 | 13 | Model Interpretability Missing | Medium | Medium |
| P2 | 12 | Multi-class Classification Not Run | Low | Medium |
| P2 | 10 | Missing Energy Efficiency Analysis | Medium | Medium |
| P3 | 4 | Missing Data File | Low | Low |
| P3 | 5 | No Unit Tests | Medium | Low |
| P3 | 6 | Hardcoded Paths | Low | Low |
| P3 | 7 | Limited Error Handling | Low | Low |
| P3 | 14 | No Hyperparameter Optimization | High | Medium |
| P3 | 15 | Clinical Deployment Preparation | High | Low |
| P3 | 16 | No Ensemble Methods | Medium | Medium |

---

## ✅ Resolved Issues

| Date | Issue ID | Resolution |
|------|----------|------------|
| 2026-03-19 | 1 | **Decision:** Remove SNN from project focus |
| 2026-03-19 | 2 | **Decision:** Migrate to GPU-accelerated quantum (Qiskit/custom) |

---

## 📝 Notes

- This document was auto-generated from repo analysis on 2026-03-19
- User decision: Focus on Transformers + Quantum, remove SNN
- Reference implementation for quantum: `D:\Projects\AI-Projects\EEG`
- New direction: Dual-branch deep learning with dynamic feature fusion (Swin + ConvNeXt)
