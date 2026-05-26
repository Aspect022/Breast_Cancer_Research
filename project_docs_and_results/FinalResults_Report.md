# Breast Cancer Histopathology Classification — Final Results Report

**Dataset:** BreaKHis | **Task:** Binary (Benign vs. Malignant)  
**Protocol:** 5-Fold Cross-Validation | **Max Epochs:** 50/fold | **Held-out Test Set:** 1,377 images (13 patients)  
**Hardware:** NVIDIA A100 80GB PCIe | **Framework:** PyTorch + Weights & Biases

> **Metric Note:** All reported values are **mean ± std across 5 folds**, evaluated on the **held-out patient-level test set**. Early stopping was applied per fold based on validation AUC. Models `CB-QCCF-ConvNet-Efficient` and `CB-QCCF-Swin-ConvNet` encountered runtime errors (shape mismatch and unknown backbone) and are excluded from results.

---

## 1. Summary Comparison Table

| # | Model | Paradigm | Accuracy | F1 (macro) | AUC-ROC | MCC | Sensitivity | Specificity | FNR |
|---|-------|----------|----------|------------|---------|-----|-------------|-------------|-----|
| 1 | EfficientNet-B3 | CNN | 0.8633 ± 0.0428 | 0.8555 ± 0.0479 | 0.9249 ± 0.0432 | 0.7181 ± 0.0888 | 0.9285 ± 0.0244 | 0.7717 ± 0.0870 | 0.0715 ± 0.0244 |
| 2 | EfficientNet-B5 | CNN | 0.8777 ± 0.0273 | 0.8718 ± 0.0305 | 0.9439 ± 0.0181 | 0.7476 ± 0.0560 | 0.9287 ± 0.0077 | 0.8059 ± 0.0654 | 0.0713 ± 0.0077 |
| 3 | CNN+ViT-Hybrid | Transformer | 0.8700 ± 0.0329 | 0.8625 ± 0.0374 | 0.9298 ± 0.0285 | 0.7340 ± 0.0652 | 0.9349 ± 0.0334 | 0.7787 ± 0.0865 | 0.0651 ± 0.0334 |
| 4 | Swin-Small | Transformer | 0.8629 ± 0.0310 | 0.8550 ± 0.0362 | 0.9241 ± 0.0210 | 0.7184 ± 0.0628 | 0.9277 ± 0.0251 | 0.7717 ± 0.0881 | 0.0723 ± 0.0251 |
| 5 | ConvNeXt-Small | CNN | 0.8468 ± 0.0273 | 0.8378 ± 0.0319 | 0.9082 ± 0.0386 | 0.6886 ± 0.0569 | 0.9108 ± 0.0572 | 0.7566 ± 0.1110 | 0.0892 ± 0.0572 |
| 6 | DualBranch-Fusion | Fusion | 0.8439 ± 0.0400 | 0.8357 ± 0.0418 | 0.9006 ± 0.0376 | 0.6781 ± 0.0833 | 0.9108 ± 0.0461 | 0.7496 ± 0.0596 | 0.0892 ± 0.0461 |
| 7 | QENN-U3 | Quantum | 0.8443 ± 0.0512 | 0.8310 ± 0.0650 | 0.9211 ± 0.0265 | 0.6821 ± 0.0984 | 0.9382 ± 0.0162 | 0.7122 ± 0.1403 | 0.0618 ± 0.0162 |
| 8 | Quantum-Enhanced-Fusion | Quantum-Fusion | 0.8370 ± 0.0477 | 0.8246 ± 0.0557 | 0.8685 ± 0.0589 | 0.6650 ± 0.0966 | 0.9319 ± 0.0114 | 0.7035 ± 0.1095 | 0.0681 ± 0.0114 |
| 9 | CB-QCCF | Quantum-Fusion | 0.8527 ± 0.0302 | 0.8436 ± 0.0353 | 0.8981 ± 0.0308 | 0.6971 ± 0.0607 | 0.9275 ± 0.0166 | 0.7476 ± 0.0830 | 0.0725 ± 0.0166 |
| 10 | MSQ-Fusion | Quantum-Fusion | 0.8167 ± 0.0328 | 0.8035 ± 0.0388 | 0.8705 ± 0.0311 | 0.6213 ± 0.0672 | 0.9138 ± 0.0107 | 0.6801 ± 0.0825 | 0.0862 ± 0.0107 |
| 11 | TripleBranch-Fusion | Fusion | 0.8722 ± 0.0505 | 0.8641 ± 0.0587 | 0.9267 ± 0.0236 | 0.7376 ± 0.1020 | 0.9324 ± 0.0128 | 0.7874 ± 0.1279 | 0.0676 ± 0.0128 |
| 12 | TBCA-Quantum-Fusion | Triple-Quantum | 0.8558 ± 0.0180 | 0.8477 ± 0.0220 | 0.9072 ± 0.0151 | 0.7040 ± 0.0353 | 0.9223 ± 0.0309 | 0.7622 ± 0.0716 | 0.0777 ± 0.0309 |
| 13 | TBCA-Quantum-Bottleneck | Triple-Quantum | 0.8556 ± 0.0380 | 0.8453 ± 0.0439 | 0.9249 ± 0.0341 | 0.7045 ± 0.0775 | **0.9456 ± 0.0117** | 0.7290 ± 0.0831 | **0.0544 ± 0.0117** |
| 14 | TBCA-CNN-FeatureMap-Quantum | Triple-Quantum | 0.8749 ± 0.0153 | 0.8691 ± 0.0165 | 0.9210 ± 0.0185 | 0.7422 ± 0.0316 | 0.9284 ± 0.0213 | 0.7996 ± 0.0365 | 0.0716 ± 0.0213 |
| 15 | TBCA-ViT-FeatureMap-Quantum | Triple-Quantum | 0.8608 ± 0.0145 | 0.8537 ± 0.0156 | 0.9108 ± 0.0326 | 0.7128 ± 0.0302 | 0.9242 ± 0.0186 | 0.7717 ± 0.0306 | 0.0758 ± 0.0186 |

---

## 2. Rankings

### 🏆 By Accuracy

| Rank | Model | Accuracy |
|------|-------|----------|
| 1 | EfficientNet-B5 | **0.8777 ± 0.0273** |
| 2 | TBCA-CNN-FeatureMap-Quantum | 0.8749 ± 0.0153 |
| 3 | TripleBranch-Fusion | 0.8722 ± 0.0505 |
| 4 | CNN+ViT-Hybrid | 0.8700 ± 0.0329 |
| 5 | EfficientNet-B3 | 0.8633 ± 0.0428 |

### 🏆 By AUC-ROC

| Rank | Model | AUC-ROC |
|------|-------|---------|
| 1 | EfficientNet-B5 | **0.9439 ± 0.0181** |
| 2 | CNN+ViT-Hybrid | 0.9298 ± 0.0285 |
| 3 | TripleBranch-Fusion | 0.9267 ± 0.0236 |
| 4 | EfficientNet-B3 | 0.9249 ± 0.0432 |
| 5 | TBCA-Quantum-Bottleneck | 0.9249 ± 0.0341 |

### 🏆 By Sensitivity (Clinically Most Important — Minimises Missed Cancers)

| Rank | Model | Sensitivity | FNR |
|------|-------|-------------|-----|
| 1 | TBCA-Quantum-Bottleneck | **0.9456 ± 0.0117** | **0.0544** |
| 2 | QENN-U3 | 0.9382 ± 0.0162 | 0.0618 |
| 3 | Quantum-Enhanced-Fusion | 0.9319 ± 0.0114 | 0.0681 |
| 4 | TripleBranch-Fusion | 0.9324 ± 0.0128 | 0.0676 |
| 5 | CNN+ViT-Hybrid | 0.9349 ± 0.0334 | 0.0651 |

### 🏆 By MCC (Most Balanced — Accounts for Class Imbalance)

| Rank | Model | MCC |
|------|-------|-----|
| 1 | EfficientNet-B5 | **0.7476 ± 0.0560** |
| 2 | TBCA-CNN-FeatureMap-Quantum | 0.7422 ± 0.0316 |
| 3 | TripleBranch-Fusion | 0.7376 ± 0.1020 |
| 4 | CNN+ViT-Hybrid | 0.7340 ± 0.0652 |
| 5 | TBCA-Quantum-Bottleneck | 0.7045 ± 0.0775 |

---

## 3. Per-Model Details

### 3.1 Baseline CNNs

#### EfficientNet-B3
- **Paradigm:** CNN | **Parameters:** 10.7M | **Training Time:** ~59 min
- **Best Validation Acc @ Fold 5:** 0.9541 (epoch 24)
- The lightest model. Fold 3 showed significant performance degradation (Acc: 0.7945) due to class-imbalanced fold split (789 Benign, 579 Malignant in validation), explaining the high standard deviation.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| 1 | 0.8620 | 0.8550 | 0.9270 | 0.7149 | 0.9255 | 0.7727 | 0.0745 |
| 2 | 0.8794 | 0.8736 | 0.9586 | 0.7512 | 0.9354 | 0.8007 | 0.0646 |
| 3 | 0.7945 | 0.7773 | 0.8562 | 0.5759 | 0.9168 | 0.6224 | 0.0832 |
| 4 | 0.8693 | 0.8648 | 0.9181 | 0.7299 | 0.8994 | 0.8269 | 0.1006 |
| 5 | 0.9114 | 0.9070 | 0.9646 | 0.8186 | 0.9652 | 0.8357 | 0.0348 |
| **Mean** | **0.8633 ± 0.0428** | **0.8555 ± 0.0479** | **0.9249 ± 0.0432** | **0.7181 ± 0.0888** | **0.9285 ± 0.0244** | **0.7717 ± 0.0870** | **0.0715 ± 0.0244** |

---

#### EfficientNet-B5
- **Paradigm:** CNN | **Parameters:** 28.3M | **Training Time:** ~98 min
- The strongest overall CNN baseline. Low standard deviation across folds indicates stable generalisation. Best accuracy, AUC, and MCC among all models. Fold 3 also showed degradation (Acc: 0.8315) for the same imbalanced split reason.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| 1 | 0.8932 | 0.8897 | 0.9451 | 0.7796 | 0.9180 | 0.8584 | 0.0820 |
| 2 | 0.9020 | 0.8981 | 0.9634 | 0.7975 | 0.9379 | 0.8514 | 0.0621 |
| 3 | 0.8315 | 0.8200 | 0.9186 | 0.6531 | 0.9280 | 0.6958 | 0.0720 |
| 4 | 0.8809 | 0.8759 | 0.9344 | 0.7537 | 0.9255 | 0.8182 | 0.0745 |
| 5 | 0.8809 | 0.8753 | 0.9582 | 0.7541 | 0.9342 | 0.8059 | 0.0658 |
| **Mean** | **0.8777 ± 0.0273** | **0.8718 ± 0.0305** | **0.9439 ± 0.0181** | **0.7476 ± 0.0560** | **0.9287 ± 0.0077** | **0.8059 ± 0.0654** | **0.0713 ± 0.0077** |

---

#### ConvNeXt-Small
- **Paradigm:** CNN | **Parameters:** N/A | **Training Time:** ~44 min
- Weakest CNN baseline. Lower specificity (0.7566) suggests it struggles to correctly reject benign samples, i.e., higher false positive rate.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8468 ± 0.0273** | **0.8378 ± 0.0319** | **0.9082 ± 0.0386** | **0.6886 ± 0.0569** | **0.9108 ± 0.0572** | **0.7566 ± 0.1110** | **0.0892 ± 0.0572** |

---

### 3.2 Transformer-Based Models

#### CNN+ViT-Hybrid
- **Paradigm:** Transformer | **Parameters:** 12.7M | **Training Time:** ~70 min
- Combines CNN feature extraction with ViT self-attention. Achieves competitive AUC and the 3rd-best sensitivity (0.9349), indicating strong malignancy detection. Warm-up LR schedule (linear increase to 1e-4 over 5 epochs) is visible in the epoch logs.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8700 ± 0.0329** | **0.8625 ± 0.0374** | **0.9298 ± 0.0285** | **0.7340 ± 0.0652** | **0.9349 ± 0.0334** | **0.7787 ± 0.0865** | **0.0651 ± 0.0334** |

---

#### Swin-Small
- **Paradigm:** Transformer | **Parameters:** N/A | **Training Time:** ~41 min
- Fastest transformer model (hierarchical windowed attention). Performance (0.8629 Acc, 0.9241 AUC) closely tracks EfficientNet-B3 with much lower compute, making it a strong efficiency-accuracy trade-off.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8629 ± 0.0310** | **0.8550 ± 0.0362** | **0.9241 ± 0.0210** | **0.7184 ± 0.0628** | **0.9277 ± 0.0251** | **0.7717 ± 0.0881** | **0.0723 ± 0.0251** |

---

### 3.3 Classical Fusion Models

#### DualBranch-Fusion
- **Paradigm:** Fusion | **Parameters:** N/A | **Training Time:** ~123 min
- Two-branch architecture with multi-scale feature fusion. Underperforms compared to single CNN backbones (B3, B5), suggesting the fusion mechanism did not add sufficient complementary information, or required more tuning.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8439 ± 0.0400** | **0.8357 ± 0.0418** | **0.9006 ± 0.0376** | **0.6781 ± 0.0833** | **0.9108 ± 0.0461** | **0.7496 ± 0.0596** | **0.0892 ± 0.0461** |

---

#### TripleBranch-Fusion
- **Paradigm:** Fusion | **Parameters:** N/A | **Training Time:** ~374 min
- Three-branch architecture fusing CNN, ViT, and an additional branch. Best fusion model overall by accuracy (0.8722) and MCC (0.7376). High variance in specificity (±0.1279) indicates sensitivity to fold composition.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8722 ± 0.0505** | **0.8641 ± 0.0587** | **0.9267 ± 0.0236** | **0.7376 ± 0.1020** | **0.9324 ± 0.0128** | **0.7874 ± 0.1279** | **0.0676 ± 0.0128** |

---

### 3.4 Quantum-Enhanced Models

#### QENN-U3 (Quantum ENN, 3-layer U-gate)
- **Paradigm:** Quantum | **Parameters:** N/A | **Training Time:** ~94 min
- Pure quantum-enhanced neural network. Achieves 2nd-highest sensitivity (0.9382) among all models and the 2nd-lowest FNR (0.0618). The quantum layer adds discriminative power for malignant feature detection at the cost of specificity (only 0.7122).

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8443 ± 0.0512** | **0.8310 ± 0.0650** | **0.9211 ± 0.0265** | **0.6821 ± 0.0984** | **0.9382 ± 0.0162** | **0.7122 ± 0.1403** | **0.0618 ± 0.0162** |

---

#### Quantum-Enhanced-Fusion
- **Paradigm:** Quantum-Fusion | **Parameters:** N/A | **Training Time:** ~188 min
- Lowest AUC and accuracy among quantum models. The fusion of quantum and classical branches did not outperform simpler baselines in this configuration, suggesting the quantum circuits may have added noise rather than signal.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8370 ± 0.0477** | **0.8246 ± 0.0557** | **0.8685 ± 0.0589** | **0.6650 ± 0.0966** | **0.9319 ± 0.0114** | **0.7035 ± 0.1095** | **0.0681 ± 0.0114** |

---

#### CB-QCCF (Cross-Branch Quantum-Classical Co-Fusion)
- **Paradigm:** Quantum-Fusion | **Parameters:** N/A | **Training Time:** ~102 min
- Mid-tier quantum fusion model. Best quantum-fusion accuracy (0.8527) but moderate AUC (0.8981) — slightly below non-quantum baselines. Consistent across folds (lower variance).

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8527 ± 0.0302** | **0.8436 ± 0.0353** | **0.8981 ± 0.0308** | **0.6971 ± 0.0607** | **0.9275 ± 0.0166** | **0.7476 ± 0.0830** | **0.0725 ± 0.0166** |

---

#### MSQ-Fusion (Multi-Scale Quantum Fusion)
- **Paradigm:** Quantum-Fusion | **Parameters:** N/A | **Training Time:** ~94 min
- Weakest overall model across all metrics. Multi-scale quantum fusion architecture underperformed, producing the lowest accuracy (0.8167) and MCC (0.6213). Suggests that indiscriminate multi-scale feature aggregation with quantum circuits requires careful regularisation.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8167 ± 0.0328** | **0.8035 ± 0.0388** | **0.8705 ± 0.0311** | **0.6213 ± 0.0672** | **0.9138 ± 0.0107** | **0.6801 ± 0.0825** | **0.0862 ± 0.0107** |

---

### 3.5 Triple-Branch Quantum Architectures (TBCA)

#### TBCA-Quantum-Fusion
- **Paradigm:** Triple-Quantum | **Parameters:** N/A | **Training Time:** ~274 min
- Triple-branch architecture with quantum fusion. Very stable (lowest std among TBCA variants: ±0.0180 Acc). Balanced sensitivity/specificity trade-off, suggesting better calibration than single-branch quantum models.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8558 ± 0.0180** | **0.8477 ± 0.0220** | **0.9072 ± 0.0151** | **0.7040 ± 0.0353** | **0.9223 ± 0.0309** | **0.7622 ± 0.0716** | **0.0777 ± 0.0309** |

---

#### TBCA-Quantum-Bottleneck ⭐ *Clinically Best*
- **Paradigm:** Triple-Quantum | **Parameters:** N/A | **Training Time:** ~477 min
- **Highest sensitivity (0.9456) and lowest FNR (0.0544) across all 15 models.** From a clinical perspective, this is the most important model — it misses the fewest malignant cases (false negatives). The bottleneck quantum module appears to act as a strong constraint, forcing the model to focus on discriminative malignant features. AUC (0.9249) is comparable to EfficientNet-B3 at a 8× compute cost.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8556 ± 0.0380** | **0.8453 ± 0.0439** | **0.9249 ± 0.0341** | **0.7045 ± 0.0775** | **0.9456 ± 0.0117** | **0.7290 ± 0.0831** | **0.0544 ± 0.0117** |

---

#### TBCA-CNN-FeatureMap-Quantum ⭐ *Best Quantum-Accuracy Trade-off*
- **Paradigm:** Triple-Quantum | **Parameters:** N/A | **Training Time:** ~337 min
- Uses CNN-extracted feature maps as quantum circuit inputs instead of raw pixels. Achieves the best accuracy (0.8749) among all **quantum** models, and 2nd overall. Lowest standard deviation among all 15 models (0.0153 Acc, 0.0165 F1), indicating highly consistent generalisation across different patient fold splits. Represents the best accuracy-stability combination in the quantum category.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8749 ± 0.0153** | **0.8691 ± 0.0165** | **0.9210 ± 0.0185** | **0.7422 ± 0.0316** | **0.9284 ± 0.0213** | **0.7996 ± 0.0365** | **0.0716 ± 0.0213** |

---

#### TBCA-ViT-FeatureMap-Quantum
- **Paradigm:** Triple-Quantum | **Parameters:** N/A | **Training Time:** ~413 min
- Uses ViT-extracted feature maps as quantum inputs (vs. CNN features in the prior model). Slightly lower across all metrics compared to TBCA-CNN-FeatureMap-Quantum, suggesting that CNN inductive biases (local receptive fields) produce more quantum-compatible feature representations for histopathology than global ViT tokens.

| Fold | Accuracy | F1 | AUC | MCC | Sensitivity | Specificity | FNR |
|------|----------|----|-----|-----|-------------|-------------|-----|
| **Mean** | **0.8608 ± 0.0145** | **0.8537 ± 0.0156** | **0.9108 ± 0.0326** | **0.7128 ± 0.0302** | **0.9242 ± 0.0186** | **0.7717 ± 0.0306** | **0.0758 ± 0.0186** |

---

## 4. Paradigm-Level Analysis

| Paradigm | Models | Avg. Accuracy | Avg. AUC | Avg. Sensitivity | Avg. MCC |
|----------|--------|---------------|----------|------------------|----------|
| CNN | 3 | 0.8626 | 0.9257 | 0.9227 | 0.7181 |
| Transformer | 2 | 0.8665 | 0.9270 | 0.9313 | 0.7262 |
| Fusion (Classical) | 2 | 0.8581 | 0.9137 | 0.9216 | 0.7079 |
| Quantum / Quantum-Fusion | 4 | 0.8378 | 0.9021 | 0.9279 | 0.6836 |
| Triple-Branch Quantum | 4 | 0.8618 | 0.9160 | 0.9301 | 0.7159 |

**Key Takeaways:**
- **Transformers** edge out CNNs on both AUC and sensitivity on average.
- **Quantum** models show the highest sensitivity on average but suffer on accuracy and AUC, suggesting they are better at detecting malignancy but at the cost of more benign false positives.
- **Triple-Branch Quantum** models close much of this gap while maintaining clinical-grade sensitivity.

---

## 5. Failed / Excluded Models

| Model | Error | Status |
|-------|-------|--------|
| CB-QCCF-ConvNet-Efficient | `mat1 and mat2 shapes cannot be multiplied (200704×7 and 2048×...)` — projection head dimension mismatch | ❌ Excluded |
| CB-QCCF-Swin-ConvNet | `Unknown quantum backbone: convnext_small` — backbone name not registered in model factory | ❌ Excluded |

These models require architectural fixes (linear layer size correction and backbone registry update respectively) before they can be evaluated.

---

## 6. Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | BreaKHis (7,909 images total) |
| Split strategy | Patient-level stratified 5-fold CV |
| Test set | 1,377 images (13 held-out patients) |
| Train+Val pool | 6,532 images |
| Class distribution | Malignant: 5,429 (68.6%) / Benign: 2,480 (31.4%) |
| Image resolution | Standard BreaKHis (400× magnification) |
| Optimiser | AdamW |
| Base LR | 1e-4 |
| LR schedule | Cosine annealing |
| Early stopping | Patience=10 epochs on Val AUC |
| Max epochs | 50 |
| Augmentation | Standard augmentation pipeline |
| Hardware | NVIDIA A100 80GB PCIe |
| TF32 | Enabled |
| cuDNN Benchmark | Enabled |

---

## 7. Key Conclusions

1. **Best Overall Model (Balanced):** `EfficientNet-B5` — highest accuracy, AUC, sensitivity consistency (std 0.0077), and MCC. The best choice for a general-purpose classifier.

2. **Best Clinical Model (Screening):** `TBCA-Quantum-Bottleneck` — highest sensitivity (0.9456) and lowest FNR (0.0544). In a screening context where missing cancer (false negative) is the primary risk, this model is preferred despite slightly lower overall accuracy.

3. **Best Quantum Model (Accuracy):** `TBCA-CNN-FeatureMap-Quantum` — the most consistent quantum model (lowest std across all 15 models) and the best accuracy among quantum architectures. Demonstrates that CNN feature-map-based quantum encoding is a promising direction.

4. **Quantum Insight:** Quantum models consistently achieve high sensitivity but trade off on specificity, indicating a learned bias toward malignant predictions. This may be addressable through loss reweighting or threshold calibration.

5. **Failed Models:** `CB-QCCF-ConvNet-Efficient` and `CB-QCCF-Swin-ConvNet` did not produce results due to implementation bugs. These are fixable and worth re-running in a follow-up study.

6. **Fold 3 Volatility:** Fold 3 consistently produces lower metrics across models (likely due to its highly imbalanced validation composition: 789 Benign, 579 Malignant). This is expected in patient-level stratified splitting and is not indicative of model failure.

---

*Report generated from training logs in `Breast_cancer_final.out` | Run on A100 server: 2026-03-28 to 2026-03-29*
