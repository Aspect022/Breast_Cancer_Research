# Multimodal Breast Cancer Research — Master Plan
## Extending TBCA-VQC Histopathology → Genomic + Multimodal Fusion

---

## 1. Project Overview

### What We Have (Published)
- **TBCA-CNN-FM-Quantum** on BreakHis: 95.48% accuracy, 99.06% sensitivity, 0.94% FNR
- Full codebase with 18 histopathology architectures benchmarked
- Pipeline: patient-level 5-fold CV, W&B tracking, AMP training

### What We Are Building (Extension Paper)
Three parallel workstreams that culminate in a late-fusion multimodal validation:

```
Workstream A: Genomic Model Zoo (new — GEO + TCGA-BRCA)
Workstream B: TBCA Improvements (quantum placement + new variants)
Workstream C: Late-Fusion Multimodal Validation (TCGA-BRCA paired)
```

### The Clinical Shift
| Dimension | Paper 1 (Done) | Paper 2 (Extension) |
|---|---|---|
| Task | Binary: benign vs malignant | pCR prediction + subtype |
| Data | BreakHis (patches, 7,909 images) | TCGA-BRCA + GEO TNBC (patients) |
| Modality | Histopathology only | Genomic only → Late fusion |
| Unit | Image patch | Patient-level |
| Architecture scope | 18 histopathology models | Genomic model zoo + fusion |

---

## 2. High-Level Architecture Vision

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL PIPELINE                          │
│                                                                 │
│  ┌──────────────────────┐    ┌──────────────────────────────┐   │
│  │   HISTOPATHOLOGY ARM │    │       GENOMIC ARM            │   │
│  │                      │    │                              │   │
│  │  Input: H&E patches  │    │  Input: RNA-seq / GEO        │   │
│  │  (224×224, BreakHis  │    │  (top-N genes, pathway-      │   │
│  │   or TCGA-BRCA WSI)  │    │   structured)                │   │
│  │                      │    │                              │   │
│  │  TBCA (improved) +   │    │  G-Baseline-MLP              │   │
│  │  New quantum variant │    │  G-PASNet (pathway)          │   │
│  │                      │    │  G-TabTransformer            │   │
│  │  → p_image (prob)    │    │  G-GCN-PPI                   │   │
│  │  → feat_image (768d) │    │  G-Quantum-MLP               │   │
│  └──────────┬───────────┘    │  G-CrossOmics                │   │
│             │                │  G-PathFormer-Lite            │   │
│             │                │  G-Multi-Scale-1D             │   │
│             │                │  G-TNBC-DT-Hybrid            │   │
│             │                │  (+ more)                    │   │
│             │                │                              │   │
│             │                │  → p_genomic (prob)          │   │
│             │                │  → feat_genomic (Nd)         │   │
│             └────────────────┘                              │   │
│                      │                                      │   │
│              ┌───────▼────────┐                             │   │
│              │  LATE FUSION   │                             │   │
│              │  Meta-learner  │                             │   │
│              │  (MLP / XGB /  │                             │   │
│              │   quantum)     │                             │   │
│              └───────┬────────┘                             │   │
│                      │                                      │   │
│               Final prediction                              │   │
│         (pCR / subtype / survival)                          │   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. File Map

| File | Contents |
|---|---|
| `01_DATASET_PLAN.md` | All datasets, download instructions, preprocessing |
| `02_GENOMIC_MODELS.md` | All 12 genomic architectures, configs, training plan |
| `03_TBCA_QUANTUM_ANALYSIS.md` | Quantum placement analysis + 4 new TBCA variants |
| `04_MULTIMODAL_FUSION.md` | Late fusion strategy + validation plan |

---

## 4. Compute Plan (A100 Server)

All genomic models can run in parallel. Suggested order:

**Week 1–2:** Genomic baselines (G1–G4), dataset setup, preprocessing  
**Week 2–3:** Advanced genomic (G5–G9), quantum genomic (G10–G12)  
**Week 3–4:** New TBCA quantum variants (B1–B4)  
**Week 4–5:** Late fusion experiments on TCGA-BRCA paired  
**Week 5–6:** Ablations, calibration tests, paper writing  

---

## 5. Key Design Principles

1. **Mirror histopathology structure** — same paradigm progression (baseline → fusion → quantum) in genomics
2. **Quantum consistency** — VQC design (8-qubit, RY + CNOT ring) stays the same across modalities
3. **Biological grounding** — pathway-structured models are primary; pure black-box models are ablations
4. **Clinical relevance** — report Brier score + calibration alongside AUC/accuracy
5. **Reproducibility** — fixed seeds, patient-level splits, W&B tracking for all runs
