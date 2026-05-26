# Dataset Plan — Genomic + Multimodal

---

## 1. Primary Genomic Training Dataset

### GEO TNBC NAC Cohorts (pCR Prediction)

**Task: Binary — pCR (pathologic complete response) vs RD (residual disease)**

This is the primary task for genomic model training, directly parallel to your published TNBC-DT and personalized treatment papers from your group.

| Dataset | Platform | Patients | pCR | RD | Role |
|---|---|---|---|---|---|
| GSE25066 | Affymetrix HG-U133A (GPL96) | 170 | 57 | 113 | Primary training |
| GSE20271 | Affymetrix HG-U133A (GPL96) | 58 | 13 | 45 | Validation 1 |
| GSE20194 | Affymetrix HG-U133A (GPL96) | 71 | 25 | 46 | Validation 2 |
| GSE32646 | Affymetrix HG-U133 Plus 2.0 (GPL570) | 26 | 10 | 16 | Validation 3 |
| **Combined** | | **325** | **105 (32.3%)** | **220 (67.7%)** | Pooled training |

**Download:**
```bash
# Using GEOparse
pip install GEOparse
# Then in Python:
import GEOparse
for gse_id in ['GSE25066', 'GSE20271', 'GSE20194', 'GSE32646']:
    gse = GEOparse.get_GEO(geo=gse_id, destdir='./data/geo_raw/')
```

**Preprocessing pipeline (from TNBC-DT paper, your own work):**
1. Log2 transform: `y = log2(x + ε)`
2. Quantile normalization
3. ComBat batch-effect correction (use Lehmann subtype as biological covariate)
4. Z-score scaling per gene
5. Probe-to-gene collapse: keep highest inter-sample variance probe per Entrez Gene ID

**Feature engineering:**
- **Stage 1** — limma differential expression (FDR < 0.05) → ~480 genes
- **Stage 2** — batch-corrected variance filter (σ² > 0.5) → ~210 genes  
- **Stage 3** — missingness filter (< 10% missing) → **100 biologically coherent genes**
- Also test top-500 and top-1000 by coefficient of variation (CV = σ/μ)

**Class imbalance handling:**
- SMOTE during training (NOT on test folds)
- class_weight balancing in loss function
- scale_pos_weight = (220/105) ≈ 2.1 for tree models

**Lehmann TNBC subtypes:**
- Apply 101-gene algorithm to assign BL1, BL2, M, LAR subtypes
- Use as auxiliary supervision signal + prior encoding

---

## 2. Secondary Genomic Dataset

### TCGA-BRCA (Multi-task)

**Primary use: Extended task set + multimodal validation**

| Modality | Description | Patients | Access |
|---|---|---|---|
| RNA-seq (HTSeq counts) | Gene expression | ~1,093 | portal.gdc.cancer.gov |
| DNA Methylation (450k) | β-values | ~890 | portal.gdc.cancer.gov |
| CNV (GISTIC2) | Copy number | ~1,066 | portal.gdc.cancer.gov |
| Somatic mutations | SNV, indels | ~990 | portal.gdc.cancer.gov |
| Clinical | OS, DFS, stage, PAM50 | ~1,093 | portal.gdc.cancer.gov |

**Tasks available on TCGA-BRCA:**
1. **Binary classification** — Tumor (T) vs. Normal (N) tissue: ~113 normal vs ~1,100 tumor
2. **PAM50 subtype** — 5-class: LumA, LumB, HER2, Basal, Normal-like
3. **ER/PR/HER2 status** — Binary per marker
4. **Survival** — OS + DFS with Cox PH or discrete-time models

**Recommended primary task: PAM50 5-class subtype (richer, more papers to compare against)**

**Download commands:**
```bash
# Install GDC client
# https://gdc.cancer.gov/access-data/gdc-data-transfer-tool

# Create manifest for TCGA-BRCA RNA-seq
# Go to: https://portal.gdc.cancer.gov/
# Filter: Project = TCGA-BRCA, Data Category = Transcriptome Profiling
# Data Type = Gene Expression Quantification
# Download manifest → then:
gdc-client download -m gdc_manifest.txt -d ./data/tcga_brca/rnaseq/
```

**Alternative quick access via cBioPortal:**
```python
# Pandas read from cBioPortal TCGA-BRCA
import pandas as pd
# TCGA-BRCA mRNA (Z-scores) + clinical:
# https://www.cbioportal.org/study/summary?id=brca_tcga_pub2015
```

**METABRIC (secondary validation):**
- 1,904 patients, gene expression + CNA
- Access via cBioPortal: `brca_metabric`
- Use for external validation of genomic models trained on GEO/TCGA

---

## 3. Multimodal Paired Dataset (Validation)

### TCGA-BRCA Paired (WSI + Genomics)

**Goal: Validate late fusion of your TBCA histopathology model + genomic model**

| Modality | Count | Link |
|---|---|---|
| WSI (H&E, FFPE diagnostic slides) | ~1,133 slides, ~1,062 patients | GDC Portal → Diagnostic Slides |
| RNA-seq | ~1,093 patients | GDC Portal |
| **Intersection (paired)** | **~900–950 patients** | Match by patient barcode |

**Patient barcode matching:**
```python
# TCGA barcodes: TCGA-XX-XXXX
# First 12 characters identify the patient
# WSI slides: TCGA-XX-XXXX-01Z-00-DX1.svs
# RNA-seq files: TCGA-XX-XXXX-01A-11R-...
patient_id = barcode[:12]  # truncate to patient level
```

**WSI patch extraction (use CLAM toolkit):**
```bash
git clone https://github.com/mahmoodlab/CLAM.git
cd CLAM
# Step 1: Segment tissue
python create_patches_fp.py \
  --source ./data/tcga_brca/slides/ \
  --save_dir ./data/tcga_brca/patches/ \
  --patch_size 256 \
  --seg --patch --stitch

# Step 2: Extract features using your TBCA backbone
# Use EfficientNet-B5 or the full TBCA encoder as feature extractor
python extract_features_fp.py \
  --data_h5_dir ./data/tcga_brca/patches/ \
  --data_slide_dir ./data/tcga_brca/slides/ \
  --csv_path ./data/tcga_brca/slide_list.csv \
  --feat_dir ./data/tcga_brca/features/ \
  --batch_size 512 \
  --slide_ext .svs
```

**Feature extraction from TBCA:**
```python
# Load pretrained TBCA model (trained on BreakHis)
# Freeze all weights, use as feature extractor on TCGA patches
model = get_triple_branch_fusion(...)
model.load_state_dict(torch.load('best_tbca_breakhis.pt'))
model.eval()

# Extract 768-dim feature per patch
def extract_patch_features(patch_batch):
    with torch.no_grad():
        swin_f, convnext_f, effnet_f = model.extract_features(patch_batch)
        s, c, e = model.project_features(swin_f, convnext_f, effnet_f)
        fused = s * 0.33 + c * 0.33 + e * 0.33  # or use learned weights
    return fused  # (B, 768)
```

**MIL aggregation for slide-level prediction:**
```
Patches (N × 768) → ABMIL → slide_embedding (768) → probability
```
Use ABMIL from CLAM or write lightweight ABMIL head.

---

## 4. Full Dataset Summary Table

| Dataset | Task | Samples | Modalities | Use |
|---|---|---|---|---|
| GEO TNBC (pooled) | pCR binary | 325 | RNA microarray | Genomic model training |
| TCGA-BRCA genomic | PAM50 subtype | ~1,093 | RNA-seq, methyl, CNV | Genomic model (multi-task) |
| TCGA-BRCA paired | Fusion validation | ~900 | WSI + RNA-seq | Late fusion validation |
| METABRIC | Survival + subtype | 1,904 | Gene expr + CNA | External validation |
| BreakHis | Binary histopathology | 7,909 patches | H&E images | Already done (Paper 1) |

---

## 5. Data Directory Structure

```
data/
├── geo_tnbc/
│   ├── raw/
│   │   ├── GSE25066/
│   │   ├── GSE20271/
│   │   ├── GSE20194/
│   │   └── GSE32646/
│   ├── processed/
│   │   ├── combined_325_genes_100.csv    # 100-gene panel
│   │   ├── combined_325_genes_500.csv    # top-500 CV
│   │   ├── combined_325_genes_1000.csv   # top-1000 CV
│   │   ├── labels_pcr_rd.csv
│   │   └── lehmann_subtypes.csv
│   └── pathway_masks/
│       └── kegg_pathway_membership.pkl   # gene → pathway mapping
│
├── tcga_brca/
│   ├── rnaseq/                           # HTSeq counts raw
│   ├── methylation/                      # 450k beta values
│   ├── cnv/                              # GISTIC2 scores
│   ├── clinical/                         # OS, DFS, PAM50, stage
│   ├── processed/
│   │   ├── rnaseq_fpkm_top1000.csv
│   │   ├── methylation_top_sites.csv
│   │   └── multi_omics_combined.pkl
│   ├── slides/                           # WSI .svs files
│   ├── patches/                          # CLAM output h5
│   └── features/                         # TBCA 768-dim features
│
└── metabric/
    ├── expression_array.csv
    ├── cnv.csv
    └── clinical.csv
```
