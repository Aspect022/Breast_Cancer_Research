# 🔍 Gemini Reference-Finding Prompt
## Purpose: Find academic references for the Breast Cancer Classification paper
## Instructions: Copy the prompt below and paste it into Gemini (with Deep Research or 1.5 Pro)

---

## HOW TO USE THIS

1. Copy everything inside the **"--- COPY FROM HERE ---"** block below
2. Paste into **Gemini Advanced** (use "Deep Research" mode if available for best results)
3. Wait for Gemini to return a list of papers with proper citations
4. Copy the citations list and give it to Claude along with the `PAPER_CONTEXT_FOR_CLAUDE.md` file

---

--- COPY FROM HERE ---

I am writing an academic paper on deep learning and quantum-classical hybrid architectures for breast cancer histopathology binary classification (Benign vs. Malignant) using the BreakHis dataset. I need you to find and list **real, verifiable academic references** with full citation details (authors, title, journal/conference, year, DOI or arXiv link) for the following specific topics. For each reference, include:

- Full author list
- Paper title
- Journal or Conference name
- Year of publication
- DOI or arXiv link
- 1-sentence summary of why it is relevant

Please find **3–5 real papers per category** below. Only include real papers you are confident exist — do NOT hallucinate or fabricate references.

---

### CATEGORY 1: BreakHis Dataset (Original Paper)
Find the original paper introducing the BreakHis breast cancer histopathology dataset by Spanhol et al. from UFPR Brazil. Also find any benchmark papers that report results on the BreakHis binary classification task.

---

### CATEGORY 2: CNN-Based Breast Cancer Histopathology Classification
Papers using CNNs (such as ResNet, VGG, DenseNet, EfficientNet, or InceptionNet) for breast cancer classification on histopathology images. Prefer papers from 2019–2024.

---

### CATEGORY 3: Vision Transformer (ViT) for Medical Imaging / Histopathology
Papers applying Vision Transformers (ViT), Swin Transformer, or ConvNeXt to:
- Medical image classification
- Histopathology image analysis
- Breast cancer classification using transformers

Include the original ViT paper (Dosovitskiy et al. 2020), original Swin Transformer (Liu et al. 2021), and original ConvNeXt paper (Liu et al. 2022).

---

### CATEGORY 4: Multi-Branch / Fusion Architectures for Medical Imaging
Papers proposing multi-backbone, multi-branch, or ensemble fusion architectures specifically for:
- Histopathology image classification
- Medical image classification
- Any cancer classification task using feature fusion or cross-attention between multiple backbones

---

### CATEGORY 5: Cross-Attention Mechanisms in Vision Models
Papers describing or using cross-attention mechanisms between multiple vision encoders or feature streams. This includes:
- Cross-attention in multi-modal learning
- Cross-attention for feature fusion in image classification
- Any work showing cross-attention improves feature integration over simple concatenation

---

### CATEGORY 6: Quantum Machine Learning for Medical Imaging
Papers applying:
- Variational Quantum Circuits (VQC) to image classification
- Quantum-classical hybrid neural networks for medical imaging
- Quantum computing applied to cancer detection or histopathology
- PennyLane-based hybrid quantum-classical models

Include any foundational quantum machine learning papers (e.g., Biamonte et al. Nature 2017, Cerezo et al. 2021 on variational quantum algorithms).

---

### CATEGORY 7: Class Imbalance in Medical Imaging / Histopathology
Papers addressing class imbalance in breast cancer datasets, including:
- Oversampling, undersampling techniques
- Loss function modifications for imbalanced medical data
- Evaluation metrics for imbalanced data (MCC, AUC, F1)

---

### CATEGORY 8: EfficientNet
The original EfficientNet paper by Tan & Le (2019) and any follow-up works (EfficientNet-B3, B5) applied to medical imaging.

---

### CATEGORY 9: Patient-Level Cross-Validation in Medical Imaging
Papers discussing the importance of patient-level data splitting (vs. image-level) in medical imaging ML to prevent data leakage, especially for cancer classification.

---

### CATEGORY 10: State-of-the-Art on BreakHis Binary Classification
Find recent papers (2021–2024) that report high accuracy on BreakHis binary classification so I can compare my results (95.48% val accuracy) against the current state of the art.

---

Please format your response as a numbered reference list in the following format for each entry:

[N] Author(s). "Title." *Journal/Conference*, Year. DOI: xxx or arXiv: xxx.
**Relevance:** One sentence explaining why this paper is relevant to my work.

--- END COPY ---

---

## AFTER GEMINI RESPONDS

Once you have the reference list from Gemini, give Claude both:
1. `PAPER_CONTEXT_FOR_CLAUDE.md` — the full architecture and results context
2. The Gemini reference list — so Claude can correctly insert `[N]` citations into each paper section

Tell Claude: *"Use the reference list to insert citations in IEEE format. Do not cite anything not in the provided list. Where a reference is needed but not in the list, write [CITE_NEEDED: topic] as a placeholder."*
