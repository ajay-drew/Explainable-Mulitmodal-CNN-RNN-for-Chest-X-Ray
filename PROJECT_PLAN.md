# Explainable Multimodal CNN-RNN for Chest X-Ray Diagnosis — Project Plan

**Reference document:** Use this plan for all development decisions. Refer to it whenever implementing features, evaluating design choices, or validating deliverables.

---

## 1. Project Overview

The proposed system develops an **explainable multimodal CNN-RNN framework** for chest X-ray diagnosis, leveraging transfer learning for efficiency and performance.

### Goals

- **Data:** Integrate chest X-ray images and radiology reports from **MIMIC-CXR** for multi-label classification of chest diseases (e.g., Atelectasis, Cardiomegaly, Pneumonia).
- **Fusion:** Combine visual (spatial) and textual (sequential) data to overcome single-modality limitations.
- **Explainability:** Use unified XAI techniques—Grad-CAM for image heatmaps, SHAP/LIME for text attribution—so clinicians can verify decisions.
- **Architecture:** Feature extraction → multimodal fusion → XAI integration, inspired by MCX-Net and the project template (CNN-RNN hybrids + transfer learning).
- **Targets:** Diagnostic accuracy **>92%**, interpretability, and alignment with regulatory expectations (e.g., GDPR/FDA).

### Dataset

- **MIMIC-CXR:** 377,110 paired images/reports.
- Challenges: data imbalance, noise, multi-disease coexistence.
- Transfer learning keeps training feasible for a final-year college project.

---

## 2. Requirements Summary

| Category | Description |
|----------|-------------|
| **Functional** | What the system must do (data, features, fusion, classification, XAI, evaluation, optional UI). |
| **Non-functional** | Performance, scalability, reliability, security, usability, maintainability. |
| **System** | Hardware/software needs. |

---

## 3. Functional Requirements

### 3.1 Data Ingestion and Preprocessing

| Requirement | Details |
|-------------|---------|
| **Load & align** | Paired chest X-ray images and radiology reports from MIMIC-CXR. |
| **Image preprocessing** | Resize to **224×224**, normalize (mean=0.485, std=0.229), augment (random flips, rotations). |
| **Text preprocessing** | Tokenize (e.g., BERT tokenizer), clean (remove stop words, handle ambiguities), pad to fixed length (e.g., **512 tokens**). |
| **Splits** | **80% train**, **10% validation**, **10% test** (per base paper); exclude lateral views and ambiguous labels. |

### 3.2 Feature Extraction

| Requirement | Details |
|-------------|---------|
| **Image branch** | Pre-trained CNN (e.g., **ResNet50/ResNet152**), transfer learning, fine-tuned on MIMIC-CXR. |
| **Text branch** | RNN (e.g., **LSTM/GRU**, bidirectional), pre-trained embeddings (e.g., GloVe or **BioBERT**). |
| **Labels** | Multi-label for **13+ diseases** plus **"No Findings."** |

### 3.3 Multimodal Fusion

| Requirement | Details |
|-------------|---------|
| **Mechanism** | **Attention-based** (self-attention or cross-modal attention, inspired by MCX-Net transformer fusion). |
| **Output** | Unified embedding for multi-label classification via dense layer with **sigmoid** activation. |

### 3.4 Multi-Label Classification

| Requirement | Details |
|-------------|---------|
| **Loss** | **Binary cross-entropy.** |
| **Output** | Probabilities; threshold (e.g., **0.5**) for binary labels; support **co-existing diseases.** |

### 3.5 Explainable AI (XAI) Integration

| Requirement | Details |
|-------------|---------|
| **Image** | **Grad-CAM** heatmaps for disease-relevant regions. |
| **Text** | **SHAP/LIME** attributions for key phrases (e.g., "shortness of breath" → Pneumonia). |
| **Unified** | Combine image + text attributions (novel contribution); **faithfulness score** (e.g., **>92%** interpretability). |

### 3.6 Evaluation and Output

| Requirement | Details |
|-------------|---------|
| **Metrics** | Accuracy, Precision, Recall, F1-Score, **AUROC** (macro **>0.816**; outperform unimodal ResNet152 ~0.749). |
| **Visualization** | Predictions, heatmaps, attributions, faithfulness metrics. |
| **Inference** | Accept user-uploaded image–report pairs for **real-time** diagnosis. |

### 3.7 User Interface (Optional)

- Simple **web UI** (e.g., **Streamlit/Flask**) for uploads, predictions, and explanations.

---

## 4. Non-Functional Requirements

### 4.1 Performance

| Target | Value |
|--------|--------|
| **Accuracy** | **>92%** on test set. |
| **AUROC** | Macro **>0.816** (benchmark from base paper). |
| **Training time** | **<10 hours** on modest hardware (via transfer learning). |
| **Inference time** | **<5 seconds** per sample. |
| **Imbalance** | Weighted loss or oversampling. |

### 4.2 Scalability and Efficiency

- Batch processing (e.g., **batch size 32**).
- Transfer learning: e.g., freeze early CNN layers to reduce parameters.

### 4.3 Reliability and Robustness

- **Error handling:** Graceful failure for invalid inputs (e.g., non-X-ray images).
- **Bias:** Evaluate fairness across demographics (e.g., age, gender from MIMIC-CXR metadata).

### 4.4 Security and Compliance

- **Privacy:** De-identified MIMIC-CXR data (HIPAA-compliant).
- **Explainability:** XAI outputs verifiable (e.g., human studies for validation).

### 4.5 Usability

- **Explanations:** Natural language (e.g., “Model focused on lung opacity in image due to ‘consolidation’ in report”).
- **Documentation:** Code comments, user guide for replication.

### 4.6 Maintainability

- **Modular code:** Separate modules for preprocessing, models, XAI.
- **Version control:** Git for tracking changes.

---

## 5. Detailed Component Descriptions

### 5.1 Input Layer

- **Inputs:** Paired data from MIMIC-CXR (images as tensors, reports as strings).
- **Labels:** 13 disease labels + **"No Findings."**

### 5.2 Preprocessing Module

- **Images:** Grayscale/RGB, resize, normalize.
- **Text:** Lowercase, tokenize; **clinical history ("Indication")** as key input (per base paper).
- Ensures data quality before feature extraction.

### 5.3 Feature Extraction Branches

| Branch | Method | Output |
|--------|--------|--------|
| **CNN** | Pre-trained ResNet; freeze conv base, fine-tune classifier. | Flattened feature vector (e.g., **2048** dims). |
| **RNN** | Bidirectional LSTM/GRU on tokenized text; pre-trained embeddings. | Hidden states (e.g., **512** dims). |

### 5.4 Multimodal Fusion

- **Method:** Concatenate and/or **attend** over CNN + RNN outputs (e.g., **Transformer** layer, cross-modal attention per MCX-Net).
- **Role:** Integrate clinical history to focus on relevant image regions (e.g., lung opacities).

### 5.5 Classification Layer

- **Structure:** Fully connected layers with **dropout (0.5)**.
- **Loss:** **Binary cross-entropy** (multi-label).
- **Optimizer:** **Adam**, lr=**1e-4**.

### 5.6 XAI Module (Post-prediction)

| Modality | Technique | Output |
|----------|-----------|--------|
| **Image** | **Grad-CAM** | Visual saliency heatmaps. |
| **Text** | **SHAP** | Word/phrase contributions. |
| **Unified** | Novel ensemble (e.g., weighted attribution scores) | Faithfulness evaluation (e.g., perturbation tests for **>92%** score). |

### 5.7 Evaluation Module

- Compute metrics on validation/test sets.
- Visualize: ROC curves, heatmaps, attributions.
- Compare against baselines: **unimodal CNN**, **unimodal RNN**.

---

## 6. Key Benchmarks and Targets

| Metric | Target | Baseline (e.g., ResNet152) |
|--------|--------|----------------------------|
| **Accuracy** | **>92%** | — |
| **Macro AUROC** | **>0.816** | ~0.749 (unimodal) |
| **Interpretability (faithfulness)** | **>92%** | — |
| **Inference** | **<5 s** per sample | — |
| **Training** | **<10 h** (modest hardware) | — |

---

## 7. References and Conventions

- **Base paper:** MCX-Net (multimodal fusion for attention and diagnosis on MIMIC-CXR).
- **Project template:** CNN-RNN hybrids with transfer learning.
- **Dataset:** MIMIC-CXR (377,110 paired images/reports).
- **Regulatory:** GDPR/FDA-oriented transparency via XAI.

---

## 8. Transfer Learning Model Selection (CNN Backbone)

*Recommendations from web/search and documentation; prefer models trained on MIMIC-CXR for best domain alignment.*

### 8.1 Recommended: TorchXRayVision — MIMIC-CXR–trained DenseNet121

**Best fit:** Pre-trained on MIMIC-CXR, 224×224 (matches project preprocessing), PyTorch, multi-label pathology outputs.

| Attribute | Value |
|-----------|--------|
| **Library** | [TorchXRayVision](https://mlmed.org/torchxrayvision/models.html) (`pip install torchxrayvision`) |
| **Model** | `DenseNet(weights="densenet121-res224-mimic_nb")` or `"densenet121-res224-mimic_ch"` |
| **Resolution** | 224×224 (auto-resize by library) |
| **Architecture** | DenseNet-121 (~8M params); good for CXR, mitigates vanishing gradients |
| **Use** | Feature extractor: `model.features(img)`; or fine-tune classifier |
| **Targets** | Atelectasis, Consolidation, Infiltration, Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia, Pleural_Thickening, Cardiomegaly, Nodule, Mass, Hernia, Lung Lesion, Fracture, Lung Opacity, Enlarged Cardiomediastinum |
| **Weights path** | Auto-downloaded to `~/.torchxrayvision/` |

**Example (feature extraction for fusion):**
```python
import torchxrayvision as xrv
model = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
feats = model.features(img)  # for transfer learning / fusion branch
```

### 8.2 Alternative: TorchXRayVision ResNet (multi-dataset)

- **Model:** `ResNet(weights="resnet50-res512-all")` — trained on multiple CXR datasets (may include MIMIC).
- **Note:** 512×512; if using 224×224 pipeline, prefer DenseNet MIMIC above or resize/adapt.

### 8.3 Other options (not MIMIC-CXR–specific)

| Option | Description | MIMIC-CXR? |
|--------|-------------|------------|
| **CXR Foundation (Google)** | EfficientNet-L2 embeddings; pre-computed MIMIC-CXR embeddings on PhysioNet (4096-d). Good for low compute; requires credentialed access. | Embeddings derived from MIMIC-CXR; base model trained on US/India CXR. |
| **MoCo-CXR** | Contrastive (MoCo) CXR representations; strong transferability to other CXR tasks/datasets. | Not MIMIC-only; general CXR. |
| **CheXpert DenseNet** | `densenet121-res224-chex` in TorchXRayVision; Stanford CheXpert. Often used with MIMIC in multi-dataset setups. | No; complementary. |
| **CheXagent / foundation models** | Vision–language CXR models (2023–2024); higher capability, heavier. | Trained on multiple datasets including MIMIC-related. |

### 8.4 Decision for this project

- **Primary CNN backbone:** Use **TorchXRayVision DenseNet121 with MIMIC-CXR weights** (`densenet121-res224-mimic_nb` or `mimic_ch`) for the image branch.
- **Rationale:** Same dataset (MIMIC-CXR), same resolution (224×224), PyTorch feature extraction, pathology list aligned with multi-label targets; supports Grad-CAM (DenseNet compatible).
- **Fallback:** If DenseNet is swapped for ResNet in experiments, use TorchXRayVision ResNet or standard torchvision ResNet50/152 with ImageNet → fine-tune on MIMIC-CXR.

---

## 9. Text Branch (RNN / Encoder) — Hugging Face Hub

*For the report/text branch, use a pretrained encoder from Hugging Face Hub. Transformer encoders (BERT/RoBERTa) provide sequential representations and are the standard for clinical/radiology text; they can replace or sit alongside LSTM/GRU in the fusion pipeline.*

### 9.1 Recommended: RadBERT-RoBERTa-4m (radiology-specific)

**Best fit:** Trained on **4 million radiology reports** (VA, de-identified); outperforms BioBERT, Clinical-BERT, BLUE-BERT on radiology NLP tasks (abnormal sentence classification, report coding, summarization).

| Attribute | Value |
|-----------|--------|
| **HF Hub ID** | [`UCSD-VA-health/RadBERT-RoBERTa-4m`](https://huggingface.co/UCSD-VA-health/RadBERT-RoBERTa-4m) |
| **Architecture** | RoBERTa (transformer encoder) |
| **Training data** | 4M radiology reports (US VA) |
| **Library** | `transformers` (PyTorch); pipeline_tag: fill-mask; use as encoder via `AutoModel` |
| **License** | Apache-2.0 |
| **Use** | Tokenize report → `model(**inputs)` → use `last_hidden_state` or `pooler_output` as text features for fusion |

**Example (feature extraction for fusion):**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("UCSD-VA-health/RadBERT-RoBERTa-4m")
model = AutoModel.from_pretrained("UCSD-VA-health/RadBERT-RoBERTa-4m")
encoded = tokenizer(report_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**encoded)
# Use outputs.last_hidden_state (seq, hidden) or outputs.pooler_output (cls) for fusion
text_features = outputs.pooler_output  # [batch, 768] or last_hidden_state for sequence
```

### 9.2 Alternative: Bio_ClinicalBERT (MIMIC clinical notes)

**Best fit when aligning with MIMIC:** Trained on **MIMIC-III** clinical notes (~880M words, Beth Israel), same hospital ecosystem as MIMIC-CXR.

| Attribute | Value |
|-----------|--------|
| **HF Hub ID** | [`emilyalsentzer/Bio_ClinicalBERT`](https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT) |
| **Architecture** | BERT-base (L-12, H-768, A-12) |
| **Training data** | All MIMIC-III NOTEEVENTS (~880M words) |
| **Max length** | 128 (pretrain); can use 512 with truncation for reports |
| **Use** | Same pattern: `AutoModel` → `last_hidden_state` or `pooler_output` for fusion |

**Example:**
```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
encoded = tokenizer(report_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**encoded)
text_features = outputs.pooler_output  # [batch, 768]
```

### 9.3 Other HF Hub options

| Model | HF Hub ID | Notes |
|-------|-----------|--------|
| **BlueBERT (PubMed + MIMIC)** | `bionlp/bluebert_pubmed_mimic_uncased_L-24_H-1024_A-16` | Larger (L-24, H-1024); PubMed + MIMIC-III. |
| **BioClinical BERT ft MIMIC-III (lung cancer)** | `sarahmiller137/bioclinical-bert-ft-m3-lc` | Fine-tuned on MIMIC-III radiology for lung cancer; good for disease-specific downstream. |
| **Gilbert (Harvard)** | `rajpurkarlab/gilbert` | BioBERT fine-tuned for radiology token classification (e.g. prior removal). |

### 9.4 Decision for this project

- **Primary text encoder:** Use **RadBERT-RoBERTa-4m** (`UCSD-VA-health/RadBERT-RoBERTa-4m`) for the report branch.
- **Rationale:** Radiology-specific, 4M reports, strong on report coding/classification; 512 token input fits PROJECT_PLAN; use `pooler_output` (e.g. 768-d) or mean-pooled `last_hidden_state` as text embedding for multimodal fusion.
- **Alternative:** Use **Bio_ClinicalBERT** (`emilyalsentzer/Bio_ClinicalBERT`) if MIMIC ecosystem alignment is preferred; same usage pattern, 768-d output.
- **Note:** PROJECT_PLAN mentions "LSTM/GRU"; in practice HF Hub offers transformer encoders. Use RadBERT/Bio_ClinicalBERT as the text encoder and feed their output into the fusion layer (optionally add a small LSTM/GRU on top of token-level `last_hidden_state` if sequence-level refinement is desired).

---

## 10. XAI for the Text Branch (BERT Encoder)

**Yes, XAI works for the BERT encoder model.** The text branch (RadBERT / Bio_ClinicalBERT) is a transformer encoder; standard explainability methods for NLP apply and produce **token-level attributions** (which words/phrases drove the prediction). These can be combined with image Grad-CAM in the unified XAI module.

### 10.1 How it works with BERT

- **Input:** Tokenized report (e.g. 512 tokens) → BERT encoder → pooled or sequence representation → fusion → classifier → disease probabilities.
- **Goal:** Attribute the **classification output** (or the **text branch’s contribution** before fusion) to **input tokens**.
- **Output:** Per-token importance scores (and optionally highlighted phrases, e.g. “shortness of breath” → Pneumonia), which you then combine with image heatmaps in the unified explanation.

All of the methods below are **model-agnostic or gradient-based** and work with any differentiable BERT encoder (RadBERT, Bio_ClinicalBERT, etc.).

### 10.2 Recommended methods (all work with BERT)

| Method | How it works with BERT | Pros | Cons |
|--------|------------------------|------|------|
| **SHAP** | Token-level Shapley values: treat each token as a “player”; attribute prediction to tokens (e.g. mask/ablate tokens, or use gradient-based SHAP). Libraries: `shap` with custom tokenizer/masker; **TransSHAP** extends SHAP to BERT. | Theoretically grounded, consistent across instances. | Can be slow (many forward passes or approximations). |
| **LIME** | Perturb input (mask/remove or replace tokens), observe prediction change, fit local linear model to get token weights. | Intuitive, case-by-case; often faster than exact SHAP. | Local approximation only; sensitive to perturbation design. |
| **Integrated Gradients (IG)** | Gradient of output w.r.t. input embeddings along a path from baseline (e.g. zero embedding) to actual input; sum to get attribution per token. | Well-suited to neural NLP; used in medical/clinical BERT explainability. | Needs a sensible baseline; straight-line path can be improved (see DIG/SIG). |
| **Attention visualization** | Plot BERT self-attention weights (which tokens attend to which). | Easy to get from the model. | **Not always faithful** to the true importance; use as a supplement, not the main text XAI. |

### 10.3 Practical choices for this project

1. **Primary text XAI:** **SHAP** or **Integrated Gradients** for token attributions.
   - **SHAP:** Use `shap` with a BERT-friendly masker (e.g. token masking) or gradient-based explainer; or TransSHAP for BERT. Output: one importance score per token → highlight top phrases (e.g. “consolidation”, “shortness of breath”) and link to predicted diseases.
   - **Integrated Gradients:** Implement IG (or use `captum`) on the **classification logit(s)** (or text-branch output) w.r.t. **input embeddings**; map gradients back to tokens (e.g. sum over embedding dims per token). Used in medical BERT explainability (e.g. BioClinical BERT).
2. **Secondary / optional:** **LIME** for an alternative local explanation; **attention** for optional visualization (with a note that it may not reflect true importance).
3. **Unified XAI:** Combine **image Grad-CAM** (spatial heatmap) with **text token attributions** (SHAP/IG scores) into one explanation (e.g. “Model focused on lung opacity in image and phrases ‘consolidation’ / ‘shortness of breath’ in report for Pneumonia”), and compute **faithfulness** (e.g. >92%) via perturbation tests (e.g. remove top-attributed tokens and check prediction change).

### 10.4 Implementation notes

- **Differentiability:** BERT encoder is fully differentiable; gradient-based methods (IG, gradient × input, gradient-based SHAP) work out of the box.
- **Scope:** Apply XAI to the **full pipeline** (encoder → fusion → classifier) so attributions reflect what actually drove the **final** disease prediction, not just the encoder’s internal representation.
- **Libraries:** `shap` (text + custom tokenizer), `captum` (Integrated Gradients, gradient methods), `transformers` (attention weights). TransSHAP and clinical-BERT explainability repos can be used as references.

### 10.5 Summary

- **XAI for the “RNN” (text) branch works with the BERT encoder.** Use **SHAP** and/or **Integrated Gradients** for token-level explanations; optionally **LIME** and **attention**. Combine text attributions with **Grad-CAM** on the image branch in the unified XAI module and evaluate faithfulness (e.g. >92%) as in §3.5 and §5.6.

---

*This document is the single source of truth for the explainable multimodal CNN-RNN chest X-ray project. Refer to it for every implementation and evaluation step.*
