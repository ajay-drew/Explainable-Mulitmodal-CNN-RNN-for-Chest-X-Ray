# Explainable Multimodal CNN-Transformer Framework for Chest X-Ray Analysis

A hybrid CNN (image) + Transformer (text) multimodal classifier with full XAI support.
Operates in two swappable modes: **Medical Imaging** (15-class chest pathology) and
**Sentiment Analysis** (binary tweet sentiment). Runs zero-shot with no training required.

---

## Architecture

```
                        +-------------------------------------+
                        |        STREAMLIT DEMO APP          |
                        +----------------+--------------------+
                                         |
          +--------------------------+---+---+---------------------+
          |                          |                             |
   MODE A (CNN)               MODE B (RNN)                 XAI Pipeline
          |                          |                             |
  +-------+--------+        +--------+-------+        +-----------+-------+
  | TorchXRayVision|        | TwitterRoBERTa |        | Grad-CAM          |
  | DenseNet121    |        | (3-class head) |        | SHAP (masking)    |
  | (18 -> 15 lbl) |        | binary pred    |        | Integrated Grad   |
  | FROZEN         |        | FROZEN         |        | Attention Viz     |
  +-------+--------+        +----------------+        | NL Summaries      |
          |                                           +-------------------+
  +-------+--------+
  | RadBERT        | <- XAI only, not for predictions
  | FROZEN         |
  +----------------+
```

---

## Installation

```bash
git clone <repo-url>
cd Explainable-Mulitmodal-CNN-RNN-for-Chest-X-Ray
pip install -r requirements.txt
```

---

## Dataset Setup

### ChestX-ray14 (NIH)

```bash
# Print download instructions
python data/dataset.py --download

# After downloading, verify setup:
python data/dataset.py --verify data/chestxray14
```

Expected layout:
```
data/chestxray14/
├── Data_Entry_2017.csv
├── train_val_list.txt
├── test_list.txt
└── images/
    └── *.png  (~112,000 files)
```

### Sentiment140

Download from https://www.kaggle.com/datasets/kazanova/sentiment140

```bash
mkdir -p data/sentiment140
# Place training.1600000.processed.noemoticon.csv in data/sentiment140/
```

---

## Quick Start: Zero-Shot Demo (No Training Required)

```bash
# Launch Streamlit app — downloads pretrained weights (~2 GB) on first run
streamlit run app/streamlit_app.py
```

Open http://localhost:8501, select **Medical Imaging** mode, upload any
chest X-ray JPG, and click **Analyze**. Predictions and Grad-CAM appear in seconds.

---

## Single Image Inference

```bash
# Chest X-ray — zero-shot, no fine-tuning
python scripts/inference.py --mode cnn --image test_images/pneumonia.jpg

# With radiology report text for XAI attribution
python scripts/inference.py --mode cnn --image test_images/pneumonia.jpg \
    --text "Bilateral infiltrates noted in both lower lobes."

# Save Grad-CAM overlay + JSON explanation
python scripts/inference.py --mode cnn --image test_images/pneumonia.jpg \
    --output-dir results/

# Sentiment analysis
python scripts/inference.py --mode rnn --text "I feel great today!"
```

---

## Running Baselines Comparison

```bash
# Requires ChestX-ray14 test set
python scripts/run_baselines.py --config config/chestxray14.yaml

# Quick test with 100 samples
python scripts/run_baselines.py --config config/chestxray14.yaml --n-samples 100
```

Output: `results/baseline_comparison.csv`

| Model          | AUROC | F1   | Params Trained | Time (ms) |
|----------------|-------|------|----------------|-----------|
| ZeroShot (XRV) | ~0.80 | ~0.35 | 0             | ~50       |
| CNN-Only (XRV) | ~0.80 | ~0.35 | 0             | ~50       |

---

## Generating Evaluation Plots

```bash
# Given a predictions CSV with true_<label> and pred_<label> columns
python evaluation/visualize.py --results results/ --preds predictions.csv
```

Generated in `results/plots/`:
- `roc_curves.png` — per-class ROC + macro average
- `pr_curves.png` — precision-recall curves
- `training_history.png` — loss + AUROC curves

---

## Fairness Evaluation

```bash
python evaluation/fairness.py \
    --predictions results/preds.csv \
    --demographics data/chestxray14/Data_Entry_2017.csv \
    --output results/fairness_results.csv
```

Flags AUROC gaps > 0.05 between gender and age groups with a `[BIAS ALERT]`.

---

## File Structure

```
project/
├── app/
│   └── streamlit_app.py          Streamlit interactive demo
├── config/
│   ├── config.py                 Config dataclass + YAML support
│   ├── chestxray14.yaml          ChestX-ray14 config
│   ├── mimic.yaml                MIMIC-CXR-JPG config
│   └── sentiment.yaml            Sentiment140 config
├── data/
│   ├── dataset.py                ChestXray14Dataset, MIMICCXRDataset, SentimentDataset
│   ├── dataloader.py             DataLoader factories
│   └── preprocessing.py          XRV-compatible normalization + tokenizers
├── models/
│   ├── image_encoder.py          Frozen TorchXRayVision DenseNet121
│   ├── text_encoder.py           Frozen RadBERT + TwitterRoBERTa
│   └── baselines.py              ZeroShot, CNN-Only, Text-Only baselines
├── xai/
│   ├── gradcam.py                Grad-CAM / Grad-CAM++ / EigenCAM
│   ├── text_attribution.py       SHAP, Integrated Gradients, LIME
│   ├── unified.py                UnifiedExplainer orchestrator
│   └── nlp_summary.py            Rule-based NL explanation templates
├── evaluation/
│   ├── evaluate.py               AUROC, mAP, F1, precision, recall
│   ├── visualize.py              ROC, PR, XAI comparison plots
│   └── fairness.py               Demographic stratification + bias detection
├── scripts/
│   ├── inference.py              Single image/text inference CLI
│   └── run_baselines.py          Baseline comparison runner
├── requirements.txt
└── README.md
```

---

## Known Limitations

1. **TorchXRayVision label mapping**: The XRV `densenet121-res224-all` model outputs
   18 pathologies. Some labels (Infiltration, Fibrosis, Hernia) may not appear in all
   XRV weight variants and default to 0 when absent.

2. **RadBERT for XAI only**: Text features are used for SHAP/IG attribution only.
   Modality contribution is computed as a cosine norm proxy, not a learned weight.

3. **SHAP speed**: Token-masking SHAP runs O(seq_len) forward passes. On CPU with
   max_length=512 this can take 10-30 seconds per radiology report.

4. **LIME**: Requires `pip install lime`. Disabled gracefully if not installed.

5. **CPU inference**: All components work on CPU. GPU is recommended for SHAP/IG
   on long radiology reports (>200 tokens).

6. **ChestX-ray14 download**: Manual download required from NIH Box (no API access).
   Run `python data/dataset.py --download` for step-by-step instructions.
