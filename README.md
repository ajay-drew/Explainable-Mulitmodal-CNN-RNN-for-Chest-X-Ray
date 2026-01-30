# Explainable Multimodal CNN-RNN for Chest X-Ray Diagnosis

An explainable multimodal framework combining chest X-ray images and radiology reports for multi-label disease classification using deep learning with interpretable AI.

## Overview

This project implements a multimodal deep learning system for chest X-ray diagnosis that:

- **Fuses visual and textual data** — Combines chest X-ray images (CNN) with radiology reports (BERT) for improved diagnosis
- **Multi-label classification** — Detects 14 conditions including Pneumonia, Cardiomegaly, Atelectasis, etc.
- **Explainable AI** — Provides Grad-CAM heatmaps for images and token attribution for text
- **Transfer learning** — Uses TorchXRayVision (MIMIC-CXR trained) and RadBERT for efficiency

See [`PROJECT_PLAN.md`](PROJECT_PLAN.md) for detailed requirements and architecture.

## Project Structure

```
├── config/                 # Configuration
│   ├── __init__.py
│   └── config.py          # Hyperparameters, paths, model settings
├── data/                   # MIMIC-CXR dataset (not included)
│   └── .gitkeep
├── src/                    # Source code
│   ├── data/              # Dataset and preprocessing
│   │   ├── dataset.py     # PyTorch Dataset for MIMIC-CXR
│   │   ├── preprocessing.py # Image and text preprocessing
│   │   └── dataloader.py  # DataLoader utilities
│   ├── models/            # Model architecture
│   │   ├── image_encoder.py  # CNN (TorchXRayVision DenseNet)
│   │   ├── text_encoder.py   # BERT (RadBERT / Bio_ClinicalBERT)
│   │   ├── fusion.py         # Multimodal fusion (attention)
│   │   └── classifier.py     # Full multimodal classifier
│   ├── xai/               # Explainability
│   │   ├── gradcam.py        # Grad-CAM for images
│   │   ├── text_attribution.py # SHAP/IG for text
│   │   └── unified.py        # Unified multimodal XAI
│   ├── training/          # Training and evaluation
│   │   ├── train.py
│   │   └── evaluate.py
│   └── utils/             # Helpers
│       └── helpers.py
├── scripts/               # Entry points
│   ├── train.py          # Training script
│   └── inference.py      # Inference with explanations
├── app/                   # Web UI
│   └── streamlit_app.py  # Streamlit application
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── requirements.txt       # Dependencies
├── PROJECT_PLAN.md        # Detailed project plan
└── README.md
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd Explainable-Mulitmodal-CNN-RNN-for-Chest-X-Ray
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or: venv\Scripts\activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download MIMIC-CXR dataset:**
   - Request access at [PhysioNet](https://physionet.org/content/mimic-cxr/2.0.0/)
   - Download and extract to `data/mimic-cxr/`

## Usage

### Training

```bash
python scripts/train.py --data-root data/mimic-cxr --epochs 50
```

Options:
- `--config`: Path to custom config YAML
- `--batch-size`: Override batch size (default: 32)
- `--lr`: Override learning rate (default: 1e-4)
- `--device`: Device to use (cuda, mps, cpu)

### Inference

```bash
python scripts/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/xray.jpg \
    --report "Patient presents with cough and fever. Chest X-ray shows..."
    --explain \
    --output outputs/
```

### Web UI

```bash
streamlit run app/streamlit_app.py
```

## Models

### Image Encoder
- **TorchXRayVision DenseNet121** — Pretrained on MIMIC-CXR
- Model: `densenet121-res224-mimic_nb`
- Input: 224×224 grayscale images

### Text Encoder
- **RadBERT-RoBERTa-4m** — Trained on 4M radiology reports
- HuggingFace: `UCSD-VA-health/RadBERT-RoBERTa-4m`
- Alternative: `emilyalsentzer/Bio_ClinicalBERT`

### Fusion
- Attention-based multimodal fusion
- Combines image and text features before classification

## Explainability

### Image XAI
- **Grad-CAM**: Highlights disease-relevant regions in X-rays

### Text XAI
- **SHAP/Integrated Gradients**: Token-level attribution for report phrases

### Unified XAI
- Combined image + text explanations
- Faithfulness evaluation (target: >92%)

## Targets

| Metric | Target |
|--------|--------|
| Accuracy | >92% |
| AUROC (macro) | >0.816 |
| Faithfulness | >92% |
| Inference time | <5 seconds |

## References

- [PROJECT_PLAN.md](PROJECT_PLAN.md) — Full requirements and architecture
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/) — Dataset
- [TorchXRayVision](https://github.com/mlmed/torchxrayvision) — CXR models
- [RadBERT](https://huggingface.co/UCSD-VA-health/RadBERT-RoBERTa-4m) — Radiology BERT

## License

See [LICENSE](LICENSE) for details.
