"""
Configuration dataclass with YAML support for the Multimodal XAI Framework.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import yaml


CHESTXRAY14_LABELS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "No Finding",
    "Nodule",
    "Pleural Thickening",
    "Pneumonia",
    "Pneumothorax",
]

# TorchXRayVision densenet121-res224-all pathology list (18 labels)
XRV_PATHOLOGIES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Lung Opacity",
    "Lung Lesion",
]

# Mapping from our 14 labels → XRV pathology names
LABEL_TO_XRV: dict = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Effusion": "Effusion",
    "Emphysema": "Emphysema",
    "Fibrosis": "Fibrosis",
    "Hernia": "Hernia",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "No Finding": "No Finding",
    "Nodule": "Nodule",
    "Pleural Thickening": "Pleural_Thickening",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
}


@dataclass
class Config:
    """
    Unified configuration for training, inference, and XAI.
    All paths configurable via YAML — zero hardcoded paths.
    """

    # ── Mode ──────────────────────────────────────────────────────────────
    mode: str = "cnn"  # "cnn" | "rnn"

    # ── Paths ─────────────────────────────────────────────────────────────
    data_dir: str = "data/chestxray14"
    results_dir: str = "results"
    checkpoint_dir: str = "checkpoints"

    # ChestX-ray14 specific
    csv_path: str = "data/chestxray14/Data_Entry_2017.csv"
    train_list: str = "data/chestxray14/train_val_list.txt"
    test_list: str = "data/chestxray14/test_list.txt"
    images_dir: str = "data/chestxray14/images"

    # MIMIC-CXR-JPG specific
    mimic_chexpert_csv: str = "data/mimic/mimic-cxr-2.0.0-chexpert.csv"
    mimic_split_csv: str = "data/mimic/mimic-cxr-2.0.0-split.csv"
    mimic_metadata_csv: str = "data/mimic/mimic-cxr-2.0.0-metadata.csv"
    mimic_images_dir: str = "data/mimic/files"

    # Sentiment140
    sentiment_csv: str = "data/sentiment140/training.1600000.processed.noemoticon.csv"

    # ── Model ─────────────────────────────────────────────────────────────
    image_model: str = "densenet121-res224-all"
    radbert_model: str = "StanfordAIMI/RadBERT"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment"

    # ── Data ──────────────────────────────────────────────────────────────
    image_size: int = 224
    num_workers: int = 4
    batch_size: int = 32
    val_split: float = 0.1
    seed: int = 42

    # ── Labels ────────────────────────────────────────────────────────────
    labels: List[str] = field(default_factory=lambda: CHESTXRAY14_LABELS)
    num_classes: int = 15  # 14 pathologies + No Finding

    # ── XAI ───────────────────────────────────────────────────────────────
    gradcam_method: str = "gradcam"        # "gradcam" | "gradcam++" | "eigencam"
    n_ig_steps: int = 50
    n_shap_samples: int = 100
    disease_threshold: float = 0.5

    # ── Device ────────────────────────────────────────────────────────────
    device: str = "cpu"

    # ── Logging ───────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Dataset type ──────────────────────────────────────────────────────
    dataset: str = "chestxray14"  # "chestxray14" | "mimic" | "sentiment140"

    def __post_init__(self) -> None:
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load Config from a YAML file."""
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str) -> None:
        """Save Config to a YAML file."""
        with open(path, "w") as fh:
            yaml.safe_dump(asdict(self), fh, default_flow_style=False)

    def resolve_device(self) -> torch.device:
        """Return the best available device respecting the config preference."""
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
