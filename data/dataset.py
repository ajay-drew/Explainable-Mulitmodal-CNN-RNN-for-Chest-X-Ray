"""
Dataset classes for:
  - ChestXray14Dataset  (NIH ChestX-ray14)
  - MIMICCXRDataset     (MIMIC-CXR-JPG)
  - SentimentDataset    (Sentiment140)

All datasets work with pretrained models — no training from scratch.
"""
from __future__ import annotations

import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from data.preprocessing import (
    _clean_tweet,
    get_radbert_tokenizer,
    get_twitter_tokenizer,
    get_val_transform,
    tokenize_radiology_report,
    tokenize_tweet,
)

# 14 target labels (excluding "No Finding" for multi-label binarization)
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


# ─────────────────────────────────────────────────────────────────────────────
# ChestX-ray14
# ─────────────────────────────────────────────────────────────────────────────

class ChestXray14Dataset(Dataset):
    """
    NIH ChestX-ray14 dataset.

    Expected directory layout::

        data_dir/
        ├── Data_Entry_2017.csv
        ├── train_val_list.txt
        ├── test_list.txt
        └── images/
            ├── 00000001_000.png
            └── ...

    Args:
        data_dir:   Root directory containing the CSV and images/ folder.
        split:      One of "train", "val", "test".
        transform:  Callable applied to PIL Image → Tensor. Defaults to
                    the standard XRV val transform (no augmentation).
        val_frac:   Fraction of train_val_list to use as validation.
        seed:       Random seed for train/val split.
        labels:     Ordered list of 15 pathology label names.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        val_frac: float = 0.1,
        seed: int = 42,
        labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or get_val_transform()
        self.labels = labels or CHESTXRAY14_LABELS
        self.num_classes = len(self.labels)
        self.df = self._load_data(val_frac, seed)

    def _load_data(self, val_frac: float, seed: int) -> pd.DataFrame:
        csv_path = self.data_dir / "Data_Entry_2017.csv"
        train_list_path = self.data_dir / "train_val_list.txt"
        test_list_path = self.data_dir / "test_list.txt"

        for p in (csv_path, train_list_path, test_list_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing required file: {p}\n"
                    "Run `python data/dataset.py --download` for download instructions."
                )

        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        # Keep PA and AP views only
        if "View Position" in df.columns:
            df = df[df["View Position"].isin(["PA", "AP"])].copy()

        # Load split lists
        train_val_images = set(
            Path(train_list_path).read_text().strip().splitlines()
        )
        test_images = set(Path(test_list_path).read_text().strip().splitlines())

        # Build binary label vectors
        label_matrix = np.zeros((len(df), self.num_classes), dtype=np.float32)
        for idx, findings_str in enumerate(df["Finding Labels"]):
            findings = [f.strip() for f in str(findings_str).split("|")]
            for label_idx, label in enumerate(self.labels):
                if label in findings:
                    label_matrix[idx, label_idx] = 1.0

        label_cols = {self.labels[i]: label_matrix[:, i] for i in range(self.num_classes)}
        df = df.assign(**label_cols)

        # Split
        image_col = "Image Index"
        train_val_df = df[df[image_col].isin(train_val_images)].copy()
        test_df = df[df[image_col].isin(test_images)].copy()

        rng = np.random.default_rng(seed)
        n_val = int(len(train_val_df) * val_frac)
        val_indices = rng.choice(len(train_val_df), size=n_val, replace=False)
        val_mask = np.zeros(len(train_val_df), dtype=bool)
        val_mask[val_indices] = True

        if self.split == "train":
            result = train_val_df[~val_mask].reset_index(drop=True)
        elif self.split == "val":
            result = train_val_df[val_mask].reset_index(drop=True)
        else:
            result = test_df.reset_index(drop=True)

        return result

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image_path = self.data_dir / "images" / row["Image Index"]
        image = Image.open(image_path).convert("L")
        tensor = self.transform(image)
        labels = torch.tensor(
            [row[lbl] for lbl in self.labels], dtype=torch.float32
        )
        meta = {
            "image_id": row["Image Index"],
            "age": row.get("Patient Age", -1),
            "gender": row.get("Patient Gender", "U"),
        }
        return {"image": tensor, "labels": labels, "meta": meta}

    @classmethod
    def download(cls) -> None:
        """Print instructions for downloading ChestX-ray14."""
        instructions = textwrap.dedent("""
        ════════════════════════════════════════════════════════════════
        ChestX-ray14 Dataset Download Instructions
        ════════════════════════════════════════════════════════════════

        The NIH ChestX-ray14 dataset requires manual download from
        the NIH Box storage. Follow these steps:

        1. Visit the dataset page:
           https://nihcc.app.box.com/v/ChestXray-NIHCC

        2. Download the following files into data/chestxray14/:
           - images_001.tar.gz through images_012.tar.gz  (~42 GB total)
           - Data_Entry_2017.csv
           - train_val_list.txt
           - test_list.txt

        3. Extract all image archives:
           cd data/chestxray14
           for f in images_*.tar.gz; do tar -xzf "$f"; done

        4. Verify directory structure:
           data/chestxray14/
           ├── Data_Entry_2017.csv
           ├── train_val_list.txt
           ├── test_list.txt
           └── images/
               └── *.png  (~112,000 files)

        Alternative (academic mirrors):
          Kaggle: https://www.kaggle.com/datasets/nih-chest-xrays/data

        ════════════════════════════════════════════════════════════════
        """)
        print(instructions)


# ─────────────────────────────────────────────────────────────────────────────
# MIMIC-CXR-JPG
# ─────────────────────────────────────────────────────────────────────────────

class MIMICCXRDataset(Dataset):
    """
    MIMIC-CXR-JPG dataset using CheXpert labels.

    Required CSVs (from PhysioNet):
      - mimic-cxr-2.0.0-chexpert.csv  (labels)
      - mimic-cxr-2.0.0-split.csv     (train/validate/test split)
      - mimic-cxr-2.0.0-metadata.csv  (age, gender, view position)

    Args:
        data_dir:    Root directory containing the three CSVs and files/ folder.
        split:       One of "train", "validate", "test".
        transform:   Image transform. Defaults to XRV val transform.
        labels:      Ordered list of label names.
    """

    CHEXPERT_COLS: List[str] = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",
        "No Finding", "Pleural Effusion", "Pleural Other", "Pneumonia",
        "Pneumothorax", "Support Devices",
    ]

    # Map CheXpert column names → our 15-label names
    CHEXPERT_TO_TARGET: Dict[str, str] = {
        "Atelectasis": "Atelectasis",
        "Cardiomegaly": "Cardiomegaly",
        "Consolidation": "Consolidation",
        "Edema": "Edema",
        "Pleural Effusion": "Effusion",
        "Pneumonia": "Pneumonia",
        "Pneumothorax": "Pneumothorax",
        "No Finding": "No Finding",
    }

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        labels: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        assert split in ("train", "validate", "test"), f"Unknown split: {split}"
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform or get_val_transform()
        self.labels = labels or CHESTXRAY14_LABELS
        self.num_classes = len(self.labels)
        self.df = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        chex_csv = self.data_dir / "mimic-cxr-2.0.0-chexpert.csv"
        split_csv = self.data_dir / "mimic-cxr-2.0.0-split.csv"
        meta_csv = self.data_dir / "mimic-cxr-2.0.0-metadata.csv"

        for p in (chex_csv, split_csv, meta_csv):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing required MIMIC file: {p}\n"
                    "Download from https://physionet.org/content/mimic-cxr-jpg/2.0.0/"
                )

        chex = pd.read_csv(chex_csv)
        splits = pd.read_csv(split_csv)
        meta = pd.read_csv(meta_csv)

        # Merge on subject_id + study_id
        df = chex.merge(splits, on=["subject_id", "study_id"])
        df = df.merge(
            meta[["dicom_id", "subject_id", "study_id", "ViewPosition",
                   "PatientAge", "PatientSex"]],
            on=["subject_id", "study_id"],
            how="left",
        )

        # Keep frontal views only
        df = df[df["ViewPosition"].isin(["PA", "AP"])].copy()

        # Filter by split
        df = df[df["split"] == self.split].reset_index(drop=True)

        # U-Ones strategy: NaN → 1.0 (uncertain = positive)
        for col in self.CHEXPERT_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(1.0)

        # Build label vectors aligned to our 15 labels
        for label in self.labels:
            source_col = None
            for chex_col, target in self.CHEXPERT_TO_TARGET.items():
                if target == label and chex_col in df.columns:
                    source_col = chex_col
                    break
            if source_col is not None:
                df[label] = df[source_col].astype(np.float32)
            else:
                df[label] = 0.0

        # Build image paths: files/p{prefix}/p{subject_id}/s{study_id}/{dicom_id}.jpg
        def _image_path(row: pd.Series) -> str:
            sid = str(row["subject_id"])
            prefix = sid[:2]
            return str(
                self.data_dir
                / "files"
                / f"p{prefix}"
                / f"p{sid}"
                / f"s{row['study_id']}"
                / f"{row['dicom_id']}.jpg"
            )

        df["image_path"] = df.apply(_image_path, axis=1)
        return df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("L")
        tensor = self.transform(image)
        labels = torch.tensor(
            [row[lbl] for lbl in self.labels], dtype=torch.float32
        )
        meta = {
            "image_id": row.get("dicom_id", ""),
            "age": row.get("PatientAge", -1),
            "gender": row.get("PatientSex", "U"),
        }
        return {"image": tensor, "labels": labels, "meta": meta}


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment140
# ─────────────────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """
    Sentiment140 dataset (1.6M tweets).

    CSV columns (no header): target, ids, date, flag, user, text
    target: 0 = negative, 4 = positive → mapped to 0 / 1.

    Args:
        csv_path:    Path to the Sentiment140 CSV.
        tokenizer:   HuggingFace tokenizer. Loaded lazily if None.
        max_length:  Max token length.
        max_samples: If set, truncate dataset to this many rows (useful for dev).
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: Optional[AutoTokenizer] = None,
        max_length: int = 128,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path)
        self.tokenizer = tokenizer or get_twitter_tokenizer()
        self.max_length = max_length

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Sentiment140 CSV not found at {self.csv_path}.\n"
                "Download from https://www.kaggle.com/datasets/kazanova/sentiment140"
            )

        df = pd.read_csv(
            csv_path,
            encoding="latin-1",
            header=None,
            names=["target", "ids", "date", "flag", "user", "text"],
        )
        df["label"] = (df["target"] == 4).astype(np.int64)
        df["text"] = df["text"].astype(str).apply(_clean_tweet)
        df = df[df["text"].str.len() > 0].reset_index(drop=True)

        if max_samples is not None:
            df = df.iloc[:max_samples].reset_index(drop=True)

        self.texts: List[str] = df["text"].tolist()
        self.labels: List[int] = df["label"].tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "text": self.texts[idx],
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Print dataset download instructions",
    )
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        metavar="DATA_DIR",
        help="Verify ChestX-ray14 dataset at DATA_DIR",
    )
    args = parser.parse_args()

    if args.download:
        ChestXray14Dataset.download()
    elif args.verify:
        try:
            ds = ChestXray14Dataset(args.verify, split="train")
            print(f"Dataset loaded: {len(ds)} training samples")
            sample = ds[0]
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Labels: {sample['labels']}")
        except Exception as exc:
            print(f"Verification failed: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        parser.print_help()
