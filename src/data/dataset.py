"""
PyTorch Dataset for MIMIC-CXR paired image-report data.

Per PROJECT_PLAN §3.1:
- Load paired chest X-ray images and radiology reports
- Multi-label classification (13 diseases + No Findings)
- Exclude lateral views and ambiguous labels
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .preprocessing import ImagePreprocessor, TextPreprocessor


class MIMICCXRDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-CXR multimodal data.
    
    Loads paired chest X-ray images and radiology reports with
    multi-label disease annotations.
    """
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_preprocessor: Optional[ImagePreprocessor] = None,
        text_preprocessor: Optional[TextPreprocessor] = None,
        labels_file: str = "mimic-cxr-2.0.0-chexpert.csv",
        reports_file: str = "mimic-cxr-reports.csv",
        splits_file: str = "mimic-cxr-2.0.0-split.csv",
        exclude_lateral: bool = True,
        disease_labels: Optional[List[str]] = None,
    ):
        """
        Initialize MIMIC-CXR dataset.
        
        Args:
            data_root: Path to MIMIC-CXR data directory
            split: One of "train", "validate", "test"
            image_preprocessor: Image preprocessing pipeline
            text_preprocessor: Text preprocessing pipeline
            labels_file: Path to CheXpert labels CSV
            reports_file: Path to reports CSV
            splits_file: Path to train/val/test splits CSV
            exclude_lateral: Whether to exclude lateral view images
            disease_labels: List of disease label columns
        """
        self.data_root = Path(data_root)
        self.split = split
        self.exclude_lateral = exclude_lateral
        
        # Preprocessors (use defaults if not provided)
        self.image_preprocessor = image_preprocessor or ImagePreprocessor(
            augment=(split == "train")
        )
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        
        # Disease labels (per PROJECT_PLAN)
        self.disease_labels = disease_labels or [
            "Atelectasis",
            "Cardiomegaly",
            "Consolidation",
            "Edema",
            "Enlarged Cardiomediastinum",
            "Fracture",
            "Lung Lesion",
            "Lung Opacity",
            "No Finding",
            "Pleural Effusion",
            "Pleural Other",
            "Pneumonia",
            "Pneumothorax",
            "Support Devices",
        ]
        
        # Load and prepare data
        self.data = self._load_data(labels_file, reports_file, splits_file)
    
    def _load_data(
        self,
        labels_file: str,
        reports_file: str,
        splits_file: str,
    ) -> pd.DataFrame:
        """Load and merge data files."""
        # TODO: Implement actual data loading for MIMIC-CXR
        # This is a placeholder structure
        
        # In practice:
        # 1. Load labels CSV (CheXpert format)
        # 2. Load reports CSV (or extract from files)
        # 3. Load splits CSV
        # 4. Merge on study_id/subject_id
        # 5. Filter by split and exclude lateral views
        
        # Placeholder: return empty DataFrame with expected columns
        columns = ["dicom_id", "subject_id", "study_id", "image_path", "report"] + self.disease_labels
        return pd.DataFrame(columns=columns)
    
    def _load_image(self, image_path: str) -> Image.Image:
        """Load a chest X-ray image."""
        full_path = self.data_root / image_path
        
        # Handle DICOM vs JPG
        if str(full_path).endswith(".dcm"):
            # TODO: Load DICOM with pydicom
            raise NotImplementedError("DICOM loading not yet implemented")
        else:
            return Image.open(full_path).convert("L")
    
    def _get_labels(self, idx: int) -> torch.Tensor:
        """Get multi-label tensor for a sample."""
        row = self.data.iloc[idx]
        
        labels = []
        for disease in self.disease_labels:
            # CheXpert uses: 1=positive, 0=negative, -1=uncertain, NaN=missing
            # Convert to binary (treat uncertain as positive or use custom handling)
            value = row.get(disease, 0)
            if pd.isna(value):
                labels.append(0.0)
            elif value == -1:
                # Uncertainty handling: treat as positive (U-Ones) or 0 (U-Zeros)
                labels.append(1.0)  # U-Ones strategy
            else:
                labels.append(float(value))
        
        return torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - image: Preprocessed image tensor [C, H, W]
            - input_ids: Tokenized report input IDs [seq_len]
            - attention_mask: Attention mask [seq_len]
            - labels: Multi-label tensor [num_classes]
            - idx: Sample index
        """
        row = self.data.iloc[idx]
        
        # Load and preprocess image
        image = self._load_image(row["image_path"])
        image_tensor = self.image_preprocessor(image)
        
        # Preprocess report text
        text_encoded = self.text_preprocessor(row["report"])
        
        # Get labels
        labels = self._get_labels(idx)
        
        return {
            "image": image_tensor,
            "input_ids": text_encoded["input_ids"],
            "attention_mask": text_encoded["attention_mask"],
            "labels": labels,
            "idx": idx,
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "idx": torch.tensor([item["idx"] for item in batch]),
    }
