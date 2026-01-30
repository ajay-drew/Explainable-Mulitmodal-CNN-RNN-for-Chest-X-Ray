"""
Image and text preprocessing for MIMIC-CXR.

Per PROJECT_PLAN §3.1:
- Images: Resize to 224x224, normalize (mean=0.485, std=0.229), augment
- Text: Tokenize (BERT tokenizer), clean, pad to 512 tokens
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer


class ImagePreprocessor:
    """
    Preprocess chest X-ray images for the CNN encoder.
    
    Per PROJECT_PLAN §3.1:
    - Resize to 224x224
    - Normalize with mean=0.485, std=0.229
    - Augmentation: random flips, rotations (training only)
    """
    
    def __init__(
        self,
        image_size: int = 224,
        mean: float = 0.485,
        std: float = 0.229,
        augment: bool = False,
    ):
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.augment = augment
        
        # Build transform pipeline
        self.transform = self._build_transform()
    
    def _build_transform(self) -> transforms.Compose:
        """Build the image transformation pipeline."""
        transform_list = []
        
        # Resize
        transform_list.append(transforms.Resize((self.image_size, self.image_size)))
        
        # Augmentation (training only)
        if self.augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            ])
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean], std=[self.std]),
        ])
        
        return transforms.Compose(transform_list)
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a single image.
        
        Args:
            image: PIL Image (grayscale or RGB)
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        # Convert to grayscale if RGB
        if image.mode == "RGB":
            image = image.convert("L")
        
        # Apply transforms
        return self.transform(image)
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """Preprocess a batch of images."""
        return torch.stack([self(img) for img in images])


class TextPreprocessor:
    """
    Preprocess radiology reports for the text encoder.
    
    Per PROJECT_PLAN §9:
    - Uses RadBERT or Bio_ClinicalBERT tokenizer
    - Max length: 512 tokens
    - Returns input_ids, attention_mask
    """
    
    def __init__(
        self,
        model_name: str = "UCSD-VA-health/RadBERT-RoBERTa-4m",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def clean_text(self, text: str) -> str:
        """
        Clean radiology report text.
        
        Args:
            text: Raw report text
            
        Returns:
            Cleaned text
        """
        if text is None:
            return ""
        
        # Basic cleaning
        text = text.strip()
        text = " ".join(text.split())  # Normalize whitespace
        
        # TODO: Add more cleaning as needed
        # - Remove de-identification markers
        # - Handle section headers
        
        return text
    
    def __call__(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Tokenize and preprocess a single report.
        
        Args:
            text: Raw report text
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
        }
    
    def preprocess_batch(self, texts: list) -> Dict[str, torch.Tensor]:
        """
        Tokenize a batch of reports.
        
        Args:
            texts: List of report texts
            
        Returns:
            Dictionary with batched input_ids and attention_mask
        """
        # Clean texts
        texts = [self.clean_text(t) for t in texts]
        
        # Batch tokenize
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
