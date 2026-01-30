"""
DataLoader utilities for MIMIC-CXR dataset.

Provides train/val/test DataLoaders with proper batching and shuffling.
"""

from typing import Dict, Optional, Tuple

from torch.utils.data import DataLoader

from .dataset import MIMICCXRDataset, collate_fn
from .preprocessing import ImagePreprocessor, TextPreprocessor


def get_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 224,
    max_text_length: int = 512,
    text_model_name: str = "UCSD-VA-health/RadBERT-RoBERTa-4m",
) -> Dict[str, DataLoader]:
    """
    Create train, validation, and test DataLoaders.
    
    Args:
        data_root: Path to MIMIC-CXR data
        batch_size: Batch size (per PROJECT_PLAN: 32)
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        image_size: Target image size (per PROJECT_PLAN: 224)
        max_text_length: Max tokens (per PROJECT_PLAN: 512)
        text_model_name: HuggingFace model name for tokenizer
        
    Returns:
        Dictionary with "train", "val", "test" DataLoaders
    """
    # Create preprocessors
    train_image_preprocessor = ImagePreprocessor(
        image_size=image_size,
        augment=True,
    )
    eval_image_preprocessor = ImagePreprocessor(
        image_size=image_size,
        augment=False,
    )
    text_preprocessor = TextPreprocessor(
        model_name=text_model_name,
        max_length=max_text_length,
    )
    
    # Create datasets
    train_dataset = MIMICCXRDataset(
        data_root=data_root,
        split="train",
        image_preprocessor=train_image_preprocessor,
        text_preprocessor=text_preprocessor,
    )
    
    val_dataset = MIMICCXRDataset(
        data_root=data_root,
        split="validate",
        image_preprocessor=eval_image_preprocessor,
        text_preprocessor=text_preprocessor,
    )
    
    test_dataset = MIMICCXRDataset(
        data_root=data_root,
        split="test",
        image_preprocessor=eval_image_preprocessor,
        text_preprocessor=text_preprocessor,
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
