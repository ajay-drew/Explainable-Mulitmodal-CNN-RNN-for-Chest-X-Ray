"""
DataLoader factory for ChestXray14, MIMIC-CXR-JPG, and Sentiment140 datasets.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset import ChestXray14Dataset, MIMICCXRDataset, SentimentDataset
from data.preprocessing import get_train_transform, get_val_transform


def _chestxray14_collate(batch) -> Dict:
    """Custom collate that stacks images, labels, and preserves meta dicts."""
    images = torch.stack([item["image"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    metas = [item["meta"] for item in batch]
    return {"image": images, "labels": labels, "meta": metas}


def _sentiment_collate(batch) -> Dict:
    """Collate for Sentiment140 batches."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    texts = [item["text"] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label": labels,
        "text": texts,
    }


def get_chestxray14_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    val_frac: float = 0.1,
    seed: int = 42,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/val/test DataLoaders for ChestX-ray14.

    Args:
        data_dir:    Root directory of the ChestX-ray14 dataset.
        batch_size:  Batch size.
        num_workers: DataLoader worker count.
        val_frac:    Fraction of train_val_list used as validation.
        seed:        Random seed for the val split.
        image_size:  Spatial size of processed images.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = ChestXray14Dataset(
        data_dir, split="train",
        transform=get_train_transform(image_size),
        val_frac=val_frac, seed=seed,
    )
    val_ds = ChestXray14Dataset(
        data_dir, split="val",
        transform=get_val_transform(image_size),
        val_frac=val_frac, seed=seed,
    )
    test_ds = ChestXray14Dataset(
        data_dir, split="test",
        transform=get_val_transform(image_size),
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_chestxray14_collate,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader


def get_mimic_loaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/validate/test DataLoaders for MIMIC-CXR-JPG.

    Args:
        data_dir:    Root directory of the MIMIC-CXR-JPG dataset.
        batch_size:  Batch size.
        num_workers: DataLoader worker count.
        image_size:  Spatial size.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds = MIMICCXRDataset(
        data_dir, split="train", transform=get_train_transform(image_size)
    )
    val_ds = MIMICCXRDataset(
        data_dir, split="validate", transform=get_val_transform(image_size)
    )
    test_ds = MIMICCXRDataset(
        data_dir, split="test", transform=get_val_transform(image_size)
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_chestxray14_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
        DataLoader(test_ds, shuffle=False, **kwargs),
    )


def get_sentiment_loaders(
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    val_frac: float = 0.1,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train/val DataLoaders for Sentiment140.

    Args:
        csv_path:    Path to the Sentiment140 CSV file.
        batch_size:  Batch size.
        num_workers: DataLoader worker count.
        max_samples: Cap dataset size (useful for quick experiments).
        val_frac:    Fraction of data used as validation.
        seed:        Random seed for the split.

    Returns:
        (train_loader, val_loader)
    """
    from torch.utils.data import random_split

    full_ds = SentimentDataset(csv_path, max_samples=max_samples)
    n_val = int(len(full_ds) * val_frac)
    n_train = len(full_ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_sentiment_collate,
        pin_memory=torch.cuda.is_available(),
    )
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=False, **kwargs),
    )
