"""
Preprocessing pipeline for chest X-ray images and radiology text.

Image normalization: TorchXRayVision convention — float32, range [-1024, 1024],
single-channel grayscale, shape (B, 1, 224, 224).

Text tokenization: RadBERT for Mode A, TwitterRoBERTa for Mode B.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Image preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _to_grayscale_pil(image: Image.Image) -> Image.Image:
    """Convert any PIL Image to single-channel grayscale L."""
    return image.convert("L")


def _pil_to_xrv_tensor(image: Image.Image, size: int = 224) -> torch.Tensor:
    """
    Convert a PIL grayscale image to a TorchXRayVision-compatible tensor.

    Steps:
    1. Resize to (size, size) using LANCZOS
    2. Convert to float32 numpy array
    3. Scale pixel values from [0, 255] to [-1024, 1024]
    4. Return shape (1, size, size) float32 tensor

    Args:
        image: Grayscale PIL Image.
        size:  Target spatial size (default 224).

    Returns:
        Tensor of shape (1, H, W), dtype float32, range [-1024, 1024].
    """
    img = _to_grayscale_pil(image)
    img = img.resize((size, size), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)          # [0, 255]
    arr = (arr / 255.0) * 2048.0 - 1024.0          # [-1024, 1024]
    tensor = torch.from_numpy(arr).unsqueeze(0)     # (1, H, W)
    return tensor


def get_train_transform(size: int = 224) -> transforms.Compose:
    """
    Augmentation pipeline for training chest X-rays.
    All geometric augmentations happen on the grayscale PIL image BEFORE
    the [-1024, 1024] scaling.

    Returns:
        A torchvision Compose transform that accepts a PIL Image and
        returns a (1, size, size) float32 tensor in [-1024, 1024].
    """
    def _augment_and_scale(img: Image.Image) -> torch.Tensor:
        # Apply geometric augmentations on PIL (grayscale safe)
        aug = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((size + 16, size + 16), Image.LANCZOS),
            transforms.RandomCrop(size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
        ])
        img = aug(img)
        return _pil_to_xrv_tensor(img, size)

    return transforms.Lambda(_augment_and_scale)


def get_val_transform(size: int = 224) -> transforms.Compose:
    """
    Deterministic transform for validation/test chest X-rays.

    Returns:
        A torchvision Compose transform that accepts a PIL Image and
        returns a (1, size, size) float32 tensor in [-1024, 1024].
    """
    def _scale(img: Image.Image) -> torch.Tensor:
        return _pil_to_xrv_tensor(img, size)

    return transforms.Lambda(_scale)


def preprocess_xray_array(arr: np.ndarray, size: int = 224) -> torch.Tensor:
    """
    Preprocess a raw numpy uint8 array (H, W) or (H, W, C) to XRV tensor.

    Args:
        arr:  Numpy array, uint8 or float.
        size: Target spatial size.

    Returns:
        Tensor (1, size, size), float32, [-1024, 1024].
    """
    if arr.ndim == 3:
        img = Image.fromarray(arr.astype(np.uint8)).convert("L")
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode="L")
    return _pil_to_xrv_tensor(img, size)


# ─────────────────────────────────────────────────────────────────────────────
# Text preprocessing
# ─────────────────────────────────────────────────────────────────────────────

_RADBERT_TOKENIZER: Optional[AutoTokenizer] = None
_TWITTER_TOKENIZER: Optional[AutoTokenizer] = None


def get_radbert_tokenizer(model_name: str = "StanfordAIMI/RadBERT") -> AutoTokenizer:
    """Lazy-load and cache the RadBERT tokenizer."""
    global _RADBERT_TOKENIZER
    if _RADBERT_TOKENIZER is None:
        _RADBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _RADBERT_TOKENIZER


def get_twitter_tokenizer(
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment",
) -> AutoTokenizer:
    """Lazy-load and cache the TwitterRoBERTa tokenizer."""
    global _TWITTER_TOKENIZER
    if _TWITTER_TOKENIZER is None:
        _TWITTER_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TWITTER_TOKENIZER


def tokenize_radiology_report(
    text: str,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 512,
) -> dict:
    """
    Tokenize a radiology report for RadBERT.

    Args:
        text:       Raw report string.
        tokenizer:  AutoTokenizer instance. Loaded lazily if None.
        max_length: Maximum token length (default 512).

    Returns:
        Dict with keys 'input_ids' and 'attention_mask', each a 1D LongTensor.
    """
    if tokenizer is None:
        tokenizer = get_radbert_tokenizer()
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }


def _clean_tweet(text: str) -> str:
    """Remove URLs, mentions, hashtag symbols and HTML entities from tweet text."""
    text = re.sub(r"http\S+|www\.\S+", "", text)           # URLs
    text = re.sub(r"@\w+", "", text)                        # mentions
    text = re.sub(r"#", "", text)                           # hashtag symbol only
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_tweet(
    text: str,
    tokenizer: Optional[AutoTokenizer] = None,
    max_length: int = 128,
) -> dict:
    """
    Clean and tokenize a tweet for TwitterRoBERTa.

    Args:
        text:       Raw tweet string.
        tokenizer:  AutoTokenizer instance. Loaded lazily if None.
        max_length: Maximum token length (default 128).

    Returns:
        Dict with keys 'input_ids' and 'attention_mask', each a 1D LongTensor.
    """
    if tokenizer is None:
        tokenizer = get_twitter_tokenizer()
    text = _clean_tweet(text)
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0),
    }
