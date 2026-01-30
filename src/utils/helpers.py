"""
Utility helper functions.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """
    Get the best available device.
    
    Args:
        preferred: Preferred device ("cuda", "mps", "cpu")
        
    Returns:
        torch.device
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif preferred == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        trainable_only: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def save_model(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict] = None,
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        path: Save path
        optimizer: Optional optimizer to save
        epoch: Optional epoch number
        metrics: Optional metrics dictionary
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint["epoch"] = epoch
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_model(
    model: nn.Module,
    path: str,
    device: str = "cpu",
    strict: bool = True,
) -> dict:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model (architecture must match)
        path: Checkpoint path
        device: Device to load to
        strict: Strict state dict loading
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint


def format_metrics(metrics: dict, precision: int = 4) -> str:
    """Format metrics dictionary as string."""
    lines = []
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            lines.append(f"{key}: {value:.{precision}f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop
