"""
Evaluation and metrics for the multimodal classifier.

Per PROJECT_PLAN §3.6:
- Metrics: Accuracy, Precision, Recall, F1-Score, AUROC
- Target: AUROC macro >0.816, accuracy >92%
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    multilabel_confusion_matrix,
)
from tqdm import tqdm


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Multimodal classifier
        dataloader: Evaluation DataLoader
        criterion: Loss function
        device: Device
        class_names: List of class names
        threshold: Classification threshold
        
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            image = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(image, input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all predictions
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(
        all_logits.numpy(),
        all_labels.numpy(),
        class_names=class_names,
        threshold=threshold,
    )
    
    metrics["val_loss"] = total_loss / len(dataloader)
    
    return metrics


def compute_metrics(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        logits: Model logits [N, num_classes]
        labels: Ground truth labels [N, num_classes]
        class_names: List of class names
        threshold: Classification threshold
        
    Returns:
        Dictionary with metrics
    """
    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    
    # Binary predictions
    preds = (probs >= threshold).astype(int)
    
    metrics = {}
    
    # === Overall Metrics ===
    
    # AUROC (macro and per-class)
    try:
        auroc_macro = roc_auc_score(labels, probs, average="macro")
        auroc_micro = roc_auc_score(labels, probs, average="micro")
        auroc_weighted = roc_auc_score(labels, probs, average="weighted")
        metrics["auroc_macro"] = auroc_macro
        metrics["auroc_micro"] = auroc_micro
        metrics["auroc_weighted"] = auroc_weighted
    except ValueError:
        # Handle case where some classes have no positive samples
        metrics["auroc_macro"] = 0.0
        metrics["auroc_micro"] = 0.0
        metrics["auroc_weighted"] = 0.0
    
    # Average Precision (mAP)
    try:
        ap_macro = average_precision_score(labels, probs, average="macro")
        metrics["ap_macro"] = ap_macro
    except ValueError:
        metrics["ap_macro"] = 0.0
    
    # Accuracy (subset accuracy for multi-label)
    metrics["accuracy"] = accuracy_score(labels, preds)
    
    # Precision, Recall, F1 (macro)
    metrics["precision_macro"] = precision_score(labels, preds, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(labels, preds, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
    
    # Micro metrics
    metrics["precision_micro"] = precision_score(labels, preds, average="micro", zero_division=0)
    metrics["recall_micro"] = recall_score(labels, preds, average="micro", zero_division=0)
    metrics["f1_micro"] = f1_score(labels, preds, average="micro", zero_division=0)
    
    # === Per-Class Metrics ===
    num_classes = labels.shape[1]
    class_names = class_names or [f"class_{i}" for i in range(num_classes)]
    
    for i, name in enumerate(class_names):
        try:
            auroc = roc_auc_score(labels[:, i], probs[:, i])
        except ValueError:
            auroc = 0.0
        
        precision = precision_score(labels[:, i], preds[:, i], zero_division=0)
        recall = recall_score(labels[:, i], preds[:, i], zero_division=0)
        f1 = f1_score(labels[:, i], preds[:, i], zero_division=0)
        
        metrics[f"auroc_{name}"] = auroc
        metrics[f"precision_{name}"] = precision
        metrics[f"recall_{name}"] = recall
        metrics[f"f1_{name}"] = f1
    
    return metrics


def get_classification_report(
    logits: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> str:
    """Generate sklearn classification report."""
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    
    return classification_report(
        labels,
        preds,
        target_names=class_names,
        zero_division=0,
    )


def compute_confusion_matrices(
    logits: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Compute confusion matrix for each class."""
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    
    return multilabel_confusion_matrix(labels, preds)
