"""
Evaluation metrics for multilabel chest X-ray classification.

Metrics:
    - Per-class and macro AUROC
    - Per-class and macro Average Precision (mAP)
    - Per-class F1, precision, recall at a given threshold
    - Overall accuracy
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def compute_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute per-class and macro AUROC.

    Args:
        y_true:  (N, C) binary ground-truth array.
        y_score: (N, C) predicted probability array.
        labels:  Optional label names for the result dict.

    Returns:
        Dict mapping label names (or "class_i") and "macro" → AUROC float.
        Classes with fewer than 2 unique labels get NaN.
    """
    n_classes = y_true.shape[1]
    names = labels if labels else [f"class_{i}" for i in range(n_classes)]
    result: Dict[str, float] = {}
    valid_aurocs: List[float] = []

    for i, name in enumerate(names):
        col_true = y_true[:, i]
        col_score = y_score[:, i]
        if len(np.unique(col_true)) < 2:
            result[name] = float("nan")
            continue
        try:
            auc = float(roc_auc_score(col_true, col_score))
            result[name] = auc
            valid_aurocs.append(auc)
        except Exception as exc:
            logger.warning(f"AUROC failed for {name}: {exc}")
            result[name] = float("nan")

    result["macro"] = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    return result


def compute_map(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute per-class and macro Average Precision (mAP).

    Args:
        y_true:  (N, C) binary ground-truth array.
        y_score: (N, C) predicted probability array.
        labels:  Optional label names.

    Returns:
        Dict mapping label → AP float, plus "macro" key.
    """
    n_classes = y_true.shape[1]
    names = labels if labels else [f"class_{i}" for i in range(n_classes)]
    result: Dict[str, float] = {}
    valid_aps: List[float] = []

    for i, name in enumerate(names):
        col_true = y_true[:, i]
        col_score = y_score[:, i]
        if col_true.sum() == 0:
            result[name] = float("nan")
            continue
        try:
            ap = float(average_precision_score(col_true, col_score))
            result[name] = ap
            valid_aps.append(ap)
        except Exception as exc:
            logger.warning(f"AP failed for {name}: {exc}")
            result[name] = float("nan")

    result["macro"] = float(np.mean(valid_aps)) if valid_aps else float("nan")
    return result


def compute_f1_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    labels: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-class F1, precision, and recall at a fixed threshold.

    Args:
        y_true:     (N, C) binary ground-truth array.
        y_score:    (N, C) predicted probability array.
        threshold:  Binarization threshold.
        labels:     Optional label names.

    Returns:
        Dict with keys "f1", "precision", "recall", each mapping label → float.
        Also includes a "macro" key in each sub-dict.
    """
    y_pred = (y_score >= threshold).astype(np.int32)
    n_classes = y_true.shape[1]
    names = labels if labels else [f"class_{i}" for i in range(n_classes)]

    metrics: Dict[str, Dict[str, float]] = {
        "f1": {}, "precision": {}, "recall": {}
    }

    for i, name in enumerate(names):
        col_true = y_true[:, i]
        col_pred = y_pred[:, i]
        metrics["f1"][name] = float(
            f1_score(col_true, col_pred, zero_division=0)
        )
        metrics["precision"][name] = float(
            precision_score(col_true, col_pred, zero_division=0)
        )
        metrics["recall"][name] = float(
            recall_score(col_true, col_pred, zero_division=0)
        )

    # Macro averages (ignore NaN)
    for key in ("f1", "precision", "recall"):
        vals = [v for v in metrics[key].values() if not np.isnan(v)]
        metrics[key]["macro"] = float(np.mean(vals)) if vals else float("nan")

    return metrics


def evaluate_model(
    model,
    dataloader,
    device,
    labels: Optional[List[str]] = None,
    threshold: float = 0.5,
) -> Dict[str, object]:
    """
    Run full evaluation on a dataloader and return a metrics dict.

    Args:
        model:      Model with a forward() that returns dict with 'logits' key.
        dataloader: DataLoader yielding batches with 'image' and 'labels' keys.
        device:     Torch device.
        labels:     Label names for the metrics dict.
        threshold:  Binarization threshold for F1/precision/recall.

    Returns:
        Dict with keys "auroc", "map", "f1", "precision", "recall".
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm

    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            label_tensor = batch["labels"].cpu().numpy()

            out = model(images)
            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(label_tensor)

    y_score = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    auroc = compute_auroc(y_true, y_score, labels)
    ap = compute_map(y_true, y_score, labels)
    f1_metrics = compute_f1_metrics(y_true, y_score, threshold, labels)

    return {
        "auroc": auroc,
        "map": ap,
        "f1": f1_metrics["f1"],
        "precision": f1_metrics["precision"],
        "recall": f1_metrics["recall"],
        "y_true": y_true,
        "y_score": y_score,
    }


def print_evaluation_table(metrics: Dict, labels: List[str]) -> None:
    """Print a formatted evaluation table to stdout."""
    header = f"{'Label':<25} {'AUROC':>8} {'AP':>8} {'F1':>8}"
    print(header)
    print("-" * len(header))
    for lbl in labels:
        auroc = metrics["auroc"].get(lbl, float("nan"))
        ap = metrics["map"].get(lbl, float("nan"))
        f1 = metrics["f1"].get(lbl, float("nan"))
        print(f"{lbl:<25} {auroc:>8.4f} {ap:>8.4f} {f1:>8.4f}")
    print("-" * len(header))
    print(
        f"{'Macro':<25} "
        f"{metrics['auroc'].get('macro', float('nan')):>8.4f} "
        f"{metrics['map'].get('macro', float('nan')):>8.4f} "
        f"{metrics['f1'].get('macro', float('nan')):>8.4f}"
    )
