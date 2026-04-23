"""
Visualization utilities: ROC curves, PR curves, confusion matrix,
XAI comparison panels, training history, fairness bars, and token attribution.

All plots saved as PNG at 150 dpi.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve

logger = logging.getLogger(__name__)

FIG_DPI = 150


def _save(fig: plt.Figure, path: str) -> None:
    """Save figure to path and close it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot: {path}")


def plot_roc_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-class ROC curves and macro average.

    Args:
        y_true:    (N, C) binary ground-truth array.
        y_score:   (N, C) predicted probability array.
        labels:    List of class label strings.
        save_path: If provided, save figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    all_fpr = np.linspace(0, 1, 200)
    tprs = []

    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    for i, (label, color) in enumerate(zip(labels, colors)):
        col_true = y_true[:, i]
        col_score = y_score[:, i]
        if len(np.unique(col_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(col_true, col_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1, color=color, alpha=0.7,
                label=f"{label} (AUC={roc_auc:.3f})")
        tprs.append(np.interp(all_fpr, fpr, tpr))

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        macro_auc = auc(all_fpr, mean_tpr)
        ax.plot(all_fpr, mean_tpr, lw=2.5, color="black", linestyle="--",
                label=f"Macro Avg (AUC={macro_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Per Class and Macro Average", fontsize=14)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    if save_path:
        _save(fig, save_path)
    return fig


def plot_pr_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    labels: List[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot per-class precision-recall curves.

    Args:
        y_true:    (N, C) binary ground-truth.
        y_score:   (N, C) predicted probabilities.
        labels:    Label names.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(labels)))

    for i, (label, color) in enumerate(zip(labels, colors)):
        col_true = y_true[:, i]
        col_score = y_score[:, i]
        if col_true.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(col_true, col_score)
        ap = average_precision_score(col_true, col_score)
        ax.plot(recall, precision, lw=1, color=color, alpha=0.7,
                label=f"{label} (AP={ap:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    if save_path:
        _save(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot a normalised confusion matrix for a single binary label.

    Args:
        y_true:    (N,) binary ground-truth vector.
        y_pred:    (N,) binary prediction vector.
        label:     Label name for title.
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {label}", fontsize=12)

    if save_path:
        _save(fig, save_path)
    return fig


def plot_xai_comparison(
    original_pil,
    heatmap_pil,
    token_attributions: List[Tuple[str, float]],
    title: str = "XAI Comparison",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3-panel figure: original X-ray, Grad-CAM overlay, token attribution bar chart.

    Args:
        original_pil:       PIL Image of the original chest X-ray.
        heatmap_pil:        PIL Image of the Grad-CAM overlay.
        token_attributions: List of (token, score) pairs.
        title:              Figure title.
        save_path:          Optional save path.

    Returns:
        Matplotlib Figure.
    """
    import numpy as np

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_pil, cmap="gray")
    axes[0].set_title("Original X-Ray", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap_pil)
    axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
    axes[1].axis("off")

    # Token attribution bar chart
    if token_attributions:
        top = token_attributions[:15]
        tokens = [t for t, _ in top]
        scores = [s for _, s in top]
        colors = ["red" if s > 0 else "steelblue" for s in scores]
        y_pos = np.arange(len(tokens))
        axes[2].barh(y_pos, scores, color=colors, alpha=0.8)
        axes[2].set_yticks(y_pos)
        axes[2].set_yticklabels(tokens, fontsize=9)
        axes[2].axvline(0, color="black", lw=0.8)
        axes[2].set_xlabel("Attribution Score", fontsize=11)
        axes[2].set_title("Token Attribution (SHAP)", fontsize=12)
        axes[2].grid(True, alpha=0.3, axis="x")
    else:
        axes[2].text(0.5, 0.5, "No report text provided",
                     ha="center", va="center", transform=axes[2].transAxes)
        axes[2].set_title("Token Attribution", fontsize=12)
        axes[2].axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Dual-axis plot: loss (left y-axis) and AUROC (right y-axis).

    Args:
        history:   Dict with keys "train_loss", "val_loss", "val_auroc".
        save_path: Optional save path.

    Returns:
        Matplotlib Figure.
    """
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, ax1 = plt.subplots(figsize=(10, 5))

    if "train_loss" in history:
        ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", ms=5)
    if "val_loss" in history:
        ax1.plot(epochs, history["val_loss"], "b--s", label="Val Loss", ms=5)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    ax2 = ax1.twinx()
    if "val_auroc" in history:
        ax2.plot(epochs, history["val_auroc"], "r-^", label="Val AUROC", ms=5)
    ax2.set_ylabel("AUROC", fontsize=12, color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim([0, 1])

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("Training History", fontsize=14)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


def plot_fairness_comparison(
    fairness_df,
    metric: str = "auroc",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Grouped bar chart of a metric by demographic slice.

    Args:
        fairness_df: DataFrame with columns [group, subgroup, <metric>].
        metric:      Metric column name to plot.
        save_path:   Optional save path.

    Returns:
        Matplotlib Figure.
    """
    import pandas as pd

    if not hasattr(fairness_df, "columns"):
        raise ValueError("fairness_df must be a pandas DataFrame.")

    fig, ax = plt.subplots(figsize=(10, 5))
    groups = fairness_df["group"].unique() if "group" in fairness_df.columns else []
    subgroups = (
        fairness_df["subgroup"].unique() if "subgroup" in fairness_df.columns else []
    )

    x = np.arange(len(subgroups))
    width = 0.8 / max(len(groups), 1)

    for i, grp in enumerate(groups):
        sub = fairness_df[fairness_df["group"] == grp]
        vals = [
            float(sub[sub["subgroup"] == sg][metric].values[0])
            if sg in sub["subgroup"].values else 0.0
            for sg in subgroups
        ]
        ax.bar(x + i * width - 0.4, vals, width, label=str(grp), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(subgroups, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(f"Fairness Comparison — {metric.upper()} by Demographic Slice", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0, 1])
    plt.tight_layout()

    if save_path:
        _save(fig, save_path)
    return fig


def save_all_plots(
    results_dir: str,
    y_true: Optional[np.ndarray] = None,
    y_score: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
    history: Optional[Dict] = None,
    fairness_df=None,
) -> None:
    """
    Generate and save all standard plots to results_dir/plots/.

    Args:
        results_dir: Root results directory.
        y_true:      (N, C) ground-truth array.
        y_score:     (N, C) predicted probability array.
        labels:      Class label names.
        history:     Training history dict.
        fairness_df: Fairness results DataFrame.
    """
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    if y_true is not None and y_score is not None and labels is not None:
        plot_roc_curves(y_true, y_score, labels,
                        save_path=os.path.join(plots_dir, "roc_curves.png"))
        plot_pr_curves(y_true, y_score, labels,
                       save_path=os.path.join(plots_dir, "pr_curves.png"))

    if history:
        plot_training_history(history,
                              save_path=os.path.join(plots_dir, "training_history.png"))

    if fairness_df is not None:
        plot_fairness_comparison(fairness_df,
                                 save_path=os.path.join(plots_dir, "fairness.png"))

    logger.info(f"All plots saved to {plots_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Generate evaluation plots")
    parser.add_argument("--results", required=True,
                        help="Results directory containing predictions CSV")
    parser.add_argument("--preds", default="predictions.csv",
                        help="Predictions CSV filename inside results dir")
    args = parser.parse_args()

    preds_path = os.path.join(args.results, args.preds)
    if not os.path.exists(preds_path):
        print(f"Predictions file not found: {preds_path}")
        raise SystemExit(1)

    df = pd.read_csv(preds_path)
    # Expect columns: label_cols (ground truth with prefix 'true_')
    # and score cols (with prefix 'pred_')
    label_cols = [c for c in df.columns if c.startswith("true_")]
    score_cols = [c.replace("true_", "pred_") for c in label_cols]
    lbl_names = [c.replace("true_", "") for c in label_cols]

    if not label_cols or not all(c in df.columns for c in score_cols):
        print("Expected columns: true_<label> and pred_<label> for each label.")
        raise SystemExit(1)

    y_true = df[label_cols].values
    y_score = df[score_cols].values

    save_all_plots(args.results, y_true, y_score, lbl_names)
    print(f"Plots saved to {args.results}/plots/")
