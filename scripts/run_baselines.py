"""
Run all baseline models on the test set and print a comparison table.

Usage:
    python scripts/run_baselines.py --config config/chestxray14.yaml

Outputs:
    results/baseline_comparison.csv
    stdout table: Model | AUROC | F1 | Params Trained | Inference Time (ms)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _run_model(model, dataloader, device):
    """
    Run model inference on a dataloader.

    Returns:
        (y_true, y_score, avg_inference_time_ms)
    """
    model.eval()
    all_probs = []
    all_labels = []
    times = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["labels"].cpu().numpy()

            t0 = time.perf_counter()
            out = model(images)
            t1 = time.perf_counter()

            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out

            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
            times.append((t1 - t0) * 1000 / images.shape[0])

    y_score = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    avg_time = float(np.mean(times))
    return y_true, y_score, avg_time


def main() -> None:
    """Entry point for baseline comparison."""
    parser = argparse.ArgumentParser(description="Run baseline model comparison")
    parser.add_argument("--config", default="config/chestxray14.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Limit test set to N samples for quick testing")
    args = parser.parse_args()

    from config.config import Config, seed_everything, CHESTXRAY14_LABELS

    cfg = Config.from_yaml(args.config)
    seed_everything(cfg.seed)
    device = cfg.resolve_device()
    logger.info(f"Using device: {device}")

    # ── Load dataset ────────────────────────────────────────────────────────
    from data.dataset import ChestXray14Dataset
    from data.preprocessing import get_val_transform
    from torch.utils.data import DataLoader, Subset

    test_ds = ChestXray14Dataset(
        cfg.data_dir, split="test", transform=get_val_transform(cfg.image_size)
    )
    if args.n_samples:
        idx = list(range(min(args.n_samples, len(test_ds))))
        test_ds = Subset(test_ds, idx)

    from data.dataloader import _chestxray14_collate
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, collate_fn=_chestxray14_collate,
    )
    logger.info(f"Test set: {len(test_ds)} samples")

    # ── Load baselines ───────────────────────────────────────────────────────
    from models.baselines import ZeroShotBaseline, CNNOnlyBaseline, benchmark_inference_time

    dummy = torch.randn(1, 1, 224, 224).to(device)

    baselines = {
        "ZeroShot (XRV)": ZeroShotBaseline().to(device),
        "CNN-Only (XRV)": CNNOnlyBaseline().to(device),
    }

    # ── Evaluate ─────────────────────────────────────────────────────────────
    from evaluation.evaluate import compute_auroc, compute_f1_metrics

    rows = []
    for model_name, model in baselines.items():
        logger.info(f"Evaluating {model_name} …")
        try:
            y_true, y_score, avg_time = _run_model(model, test_loader, device)
        except Exception as exc:
            logger.error(f"{model_name} evaluation failed: {exc}")
            continue

        auroc = compute_auroc(y_true, y_score, CHESTXRAY14_LABELS)
        f1_metrics = compute_f1_metrics(y_true, y_score, 0.5, CHESTXRAY14_LABELS)

        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        inf_ms = benchmark_inference_time(model, dummy)

        rows.append({
            "Model": model_name,
            "AUROC (macro)": round(auroc["macro"], 4),
            "F1 (macro)": round(f1_metrics["f1"]["macro"], 4),
            "Params Trained": n_trainable,
            "Inference Time (ms)": round(inf_ms, 2),
        })

        logger.info(
            f"  AUROC={auroc['macro']:.4f}  F1={f1_metrics['f1']['macro']:.4f}  "
            f"Params={n_trainable}  Time={inf_ms:.1f}ms"
        )

    # ── Print table ───────────────────────────────────────────────────────────
    if not rows:
        logger.error("No baseline results to report.")
        sys.exit(1)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    import os
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_path = Path(cfg.results_dir) / "baseline_comparison.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
