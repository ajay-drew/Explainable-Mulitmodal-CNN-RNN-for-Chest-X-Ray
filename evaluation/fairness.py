"""
Demographic fairness evaluation for chest X-ray classification models.

Evaluates AUROC and F1 stratified by patient gender and age group,
then flags groups with AUROC gaps > 0.05.

CLI usage:
    python evaluation/fairness.py --predictions results/preds.csv
                                   --demographics data/metadata.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_demographics(metadata_csv: str) -> pd.DataFrame:
    """
    Load patient demographics from a CSV file.

    Expected columns (flexible): subject_id / patient_id, age, gender / sex.
    Returns a DataFrame with normalised columns: subject_id, age, gender.

    Args:
        metadata_csv: Path to demographics CSV.

    Returns:
        DataFrame with columns ['subject_id', 'age', 'gender'].
    """
    path = Path(metadata_csv)
    if not path.exists():
        raise FileNotFoundError(f"Demographics CSV not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    df.columns = df.columns.str.strip().str.lower()

    # Normalise subject identifier
    if "subject_id" not in df.columns:
        for alias in ("patient_id", "patientid", "id"):
            if alias in df.columns:
                df = df.rename(columns={alias: "subject_id"})
                break

    # Normalise age column
    if "age" not in df.columns:
        for alias in ("patientage", "patient_age"):
            if alias in df.columns:
                df = df.rename(columns={alias: "age"})
                break

    # Normalise gender column
    if "gender" not in df.columns:
        for alias in ("sex", "patientsex", "patient_gender"):
            if alias in df.columns:
                df = df.rename(columns={alias: "gender"})
                break

    keep = [c for c in ("subject_id", "age", "gender") if c in df.columns]
    return df[keep].drop_duplicates(subset="subject_id" if "subject_id" in keep else None)


def bin_age(age) -> str:
    """
    Map a numeric age to a standardised age group string.

    Args:
        age: Numeric or string age value.

    Returns:
        One of "18-40", "41-60", "61-80", "81+", or "Unknown".
    """
    try:
        age_int = int(float(str(age).replace("Y", "").replace("y", "").strip()))
    except (ValueError, TypeError):
        return "Unknown"

    if age_int < 18:
        return "<18"
    elif age_int <= 40:
        return "18-40"
    elif age_int <= 60:
        return "41-60"
    elif age_int <= 80:
        return "61-80"
    else:
        return "81+"


def stratify_and_evaluate(
    predictions_df: pd.DataFrame,
    demographics_df: pd.DataFrame,
    metric_fn: Optional[Callable] = None,
    label_cols: Optional[List[str]] = None,
    score_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Stratify predictions by gender and age group, compute AUROC and F1.

    Args:
        predictions_df:  DataFrame with columns [image_id / subject_id,
                         true_<label>, pred_<label>, ...].
        demographics_df: DataFrame with [subject_id, age, gender].
        metric_fn:       Optional custom metric function (unused; AUROC/F1 used).
        label_cols:      Ground-truth column names (auto-detected if None).
        score_cols:      Prediction score column names (auto-detected if None).

    Returns:
        DataFrame with columns [group, subgroup, auroc, f1].
    """
    from sklearn.metrics import f1_score, roc_auc_score

    if label_cols is None:
        label_cols = [c for c in predictions_df.columns if c.startswith("true_")]
    if score_cols is None:
        score_cols = [c.replace("true_", "pred_") for c in label_cols]

    # Merge on subject/image ID
    id_col = None
    for candidate in ("subject_id", "patient_id", "image_id", "image_index"):
        if candidate in predictions_df.columns:
            id_col = candidate
            break
    if id_col is None:
        raise ValueError(
            "predictions_df must have one of: subject_id, patient_id, image_id, image_index"
        )

    demo_id = "subject_id" if "subject_id" in demographics_df.columns else demographics_df.columns[0]
    merged = predictions_df.merge(
        demographics_df, left_on=id_col, right_on=demo_id, how="left"
    )

    # Bin age
    if "age" in merged.columns:
        merged["age_group"] = merged["age"].apply(bin_age)
    else:
        merged["age_group"] = "Unknown"

    if "gender" not in merged.columns:
        merged["gender"] = "Unknown"

    results = []

    def _safe_auroc(y_true, y_score):
        if len(np.unique(y_true)) < 2:
            return float("nan")
        try:
            return float(roc_auc_score(y_true, y_score))
        except Exception:
            return float("nan")

    def _macro_auroc(sub_df):
        aurocs = []
        for lbl, scr in zip(label_cols, score_cols):
            if lbl in sub_df.columns and scr in sub_df.columns:
                a = _safe_auroc(sub_df[lbl].values, sub_df[scr].values)
                if not np.isnan(a):
                    aurocs.append(a)
        return float(np.mean(aurocs)) if aurocs else float("nan")

    def _macro_f1(sub_df, thresh=0.5):
        f1s = []
        for lbl, scr in zip(label_cols, score_cols):
            if lbl in sub_df.columns and scr in sub_df.columns:
                y_true = sub_df[lbl].values
                y_pred = (sub_df[scr].values >= thresh).astype(int)
                f1s.append(float(f1_score(y_true, y_pred, zero_division=0)))
        return float(np.mean(f1s)) if f1s else float("nan")

    # Stratify by gender
    for gender_val in merged["gender"].dropna().unique():
        sub = merged[merged["gender"] == gender_val]
        if len(sub) < 10:
            continue
        results.append({
            "group": "Gender",
            "subgroup": str(gender_val),
            "n": len(sub),
            "auroc": _macro_auroc(sub),
            "f1": _macro_f1(sub),
        })

    # Stratify by age group
    for age_grp in ["<18", "18-40", "41-60", "61-80", "81+", "Unknown"]:
        sub = merged[merged["age_group"] == age_grp]
        if len(sub) < 10:
            continue
        results.append({
            "group": "Age",
            "subgroup": age_grp,
            "n": len(sub),
            "auroc": _macro_auroc(sub),
            "f1": _macro_f1(sub),
        })

    return pd.DataFrame(results)


def fairness_report(results_df: pd.DataFrame) -> Dict:
    """
    Print a formatted fairness table and flag AUROC gaps > 0.05.

    Args:
        results_df: Output DataFrame from stratify_and_evaluate().

    Returns:
        Summary dict with keys 'bias_alerts' and 'max_auroc_gap'.
    """
    if results_df.empty:
        print("No fairness results to display.")
        return {"bias_alerts": [], "max_auroc_gap": 0.0}

    print("\n" + "=" * 60)
    print("FAIRNESS EVALUATION REPORT")
    print("=" * 60)
    print(f"{'Group':<10} {'Subgroup':<12} {'N':>8} {'AUROC':>8} {'F1':>8}")
    print("-" * 60)

    for _, row in results_df.iterrows():
        auroc_str = f"{row['auroc']:.4f}" if not np.isnan(row["auroc"]) else "  N/A"
        f1_str = f"{row['f1']:.4f}" if not np.isnan(row["f1"]) else "  N/A"
        print(
            f"{row['group']:<10} {str(row['subgroup']):<12} "
            f"{int(row['n']):>8} {auroc_str:>8} {f1_str:>8}"
        )

    print("=" * 60)

    # Detect bias
    alerts = []
    max_gap = 0.0
    for group_name in results_df["group"].unique():
        sub = results_df[results_df["group"] == group_name]["auroc"].dropna()
        if len(sub) < 2:
            continue
        gap = float(sub.max() - sub.min())
        max_gap = max(max_gap, gap)
        if gap > 0.05:
            grp_rows = results_df[results_df["group"] == group_name]
            best = grp_rows.loc[grp_rows["auroc"].idxmax(), "subgroup"]
            worst = grp_rows.loc[grp_rows["auroc"].idxmin(), "subgroup"]
            alert = (
                f"[BIAS ALERT] {group_name}: AUROC gap={gap:.3f} "
                f"({best} vs {worst})"
            )
            alerts.append(alert)
            print(alert)

    if not alerts:
        print("[OK] No AUROC gaps > 0.05 detected across groups.")

    return {"bias_alerts": alerts, "max_auroc_gap": max_gap}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run fairness evaluation")
    parser.add_argument("--predictions", required=True,
                        help="CSV file with true_<label> and pred_<label> columns")
    parser.add_argument("--demographics", required=True,
                        help="CSV file with subject_id, age, gender columns")
    parser.add_argument("--output", default=None,
                        help="Optional output CSV path for results DataFrame")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        preds_df = pd.read_csv(args.predictions)
    except Exception as exc:
        print(f"Failed to load predictions: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        demo_df = load_demographics(args.demographics)
    except Exception as exc:
        print(f"Failed to load demographics: {exc}", file=sys.stderr)
        sys.exit(1)

    results = stratify_and_evaluate(preds_df, demo_df)
    summary = fairness_report(results)

    if args.output:
        results.to_csv(args.output, index=False)
        print(f"\nFairness results saved to {args.output}")

    if summary["bias_alerts"]:
        sys.exit(1)  # Non-zero exit if bias detected
