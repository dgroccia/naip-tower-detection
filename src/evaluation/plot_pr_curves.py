"""
plot_pr_curves.py

Reads outputs/results/raw_predictions.csv (produced by
collect_raw_predictions.py) and computes a pooled precision-recall curve
per model by sweeping the confidence threshold. Does not render a figure —
figures are R/ggplot2 only per project convention; this script only writes
the curve data the R script will plot.

Ground-truth recall denominator: raw_predictions.csv repeats n_gt on every
detection row for a given image, so summing n_gt directly over all rows
would multiply each image's ground-truth count by however many detections
it has. The true per-model total ground truth is the sum of n_gt over each
image counted once (including zero-detection placeholder rows, confidence
== -1, which exist specifically so no image's ground truth is lost).

Usage (from project root in WSL):
    conda activate thesis
    python src/evaluation/plot_pr_curves.py

Output:
    outputs/results/pr_curve_data.csv
    Columns: model, threshold, precision, recall, f1
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PREDICTIONS_CSV = PROJECT_ROOT / "outputs/results/raw_predictions.csv"
OUTPUT_CSV           = PROJECT_ROOT / "outputs/results/pr_curve_data.csv"


def total_ground_truth(model_df: pd.DataFrame) -> int:
    """Total ground-truth box count for one model, counted once per image.

    Args:
        model_df: Rows of raw_predictions.csv for a single model.

    Returns:
        Sum of n_gt across unique filenames.
    """
    return int(model_df.drop_duplicates("filename")["n_gt"].sum())


def compute_pr_curve(model_df: pd.DataFrame, n_gt_total: int) -> pd.DataFrame:
    """Compute a pooled precision-recall curve by sweeping confidence threshold.

    Pools every real detection (confidence >= 0) across all images for this
    model, sorts descending by confidence, and walks the cumulative TP/FP
    counts. Each unique confidence value in the sorted detections becomes
    one threshold point: keeping only detections with confidence >= that
    value gives the precision/recall/F1 at that operating point.

    Args:
        model_df: Rows of raw_predictions.csv for a single model.
        n_gt_total: Total ground-truth box count for this model (see
            total_ground_truth), used as the fixed recall denominator.

    Returns:
        DataFrame with columns threshold, precision, recall, f1, sorted by
        descending threshold (i.e. increasing recall).
    """
    detections = model_df[model_df["confidence"] >= 0].sort_values(
        "confidence", ascending=False
    ).reset_index(drop=True)

    if detections.empty or n_gt_total == 0:
        return pd.DataFrame(columns=["threshold", "precision", "recall", "f1"])

    tp_cum = detections["is_tp"].cumsum().to_numpy()
    fp_cum = (1 - detections["is_tp"]).cumsum().to_numpy()

    precision = tp_cum / (tp_cum + fp_cum)
    recall = tp_cum / n_gt_total

    curve = pd.DataFrame({
        "threshold": detections["confidence"].to_numpy(),
        "precision": precision,
        "recall": recall,
    })

    # Collapse ties at the same confidence value to a single point (the last
    # -- i.e. most-inclusive -- cumulative counts at that confidence).
    curve = curve.groupby("threshold", as_index=False).last().sort_values(
        "threshold", ascending=False
    )

    curve["f1"] = np.where(
        (curve["precision"] + curve["recall"]) > 0,
        2 * curve["precision"] * curve["recall"] / (curve["precision"] + curve["recall"]),
        0.0,
    )

    return curve.reset_index(drop=True)


def main() -> None:
    if not RAW_PREDICTIONS_CSV.exists():
        raise FileNotFoundError(
            f"{RAW_PREDICTIONS_CSV} not found — run collect_raw_predictions.py first."
        )

    df = pd.read_csv(RAW_PREDICTIONS_CSV)
    log.info(f"Loaded {len(df)} rows across {df['model'].nunique()} models.")

    all_curves = []
    best_operating_points = []

    for model_name, model_df in df.groupby("model"):
        n_gt_total = total_ground_truth(model_df)
        curve = compute_pr_curve(model_df, n_gt_total)
        if curve.empty:
            log.warning(f"{model_name}: no detections or no ground truth, skipping.")
            continue

        curve.insert(0, "model", model_name)
        all_curves.append(curve)

        best = curve.loc[curve["f1"].idxmax()]
        best_operating_points.append({
            "model": model_name,
            "threshold": round(float(best["threshold"]), 4),
            "precision": round(float(best["precision"]), 4),
            "recall": round(float(best["recall"]), 4),
            "f1": round(float(best["f1"]), 4),
        })

    curve_df = pd.concat(all_curves, ignore_index=True)
    curve_df[["threshold", "precision", "recall", "f1"]] = curve_df[
        ["threshold", "precision", "recall", "f1"]
    ].round(6)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    curve_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"Saved {len(curve_df)} rows to {OUTPUT_CSV}")

    best_df = pd.DataFrame(best_operating_points).sort_values("f1", ascending=False)
    print("\nMax-F1 operating point per model:")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
