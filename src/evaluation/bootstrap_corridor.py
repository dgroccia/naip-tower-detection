"""
bootstrap_corridor.py

Cluster bootstrap resampling at the corridor level rather than image level.
Addresses spatial autocorrelation within corridors by treating each corridor
as the unit of resampling rather than individual patches.

A corridor is defined as a unique clipped GeoTIFF source, derived from the
filename stem by stripping the _rXXX_cXXX grid index suffix.

Usage:
    conda activate thesis
    python src/evaluation/bootstrap_corridor.py
"""
import re
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PREDICTIONS_CSV = Path("outputs/results/per_image_predictions.csv")
OUTPUT_CSV      = Path("outputs/results/bootstrap_corridor_results.csv")
N_ITER          = 1000   # more iterations since fewer clusters
SEED            = 42
RESULTS_DIR     = Path("outputs/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Verify checkpoint paths exist for expected models
WEIGHTS = {
    "yolov8m":     Path("outputs/weights/naip_yolov8m_coco/weights/best.pt"),
    "yolov9m":     Path("outputs/weights/naip_yolov9m_coco/weights/best.pt"),
    "yolov10m":    Path("outputs/weights/naip_yolov10m_coco/weights/best.pt"),
    "yolov11m":    Path("outputs/weights/naip_yolov11m_coco/weights/best.pt"),
    "retinanet":   Path("outputs/weights/naip_retinanet/best.pt"),
    "faster_rcnn": Path("outputs/weights/naip_faster_rcnn/best.pt"),
    "detr":        Path("outputs/weights/naip_detr/best"),
}

PATTERN = re.compile(r"^(.+_clipped)_r\d+_c\d+\.jpg$")


def extract_corridor(filename):
    """Extract corridor identifier from patch filename."""
    m = PATTERN.match(filename)
    if m:
        return m.group(1)
    return filename  # fallback: treat as own corridor


def corridor_bootstrap(ap50_by_corridor, n_iter=N_ITER, seed=SEED):
    """
    Bootstrap by resampling corridors with replacement.
    Each resample draws n_corridors corridors, then pools all
    patch AP50 values from those corridors and takes the mean.
    Returns array of bootstrapped means.
    """
    rng = np.random.default_rng(seed)
    corridors = list(ap50_by_corridor.keys())
    n = len(corridors)
    means = np.empty(n_iter)
    for i in range(n_iter):
        sampled = rng.choice(corridors, size=n, replace=True)
        pooled = np.concatenate([ap50_by_corridor[c] for c in sampled])
        means[i] = pooled.mean()
    return means


def main():
    df = pd.read_csv(PREDICTIONS_CSV)
    log.info(f"Loaded {len(df)} prediction rows")

    # Derive corridor identifier from filename
    df["corridor"] = df["filename"].apply(extract_corridor)

    # Report corridor counts
    n_corridors = df["corridor"].nunique()
    n_images    = df["filename"].nunique()
    log.info(f"Unique corridors: {n_corridors}")
    log.info(f"Unique images: {n_images}")
    log.info(f"Mean images per corridor: {n_images/n_corridors:.1f}")

    # Show corridor breakdown
    corridor_counts = df.groupby("corridor")["filename"].nunique()
    log.info("\nImages per corridor:")
    for c, n in corridor_counts.items():
        log.info(f"  {c}: {n} images")

    results = []

    for model_name in WEIGHTS:
        sub = df[df["model"] == model_name]
        if len(sub) == 0:
            log.warning(f"No predictions found for {model_name}, skipping")
            continue

        # Build corridor -> array of AP50 values mapping
        ap50_by_corridor = {
            corridor: grp["ap50"].values
            for corridor, grp in sub.groupby("corridor")
        }

        n_corr = len(ap50_by_corridor)
        means  = corridor_bootstrap(ap50_by_corridor, N_ITER, SEED)
        mean   = means.mean()
        se     = means.std()

        log.info(f"\n{model_name}: corridor-bootstrapped AP@0.5 = "
                 f"{mean*100:.2f} +/- {se*100:.2f}% "
                 f"({n_corr} corridors, {len(sub)//n_corr:.0f} images/corridor avg)")

        results.append({
            "model":        model_name,
            "phase":        "fine_tuned",
            "ap50_mean":    round(mean, 4),
            "ap50_se":      round(se, 4),
            "n_corridors":  n_corr,
            "n_images":     len(sub),
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\nSaved to {OUTPUT_CSV}")
    log.info("\n" + out_df.to_string(index=False))


if __name__ == "__main__":
    main()
