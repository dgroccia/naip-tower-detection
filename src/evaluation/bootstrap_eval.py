"""
Bootstrapped evaluation — 5 models (yolov8m, yolov9m, yolov10m, yolov11m,
faster_rcnn). Fine-tuned phase only; no zero-shot phase in this model set.

200 iterations, seed 42. Reports mean ± SE for AP@0.5.

Reads per-image AP50 values from outputs/results/per_image_predictions.csv
(produced by per_image_inference.py) and bootstraps by resampling images
with replacement, per model. This is a bootstrap over the mean of per-image
AP50 — NOT a re-pooled corpus-level mAP recomputed via torchmetrics per
iteration, since the per-image AP is already the unit of resampling.

Usage:
    conda activate thesis
    python src/evaluation/bootstrap_eval.py
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR  = PROJECT_ROOT / "outputs/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PER_IMAGE_CSV = RESULTS_DIR / "per_image_predictions.csv"

N_ITER = 200
SEED   = 42

# Fine-tuned weight paths — used only to verify the expected model set
# exists on disk before trusting the CSV; bootstrapping itself reads
# exclusively from PER_IMAGE_CSV.
WEIGHTS = {
    "yolov8m":     PROJECT_ROOT / "outputs/weights/naip_yolov8m_coco/weights/best.pt",
    "yolov9m":     PROJECT_ROOT / "outputs/weights/naip_yolov9m_coco/weights/best.pt",
    "yolov10m":    PROJECT_ROOT / "outputs/weights/naip_yolov10m_coco/weights/best.pt",
    "yolov11m":    PROJECT_ROOT / "outputs/weights/naip_yolov11m_coco/weights/best.pt",
    "retinanet":   PROJECT_ROOT / "outputs/weights/naip_retinanet/best.pt",
    "faster_rcnn": PROJECT_ROOT / "outputs/weights/naip_faster_rcnn/best.pt",
    "detr":        PROJECT_ROOT / "outputs/weights/naip_detr/best",
}


def bootstrap_ap50(ap50_values: np.ndarray, n_iter=N_ITER, seed=SEED):
    """Bootstrap the mean of per-image AP50 by resampling images with replacement."""
    rng = np.random.default_rng(seed)
    n = len(ap50_values)
    means = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = ap50_values[idx].mean()
    return {"mean": float(np.mean(means)), "se": float(np.std(means) / np.sqrt(n_iter))}


def main():
    for name, path in WEIGHTS.items():
        if not path.exists():
            log.warning(f"Weights not found for {name} at {path} — its rows in "
                        f"{PER_IMAGE_CSV.name} (if any) will still be bootstrapped, "
                        f"but the checkpoint that produced them can't be verified.")

    if not PER_IMAGE_CSV.exists():
        raise FileNotFoundError(
            f"{PER_IMAGE_CSV} not found. Run per_image_inference.py first — "
            f"bootstrap_eval.py depends on its output."
        )

    df = pd.read_csv(PER_IMAGE_CSV)
    missing_cols = {"model", "phase", "ap50"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{PER_IMAGE_CSV} is missing expected columns: {missing_cols}")

    df = df[df["phase"] == "fine_tuned"]

    all_results = []
    for name in WEIGHTS:
        sub = df[df["model"] == name]
        if sub.empty:
            log.error(f"  {name}: no rows found in {PER_IMAGE_CSV.name} — skipping")
            all_results.append({"model": name, "phase": "fine_tuned",
                                 "ap50_mean": None, "ap50_se": None, "n_images": 0})
            continue

        log.info(f"\nBootstrapping {name} (fine_tuned) over {len(sub)} images...")
        bs = bootstrap_ap50(sub["ap50"].to_numpy(), n_iter=N_ITER, seed=SEED)
        log.info(f"  AP@0.5 = {bs['mean']*100:.2f} ± {bs['se']*100:.2f}%")
        all_results.append({
            "model":    name,
            "phase":    "fine_tuned",
            "ap50_mean": round(bs["mean"], 4),
            "ap50_se":   round(bs["se"],   4),
            "n_images":  len(sub),
        })

    out_df = pd.DataFrame(all_results)
    out_df.to_csv(RESULTS_DIR / "bootstrap_results.csv", index=False)
    log.info(f"\nSaved to {RESULTS_DIR}/bootstrap_results.csv")
    log.info("\n" + out_df.to_string(index=False))


if __name__ == "__main__":
    main()
