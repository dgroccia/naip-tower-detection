"""
bootstrap_corridor_conditional.py

Corridor-level bootstrap resampling for conditional evaluation strata.
Computes bootstrapped mean AP@0.5 and SE for each model x condition
combination (voltage_tier and land_use) by resampling corridors within
each stratum rather than individual images.

Usage:
    conda activate thesis
    cd ~/projects/thesis_infrastructure_detection
    python src/evaluation/bootstrap_corridor_conditional.py
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
METADATA_CSV    = Path("data/collected/annotated/tower_metadata.csv")
OUTPUT_CSV      = Path("outputs/results/bootstrap_corridor_conditional_results.csv")

N_ITER = 1000
SEED   = 42
MIN_CORRIDORS = 3  # minimum corridors in stratum to report SE reliably

PATTERN = re.compile(r"^(.+_clipped)_r\d+_c\d+\.jpg$")

# Conditions to evaluate
CONDITIONS = {
    "voltage_tier": ["345kV", "500kV"],
    "land_use":     ["agriculture", "forest", "suburban"],
}

# Models to evaluate
MODELS = ["yolov8m", "yolov9m", "yolov10m", "yolov11m",
          "retinanet", "faster_rcnn", "detr"]


def extract_corridor(filename):
    m = PATTERN.match(filename)
    return m.group(1) if m else filename


def corridor_bootstrap(ap50_by_corridor, n_iter=N_ITER, seed=SEED):
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
    # Load predictions
    preds = pd.read_csv(PREDICTIONS_CSV)
    preds["corridor"] = preds["filename"].apply(extract_corridor)
    log.info(f"Loaded {len(preds)} prediction rows, "
             f"{preds['corridor'].nunique()} corridors")

    # Load metadata for condition assignments
    meta = pd.read_csv(METADATA_CSV)
    meta["filename_stem"] = meta["filename"].str.replace(".jpg", "",
                                                          regex=False)

    # Join metadata to predictions on filename stem
    preds["filename_stem"] = (preds["filename"]
                               .str.replace(".jpg", "", regex=False))
    # Deduplicate metadata to one row per image (take first)
    meta_dedup = (meta.groupby("filename_stem")
                      .first()
                      .reset_index()[["filename_stem", "voltage_tier",
                                      "land_use", "state"]])

    preds = preds.merge(meta_dedup, on="filename_stem", how="left",
                        suffixes=("_pred", "_meta"))

    # Use metadata columns where available, fall back to prediction columns
    for col in ["voltage_tier", "land_use"]:
        pred_col  = f"{col}_pred" if f"{col}_pred" in preds.columns else col
        meta_col  = f"{col}_meta" if f"{col}_meta" in preds.columns else col
        if pred_col in preds.columns and meta_col in preds.columns:
            preds[col] = preds[meta_col].fillna(preds[pred_col])
        elif meta_col in preds.columns:
            preds[col] = preds[meta_col]

    results = []

    for dim, groups in CONDITIONS.items():
        log.info(f"\n=== {dim} ===")
        for group in groups:
            stratum = preds[preds[dim].str.lower() == group.lower()]
            if len(stratum) == 0:
                log.warning(f"  {group}: no images found, skipping")
                continue

            n_images_total = stratum["filename"].nunique()
            log.info(f"  {group}: {n_images_total} images")

            for model in MODELS:
                sub = stratum[stratum["model"] == model]
                if len(sub) == 0:
                    continue

                ap50_by_corridor = {
                    c: grp["ap50"].values
                    for c, grp in sub.groupby("corridor")
                }
                n_corridors = len(ap50_by_corridor)
                n_images    = sub["filename"].nunique()

                if n_corridors < MIN_CORRIDORS:
                    log.warning(
                        f"    {model}/{group}: only {n_corridors} corridors, "
                        f"SE unreliable -- reporting mean only"
                    )
                    mean = sub["ap50"].mean()
                    se   = float("nan")
                else:
                    means = corridor_bootstrap(ap50_by_corridor, N_ITER, SEED)
                    mean  = means.mean()
                    se    = means.std()

                log.info(
                    f"    {model}: {mean*100:.1f}% +/- {se*100:.2f}% "
                    f"({n_corridors} corridors, {n_images} images)"
                )

                results.append({
                    "model":       model,
                    "dimension":   dim,
                    "group":       group,
                    "ap50_mean":   round(mean, 4),
                    "ap50_se":     round(se, 4) if not np.isnan(se) else None,
                    "n_corridors": n_corridors,
                    "n_images":    n_images,
                    "phase":       "fine_tuned",
                })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    log.info(f"\nSaved to {OUTPUT_CSV}")
    log.info("\n" + out_df.to_string(index=False))


if __name__ == "__main__":
    main()
