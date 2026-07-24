"""
integrate_ga.py
Integrates GA 500kV annotations into the merged dataset and train split.
Run from project root: python src/data/integrate_ga.py
"""
import re
import shutil
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
GA_PATCHES_DIR = Path("data/naip/patches/GA_500kV/images")
GA_LABELS_DIR  = Path("data/collected/annotated/GA_500kV/obj_train_data")
CLIPPED_DIR    = Path("data/naip/clipped")
MERGED_IMAGES  = Path("data/collected/annotated/merged/images")
MERGED_LABELS  = Path("data/collected/annotated/merged/labels")
METADATA_CSV   = Path("data/collected/annotated/tower_metadata.csv")
TRAIN_IMAGES   = Path("data/splits/train/images")
TRAIN_LABELS   = Path("data/splits/train/labels")

PATCH_SIZE = 640
OVERLAP    = 0.1
STRIDE     = int(PATCH_SIZE * (1 - OVERLAP))
# ─────────────────────────────────────────────────────────────────────────────

def get_patch_centroid(tif_path, row_i, col_i):
    """Approximate centroid lon/lat for a patch using row/col grid index."""
    with rasterio.open(tif_path) as src:
        x = col_i * STRIDE
        y = row_i * STRIDE
        window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
        transform = src.window_transform(window)
        lon, lat = transform * (PATCH_SIZE / 2, PATCH_SIZE / 2)
        if src.crs.to_string() != "EPSG:4326":
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(lon, lat)
        return lon, lat


def main():
    MERGED_IMAGES.mkdir(parents=True, exist_ok=True)
    MERGED_LABELS.mkdir(parents=True, exist_ok=True)
    TRAIN_IMAGES.mkdir(parents=True, exist_ok=True)
    TRAIN_LABELS.mkdir(parents=True, exist_ok=True)

    # Step 1 — find GA patches with at least one tower annotation
    ga_label_files = sorted(GA_LABELS_DIR.glob("*.txt"))
    nonempty = [lf.stem for lf in ga_label_files if lf.stat().st_size > 0]
    print(f"GA patches with >=1 tower: {len(nonempty)} / {len(ga_label_files)}")

    # Step 2 — copy images + labels into merged/ and splits/train/
    copied = 0
    pattern = re.compile(r"^(.*_clipped)_r(\d+)_c(\d+)$")
    meta_rows = []

    for stem in nonempty:
        img_src = GA_PATCHES_DIR / f"{stem}.jpg"
        lbl_src = GA_LABELS_DIR / f"{stem}.txt"

        if not img_src.exists():
            print(f"  WARNING: missing image for {stem}")
            continue

        shutil.copy(img_src, MERGED_IMAGES / f"{stem}.jpg")
        shutil.copy(lbl_src, MERGED_LABELS / f"{stem}.txt")
        shutil.copy(img_src, TRAIN_IMAGES / f"{stem}.jpg")
        shutil.copy(lbl_src, TRAIN_LABELS / f"{stem}.txt")
        copied += 1

        # Step 3 — approximate centroid for metadata
        m = pattern.match(stem)
        if not m:
            print(f"  WARNING: could not parse grid index from {stem}")
            continue
        tif_stem, row_i, col_i = m.group(1), int(m.group(2)), int(m.group(3))
        tif_path = CLIPPED_DIR / f"{tif_stem}.tif"
        if not tif_path.exists():
            print(f"  WARNING: source tif missing for {stem}")
            continue

        lon, lat = get_patch_centroid(tif_path, row_i, col_i)
        meta_rows.append({
            "filename":     stem,
            "state":        "GA",
            "split":        "train",
            "lon":          lon,
            "lat":          lat,
            "nlcd_code":    None,
            "nlcd_class":   "pending",
            "land_use":     "pending",
            "voltage_tier": "500kV",
            "dist_m":       None,
        })

    print(f"\nCopied {copied} GA images+labels into merged/ and splits/train/")

    # Step 4 — append to tower_metadata.csv
    new_meta = pd.DataFrame(meta_rows)
    existing = pd.read_csv(METADATA_CSV)
    combined = pd.concat([existing, new_meta], ignore_index=True)
    combined.to_csv(METADATA_CSV, index=False)
    print(f"Updated {METADATA_CSV} — {len(existing)} -> {len(combined)} rows")

    # Sanity check
    total_instances = sum(
        sum(1 for _ in open(lbl)) for lbl in MERGED_LABELS.glob("*.txt")
    )
    print(f"\nTotal instances in merged/labels/: {total_instances}")
    print(f"Total images in merged/images/:   {len(list(MERGED_IMAGES.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
