"""
integrate_ny.py
Integrates NY 345kV annotations into the merged dataset and train split.
Run from project root: python src/data/integrate_ny.py
"""
import re
import shutil
from pathlib import Path

import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
NY_PATCHES_DIR = Path("data/naip/patches/NY_345kV/images")
NY_LABELS_DIR  = Path("data/collected/annotated/NY_345kV/obj_train_data")
CLIPPED_DIR    = Path("data/naip/clipped_ny")   # NY tiles were clipped into a dedicated folder
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

    ny_label_files = sorted(NY_LABELS_DIR.glob("*.txt"))
    nonempty = [lf.stem for lf in ny_label_files if lf.stat().st_size > 0]
    print(f"NY patches with >=1 tower: {len(nonempty)} / {len(ny_label_files)}")

    copied = 0
    pattern = re.compile(r"^(.*_clipped)_r(\d+)_c(\d+)$")
    meta_rows = []

    for stem in nonempty:
        img_src = NY_PATCHES_DIR / f"{stem}.jpg"
        lbl_src = NY_LABELS_DIR / f"{stem}.txt"

        if not img_src.exists():
            print(f"  WARNING: missing image for {stem}")
            continue

        shutil.copy(img_src, MERGED_IMAGES / f"{stem}.jpg")
        shutil.copy(lbl_src, MERGED_LABELS / f"{stem}.txt")
        shutil.copy(img_src, TRAIN_IMAGES / f"{stem}.jpg")
        shutil.copy(lbl_src, TRAIN_LABELS / f"{stem}.txt")
        copied += 1

        m = pattern.match(stem)
        if not m:
            print(f"  WARNING: could not parse grid index from {stem}")
            continue
        tif_stem, row_i, col_i = m.group(1), int(m.group(2)), int(m.group(3))
        tif_path = CLIPPED_DIR / f"{tif_stem}.tif"
        if not tif_path.exists():
            print(f"  WARNING: source tif missing for {stem} (looked in {tif_path})")
            continue

        lon, lat = get_patch_centroid(tif_path, row_i, col_i)
        meta_rows.append({
            "filename":     stem,
            "state":        "NY",
            "split":        "train",
            "lon":          lon,
            "lat":          lat,
            "nlcd_code":    None,
            "nlcd_class":   "pending",
            "land_use":     "pending",
            "voltage_tier": "345kV",
            "dist_m":       None,
        })

    print(f"\nCopied {copied} NY images+labels into merged/ and splits/train/")

    new_meta = pd.DataFrame(meta_rows)
    existing = pd.read_csv(METADATA_CSV)
    combined = pd.concat([existing, new_meta], ignore_index=True)
    combined.to_csv(METADATA_CSV, index=False)
    print(f"Updated {METADATA_CSV} — {len(existing)} -> {len(combined)} rows")

    total_instances = sum(
        sum(1 for _ in open(lbl)) for lbl in MERGED_LABELS.glob("*.txt")
    )
    print(f"\nTotal instances in merged/labels/: {total_instances}")
    print(f"Total images in merged/images/:   {len(list(MERGED_IMAGES.glob('*.jpg')))}")


if __name__ == "__main__":
    main()
