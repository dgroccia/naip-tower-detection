"""
integrate_az_nm_metadata.py

Adds metadata rows for AZ 500kV and NM 345kV test patches to
tower_metadata.csv so conditional_eval.py can join against them.

Same pattern as integrate_ga.py and integrate_ny.py -- one row per
annotated image, patch centroid lon/lat derived from clipped GeoTIFF
and row/col grid index parsed from filename stem.

Usage:
    conda activate thesis
    cd ~/projects/thesis_infrastructure_detection
    python src/data/integrate_az_nm_metadata.py
"""
import re
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Transformer
from pathlib import Path

METADATA_CSV = Path("data/collected/annotated/tower_metadata.csv")
BACKUP_CSV   = Path("data/collected/annotated/tower_metadata_backup_pre_az_nm.csv")

PATCH_SIZE = 640
OVERLAP    = 0.1
STRIDE     = int(PATCH_SIZE * (1 - OVERLAP))

CONFIGS = [
    {
        "state":        "AZ",
        "voltage_tier": "500kV",
        "split":        "test",
        "nlcd_class":   "pending",
        "land_use":     "shrubland",   # desert scrub -- closest NLCD analog
        "labels_dir":   Path("data/splits/test/labels"),
        "clipped_dir":  Path("data/naip/clipped_az_test_500"),
        "stem_prefix":  "az_",
    },
    {
        "state":        "NM",
        "voltage_tier": "345kV",
        "split":        "test",
        "nlcd_class":   "pending",
        "land_use":     "shrubland",
        "labels_dir":   Path("data/splits/test/labels"),
        "clipped_dir":  Path("data/naip/clipped_nm_test_345"),
        "stem_prefix":  "nm_",
    },
]

PATTERN = re.compile(r"^(.*_clipped)_r(\d+)_c(\d+)$")


def get_patch_centroid(tif_path, row_i, col_i):
    with rasterio.open(tif_path) as src:
        x = col_i * STRIDE
        y = row_i * STRIDE
        window = Window(x, y, PATCH_SIZE, PATCH_SIZE)
        transform = src.window_transform(window)
        lon, lat = transform * (PATCH_SIZE / 2, PATCH_SIZE / 2)
        if src.crs.to_epsg() != 4326:
            transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            lon, lat = transformer.transform(lon, lat)
        return lon, lat


def main():
    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded metadata: {len(df)} rows")
    df.to_csv(BACKUP_CSV, index=False)
    print(f"Backed up to {BACKUP_CSV}")

    new_rows = []

    for cfg in CONFIGS:
        state    = cfg["state"]
        prefix   = cfg["stem_prefix"]
        lbl_dir  = cfg["labels_dir"]
        clip_dir = cfg["clipped_dir"]

        # Find annotated test label files for this state
        label_files = [
            f for f in lbl_dir.glob(f"{prefix}*.txt")
            if f.stat().st_size > 0
        ]
        print(f"\n{state}: {len(label_files)} annotated test images")

        missing_tif = 0
        for lbl in label_files:
            stem = lbl.stem
            m = PATTERN.match(stem)
            if not m:
                print(f"  WARNING: cannot parse grid index from {stem}")
                continue

            tif_stem, row_i, col_i = m.group(1), int(m.group(2)), int(m.group(3))
            tif_path = clip_dir / f"{tif_stem}.tif"
            if not tif_path.exists():
                missing_tif += 1
                lon, lat = None, None
            else:
                lon, lat = get_patch_centroid(tif_path, row_i, col_i)

            # Count instances in this label file
            lines = [l.strip() for l in lbl.read_text().splitlines() if l.strip()]
            n_instances = len(lines)

            for _ in range(n_instances):
                new_rows.append({
                    "filename":     stem,
                    "state":        state,
                    "split":        cfg["split"],
                    "lon":          lon,
                    "lat":          lat,
                    "nlcd_code":    None,
                    "nlcd_class":   cfg["nlcd_class"],
                    "land_use":     cfg["land_use"],
                    "voltage_tier": cfg["voltage_tier"],
                    "dist_m":       None,
                })

        if missing_tif:
            print(f"  WARNING: {missing_tif} label files had no matching clipped TIF")

    new_meta = pd.DataFrame(new_rows)
    print(f"\nNew metadata rows: {len(new_meta)}")
    print(f"  AZ: {len(new_meta[new_meta['state'] == 'AZ'])}")
    print(f"  NM: {len(new_meta[new_meta['state'] == 'NM'])}")

    combined = pd.concat([df, new_meta], ignore_index=True)
    combined.to_csv(METADATA_CSV, index=False)
    print(f"\nSaved updated metadata: {len(combined)} total rows")
    print(f"Expected: {len(df)} + {len(new_meta)} = {len(df) + len(new_meta)}")


if __name__ == "__main__":
    main()
