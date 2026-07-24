"""
propagate_test_metadata.py

Propagates metadata from the 97 base test images to all 3,492
augmented test images. Each augmented image inherits the voltage_tier,
nlcd_class, land_use, state, lon, lat, and dist_m from its base image.

Usage:
    conda activate thesis
    cd ~/projects/thesis_infrastructure_detection
    python src/data/propagate_test_metadata.py
"""
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

METADATA_CSV = Path("data/collected/annotated/tower_metadata.csv")
AUG_IMAGES   = Path("data/augmented/test/images")
OUTPUT_CSV   = Path("data/collected/annotated/tower_metadata_aug_test.csv")

# Load existing metadata
meta = pd.read_csv(METADATA_CSV)
test_meta = meta[meta["split"] == "test"].copy()
log.info(f"Base test metadata rows: {len(test_meta)}")

# Get all augmented test image stems
aug_stems = [p.stem for p in sorted(AUG_IMAGES.glob("*.jpg"))]
log.info(f"Augmented test images: {len(aug_stems)}")

# For each augmented image, find its base image stem and copy metadata
new_rows = []
for stem in aug_stems:
    # Augmented stem format: {base_stem}_aug_{angle:03d}
    # Base stem is everything before _aug_
    base_stem = "_aug_".join(stem.split("_aug_")[:-1])
    base_rows = test_meta[test_meta["filename"] == base_stem]
    if base_rows.empty:
        log.warning(f"No metadata found for base: {base_stem}")
        continue
    for _, row in base_rows.iterrows():
        new_row = row.copy()
        new_row["filename"] = stem
        new_rows.append(new_row)

aug_meta = pd.DataFrame(new_rows)
log.info(f"Augmented test metadata rows generated: {len(aug_meta)}")

# Combine original metadata with augmented test metadata
combined = pd.concat([meta, aug_meta], ignore_index=True)
combined.to_csv(OUTPUT_CSV, index=False)
log.info(f"Saved combined metadata to {OUTPUT_CSV}")
log.info(f"Total rows: {len(combined)}")

# Sanity check
aug_test_check = combined[combined["filename"].str.contains("_aug_")]
log.info(f"Sanity check - augmented rows in output: {len(aug_test_check)}")
