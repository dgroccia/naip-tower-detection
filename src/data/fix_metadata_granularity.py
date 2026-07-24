"""
fix_metadata_granularity.py

Fixes GA/NY metadata rows to match instance-level granularity used by the
original thesis metadata (one row per annotated tower, not per image).

Removes the incorrectly image-level GA/NY rows and regenerates them at
instance level by reading the actual label files in merged/labels/.
"""
from pathlib import Path
import pandas as pd

METADATA_CSV  = Path("data/collected/annotated/tower_metadata.csv")
MERGED_LABELS = Path("data/collected/annotated/merged/labels")
BACKUP_CSV    = Path("data/collected/annotated/tower_metadata_backup_pre_fix.csv")


def main():
    df = pd.read_csv(METADATA_CSV)
    print(f"Loaded {len(df)} rows from {METADATA_CSV}")

    # Backup before modifying
    df.to_csv(BACKUP_CSV, index=False)
    print(f"Backed up original to {BACKUP_CSV}")

    # Split into rows to keep (everything not GA/NY) and rows to fix
    keep = df[~df["state"].isin(["GA", "NY"])].copy()
    to_fix = df[df["state"].isin(["GA", "NY"])].copy()
    print(f"\nRows to keep as-is (TX/TN/WV): {len(keep)}")
    print(f"Rows to fix (GA/NY, currently image-level): {len(to_fix)}")

    if len(to_fix) == 0:
        print("No GA/NY rows found — nothing to fix.")
        return

    # For each GA/NY row, read its label file and expand to N rows
    # where N = number of bounding boxes in that image's label file
    expanded_rows = []
    missing_labels = []

    for _, row in to_fix.iterrows():
        filename = row["filename"]
        label_path = MERGED_LABELS / f"{filename}.txt"

        if not label_path.exists():
            missing_labels.append(filename)
            n_instances = 1  # fallback — keep at least the one row
        else:
            with open(label_path) as f:
                n_instances = sum(1 for line in f if line.strip())
            if n_instances == 0:
                n_instances = 1  # shouldn't happen since we only integrated non-empty labels

        for _ in range(n_instances):
            expanded_rows.append(row.to_dict())

    if missing_labels:
        print(f"\nWARNING: {len(missing_labels)} label files not found, defaulted to 1 row:")
        for m in missing_labels[:10]:
            print(f"  {m}")

    expanded_df = pd.DataFrame(expanded_rows)
    print(f"\nExpanded GA/NY rows: {len(to_fix)} -> {len(expanded_df)}")

    ga_before = len(to_fix[to_fix["state"] == "GA"])
    ga_after  = len(expanded_df[expanded_df["state"] == "GA"])
    ny_before = len(to_fix[to_fix["state"] == "NY"])
    ny_after  = len(expanded_df[expanded_df["state"] == "NY"])
    print(f"  GA: {ga_before} -> {ga_after} (expected 60)")
    print(f"  NY: {ny_before} -> {ny_after} (expected 65)")

    # Combine and save
    fixed = pd.concat([keep, expanded_df], ignore_index=True)
    fixed.to_csv(METADATA_CSV, index=False)
    print(f"\nSaved corrected metadata: {len(fixed)} total rows")
    print(f"Expected total: 552 (thesis) + 60 (GA) + 65 (NY) = 677")


if __name__ == "__main__":
    main()
