"""
download_naip_desert.py

Downloads NAIP tiles for AZ 500kV and NM 345kV test corridors.
Desert environments -- genuinely out-of-distribution from training data.

Usage:
    conda activate thesis
    cd ~/projects/thesis_infrastructure_detection
    python src/data/download_naip_desert.py
"""
import requests
import geopandas as gpd
import planetary_computer
import pystac_client
from pathlib import Path
import time

OUTPUT_DIR = Path("data/naip/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "thesis-research-script/1.0 (academic use, GMU)",
    "Content-Type": "application/x-www-form-urlencoded",
}

CORRIDORS = [
    ("az", "data/collected/annotated/AZ_500kv_test_set.shp", "500kV"),
    ("nm", "data/collected/annotated/NM_345kv_test_set.shp", "345kV"),
]

NAIP_YEARS = ["2022", "2021", "2020"]
MAX_TILES  = 15


def main():
    print("Connecting to Planetary Computer...")
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    for state_code, shp_path, voltage in CORRIDORS:
        print(f"\n{'='*60}")
        print(f"Processing {state_code.upper()} {voltage}")

        gdf  = gpd.read_file(shp_path).to_crs("EPSG:4326")
        bbox = gdf.total_bounds
        print(f"  Bounding box: {bbox}")
        print(f"  Polygons: {len(gdf)}")

        items = []
        used_year = None
        for year in NAIP_YEARS:
            search = catalog.search(
                collections=["naip"],
                bbox=bbox.tolist(),
                query={"naip:state": {"eq": state_code}},
                datetime=f"{year}-01-01/{year}-12-31",
            )
            items = list(search.items())
            if items:
                used_year = year
                print(f"  Found {len(items)} tiles for {year}")
                break
            print(f"  No tiles for {year}, trying earlier...")
            time.sleep(1)

        if not items:
            print(f"  WARNING: No NAIP tiles found for {state_code.upper()}")
            continue

        downloaded = 0
        for item in items[:MAX_TILES]:
            if "image" not in item.assets:
                continue
            fname = OUTPUT_DIR / f"{item.id}.tif"
            if fname.exists():
                print(f"    Already exists: {fname.name}")
                downloaded += 1
                continue
            print(f"    Downloading {item.id}...", end=" ", flush=True)
            try:
                r = requests.get(item.assets["image"].href,
                                 stream=True, timeout=120)
                r.raise_for_status()
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                size_mb = fname.stat().st_size / (1024 * 1024)
                print(f"done ({size_mb:.1f} MB)")
                downloaded += 1
            except Exception as e:
                print(f"FAILED: {e}")

        print(f"  Downloaded {downloaded} tiles (year={used_year})")

    print("\nDone. Next: filter overlapping tiles, clip, tile, annotate.")


if __name__ == "__main__":
    main()
