"""
download_naip_test_500.py
Downloads NAIP tiles for WV 500kV new test corridors.
"""
import requests
import geopandas as gpd
import planetary_computer
import pystac_client
from pathlib import Path

CORRIDORS_SHP = Path("data/collected/500_test_set.shp")
OUTPUT_DIR    = Path("data/naip/raw")
NAIP_YEAR     = "2022"
MAX_TILES     = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "thesis-research-script/1.0 (academic use, GMU)",
    "Content-Type": "application/x-www-form-urlencoded",
}

print("Loading 500kV test corridor polygons...")
gdf = gpd.read_file(CORRIDORS_SHP).to_crs("EPSG:4326")
bbox = gdf.total_bounds
print(f"Bounding box (WGS84): {bbox}")

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print(f"Searching for NAIP {NAIP_YEAR} tiles (WV)...")
search = catalog.search(
    collections=["naip"],
    bbox=bbox.tolist(),
    query={"naip:state": {"eq": "wv"}},
    datetime=f"{NAIP_YEAR}-01-01/{NAIP_YEAR}-12-31",
)
items = list(search.items())
print(f"Found {len(items)} tiles")

if len(items) == 0:
    print("Trying 2021...")
    search = catalog.search(
        collections=["naip"],
        bbox=bbox.tolist(),
        query={"naip:state": {"eq": "wv"}},
        datetime="2021-01-01/2021-12-31",
    )
    items = list(search.items())
    print(f"Found {len(items)} tiles for 2021")

downloaded = 0
for item in items[:MAX_TILES]:
    if "image" not in item.assets:
        continue
    fname = OUTPUT_DIR / f"{item.id}.tif"
    if fname.exists():
        print(f"  Already exists: {fname.name}")
        downloaded += 1
        continue
    print(f"  Downloading {item.id}...", end=" ", flush=True)
    try:
        r = requests.get(item.assets["image"].href, stream=True, timeout=120)
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
        print(f"done ({fname.stat().st_size/1024/1024:.1f} MB)")
        downloaded += 1
    except Exception as e:
        print(f"FAILED: {e}")

print(f"\nDownloaded {downloaded} tiles to {OUTPUT_DIR}/")
