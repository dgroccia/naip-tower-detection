"""
download_naip_ny.py
Downloads NAIP tiles covering the NY 345kV corridor polygon.
"""
import requests
import geopandas as gpd
import planetary_computer
import pystac_client
from pathlib import Path

CORRIDORS_SHP  = Path("data/collected/NY_345kv_corridor.shp")
OUTPUT_DIR     = Path("data/naip/raw")
NAIP_YEAR      = "2022"
MAX_TILES      = 20

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading corridor polygon...")
gdf = gpd.read_file(CORRIDORS_SHP).to_crs("EPSG:4326")
bbox = gdf.total_bounds
print(f"Bounding box (WGS84): {bbox}")

print("\nConnecting to Planetary Computer...")
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

print(f"Searching for NAIP {NAIP_YEAR} tiles...")
search = catalog.search(
    collections=["naip"],
    bbox=bbox.tolist(),
    query={"naip:state": {"eq": "ny"}},
    datetime=f"{NAIP_YEAR}-01-01/{NAIP_YEAR}-12-31",
)
items = list(search.items())
print(f"Found {len(items)} NAIP tiles")

if len(items) == 0:
    print("Trying 2021...")
    search = catalog.search(
        collections=["naip"],
        bbox=bbox.tolist(),
        query={"naip:state": {"eq": "ny"}},
        datetime="2021-01-01/2021-12-31",
    )
    items = list(search.items())
    print(f"Found {len(items)} tiles for 2021")

if len(items) == 0:
    print("No NAIP tiles found.")
    exit(1)

print(f"\nDownloading up to {MAX_TILES} tiles to {OUTPUT_DIR}...")
downloaded = 0
for item in items[:MAX_TILES]:
    if "image" not in item.assets:
        continue
    asset = item.assets["image"]
    fname = OUTPUT_DIR / f"{item.id}.tif"
    if fname.exists():
        print(f"  Already exists: {fname.name}")
        downloaded += 1
        continue
    print(f"  Downloading {item.id}...", end=" ", flush=True)
    try:
        r = requests.get(asset.href, stream=True, timeout=120)
        r.raise_for_status()
        with open(fname, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        size_mb = fname.stat().st_size / (1024 * 1024)
        print(f"done ({size_mb:.1f} MB)")
        downloaded += 1
    except Exception as e:
        print(f"FAILED: {e}")

print(f"\nDownloaded {downloaded} tiles to {OUTPUT_DIR}/")
