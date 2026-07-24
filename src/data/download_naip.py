"""
download_naip.py
----------------
Downloads NAIP tiles for specified corridor bounding boxes via
Microsoft Planetary Computer STAC API.

No account or API key required — Planetary Computer is free and public.

Usage:
    conda activate thesis
    python src/data/download_naip.py

Edit the CORRIDORS dict below with your actual bounding boxes.
Get bbox coordinates from ArcPro:
    - Zoom to your corridor polygon
    - Right-click the corridors layer > Properties > Extent tab
    - Note the Top, Bottom, Left, Right values (these are your bbox)
    - Convert to WGS84 decimal degrees if needed (check CRS in Properties)
"""

import logging
from pathlib import Path

import planetary_computer
import pystac_client
import requests
from odc.stac import load
import rasterio
from rasterio.crs import CRS
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── EDIT THIS SECTION ─────────────────────────────────────────────────────────
# Add your 5 corridor bounding boxes here.
# Format: [min_lon, min_lat, max_lon, max_lat] in WGS84 decimal degrees.
#
# To get bbox from ArcPro:
#   1. Right-click corridors layer in Contents > Properties > Extent
#   2. Note Left/Right/Top/Bottom values
#   3. If your map CRS is Web Mercator, reproject to WGS84:
#      ArcPro > Analysis Tools > Project (or just read coords off the
#      bottom status bar when hovering over corners of your polygon —
#      it shows lat/lon if your display is set to degrees)

CORRIDORS = {
    "site_01_TN": [-84.794931, 35.533089, -84.75412,  35.596303],  # ~4.5 x 8.7 km
    "site_01_WV": [-79.080633, 39.160522, -78.95963,  39.188335],  # ~13.5 x 4.0 km
    "site_02_WV": [-79.198853, 39.183525, -79.080377, 39.20462 ],  # ~13.2 x 3.0 km
    "site_03_WV": [-79.267508, 39.203293, -79.184283, 39.218985],  # ~9.3 x 2.3 km
    "site_02_TN": [-84.623728, 35.670679, -84.486444, 35.700824],  # ~15.3 x 4.1 km
    "site_03_TN": [-84.760667, 35.632555, -84.622718, 35.675207],  # ~15.4 x 5.8 km
    "site_01_TX": [-97.559851, 33.291533, -97.508548, 33.327426],  # ~5.7 x 4.8 km
    "site_03_TX": [-97.489439, 33.317079, -97.391241, 33.342468],  # ~10.9 x 3.4 km
    "site_04_TN": [-84.780858, 35.577625, -84.756121, 35.634937],  # ~2.8 x 7.8 km
    "site_05_TN": [-84.083316, 35.874944, -84.031929, 35.914185],  # ~5.7 x 5.4 km
    "site_06_TN": [-84.136936, 35.919161, -84.082032, 35.986059],  # ~6.1 x 9.2 km
}

# Target year — use most recent available (2022 preferred, 2023 if available)
TARGET_YEAR = "2022"

# Output directory
OUTPUT_DIR = Path("data/naip/raw")

# ─────────────────────────────────────────────────────────────────────────────


def download_naip_for_corridor(
    name: str,
    bbox: list[float],
    output_dir: Path,
    target_year: str = "2022",
) -> Path | None:
    """
    Download the most recent NAIP tile(s) covering a corridor bbox.

    Args:
        name:        Corridor name (used for output filename).
        bbox:        [min_lon, min_lat, max_lon, max_lat] in WGS84.
        output_dir:  Directory to save downloaded GeoTIFF.
        target_year: Preferred acquisition year.

    Returns:
        Path to downloaded GeoTIFF, or None if no data found.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{name}.tif"

    if output_path.exists():
        log.info(f"{name}: already downloaded at {output_path}, skipping")
        return output_path

    log.info(f"{name}: searching Planetary Computer for NAIP {target_year}...")

    # Connect to Planetary Computer STAC API
    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Search for NAIP items covering our bbox
    search = catalog.search(
        collections=["naip"],
        bbox=bbox,
        datetime=f"{target_year}-01-01/{target_year}-12-31",
        max_items=50,
    )

    items = list(search.items())

    if not items:
        # Try the year before
        fallback_year = str(int(target_year) - 1)
        log.warning(f"{name}: no NAIP found for {target_year}, trying {fallback_year}...")
        search = catalog.search(
            collections=["naip"],
            bbox=bbox,
            datetime=f"{fallback_year}-01-01/{fallback_year}-12-31",
            max_items=50,
        )
        items = list(search.items())

    if not items:
        log.error(f"{name}: no NAIP tiles found for this bbox — check coordinates")
        return None

    log.info(f"{name}: found {len(items)} NAIP tile(s)")

    # Load as xarray dataset — odc-stac handles mosaicking multiple tiles
    ds = load(
        items,
        bbox=bbox,
        bands=["red", "green", "blue", "nir"],
        resolution=0.6,  # native NAIP resolution
        groupby="solar_day",
    )

    # Take the most recent acquisition
    ds_single = ds.isel(time=-1)

    # Stack bands into array: (4, H, W)
    red   = ds_single["red"].values.astype(np.uint8)
    green = ds_single["green"].values.astype(np.uint8)
    blue  = ds_single["blue"].values.astype(np.uint8)
    nir   = ds_single["nir"].values.astype(np.uint8)

    data = np.stack([red, green, blue, nir], axis=0)  # (4, H, W)

    # Get geotransform from dataset
    transform = rasterio.transform.from_bounds(
        *bbox,
        width=data.shape[2],
        height=data.shape[1],
    )

    # Write GeoTIFF
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=4,
        dtype=np.uint8,
        crs=CRS.from_epsg(4326),
        transform=transform,
        compress="lzw",  # compress to keep file sizes manageable
    ) as dst:
        dst.write(data)
        dst.update_tags(
            corridor=name,
            naip_year=target_year,
            bands="R,G,B,NIR",
        )

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"{name}: saved to {output_path} ({size_mb:.1f} MB)")
    return output_path


def verify_download(tif_path: Path) -> bool:
    """
    Quick verification that the downloaded GeoTIFF is valid.

    Args:
        tif_path: Path to the GeoTIFF to verify.

    Returns:
        True if valid, False if something looks wrong.
    """
    with rasterio.open(tif_path) as src:
        bands     = src.count
        res       = src.res[0]
        valid_pct = np.sum(src.read(1) > 0) / (src.width * src.height) * 100

        ok = bands == 4 and res < 1.5 and valid_pct > 80

        status = "OK" if ok else "WARNING"
        log.info(
            f"  [{status}] {tif_path.name}: "
            f"{bands} bands, {res:.3f} m/px, {valid_pct:.1f}% valid pixels"
        )
        return ok


def main() -> None:
    log.info(f"Downloading NAIP for {len(CORRIDORS)} corridors...")
    log.info(f"Output directory: {OUTPUT_DIR}")

    results = {}
    for name, bbox in CORRIDORS.items():
        path = download_naip_for_corridor(
            name=name,
            bbox=bbox,
            output_dir=OUTPUT_DIR,
            target_year=TARGET_YEAR,
        )
        results[name] = path

    log.info("\n── Verification ─────────────────────────────────────────")
    all_ok = True
    for name, path in results.items():
        if path is None:
            log.error(f"  {name}: FAILED — no data downloaded")
            all_ok = False
        else:
            ok = verify_download(path)
            if not ok:
                all_ok = False

    if all_ok:
        log.info("\nAll corridors downloaded and verified. Ready for slicing.")
        log.info("Next step:")
        log.info("  python src/data/slice_naip_tiles.py --input_dir data/naip/raw")
    else:
        log.warning("\nSome corridors had issues — check warnings above.")
        log.warning("Fix bbox coordinates for failed corridors and re-run.")


if __name__ == "__main__":
    main()
