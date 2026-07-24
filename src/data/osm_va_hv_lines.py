"""
osm_va_hv_lines.py
Step 1: Pull power=line ways in Virginia with voltage tags, filter to 345kV+.
Step 2: Spatially associate tower nodes (already pulled) to these HV lines.
"""
import requests
import json
import time
import re
from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point

# ── Step 1: query HV line ways ────────────────────────────────────────────────
QUERY = """
[out:json][timeout:180];
area["ISO3166-2"="US-VA"]["admin_level"="4"]->.va;
(
  way(area.va)["power"="line"]["voltage"];
);
out geom;
"""

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
    "https://overpass.osm.ch/api/interpreter",
]

HEADERS = {
    "User-Agent": "thesis-research-script/1.0 (academic use, GMU)",
    "Content-Type": "application/x-www-form-urlencoded",
}

MIN_VOLTAGE = 345_000  # 345kV in volts — OSM voltage tags are in volts


def fetch_lines(max_retries_per_endpoint=2):
    for endpoint in OVERPASS_ENDPOINTS:
        for attempt in range(max_retries_per_endpoint):
            print(f"Trying {endpoint} (attempt {attempt + 1})...")
            try:
                resp = requests.post(
                    endpoint, data={"data": QUERY}, headers=HEADERS, timeout=190,
                )
                if resp.status_code == 429:
                    print("  Rate limited. Waiting 15s...")
                    time.sleep(15)
                    continue
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                print(f"  Failed: {e}")
                time.sleep(3)
    raise RuntimeError("All Overpass endpoints failed after retries")


def parse_voltage(voltage_str):
    """
    OSM voltage tags can be a single value '345000' or multiple
    semicolon-separated values for shared corridors '345000;500000'.
    Returns the max parsed voltage, or None if unparseable.
    """
    if not voltage_str:
        return None
    parts = re.split(r"[;,]", voltage_str)
    values = []
    for p in parts:
        p = p.strip()
        if p.isdigit():
            values.append(int(p))
    return max(values) if values else None


def main():
    print("Step 1: Fetching power=line ways with voltage tags in Virginia...")
    data = fetch_lines()
    elements = data.get("elements", [])
    print(f"Total tagged line ways found: {len(elements)}")

    # Filter to 345kV+ and build LineString geometries
    hv_lines = []
    for el in elements:
        if el.get("type") != "way" or "geometry" not in el:
            continue
        voltage_raw = el.get("tags", {}).get("voltage", "")
        max_v = parse_voltage(voltage_raw)
        if max_v is None or max_v < MIN_VOLTAGE:
            continue
        coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"] if pt]
        if len(coords) < 2:
            continue
        hv_lines.append({
            "way_id": el["id"],
            "voltage": max_v,
            "geometry": LineString(coords),
        })

    print(f"Lines at or above 345kV: {len(hv_lines)}")

    if not hv_lines:
        print("No HV lines found — check voltage tag parsing or query.")
        return

    lines_gdf = gpd.GeoDataFrame(hv_lines, crs="EPSG:4326")
    lines_gdf.to_file("data/collected/osm_va_hv_lines.geojson", driver="GeoJSON")
    print("Saved HV lines to data/collected/osm_va_hv_lines.geojson")

    # ── Step 2: associate tower nodes to HV lines ──────────────────────────────
    print("\nStep 2: Loading previously pulled tower nodes...")
    tower_path = Path("data/collected/osm_va_towers_clipped.json")
    if not tower_path.exists():
        print(f"Tower file not found at {tower_path} — run osm_va_towers.py first.")
        return

    with open(tower_path) as f:
        tower_data = json.load(f)

    towers = [
        Point(e["lon"], e["lat"])
        for e in tower_data.get("elements", [])
        if e.get("type") == "node" and "lon" in e and "lat" in e
    ]
    towers_gdf = gpd.GeoDataFrame({"geometry": towers}, crs="EPSG:4326")
    print(f"Total tower nodes loaded: {len(towers_gdf)}")

    # Reproject to a metric CRS for buffering (Virginia — UTM zone 17/18N is reasonable)
    lines_proj  = lines_gdf.to_crs("EPSG:32617")
    towers_proj = towers_gdf.to_crs("EPSG:32617")

    BUFFER_M = 75  # towers should sit very close to the line centerline
    lines_buffered = lines_proj.copy()
    lines_buffered["geometry"] = lines_proj.geometry.buffer(BUFFER_M)

    # Union all HV line buffers into one geometry for a fast spatial filter
    hv_corridor = lines_buffered.geometry.union_all()

    towers_proj["near_hv_line"] = towers_proj.geometry.within(hv_corridor)
    hv_tower_count = towers_proj["near_hv_line"].sum()

    print(f"\nTower nodes within {BUFFER_M}m of a 345kV+ line: {hv_tower_count}")
    print(f"Out of total tower nodes in VA: {len(towers_proj)}")

    # Save the filtered HV towers
    hv_towers = towers_proj[towers_proj["near_hv_line"]].to_crs("EPSG:4326")
    hv_towers.to_file("data/collected/osm_va_hv_towers.geojson", driver="GeoJSON")
    print("Saved filtered HV towers to data/collected/osm_va_hv_towers.geojson")


if __name__ == "__main__":
    main()
