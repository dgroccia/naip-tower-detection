"""
osm_va_towers.py (v3)
Pulls power transmission tower point locations for Virginia from OpenStreetMap,
clipped to Virginia's actual administrative boundary (not a rectangular bbox).
"""
import requests
import json
import time

# Area-based query — uses VA's actual ISO boundary, not a rectangle
QUERY = """
[out:json][timeout:180];
area["ISO3166-2"="US-VA"]["admin_level"="4"]->.va;
(
  node(area.va)["power"="tower"];
);
out body;
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

def fetch_towers(max_retries_per_endpoint=2):
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


def main():
    print("Querying OSM for power towers within VA's actual boundary...")
    data = fetch_towers()
    elements = data.get("elements", [])
    print(f"\nTotal towers found (VA boundary, not bbox): {len(elements)}")

    out_path = "data/collected/osm_va_towers_clipped.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved to {out_path}")

    has_voltage = sum(1 for e in elements if "voltage" in e.get("tags", {}))
    print(f"Towers with voltage tag on the node itself: {has_voltage} / {len(elements)}")
    print("\nNote: voltage is typically tagged on the LINE way, not the tower node.")
    print("A separate query against power=line ways with voltage tags is needed")
    print("to isolate towers comparable to your 345kV/500kV annotated dataset.")


if __name__ == "__main__":
    main()
