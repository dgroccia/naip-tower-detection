"""
osm_va_towers.py (v2)
Pulls power transmission tower point locations for Virginia from OpenStreetMap
via the Overpass API. Free, no auth required. Includes proper headers and
multiple fallback mirrors with retry/backoff.
"""
import requests
import json
import time

VA_BBOX = "36.5,-83.7,39.5,-75.2"

QUERY = f"""
[out:json][timeout:120];
(
  node["power"="tower"]({VA_BBOX});
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
                    endpoint,
                    data={"data": QUERY},
                    headers=HEADERS,
                    timeout=130,
                )
                if resp.status_code == 429:
                    print("  Rate limited (429). Waiting 15s before retry...")
                    time.sleep(15)
                    continue
                resp.raise_for_status()
                data = resp.json()
                print(f"  Success.")
                return data
            except Exception as e:
                print(f"  Failed: {e}")
                time.sleep(3)
    raise RuntimeError("All Overpass endpoints failed after retries")


def main():
    print(f"Querying OSM for power towers in Virginia (bbox: {VA_BBOX})...")
    data = fetch_towers()
    elements = data.get("elements", [])
    print(f"\nTotal towers found: {len(elements)}")

    out_path = "data/collected/osm_va_towers.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved raw results to {out_path}")

    has_voltage = sum(1 for e in elements if "voltage" in e.get("tags", {}))
    has_line_ref = sum(1 for e in elements if "line" in e.get("tags", {}) or "ref" in e.get("tags", {}))
    print(f"\nTowers with voltage tag: {has_voltage} / {len(elements)}")
    print(f"Towers with line/ref tag: {has_line_ref} / {len(elements)}")

    print("\nSample tags from first 3 towers:")
    for e in elements[:3]:
        print(f"  ID {e['id']}: {e.get('tags', {})}")


if __name__ == "__main__":
    main()
