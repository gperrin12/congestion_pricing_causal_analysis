"""Build the official NYC Congestion Relief Zone (CRZ) boundary polygon.

Downloads the MTA Central Business District geofence from data.ny.gov
(srxy-5nxn), unions its component polygons into a single geometry used for
Panel A station-complex zone assignment (analysis_plan.md §4), and writes:

- data/raw/crz/mta_cbd_geofence_raw.geojson  (source features as published)
- data/raw/crz/crz_boundary.geojson          (unary union, CRS EPSG:4326)

Usage
-----
    python -m cpca.geo.build_crz_polygon
    python -m cpca.geo.build_crz_polygon --force
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parents[3]


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def download_geojson(url: str, timeout: int = 120) -> dict:
    print(f"Downloading {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("type") != "FeatureCollection" or not data.get("features"):
        raise ValueError("Expected a non-empty GeoJSON FeatureCollection")
    return data


def union_features(feature_collection: dict) -> dict:
    geoms = [shape(feat["geometry"]) for feat in feature_collection["features"]]
    if not geoms:
        raise ValueError("No geometries to union")
    merged = unary_union(geoms)
    if merged.is_empty:
        raise ValueError("Union of CRZ geofence polygons is empty")

    return {
        "type": "FeatureCollection",
        "name": "nyc_congestion_relief_zone",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Congestion Relief Zone",
                    "source": "MTA Central Business District Geofence (srxy-5nxn)",
                    "definition": (
                        "Manhattan at/south of 60th Street, excluding FDR Drive, "
                        "West Side Highway/Route 9A, Battery Park Underpass, and "
                        "Hugh L. Carey Tunnel surface connections to West Street"
                    ),
                    "n_source_polygons": len(geoms),
                    "geom_type": merged.geom_type,
                },
                "geometry": mapping(merged),
            }
        ],
    }


def write_geojson(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.rename(path)


def summarize(feature_collection: dict) -> None:
    geom = shape(feature_collection["features"][0]["geometry"])
    minx, miny, maxx, maxy = geom.bounds
    print(
        f"union geom_type={geom.geom_type} "
        f"bounds=[{minx:.5f}, {miny:.5f}, {maxx:.5f}, {maxy:.5f}]"
    )
    # Rough area in km^2 using equirectangular meters near NYC latitude.
    import math

    lat0 = (miny + maxy) / 2.0
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))
    area_km2 = geom.area * m_per_deg_lat * m_per_deg_lon / 1e6
    print(f"approx area ≈ {area_km2:.1f} km²")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download/rebuild even if outputs already exist",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_settings()["crz_boundary"]
    raw_path = ROOT / cfg["raw_path"]
    union_path = ROOT / cfg["union_path"]

    if raw_path.exists() and union_path.exists() and not args.force:
        print(f"Outputs already exist (use --force to refresh):\n  {raw_path}\n  {union_path}")
        return 0

    raw = download_geojson(cfg["geojson_url"])
    write_geojson(raw_path, raw)
    print(f"Wrote {len(raw['features'])} source polygons → {raw_path}")

    unioned = union_features(raw)
    write_geojson(union_path, unioned)
    print(f"Wrote unioned CRZ boundary → {union_path}")
    summarize(unioned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
