"""CRZ geometry helpers for station zone assignment."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CRZ_PATH = ROOT / "data" / "raw" / "crz" / "crz_boundary.geojson"


@lru_cache(maxsize=1)
def load_crz_polygon(path: str | Path | None = None) -> BaseGeometry:
    """Load the unioned CRZ boundary (EPSG:4326 lon/lat)."""
    geo_path = Path(path) if path is not None else DEFAULT_CRZ_PATH
    if not geo_path.exists():
        raise FileNotFoundError(
            f"CRZ boundary not found at {geo_path}. "
            "Run: PYTHONPATH=src python -m cpca.geo.build_crz_polygon"
        )
    payload = json.loads(geo_path.read_text())
    features = payload.get("features") or []
    if not features:
        raise ValueError(f"No features in {geo_path}")
    return shape(features[0]["geometry"])


def point_in_crz(lon: float, lat: float, path: str | Path | None = None) -> bool:
    """True if (lon, lat) falls inside the Congestion Relief Zone."""
    return load_crz_polygon(path).covers(Point(lon, lat))
