"""Assign subway station complexes to CRZ / border / control zones.

Rules (analysis_plan.md §4, config/treatment.yaml)
-------------------------------------------------
- crz:     centroid inside the official CRZ polygon
- border:  Manhattan centroid between 60th and 96th St (approx lat band),
           and not inside the CRZ
- control: all other complexes with dist_to_boundary >= control_min_distance_km
- near:    remaining complexes within the min-distance buffer (excluded from
           primary DiD control pool)

Distance is computed in EPSG:2263 (NAD83 / New York Long Island), in meters.
Signed distance is negative inside the CRZ and positive outside.

Usage
-----
    python -m cpca.geo.assign_zones
    python -m cpca.geo.assign_zones --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml
from shapely.geometry import Point
from shapely.ops import nearest_points

from cpca.geo.crz import load_crz_polygon

ROOT = Path(__file__).resolve().parents[3]
# NAD83 / New York Long Island — coordinates are US survey feet.
NY_LONG_ISLAND = "EPSG:2263"
US_SURVEY_FOOT_TO_M = 0.304800609601219


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def station_centroids_from_ridership(ridership_dir: Path) -> pd.DataFrame:
    """One row per station_complex_id from hourly ridership partitions."""
    paths = sorted(ridership_dir.glob("year=*/month=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"No ridership partitions under {ridership_dir}")

    frames = [
        pd.read_parquet(
            p,
            columns=[
                "station_complex_id",
                "station_complex",
                "borough",
                "latitude",
                "longitude",
            ],
        )
        for p in paths
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["station_complex_id"] = raw["station_complex_id"].astype(str)

    # Median lat/lon absorbs occasional coordinate drift across months.
    stations = (
        raw.groupby("station_complex_id", as_index=False)
        .agg(
            station_complex=("station_complex", "first"),
            borough=("borough", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            n_obs=("station_complex_id", "size"),
        )
        .sort_values("station_complex_id")
        .reset_index(drop=True)
    )
    return stations


def assign_zones(
    stations: pd.DataFrame,
    *,
    border_north_lat: float,
    border_south_lat: float,
    control_min_distance_km: float,
    crz_path: Path | None = None,
) -> gpd.GeoDataFrame:
    crz = load_crz_polygon(crz_path)
    gdf = gpd.GeoDataFrame(
        stations.copy(),
        geometry=[
            Point(xy) for xy in zip(stations["longitude"], stations["latitude"], strict=True)
        ],
        crs="EPSG:4326",
    )

    gdf["in_crz"] = gdf.geometry.apply(lambda p: bool(crz.covers(p)))

    # Project for metric distances (EPSG:2263 is US survey feet).
    gdf_ft = gdf.to_crs(NY_LONG_ISLAND)
    crz_ft = gpd.GeoSeries([crz], crs="EPSG:4326").to_crs(NY_LONG_ISLAND).iloc[0]
    boundary_ft = crz_ft.boundary

    dist_ft = gdf_ft.geometry.distance(boundary_ft)
    # Signed: negative inside polygon.
    signed_ft = dist_ft.where(~gdf["in_crz"], -dist_ft)
    gdf["dist_to_crz_boundary_m"] = signed_ft.to_numpy() * US_SURVEY_FOOT_TO_M
    gdf["dist_to_crz_boundary_km"] = gdf["dist_to_crz_boundary_m"] / 1000.0
    gdf["abs_dist_to_crz_boundary_km"] = gdf["dist_to_crz_boundary_km"].abs()

    manhattan = gdf["borough"].str.lower().eq("manhattan")
    # East Side ~60th sits south of West Side ~60th in lat/lon, so the south
    # edge of the border band is looser than the CRZ's western north edge.
    # Roosevelt Island is geographically in-band but not on the 60th–96th
    # Manhattan street grid — keep it out of border.
    roosevelt_island = gdf["station_complex"].str.contains(
        r"Roosevelt|Tramway", case=False, na=False
    )
    in_border_band = (
        manhattan
        & ~gdf["in_crz"]
        & ~roosevelt_island
        & (gdf["latitude"] >= border_south_lat)
        & (gdf["latitude"] <= border_north_lat)
    )

    zone = pd.Series("near", index=gdf.index, dtype="object")
    zone = zone.mask(gdf["abs_dist_to_crz_boundary_km"] >= control_min_distance_km, "control")
    zone = zone.mask(in_border_band, "border")
    zone = zone.mask(gdf["in_crz"], "crz")
    gdf["zone"] = zone

    # Nearest boundary point (lon/lat) for QA maps.
    nearest = []
    crz_ll = crz  # lon/lat
    for pt in gdf.geometry:
        _, bpt = nearest_points(pt, crz_ll.boundary)
        nearest.append(bpt)
    gdf["nearest_boundary_lon"] = [p.x for p in nearest]
    gdf["nearest_boundary_lat"] = [p.y for p in nearest]

    return gdf


def parse_args() -> argparse.Namespace:
    settings = load_yaml(ROOT / "config" / "settings.yaml")
    treatment = load_yaml(ROOT / "config" / "treatment.yaml")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ridership-dir",
        default=settings["mta_ridership"]["output_dir"],
        help="Directory of monthly ridership parquet partitions",
    )
    p.add_argument(
        "--out",
        default=settings.get("station_zones", {}).get(
            "output_path", "data/processed/station_zones.parquet"
        ),
        help="Output parquet path",
    )
    p.add_argument(
        "--geojson-out",
        default=settings.get("station_zones", {}).get(
            "geojson_path", "data/processed/station_zones.geojson"
        ),
        help="Output GeoJSON path",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--border-north-lat",
        type=float,
        default=float(treatment["zone_bands"]["border_north_lat_approx"]),
    )
    p.add_argument(
        "--border-south-lat",
        type=float,
        default=float(treatment["zone_bands"]["border_south_lat_approx"]),
    )
    p.add_argument(
        "--control-min-km",
        type=float,
        default=float(treatment["zone_bands"]["control_min_distance_km"]),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = ROOT / args.out
    geojson_path = ROOT / args.geojson_out
    if out_path.exists() and not args.force:
        print(f"Output already exists (use --force to refresh): {out_path}")
        return 0

    stations = station_centroids_from_ridership(ROOT / args.ridership_dir)
    print(f"Station complexes: {len(stations):,}")

    gdf = assign_zones(
        stations,
        border_north_lat=args.border_north_lat,
        border_south_lat=args.border_south_lat,
        control_min_distance_km=args.control_min_km,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "station_complex_id",
        "station_complex",
        "borough",
        "latitude",
        "longitude",
        "n_obs",
        "in_crz",
        "zone",
        "dist_to_crz_boundary_m",
        "dist_to_crz_boundary_km",
        "abs_dist_to_crz_boundary_km",
        "nearest_boundary_lon",
        "nearest_boundary_lat",
    ]
    table = pd.DataFrame(gdf[cols])
    tmp = out_path.with_suffix(".parquet.tmp")
    table.to_parquet(tmp, index=False)
    tmp.rename(out_path)

    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    gdf[cols + ["geometry"]].to_file(geojson_path, driver="GeoJSON")

    print(f"Wrote {out_path}")
    print(f"Wrote {geojson_path}")
    print("\nZone counts:")
    print(table["zone"].value_counts().to_string())
    print("\nBy borough x zone:")
    print(pd.crosstab(table["borough"], table["zone"]).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
