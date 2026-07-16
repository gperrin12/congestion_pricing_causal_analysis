"""Build Panel A: station_complex × ISO week.

Consumes
--------
- data/raw/mta_subway_hourly/**/part.parquet
- data/processed/station_zones.parquet
- data/raw/noaa_ghcn_central_park/daily.parquet
- config/treatment.yaml, config/settings.yaml

Produces
--------
- data/processed/panel_station_week.parquet
- data/processed/manifest.json  (merged/updated panel entry)

Outcomes follow analysis_plan.md §4:
- weekly_entries, weekday_peak_entries, weekend_entries
- log1p counterparts
- weekly precip / mean temp, holiday_week, fare_change
- zone + distance columns from station_zones
- period labels (pre / washout / post)

Usage
-----
    python -m cpca.panel.build_station_week
    python -m cpca.panel.build_station_week --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]

# US federal holiday observed dates in the sample window (NYC).
# Kept explicit so the panel build does not depend on an external holidays lib.
FEDERAL_HOLIDAYS = pd.to_datetime(
    [
        # 2022
        "2022-01-17",
        "2022-02-21",
        "2022-05-30",
        "2022-06-20",
        "2022-07-04",
        "2022-09-05",
        "2022-10-10",
        "2022-11-11",
        "2022-11-24",
        "2022-12-26",
        # 2023
        "2023-01-02",
        "2023-01-16",
        "2023-02-20",
        "2023-05-29",
        "2023-06-19",
        "2023-07-04",
        "2023-09-04",
        "2023-10-09",
        "2023-11-10",
        "2023-11-23",
        "2023-12-25",
        # 2024
        "2024-01-01",
        "2024-01-15",
        "2024-02-19",
        "2024-05-27",
        "2024-06-19",
        "2024-07-04",
        "2024-09-02",
        "2024-10-14",
        "2024-11-11",
        "2024-11-28",
        "2024-12-25",
        # 2025
        "2025-01-01",
        "2025-01-20",
        "2025-02-17",
        "2025-05-26",
        "2025-06-19",
        "2025-07-04",
        "2025-09-01",
        "2025-10-13",
        "2025-11-11",
        "2025-11-27",
        "2025-12-25",
        # 2026
        "2026-01-01",
        "2026-01-19",
        "2026-02-16",
        "2026-05-25",
        "2026-06-19",
        "2026-07-03",
    ]
)


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def iso_week_start(ts: pd.Series) -> pd.Series:
    """Monday-start ISO week date for each timestamp."""
    days = ts.dt.normalize()
    return days - pd.to_timedelta(days.dt.dayofweek, unit="D")


def assign_period(week_start: pd.Series, windows: dict) -> pd.Series:
    pre_end = pd.Timestamp(windows["pre_end"])
    washout_end = pd.Timestamp(windows["washout_end"])
    out = pd.Series("post", index=week_start.index, dtype="object")
    out = out.mask(week_start <= pre_end, "pre")
    out = out.mask(
        (week_start > pre_end) & (week_start <= washout_end),
        "washout",
    )
    return out


def aggregate_ridership_weeks(
    ridership_dir: Path,
    *,
    start: date,
    end: date,
    peak_hours: list[int],
) -> pd.DataFrame:
    """Month-wise aggregation to station-week outcomes."""
    paths = sorted(ridership_dir.glob("year=*/month=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"No ridership partitions under {ridership_dir}")

    peak = set(int(h) for h in peak_hours)
    pieces: list[pd.DataFrame] = []

    for path in paths:
        df = pd.read_parquet(
            path,
            columns=["transit_timestamp", "station_complex_id", "ridership"],
        )
        ts = pd.to_datetime(df["transit_timestamp"])
        day = ts.dt.normalize()
        keep = (day.dt.date >= start) & (day.dt.date <= end)
        if not keep.any():
            continue

        sid = df.loc[keep, "station_complex_id"].astype(str)
        rides = pd.to_numeric(df.loc[keep, "ridership"], errors="coerce").fillna(0.0)
        ts_k = ts.loc[keep]
        week_start = iso_week_start(ts_k)
        dow = ts_k.dt.dayofweek
        hour = ts_k.dt.hour
        is_weekend = dow >= 5
        is_peak = (~is_weekend) & hour.isin(peak)

        tmp = pd.DataFrame(
            {
                "station_complex_id": sid.to_numpy(),
                "week_start": week_start.to_numpy(),
                "weekly_entries": rides.to_numpy(),
                "weekday_peak_entries": np.where(is_peak.to_numpy(), rides.to_numpy(), 0.0),
                "weekend_entries": np.where(is_weekend.to_numpy(), rides.to_numpy(), 0.0),
            }
        )
        pieces.append(
            tmp.groupby(["station_complex_id", "week_start"], as_index=False).sum()
        )

    if not pieces:
        raise ValueError("No ridership rows in the requested date window")

    out = (
        pd.concat(pieces, ignore_index=True)
        .groupby(["station_complex_id", "week_start"], as_index=False)
        .sum()
    )
    out["week_start"] = pd.to_datetime(out["week_start"]).dt.normalize()
    return out


def weekly_weather(weather_path: Path, start: date, end: date) -> pd.DataFrame:
    w = pd.read_parquet(weather_path, columns=["date", "prcp", "tavg"])
    w["date"] = pd.to_datetime(w["date"]).dt.normalize()
    mask = (w["date"].dt.date >= start) & (w["date"].dt.date <= end)
    w = w.loc[mask].copy()
    w["week_start"] = iso_week_start(w["date"])
    return (
        w.groupby("week_start", as_index=False)
        .agg(precip_mm=("prcp", "sum"), tavg_c=("tavg", "mean"))
        .sort_values("week_start")
    )


def holiday_weeks(week_starts: pd.Series) -> pd.Series:
    hol = FEDERAL_HOLIDAYS
    hol_weeks = set(iso_week_start(pd.Series(hol)).dt.normalize())
    return week_starts.dt.normalize().isin(hol_weeks)


def drop_unstable_stations(
    panel: pd.DataFrame,
    *,
    min_week_coverage: float,
    sample_start: date,
    sample_end: date,
) -> tuple[pd.DataFrame, list[dict]]:
    """Drop complexes that opened/closed mid-sample or have sparse coverage."""
    n_weeks = panel["week_start"].nunique()
    min_weeks = int(np.ceil(min_week_coverage * n_weeks))

    first_ok = pd.Timestamp(sample_start) + pd.Timedelta(days=42)
    last_ok = pd.Timestamp(sample_end) - pd.Timedelta(days=42)

    stats = panel.groupby("station_complex_id").agg(
        n_weeks=("week_start", "nunique"),
        first_week=("week_start", "min"),
        last_week=("week_start", "max"),
        station_complex=("station_complex", "first"),
    )
    keep_mask = (
        (stats["n_weeks"] >= min_weeks)
        & (stats["first_week"] <= first_ok)
        & (stats["last_week"] >= last_ok)
    )
    dropped = stats.loc[~keep_mask].reset_index()
    dropped_records = dropped.to_dict(orient="records")
    for row in dropped_records:
        row["first_week"] = str(pd.Timestamp(row["first_week"]).date())
        row["last_week"] = str(pd.Timestamp(row["last_week"]).date())
        row["n_weeks"] = int(row["n_weeks"])

    keep_ids = set(stats.index[keep_mask].astype(str))
    return panel.loc[panel["station_complex_id"].isin(keep_ids)].copy(), dropped_records


def build_panel(
    *,
    settings: dict,
    treatment: dict,
) -> tuple[pd.DataFrame, dict]:
    windows = treatment["sample_windows"]
    start = date.fromisoformat(windows["pre_start"])
    end = date.fromisoformat(windows["post_end"])
    panel_cfg = settings["panel_station_week"]
    peak_hours = panel_cfg["weekday_peak_hours"]

    ridership = aggregate_ridership_weeks(
        ROOT / settings["mta_ridership"]["output_dir"],
        start=start,
        end=end,
        peak_hours=peak_hours,
    )
    zones = pd.read_parquet(ROOT / settings["station_zones"]["output_path"])
    zones["station_complex_id"] = zones["station_complex_id"].astype(str)

    weather = weekly_weather(
        ROOT / settings["noaa_ghcn"]["output_path"],
        start=start,
        end=end,
    )

    zone_cols = [
        "station_complex_id",
        "station_complex",
        "borough",
        "latitude",
        "longitude",
        "zone",
        "in_crz",
        "dist_to_crz_boundary_km",
        "abs_dist_to_crz_boundary_km",
    ]
    panel = ridership.merge(zones[zone_cols], on="station_complex_id", how="inner")
    panel = panel.merge(weather, on="week_start", how="left")

    panel["log_weekly_entries"] = np.log1p(panel["weekly_entries"])
    panel["log_weekday_peak_entries"] = np.log1p(panel["weekday_peak_entries"])
    panel["log_weekend_entries"] = np.log1p(panel["weekend_entries"])

    panel["iso_year"] = panel["week_start"].dt.isocalendar().year.astype(int)
    panel["iso_week"] = panel["week_start"].dt.isocalendar().week.astype(int)
    panel["holiday_week"] = holiday_weeks(panel["week_start"])

    fare_date = pd.Timestamp(treatment["treatment"]["fare_change_date"])
    fare_week = iso_week_start(pd.Series([fare_date])).iloc[0]
    panel["fare_change"] = panel["week_start"] >= fare_week

    panel["period"] = assign_period(panel["week_start"], windows)
    panel["post"] = panel["period"].eq("post")
    t0 = pd.Timestamp(treatment["treatment"]["t0"])
    panel["weeks_since_t0"] = ((panel["week_start"] - iso_week_start(pd.Series([t0])).iloc[0]).dt.days // 7).astype(int)

    panel, dropped = drop_unstable_stations(
        panel,
        min_week_coverage=float(panel_cfg["min_week_coverage"]),
        sample_start=start,
        sample_end=end,
    )

    panel = panel.sort_values(["station_complex_id", "week_start"]).reset_index(drop=True)

    # Dtypes for a stable on-disk schema.
    panel["station_complex_id"] = panel["station_complex_id"].astype(str)
    for col in ("station_complex", "borough", "zone", "period"):
        panel[col] = panel[col].astype(str)
    panel["in_crz"] = panel["in_crz"].astype(bool)
    panel["holiday_week"] = panel["holiday_week"].astype(bool)
    panel["fare_change"] = panel["fare_change"].astype(bool)
    panel["post"] = panel["post"].astype(bool)

    manifest = {
        "panel": "panel_station_week",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(panel)),
        "n_stations": int(panel["station_complex_id"].nunique()),
        "n_weeks": int(panel["week_start"].nunique()),
        "week_start_min": str(panel["week_start"].min().date()),
        "week_start_max": str(panel["week_start"].max().date()),
        "zone_counts": panel.groupby("zone")["station_complex_id"].nunique().to_dict(),
        "period_counts": panel["period"].value_counts().to_dict(),
        "dropped_stations": dropped,
        "min_week_coverage": float(panel_cfg["min_week_coverage"]),
        "weekday_peak_hours": peak_hours,
        "sample_windows": windows,
        "t0": treatment["treatment"]["t0"],
        "fare_change_date": treatment["treatment"]["fare_change_date"],
        "inputs": {
            "ridership_dir": settings["mta_ridership"]["output_dir"],
            "n_ridership_partitions": len(
                list((ROOT / settings["mta_ridership"]["output_dir"]).glob("year=*/month=*/part.parquet"))
            ),
            "weather_path": settings["noaa_ghcn"]["output_path"],
            "weather_sha256": file_sha256(ROOT / settings["noaa_ghcn"]["output_path"]),
            "zones_path": settings["station_zones"]["output_path"],
            "zones_sha256": file_sha256(ROOT / settings["station_zones"]["output_path"]),
        },
    }
    return panel, manifest


def update_manifest(manifest_path: Path, panel_manifest: dict) -> None:
    payload: dict
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text())
    else:
        payload = {}
    payload[panel_manifest["panel"]] = panel_manifest
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    tmp.rename(manifest_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_yaml(ROOT / "config" / "settings.yaml")
    treatment = load_yaml(ROOT / "config" / "treatment.yaml")
    out_path = ROOT / settings["panel_station_week"]["output_path"]
    manifest_path = ROOT / settings["panel_station_week"]["manifest_path"]

    if out_path.exists() and not args.force:
        print(f"Output already exists (use --force to refresh): {out_path}")
        return 0

    panel, manifest = build_panel(settings=settings, treatment=treatment)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    panel.to_parquet(tmp, index=False)
    tmp.rename(out_path)

    manifest["panel_sha256"] = file_sha256(out_path)
    update_manifest(manifest_path, manifest)

    print(f"Wrote {out_path}")
    print(f"Updated {manifest_path}")
    print(
        f"rows={manifest['n_rows']:,} stations={manifest['n_stations']:,} "
        f"weeks={manifest['n_weeks']:,} "
        f"({manifest['week_start_min']} → {manifest['week_start_max']})"
    )
    print("zone station counts:", manifest["zone_counts"])
    print("dropped stations:", len(manifest["dropped_stations"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
