"""Download NOAA GHCN-Daily weather for NYC Central Park.

Station: USW00094728 (NY CITY CENTRAL PARK, NY US).
Source: public NCEI GHCN-Daily station CSV (no API token required).

Keeps the covariates used in Panel A (analysis_plan.md §4): precipitation and
temperature. Converts GHCN units to mm / °C and filters to the configured
sample window.

Usage
-----
    python -m cpca.ingest.download_noaa_weather
    python -m cpca.ingest.download_noaa_weather --start 2022-01-03 --end 2026-06-30
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[3]

# GHCN tenths → standard units
VALUE_COLS = ("prcp", "tmax", "tmin", "tavg", "snow", "snwd", "awnd")
TENTHS_MM = {"prcp", "snow", "snwd"}
TENTHS_C = {"tmax", "tmin", "tavg"}
TENTHS_MS = {"awnd"}  # tenths of m/s


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def download_station_csv(url: str, timeout: int = 300) -> pd.DataFrame:
    print(f"Downloading {url}")
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    from io import BytesIO

    return pd.read_csv(BytesIO(resp.content), low_memory=False)


def tidy_ghcn(raw: pd.DataFrame, columns: list[str], start: date, end: date) -> pd.DataFrame:
    keep = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", *columns]
    missing = [c for c in keep if c not in raw.columns]
    if missing:
        raise ValueError(f"Expected GHCN columns missing: {missing}")

    df = raw[keep].copy()
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
    df = df.loc[mask].copy()

    for col in VALUE_COLS:
        if col not in df.columns:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if col in TENTHS_MM:
            df[col] = df[col] / 10.0  # tenths of mm → mm
        elif col in TENTHS_C:
            df[col] = df[col] / 10.0  # tenths of °C → °C
        elif col in TENTHS_MS:
            df[col] = df[col] / 10.0  # tenths of m/s → m/s

    # Prefer reported TAVG; fall back to midpoint of TMAX/TMIN when missing.
    if {"tmax", "tmin", "tavg"}.issubset(df.columns):
        midpoint = (df["tmax"] + df["tmin"]) / 2.0
        df["tavg"] = df["tavg"].fillna(midpoint)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    cfg = settings["noaa_ghcn"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=cfg["start_date"], help="YYYY-MM-DD")
    p.add_argument("--end", default=cfg["end_date"], help="YYYY-MM-DD")
    p.add_argument(
        "--out",
        default=cfg["output_path"],
        help="Output parquet path",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output already exists",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    cfg = settings["noaa_ghcn"]

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        print("--end must be >= --start", file=sys.stderr)
        return 1

    out_path = ROOT / args.out
    if out_path.exists() and not args.force:
        print(f"Output already exists (use --force to refresh): {out_path}")
        return 0

    raw = download_station_csv(cfg["url"])
    daily = tidy_ghcn(raw, columns=cfg["columns"], start=start, end=end)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    daily.to_parquet(tmp, index=False)
    tmp.rename(out_path)

    print(
        f"Wrote {len(daily):,} days "
        f"({daily['date'].min().date()} → {daily['date'].max().date()}) "
        f"→ {out_path}"
    )
    print(daily[["date", "prcp", "tavg", "tmax", "tmin"]].describe().round(2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
