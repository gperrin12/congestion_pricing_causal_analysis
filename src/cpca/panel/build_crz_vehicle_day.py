"""Build daily CRZ vehicle-entry panel (descriptive first-stage companion).

Consumes
--------
- data/raw/crz_vehicle_entries/**/part.parquet
- data/raw/noaa_ghcn_central_park/daily.parquet
- config/treatment.yaml, config/settings.yaml

Produces
--------
- data/processed/panel_crz_vehicle_day.parquet
- updates data/processed/manifest.json

The public CRZ series begins at T0 (2025-01-05). This panel is for descriptive
reconciliation against MTA benchmarks; S6 CausalImpact runs on panel_bt_day.

Usage
-----
    python -m cpca.panel.build_crz_vehicle_day
    python -m cpca.panel.build_crz_vehicle_day --force
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

FEDERAL_HOLIDAYS = pd.to_datetime(
    [
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


def assign_period(dates: pd.Series, windows: dict) -> pd.Series:
    pre_end = pd.Timestamp(windows["pre_end"])
    washout_end = pd.Timestamp(windows["washout_end"])
    out = pd.Series("post", index=dates.index, dtype="object")
    out = out.mask(dates <= pre_end, "pre")
    out = out.mask((dates > pre_end) & (dates <= washout_end), "washout")
    return out


def load_crz_daily(raw_dir: Path, start: date, end: date) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("year=*/month=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"No CRZ vehicle partitions under {raw_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        df["toll_date"] = pd.to_datetime(df["toll_date"]).dt.normalize()
        mask = (df["toll_date"].dt.date >= start) & (df["toll_date"].dt.date <= end)
        if mask.any():
            frames.append(df.loc[mask].copy())
    if not frames:
        raise ValueError("No CRZ vehicle rows in the requested date window")

    raw = pd.concat(frames, ignore_index=True)
    daily = (
        raw.groupby("toll_date", as_index=False)
        .agg(
            crz_entries=("crz_entries", "sum"),
            excluded_roadway_entries=("excluded_roadway_entries", "sum"),
        )
        .rename(columns={"toll_date": "date"})
    )
    return daily


def build_panel(*, settings: dict, treatment: dict) -> tuple[pd.DataFrame, dict]:
    windows = treatment["sample_windows"]
    # Series begins at T0; still label with full window config.
    start = date.fromisoformat(treatment["treatment"]["t0"])
    end = date.fromisoformat(windows["post_end"])

    panel = load_crz_daily(
        ROOT / settings["crz_vehicle_entries"]["output_dir"], start, end
    )

    weather = pd.read_parquet(
        ROOT / settings["noaa_ghcn"]["output_path"], columns=["date", "prcp", "tavg"]
    )
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    weather = weather.rename(columns={"prcp": "precip_mm", "tavg": "tavg_c"})
    panel = panel.merge(weather, on="date", how="left")

    panel["log_crz_entries"] = np.log1p(panel["crz_entries"])
    holiday_set = set(pd.to_datetime(FEDERAL_HOLIDAYS).normalize())
    panel["holiday"] = panel["date"].isin(holiday_set)

    fare_date = pd.Timestamp(treatment["treatment"]["fare_change_date"])
    panel["fare_change"] = panel["date"] >= fare_date
    panel["period"] = assign_period(panel["date"], windows)
    panel["post"] = panel["period"].eq("post")
    t0 = pd.Timestamp(treatment["treatment"]["t0"])
    panel["days_since_t0"] = (panel["date"] - t0).dt.days.astype(int)

    panel = panel.sort_values("date").reset_index(drop=True)
    panel["period"] = panel["period"].astype(str)
    panel["holiday"] = panel["holiday"].astype(bool)
    panel["fare_change"] = panel["fare_change"].astype(bool)
    panel["post"] = panel["post"].astype(bool)

    expected = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    missing = expected.difference(panel["date"])

    manifest = {
        "panel": "panel_crz_vehicle_day",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(panel)),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "n_missing_calendar_days": int(len(missing)),
        "period_counts": panel["period"].value_counts().to_dict(),
        "mean_crz_entries": float(panel["crz_entries"].mean()),
        "note": "Series begins at T0; no pre-period. S6 primary uses panel_bt_day.",
        "sample_windows": windows,
        "t0": treatment["treatment"]["t0"],
        "fare_change_date": treatment["treatment"]["fare_change_date"],
        "inputs": {
            "crz_dir": settings["crz_vehicle_entries"]["output_dir"],
            "n_crz_partitions": len(
                list(
                    (ROOT / settings["crz_vehicle_entries"]["output_dir"]).glob(
                        "year=*/month=*/part.parquet"
                    )
                )
            ),
            "weather_path": settings["noaa_ghcn"]["output_path"],
            "weather_sha256": file_sha256(ROOT / settings["noaa_ghcn"]["output_path"]),
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
    out_path = ROOT / settings["panel_crz_vehicle_day"]["output_path"]
    manifest_path = ROOT / settings["panel_crz_vehicle_day"]["manifest_path"]

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
    print(
        f"rows={manifest['n_rows']:,} "
        f"({manifest['date_min']} to {manifest['date_max']}) "
        f"mean daily CRZ entries={round(manifest['mean_crz_entries']):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
