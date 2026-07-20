"""Build daily B&T panel for S6 first-stage (into-Manhattan crossings).

Consumes
--------
- data/raw/bt_crossings/**/part.parquet
- data/raw/noaa_ghcn_central_park/daily.parquet
- config/treatment.yaml (bsts.bt_treated / bt_controls), config/settings.yaml

Produces
--------
- data/processed/panel_bt_day.parquet
- updates data/processed/manifest.json

Treated outcome is the sum of configured into-Manhattan facility×direction
flows. Control facilities are total daily crossings (both directions) used as
BSTS covariates.

Usage
-----
    python -m cpca.panel.build_bt_day
    python -m cpca.panel.build_bt_day --force
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

# Shared with station-week panel (US federal holidays observed in-sample).
FEDERAL_HOLIDAYS = pd.to_datetime(
    [
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


def load_bt_daily(raw_dir: Path, start: date, end: date) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("year=*/month=*/part.parquet"))
    if not paths:
        raise FileNotFoundError(f"No B&T partitions under {raw_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        mask = (df["date"].dt.date >= start) & (df["date"].dt.date <= end)
        if mask.any():
            frames.append(df.loc[mask].copy())
    if not frames:
        raise ValueError("No B&T rows in the requested date window")

    out = pd.concat(frames, ignore_index=True)
    out["facility_id"] = out["facility_id"].astype(str)
    out["traffic_count"] = pd.to_numeric(out["traffic_count"], errors="coerce").fillna(0)
    return out


def build_panel(*, settings: dict, treatment: dict) -> tuple[pd.DataFrame, dict]:
    windows = treatment["sample_windows"]
    start = date.fromisoformat(windows["pre_start"])
    end = date.fromisoformat(windows["post_end"])
    bsts = treatment["bsts"]

    raw = load_bt_daily(ROOT / settings["bt_crossings"]["output_dir"], start, end)

    # Treated: specific facility × direction legs into Manhattan.
    treated_parts: list[pd.DataFrame] = []
    for spec in bsts["bt_treated"]:
        mask = (raw["facility_id"] == str(spec["facility_id"])) & (
            raw["direction"] == spec["direction"]
        )
        part = (
            raw.loc[mask]
            .groupby("date", as_index=False)["traffic_count"]
            .sum()
            .rename(columns={"traffic_count": spec["series"]})
        )
        treated_parts.append(part)

    treated = treated_parts[0]
    for part in treated_parts[1:]:
        treated = treated.merge(part, on="date", how="outer")

    treated_cols = [s["series"] for s in bsts["bt_treated"]]
    for col in treated_cols:
        treated[col] = treated[col].fillna(0.0)
    treated["bt_manhattan_entries"] = treated[treated_cols].sum(axis=1)

    # Controls: both directions summed per facility.
    control_parts: list[pd.DataFrame] = []
    for spec in bsts["bt_controls"]:
        mask = raw["facility_id"] == str(spec["facility_id"])
        part = (
            raw.loc[mask]
            .groupby("date", as_index=False)["traffic_count"]
            .sum()
            .rename(columns={"traffic_count": spec["series"]})
        )
        control_parts.append(part)

    panel = treated[["date", "bt_manhattan_entries", *treated_cols]].copy()
    for part in control_parts:
        panel = panel.merge(part, on="date", how="left")

    control_cols = [s["series"] for s in bsts["bt_controls"]]
    for col in control_cols:
        panel[col] = panel[col].fillna(0.0)

    # Weather
    weather = pd.read_parquet(
        ROOT / settings["noaa_ghcn"]["output_path"], columns=["date", "prcp", "tavg"]
    )
    weather["date"] = pd.to_datetime(weather["date"]).dt.normalize()
    weather = weather.rename(columns={"prcp": "precip_mm", "tavg": "tavg_c"})
    panel = panel.merge(weather, on="date", how="left")

    panel["log_bt_manhattan_entries"] = np.log1p(panel["bt_manhattan_entries"])
    for col in control_cols:
        panel[f"log_{col}"] = np.log1p(panel[col])

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

    # Full calendar coverage check within loaded range.
    expected = pd.date_range(panel["date"].min(), panel["date"].max(), freq="D")
    missing = expected.difference(panel["date"])

    manifest = {
        "panel": "panel_bt_day",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": int(len(panel)),
        "date_min": str(panel["date"].min().date()),
        "date_max": str(panel["date"].max().date()),
        "n_missing_calendar_days": int(len(missing)),
        "period_counts": panel["period"].value_counts().to_dict(),
        "bt_treated": bsts["bt_treated"],
        "bt_controls": bsts["bt_controls"],
        "mean_bt_manhattan_entries": float(panel["bt_manhattan_entries"].mean()),
        "sample_windows": windows,
        "t0": treatment["treatment"]["t0"],
        "fare_change_date": treatment["treatment"]["fare_change_date"],
        "bsts_post_end_primary": bsts["post_end_primary"],
        "inputs": {
            "bt_dir": settings["bt_crossings"]["output_dir"],
            "n_bt_partitions": len(
                list(
                    (ROOT / settings["bt_crossings"]["output_dir"]).glob(
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
    out_path = ROOT / settings["panel_bt_day"]["output_path"]
    manifest_path = ROOT / settings["panel_bt_day"]["manifest_path"]

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
        f"missing_days={manifest['n_missing_calendar_days']}"
    )
    print("period counts:", manifest["period_counts"])
    print("mean daily into-Manhattan B&T:", round(manifest["mean_bt_manhattan_entries"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
