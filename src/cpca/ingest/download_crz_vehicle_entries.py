"""Download CRZ vehicle entries from data.ny.gov (Socrata).

Dataset
-------
- t6yz-b64h : MTA Congestion Relief Zone Vehicle Entries: Beginning 2025

Source grain is crossing location × vehicle class × 10-minute block. This
module aggregates at pull time to:

    toll_date × time_period × vehicle_class × detection_group × detection_region

with summed ``crz_entries`` and ``excluded_roadway_entries``.

Coverage note (analysis_plan.md §7)
-----------------------------------
The public series begins on T0 (2025-01-05). There is no pre-treatment window
in this dataset; first-stage BSTS (S6) will need longer-running proxies
(B&T crossings, DOT speeds) for the pre-period model, recorded in the
Deviations Log when that substitution is made.

Writes monthly parquet partitions under data/raw/crz_vehicle_entries/.
Resume-safe: skips months whose parquet already exists.

Usage
-----
    python -m cpca.ingest.download_crz_vehicle_entries
    python -m cpca.ingest.download_crz_vehicle_entries --start 2025-01-05 --end 2025-03-31
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[3]

SELECT = (
    "toll_date,time_period,vehicle_class,detection_group,detection_region,"
    "sum(crz_entries) as crz_entries,"
    "sum(excluded_roadway_entries) as excluded_roadway_entries"
)
GROUP = (
    "toll_date,time_period,vehicle_class,detection_group,detection_region"
)


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def month_starts(start: date, end: date) -> list[date]:
    months: list[date] = []
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        months.append(cur)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return months


def month_bounds(month: date, start: date, end: date) -> tuple[date, date] | None:
    n = monthrange(month.year, month.month)[1]
    month_start = date(month.year, month.month, 1)
    month_end = date(month.year, month.month, n)
    lo = max(month_start, start)
    hi = min(month_end, end)
    if hi < lo:
        return None
    return lo, hi


def fetch_month(
    session: requests.Session,
    domain: str,
    dataset_id: str,
    lo: date,
    hi: date,
    page_size: int,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch day×crossing×class×period aggregates for [lo, hi]."""
    where = (
        f"toll_date between '{lo.isoformat()}T00:00:00' "
        f"and '{hi.isoformat()}T00:00:00'"
    )
    url = f"https://{domain}/resource/{dataset_id}.json"

    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {
            "$select": SELECT,
            "$where": where,
            "$group": GROUP,
            "$order": "toll_date,detection_group,vehicle_class,time_period",
            "$limit": page_size,
            "$offset": offset,
        }
        for attempt in range(max_retries):
            try:
                resp = session.get(url, params=params, timeout=180)
                if resp.status_code == 429:
                    time.sleep(2**attempt)
                    continue
                resp.raise_for_status()
                rows = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == max_retries - 1:
                    raise RuntimeError(
                        f"Failed {lo}..{hi} after retries: {exc}"
                    ) from exc
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Rate-limited on {lo}..{hi}")

        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < page_size:
            break
        offset += page_size

    if not frames:
        return pd.DataFrame(
            columns=[
                "toll_date",
                "time_period",
                "vehicle_class",
                "detection_group",
                "detection_region",
                "crz_entries",
                "excluded_roadway_entries",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["toll_date"] = pd.to_datetime(out["toll_date"]).dt.normalize()
    for col in ("time_period", "vehicle_class", "detection_group", "detection_region"):
        out[col] = out[col].astype("string")
    for col in ("crz_entries", "excluded_roadway_entries"):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype("int64")
    return out.sort_values(
        ["toll_date", "detection_group", "vehicle_class", "time_period"]
    ).reset_index(drop=True)


def download_month(
    session: requests.Session,
    settings: dict,
    month: date,
    start: date,
    end: date,
    out_dir: Path,
) -> Path | None:
    bounds = month_bounds(month, start, end)
    if bounds is None:
        return None
    lo, hi = bounds

    out_path = out_dir / f"year={month.year}" / f"month={month.month:02d}" / "part.parquet"
    if out_path.exists():
        return out_path

    ds = settings["socrata"]["datasets"]["crz_vehicle_entries"]
    raw = fetch_month(
        session=session,
        domain=settings["socrata"]["domain"],
        dataset_id=ds["id"],
        lo=lo,
        hi=hi,
        page_size=settings["socrata"]["page_size"],
    )
    month_df = coerce_types(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".parquet.tmp")
    month_df.to_parquet(tmp, index=False)
    tmp.rename(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    cfg = settings["crz_vehicle_entries"]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default=cfg["start_date"], help="YYYY-MM-DD")
    p.add_argument("--end", default=cfg["end_date"], help="YYYY-MM-DD")
    p.add_argument(
        "--out-dir",
        default=cfg["output_dir"],
        help="Output directory for monthly parquet partitions",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download months even if parquet exists",
    )
    return p.parse_args()


def main() -> int:
    load_dotenv(ROOT / ".env")
    token = os.getenv("SOCRATA_APP_TOKEN")
    if not token:
        print("SOCRATA_APP_TOKEN not set in environment/.env", file=sys.stderr)
        return 1

    args = parse_args()
    settings = load_settings()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        print("--end must be >= --start", file=sys.stderr)
        return 1

    # Dataset begins at T0; never pull earlier even if --start is overridden low.
    dataset_start = date(2025, 1, 5)
    if start < dataset_start:
        print(f"Clamping start_date to dataset start {dataset_start.isoformat()}")
        start = dataset_start

    # Respect sample hard stop from settings (analysis_plan.md §5).
    hard_stop = datetime.strptime(
        settings["crz_vehicle_entries"]["end_date"], "%Y-%m-%d"
    ).date()
    if end > hard_stop:
        print(f"Capping end_date to sample hard stop {hard_stop.isoformat()}")
        end = hard_stop

    today = date.today()
    if end >= today:
        end = today - timedelta(days=1)
        print(f"Capping end_date to {end.isoformat()} (yesterday)")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"X-App-Token": token, "Accept": "application/json"})

    months = month_starts(start, end)
    print(
        f"Downloading CRZ vehicle entries {start} to {end} "
        f"({len(months)} months) -> {out_dir}"
    )
    print(
        "Note: source begins at T0; no pre-treatment vehicle-entry series exists."
    )

    for month in months:
        out_path = (
            out_dir / f"year={month.year}" / f"month={month.month:02d}" / "part.parquet"
        )
        if out_path.exists() and not args.force:
            print(f"skip  {month.year}-{month.month:02d} (exists)")
            continue
        if args.force and out_path.exists():
            out_path.unlink()
        path = download_month(session, settings, month, start, end, out_dir)
        if path is None:
            print(f"skip  {month.year}-{month.month:02d} (out of range)")
            continue
        df = pd.read_parquet(path)
        total = int(df["crz_entries"].sum()) if len(df) else 0
        print(
            f"wrote {month.year}-{month.month:02d}: {len(df):,} rows, "
            f"{total:,} CRZ entries -> {path}"
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
