"""Download MTA subway hourly ridership from data.ny.gov (Socrata).

Pulls the cleaned hourly ridership product (turnstile/OMNY-derived), not raw
weekly turnstile audit files. Aggregates across payment_method and
fare_class_category to station_complex × hour.

Datasets
--------
- wujg-7c2s : 2020-2024
- 5wq4-mkjj : 2025-present

Writes monthly parquet partitions under data/raw/mta_subway_hourly/.
Resume-safe: skips months whose parquet already exists.

Usage
-----
    python -m cpca.ingest.download_mta_ridership
    python -m cpca.ingest.download_mta_ridership --start 2024-01-01 --end 2025-03-31
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
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
SELECT = (
    "transit_timestamp,station_complex_id,station_complex,borough,"
    "latitude,longitude,"
    "sum(ridership) as ridership,sum(transfers) as transfers"
)
GROUP = (
    "transit_timestamp,station_complex_id,station_complex,"
    "borough,latitude,longitude"
)


def load_settings() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


def dataset_for_day(day: date, settings: dict) -> dict:
    datasets = settings["socrata"]["datasets"]
    if day.year >= 2025:
        return datasets["mta_subway_hourly_2025_present"]
    return datasets["mta_subway_hourly_2020_2024"]


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


def days_in_month(month: date, start: date, end: date) -> list[date]:
    n = monthrange(month.year, month.month)[1]
    out: list[date] = []
    for d in range(1, n + 1):
        day = date(month.year, month.month, d)
        if start <= day <= end:
            out.append(day)
    return out


def fetch_day(
    session: requests.Session,
    domain: str,
    dataset_id: str,
    day: date,
    page_size: int,
    max_retries: int = 5,
) -> pd.DataFrame:
    """Fetch one calendar day of station-hour aggregates."""
    start_ts = f"{day.isoformat()}T00:00:00"
    end_ts = f"{day.isoformat()}T23:59:59"
    where = f"transit_timestamp between '{start_ts}' and '{end_ts}'"
    url = f"https://{domain}/resource/{dataset_id}.json"

    frames: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {
            "$select": SELECT,
            "$where": where,
            "$group": GROUP,
            "$order": "transit_timestamp,station_complex_id",
            "$limit": page_size,
            "$offset": offset,
        }
        for attempt in range(max_retries):
            try:
                resp = session.get(url, params=params, timeout=120)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                rows = resp.json()
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed {day} after retries: {exc}") from exc
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"Rate-limited on {day}")

        if not rows:
            break
        frames.append(pd.DataFrame(rows))
        if len(rows) < page_size:
            break
        offset += page_size

    if not frames:
        return pd.DataFrame(
            columns=[
                "transit_timestamp",
                "station_complex_id",
                "station_complex",
                "borough",
                "latitude",
                "longitude",
                "ridership",
                "transfers",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["transit_timestamp"] = pd.to_datetime(out["transit_timestamp"], utc=False)
    out["station_complex_id"] = out["station_complex_id"].astype(str)
    for col in ("ridership", "transfers", "latitude", "longitude"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def download_month(
    session: requests.Session,
    settings: dict,
    month: date,
    start: date,
    end: date,
    out_dir: Path,
) -> Path | None:
    days = days_in_month(month, start, end)
    if not days:
        return None

    out_path = out_dir / f"year={month.year}" / f"month={month.month:02d}" / "part.parquet"
    if out_path.exists():
        return out_path

    frames: list[pd.DataFrame] = []
    for day in tqdm(days, desc=f"{month.year}-{month.month:02d}", leave=False):
        ds = dataset_for_day(day, settings)
        day_df = fetch_day(
            session=session,
            domain=settings["socrata"]["domain"],
            dataset_id=ds["id"],
            day=day,
            page_size=settings["socrata"]["page_size"],
        )
        frames.append(coerce_types(day_df))

    month_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename for crash-safety.
    tmp = out_path.with_suffix(".parquet.tmp")
    month_df.to_parquet(tmp, index=False)
    tmp.rename(out_path)
    return out_path


def parse_args() -> argparse.Namespace:
    settings = load_settings()
    cfg = settings["mta_ridership"]
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

    # Cap end at yesterday so incomplete "today" isn't written as final.
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
        f"Downloading MTA subway hourly ridership {start} → {end} "
        f"({len(months)} months) → {out_dir}"
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
        n = len(pd.read_parquet(path)) if path else 0
        print(f"wrote {month.year}-{month.month:02d}: {n:,} rows → {path}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
