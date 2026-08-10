"""Winter trough and holiday-week helpers for DiD / BSTS adjustment.

Reuses panel ``holiday`` / ``holiday_week`` flags when present. Trough bounds
come from ``treatment.yaml`` → ``holiday_adjustment`` (MM-DD strings).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _parse_md(md: str) -> tuple[int, int]:
    month_s, day_s = md.split("-")
    return int(month_s), int(day_s)


def winter_trough_mask(
    dates: pd.Series | pd.DatetimeIndex,
    start_md: str = "12-24",
    end_md: str = "01-02",
) -> pd.Series:
    """True on calendar days in the inclusive winter trough (handles year wrap)."""
    idx = pd.to_datetime(dates)
    if isinstance(idx, pd.DatetimeIndex):
        series = pd.Series(idx, index=idx)
    else:
        series = pd.Series(idx, index=getattr(dates, "index", None))

    start_m, start_d = _parse_md(start_md)
    end_m, end_d = _parse_md(end_md)
    month = series.dt.month
    day = series.dt.day

    if (start_m, start_d) <= (end_m, end_d):
        # Non-wrapping window within a single calendar year.
        after_start = (month > start_m) | ((month == start_m) & (day >= start_d))
        before_end = (month < end_m) | ((month == end_m) & (day <= end_d))
        mask = after_start & before_end
    else:
        # Wrapping window, e.g. Dec 24 through Jan 2.
        in_start_year = (month > start_m) | ((month == start_m) & (day >= start_d))
        in_end_year = (month < end_m) | ((month == end_m) & (day <= end_d))
        mask = in_start_year | in_end_year

    return mask.astype(bool)


def flag_trough_week(
    week_start: pd.Series | pd.DatetimeIndex,
    *,
    start_md: str = "12-24",
    end_md: str = "01-02",
) -> pd.Series:
    """True if any day in the ISO week [week_start, week_start+6] is in the trough."""
    starts = pd.to_datetime(week_start)
    if isinstance(starts, pd.Series):
        index = starts.index
        start_vals = starts
    else:
        index = starts
        start_vals = pd.Series(starts, index=starts)

    # Expand to 7 calendar days per week and OR the trough mask.
    trough = pd.Series(False, index=index)
    for offset in range(7):
        day = start_vals + pd.Timedelta(days=int(offset))
        day_mask = winter_trough_mask(day, start_md, end_md)
        trough = trough | day_mask.to_numpy()
    return trough.astype(bool)


def holiday_cfg(config_like: Any) -> dict[str, Any]:
    """Extract holiday_adjustment dict from TreatmentConfig or raw mapping."""
    if hasattr(config_like, "holiday_adjustment"):
        return dict(getattr(config_like, "holiday_adjustment") or {})
    if isinstance(config_like, dict):
        return dict(config_like.get("holiday_adjustment") or {})
    return {}


def att_exclusion_mask(
    dates: pd.Series | pd.DatetimeIndex,
    *,
    holiday: pd.Series | None = None,
    start_md: str = "12-24",
    end_md: str = "01-02",
    use_federal_holidays: bool = True,
) -> pd.Series:
    """True on days that should be excluded from the primary BSTS ATT average."""
    idx = pd.to_datetime(dates)
    trough = winter_trough_mask(idx, start_md, end_md)
    if use_federal_holidays and holiday is not None:
        hol = pd.Series(holiday, index=trough.index).astype(bool)
        return (trough | hol).astype(bool)
    return trough.astype(bool)
