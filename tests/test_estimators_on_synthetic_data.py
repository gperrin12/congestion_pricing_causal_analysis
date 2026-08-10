"""Synthetic-panel recovery tests for S1/S2 TWFE DiD estimators.

Build a station-week panel mirroring Panel A's shape (three zones, primary
sample flag, washout gap, did_pre_start window) with known injected effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpca.estimators.base import TreatmentConfig
from cpca.estimators.bsts import prepare_ci_frame
from cpca.estimators.did import fit_event_study, fit_static_did
from cpca.inference.placebo_did import fit_preperiod_placebo_did
from cpca.panel.holidays import att_exclusion_mask, winter_trough_mask

T0 = "2025-01-05"
DID_PRE_START = "2023-07-01"
WASHOUT_START = "2024-06-03"
WASHOUT_END = "2025-01-04"
POST_END = "2026-06-30"
PRE_END = "2024-06-02"

_HOLIDAY_ADJ = {
    "trough_start_md": "12-24",
    "trough_end_md": "01-02",
    "use_federal_holidays": True,
    "did_primary": "exclude_trough_weeks",
    "did_robustness": "treated_holiday_interactions",
    "bsts_primary_att": "exclude_trough_days",
    "bsts_covariates": ["holiday", "winter_trough"],
}


def _config() -> TreatmentConfig:
    return TreatmentConfig(
        t0=T0,
        fare_change_date="2026-01-04",
        sample_windows={
            "pre_start": "2022-01-03",
            "pre_end": PRE_END,
            "did_pre_start": DID_PRE_START,
            "washout_start": WASHOUT_START,
            "washout_end": WASHOUT_END,
            "post_start": T0,
            "post_end": POST_END,
        },
        did={
            "outcome": "log_weekly_entries",
            "contrasts": ["crz_vs_control", "border_vs_control"],
            "station_trends": True,
            "cluster": "station_complex_id",
            "joint_test_alpha": 0.10,
        },
        holiday_adjustment=dict(_HOLIDAY_ADJ),
        bsts={
            "post_end_primary": "2025-12-31",
            "outcome": "log_bt_manhattan_entries",
            "bt_controls": [
                {"series": "bt_verrazzano"},
                {"series": "bt_throgs_neck"},
            ],
        },
    )


def _week_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="W-MON")


def make_synthetic_panel(
    *,
    att: float = 0.0,
    differential_pretrend: float = 0.0,
    treated_trough_dip: float = 0.0,
    unit_trends: bool = True,
    trough_start_md: str = "12-24",
    trough_end_md: str = "01-02",
    seed: int = 20250105,
    n_crz: int = 25,
    n_border: int = 20,
    n_control: int = 50,
    noise_sd: float = 0.05,
) -> pd.DataFrame:
    """Station-week panel with optional ATT and differential pre-trend.

    Outcome is generated in levels that behave like log entries: station FE,
    week FE, optional station-specific linear trends, optional treated x post
    ATT, and optional treated x time differential slope in the pre window.
    """
    rng = np.random.default_rng(seed)
    weeks = _week_range(DID_PRE_START, POST_END)
    # Include washout weeks in the panel (estimators drop them).
    t0 = pd.Timestamp(T0)
    did_pre = pd.Timestamp(DID_PRE_START)

    stations: list[tuple[str, str]] = []
    for i in range(n_crz):
        stations.append((f"c{i:03d}", "crz"))
    for i in range(n_border):
        stations.append((f"b{i:03d}", "border"))
    for i in range(n_control):
        stations.append((f"n{i:03d}", "control"))

    station_fe = {sid: float(rng.normal(0, 0.4)) for sid, _ in stations}
    if unit_trends:
        station_trend = {sid: float(rng.normal(0, 0.002)) for sid, _ in stations}
    else:
        station_trend = {sid: 0.0 for sid, _ in stations}
    week_fe = {w: float(rng.normal(0, 0.05)) for w in weeks}
    trough_weeks = {
        w: bool(
            any(
                bool(
                    winter_trough_mask(
                        pd.Series([w + pd.Timedelta(days=int(off))]),
                        start_md=trough_start_md,
                        end_md=trough_end_md,
                    ).iloc[0]
                )
                for off in range(7)
            )
        )
        for w in weeks
    }

    rows: list[dict] = []
    for sid, zone in stations:
        treated = zone == "crz"
        for w in weeks:
            t_idx = (w - did_pre).days / 7.0
            y = station_fe[sid] + week_fe[w] + station_trend[sid] * t_idx
            # Differential slope runs through the full sample (counterfactual
            # continues); ATT is an additive post level shift on top.
            if treated and differential_pretrend != 0.0:
                y += differential_pretrend * t_idx
            if treated and w >= t0:
                y += att
            trough_week = trough_weeks[w]
            # Treated-only trough dip every winter (pre and post), so
            # treated x trough interactions can absorb it.
            if treated and treated_trough_dip != 0.0 and trough_week:
                y += treated_trough_dip
            y += float(rng.normal(0, noise_sd))
            rows.append(
                {
                    "station_complex_id": sid,
                    "week_start": w,
                    "zone": zone,
                    "primary_sample": True,
                    # Keep holiday_week False so trough is the distinct channel.
                    "holiday_week": False,
                    "log_weekly_entries": y,
                    "weekly_entries": float(np.expm1(max(y, 0.0))),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def config() -> TreatmentConfig:
    return _config()


def test_winter_trough_mask_wraps_year() -> None:
    dates = pd.to_datetime(
        ["2024-12-23", "2024-12-24", "2025-01-02", "2025-01-03", "2025-07-01"]
    )
    mask = winter_trough_mask(dates, "12-24", "01-02")
    assert list(mask.astype(bool)) == [False, True, True, False, False]


def test_clean_panel_recovers_att(config: TreatmentConfig) -> None:
    true_att = 0.12
    panel = make_synthetic_panel(att=true_att, differential_pretrend=0.0)

    s2 = fit_static_did(
        panel,
        config,
        outcome="log_weekly_entries",
        treated_zone="crz",
        station_trends=True,
        holiday_mode="exclude",
    )
    s1 = fit_event_study(
        panel,
        config,
        outcome="log_weekly_entries",
        treated_zone="crz",
        station_trends=True,
        holiday_mode="exclude",
    )

    assert abs(s2.att - true_att) < 0.03
    assert abs(s1.att - true_att) < 0.06
    assert s1.dynamic_effects is not None
    leads = s1.dynamic_effects[
        s1.dynamic_effects["is_pre"]
        & (s1.dynamic_effects["rel_month"] != s1.diagnostics["reference_rel_month"])
    ]
    # Leads should be near zero (no differential pre-trend).
    assert float(leads["coef"].abs().mean()) < 0.04
    assert s1.diagnostics["reference_rel_month"] < 0
    assert s1.diagnostics["joint_wald_pvalue"] > 0.01 or float(
        leads["coef"].abs().max()
    ) < 0.08


def test_station_trends_recover_att_under_pretrend(config: TreatmentConfig) -> None:
    true_att = 0.10
    # Steep differential pre-trend on treated stations. Use a larger, quieter
    # panel so the level shift and unit trends are separately identified across
    # the washout gap (small-N noise otherwise confounds the split).
    panel = make_synthetic_panel(
        att=true_att,
        differential_pretrend=0.004,
        noise_sd=0.02,
        n_crz=40,
        n_control=80,
        n_border=15,
    )

    with_trends = fit_static_did(
        panel, config, treated_zone="crz", station_trends=True, holiday_mode="exclude"
    )
    without = fit_static_did(
        panel, config, treated_zone="crz", station_trends=False, holiday_mode="exclude"
    )

    assert abs(with_trends.att - true_att) < 0.04
    # Unadjusted TWFE should be pulled away from the true ATT by the pre-trend.
    assert abs(without.att - true_att) > 0.06
    assert abs(without.att - true_att) > abs(with_trends.att - true_att)


def test_zero_effect_ci_covers_zero(config: TreatmentConfig) -> None:
    panel = make_synthetic_panel(
        att=0.0,
        differential_pretrend=0.0,
        noise_sd=0.08,
        n_crz=30,
        n_control=60,
    )

    for trends in (True, False):
        s2 = fit_static_did(
            panel,
            config,
            treated_zone="crz",
            station_trends=trends,
            holiday_mode="exclude",
        )
        s1 = fit_event_study(
            panel,
            config,
            treated_zone="crz",
            station_trends=trends,
            holiday_mode="exclude",
        )
        assert s2.ci_low <= 0.0 <= s2.ci_high
        assert s1.ci_low <= 0.0 <= s1.ci_high
        assert abs(s2.att) < 0.05
        assert abs(s1.att) < 0.08


def test_placebo_near_zero_without_pretrend(config: TreatmentConfig) -> None:
    panel = make_synthetic_panel(att=0.15, differential_pretrend=0.0)
    placebo = fit_preperiod_placebo_did(
        panel, config, treated_zone="crz", station_trends=True
    )
    assert placebo.ci_low <= 0.0 <= placebo.ci_high or abs(placebo.att) < 0.05


def test_holiday_trough_exclude_recovers_att(config: TreatmentConfig) -> None:
    true_att = 0.10
    # Widen the trough so enough post weeks are contaminated for a clear bias.
    trough_start, trough_end = "12-15", "01-10"
    cfg = TreatmentConfig(
        t0=config.t0,
        fare_change_date=config.fare_change_date,
        sample_windows=dict(config.sample_windows),
        did=dict(config.did),
        bsts=dict(config.bsts),
        holiday_adjustment={
            **dict(config.holiday_adjustment),
            "trough_start_md": trough_start,
            "trough_end_md": trough_end,
        },
    )
    panel = make_synthetic_panel(
        att=true_att,
        treated_trough_dip=-0.80,
        unit_trends=False,
        trough_start_md=trough_start,
        trough_end_md=trough_end,
        noise_sd=0.03,
        n_crz=35,
        n_control=70,
    )

    excluded = fit_static_did(
        panel,
        cfg,
        treated_zone="crz",
        station_trends=False,
        holiday_mode="exclude",
    )
    unadjusted = fit_static_did(
        panel,
        cfg,
        treated_zone="crz",
        station_trends=False,
        holiday_mode="none",
    )
    interacted = fit_static_did(
        panel,
        cfg,
        treated_zone="crz",
        station_trends=False,
        holiday_mode="interact",
    )

    assert abs(excluded.att - true_att) < 0.04
    assert abs(unadjusted.att - true_att) > 0.02
    assert abs(unadjusted.att - true_att) > abs(excluded.att - true_att)
    assert abs(interacted.att - true_att) < 0.04
    assert excluded.diagnostics["n_weeks_dropped"] > 0
    assert abs(interacted.att - true_att) < abs(unadjusted.att - true_att)


def test_bsts_prepare_ci_frame_adds_trough_covariates(config: TreatmentConfig) -> None:
    dates = pd.date_range("2023-07-01", "2025-12-31", freq="D")
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(
        {
            "date": dates,
            "log_bt_manhattan_entries": rng.normal(10, 0.1, len(dates)),
            "log_bt_verrazzano": rng.normal(9, 0.1, len(dates)),
            "log_bt_throgs_neck": rng.normal(9, 0.1, len(dates)),
            "precip_mm": rng.uniform(0, 5, len(dates)),
            "tavg_c": rng.uniform(0, 25, len(dates)),
            "holiday": dates.isin(pd.to_datetime(["2024-12-25", "2025-01-01"])),
        }
    )
    frame = prepare_ci_frame(
        panel, outcome="log_bt_manhattan_entries", config=config
    )
    assert "holiday" in frame.columns
    assert "winter_trough" in frame.columns
    assert float(frame.loc["2024-12-25", "winter_trough"]) == 1.0
    assert float(frame.loc["2024-12-20", "winter_trough"]) == 0.0

    excl = att_exclusion_mask(
        frame.index,
        holiday=frame["holiday"].astype(bool),
        start_md="12-24",
        end_md="01-02",
        use_federal_holidays=True,
    )
    assert bool(excl.loc["2024-12-25"])
    assert bool(excl.loc["2025-01-01"])
    assert not bool(excl.loc["2025-03-01"])
