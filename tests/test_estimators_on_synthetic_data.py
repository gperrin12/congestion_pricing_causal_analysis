"""Synthetic-panel recovery tests for S1/S2 TWFE DiD estimators.

Build a station-week panel mirroring Panel A's shape (three zones, primary
sample flag, washout gap, did_pre_start window) with known injected effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpca.estimators.base import TreatmentConfig
from cpca.estimators.did import fit_event_study, fit_static_did
from cpca.inference.placebo_did import fit_preperiod_placebo_did

T0 = "2025-01-05"
DID_PRE_START = "2023-07-01"
WASHOUT_START = "2024-06-03"
WASHOUT_END = "2025-01-04"
POST_END = "2026-06-30"
PRE_END = "2024-06-02"


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
    )


def _week_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start=start, end=end, freq="W-MON")


def make_synthetic_panel(
    *,
    att: float = 0.0,
    differential_pretrend: float = 0.0,
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
    station_trend = {sid: float(rng.normal(0, 0.002)) for sid, _ in stations}
    week_fe = {w: float(rng.normal(0, 0.05)) for w in weeks}

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
            y += float(rng.normal(0, noise_sd))
            rows.append(
                {
                    "station_complex_id": sid,
                    "week_start": w,
                    "zone": zone,
                    "primary_sample": True,
                    "log_weekly_entries": y,
                    "weekly_entries": float(np.expm1(max(y, 0.0))),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def config() -> TreatmentConfig:
    return _config()


def test_clean_panel_recovers_att(config: TreatmentConfig) -> None:
    true_att = 0.12
    panel = make_synthetic_panel(att=true_att, differential_pretrend=0.0)

    s2 = fit_static_did(
        panel, config, outcome="log_weekly_entries", treated_zone="crz", station_trends=True
    )
    s1 = fit_event_study(
        panel, config, outcome="log_weekly_entries", treated_zone="crz", station_trends=True
    )

    assert abs(s2.att - true_att) < 0.03
    assert abs(s1.att - true_att) < 0.05
    assert s1.dynamic_effects is not None
    leads = s1.dynamic_effects[s1.dynamic_effects["is_pre"] & (s1.dynamic_effects["rel_month"] != s1.diagnostics["reference_rel_month"])]
    # Leads should be near zero (no differential pre-trend).
    assert float(leads["coef"].abs().mean()) < 0.04
    assert s1.diagnostics["reference_rel_month"] < 0
    assert s1.diagnostics["joint_wald_pvalue"] > 0.01 or float(leads["coef"].abs().max()) < 0.08


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
        panel, config, treated_zone="crz", station_trends=True
    )
    without = fit_static_did(
        panel, config, treated_zone="crz", station_trends=False
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
            panel, config, treated_zone="crz", station_trends=trends
        )
        s1 = fit_event_study(
            panel, config, treated_zone="crz", station_trends=trends
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
