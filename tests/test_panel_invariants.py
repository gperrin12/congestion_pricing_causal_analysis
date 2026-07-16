"""Panel A invariant tests.

These assert structural properties of `panel_station_week.parquet` that must
hold before any estimator runs. A failure means fix the panel build or the
underlying inputs - do not loosen these assertions (see AGENTS.md rule 5).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def settings() -> dict:
    with open(ROOT / "config" / "settings.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def treatment() -> dict:
    with open(ROOT / "config" / "treatment.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def panel_path(settings: dict) -> Path:
    path = ROOT / settings["panel_station_week"]["output_path"]
    if not path.exists():
        pytest.fail(
            f"Missing panel at {path}. Run: "
            "PYTHONPATH=src python -m cpca.panel.build_station_week --force"
        )
    return path


@pytest.fixture(scope="module")
def panel(panel_path: Path) -> pd.DataFrame:
    return pd.read_parquet(panel_path)


@pytest.fixture(scope="module")
def zones(settings: dict) -> pd.DataFrame:
    path = ROOT / settings["station_zones"]["output_path"]
    if not path.exists():
        pytest.fail(f"Missing station zones at {path}")
    z = pd.read_parquet(path)
    z["station_complex_id"] = z["station_complex_id"].astype(str)
    return z


@pytest.fixture(scope="module")
def manifest(settings: dict) -> dict:
    path = ROOT / settings["panel_station_week"]["manifest_path"]
    if not path.exists():
        pytest.fail(f"Missing manifest at {path}")
    return json.loads(path.read_text())


REQUIRED_COLUMNS = [
    "station_complex_id",
    "week_start",
    "weekly_entries",
    "weekday_peak_entries",
    "weekend_entries",
    "log_weekly_entries",
    "log_weekday_peak_entries",
    "log_weekend_entries",
    "station_complex",
    "borough",
    "latitude",
    "longitude",
    "zone",
    "in_crz",
    "dist_to_crz_boundary_km",
    "abs_dist_to_crz_boundary_km",
    "precip_mm",
    "tavg_c",
    "holiday_week",
    "fare_change",
    "period",
    "post",
    "weeks_since_t0",
    "iso_year",
    "iso_week",
]


def _iso_week_start(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).normalize()
    return ts - pd.Timedelta(days=int(ts.dayofweek))


class TestPanelSchema:
    def test_required_columns_present(self, panel: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in panel.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_station_id_is_string(self, panel: pd.DataFrame) -> None:
        assert pd.api.types.is_string_dtype(panel["station_complex_id"]) or panel[
            "station_complex_id"
        ].dtype == object

    def test_week_start_is_monday(self, panel: pd.DataFrame) -> None:
        week_start = pd.to_datetime(panel["week_start"])
        assert (week_start.dt.dayofweek == 0).all()

    def test_unique_panel_keys(self, panel: pd.DataFrame) -> None:
        dup = panel.duplicated(subset=["station_complex_id", "week_start"], keep=False)
        assert not dup.any(), (
            f"Duplicate station-week keys: {int(dup.sum())} rows\n"
            f"{panel.loc[dup, ['station_complex_id', 'week_start']].head()}"
        )


class TestPanelCompleteness:
    def test_no_nulls_in_required_columns(self, panel: pd.DataFrame) -> None:
        null_counts = panel[REQUIRED_COLUMNS].isna().sum()
        bad = null_counts[null_counts > 0]
        assert bad.empty, f"Nulls in required columns:\n{bad.to_string()}"

    def test_outcomes_nonnegative(self, panel: pd.DataFrame) -> None:
        for col in ("weekly_entries", "weekday_peak_entries", "weekend_entries"):
            assert (panel[col] >= 0).all(), f"{col} has negative values"

    def test_peak_and_weekend_bounded_by_weekly(self, panel: pd.DataFrame) -> None:
        # Peak and weekend are partitions of weekly hours, so each <= weekly.
        assert (panel["weekday_peak_entries"] <= panel["weekly_entries"] + 1e-6).all()
        assert (panel["weekend_entries"] <= panel["weekly_entries"] + 1e-6).all()

    def test_log1p_consistency(self, panel: pd.DataFrame) -> None:
        assert np.allclose(
            panel["log_weekly_entries"],
            np.log1p(panel["weekly_entries"]),
            rtol=0,
            atol=1e-9,
        )
        assert np.allclose(
            panel["log_weekday_peak_entries"],
            np.log1p(panel["weekday_peak_entries"]),
            rtol=0,
            atol=1e-9,
        )
        assert np.allclose(
            panel["log_weekend_entries"],
            np.log1p(panel["weekend_entries"]),
            rtol=0,
            atol=1e-9,
        )


class TestSampleWindows:
    def test_hard_stop_respected(self, panel: pd.DataFrame, treatment: dict) -> None:
        post_end = pd.Timestamp(treatment["sample_windows"]["post_end"])
        # week_start is Monday; last allowed Monday is the ISO week containing post_end.
        last_allowed = _iso_week_start(post_end)
        week_start = pd.to_datetime(panel["week_start"])
        assert week_start.max() <= last_allowed
        # No calendar day in the underlying sample may exceed post_end; week_start
        # itself must not start after post_end.
        assert week_start.max() <= post_end

    def test_sample_starts_on_or_after_pre_start(
        self, panel: pd.DataFrame, treatment: dict
    ) -> None:
        pre_start = pd.Timestamp(treatment["sample_windows"]["pre_start"])
        week_start = pd.to_datetime(panel["week_start"])
        assert week_start.min() >= _iso_week_start(pre_start)

    def test_period_labels_match_treatment_windows(
        self, panel: pd.DataFrame, treatment: dict
    ) -> None:
        windows = treatment["sample_windows"]
        pre_end = pd.Timestamp(windows["pre_end"])
        washout_end = pd.Timestamp(windows["washout_end"])
        week_start = pd.to_datetime(panel["week_start"])

        expected = pd.Series("post", index=panel.index, dtype="object")
        expected = expected.mask(week_start <= pre_end, "pre")
        expected = expected.mask(
            (week_start > pre_end) & (week_start <= washout_end),
            "washout",
        )
        assert (panel["period"] == expected).all()

    def test_post_flag_matches_period(self, panel: pd.DataFrame) -> None:
        assert (panel["post"] == panel["period"].eq("post")).all()

    def test_all_periods_present(self, panel: pd.DataFrame) -> None:
        assert set(panel["period"].unique()) == {"pre", "washout", "post"}


class TestTreatmentFlags:
    def test_fare_change_starts_on_configured_week(
        self, panel: pd.DataFrame, treatment: dict
    ) -> None:
        fare_date = pd.Timestamp(treatment["treatment"]["fare_change_date"])
        fare_week = _iso_week_start(fare_date)
        week_start = pd.to_datetime(panel["week_start"])
        expected = week_start >= fare_week
        assert (panel["fare_change"] == expected).all()
        assert panel.loc[panel["fare_change"], "week_start"].min() == fare_week

    def test_weeks_since_t0_zero_on_t0_week(
        self, panel: pd.DataFrame, treatment: dict
    ) -> None:
        t0_week = _iso_week_start(pd.Timestamp(treatment["treatment"]["t0"]))
        on_t0 = pd.to_datetime(panel["week_start"]) == t0_week
        assert on_t0.any(), "T0 week missing from panel"
        assert (panel.loc[on_t0, "weeks_since_t0"] == 0).all()


class TestZoneAssignment:
    def test_zone_labels_are_known(self, panel: pd.DataFrame) -> None:
        assert set(panel["zone"].unique()) <= {"crz", "border", "control", "near"}

    def test_in_crz_matches_zone(self, panel: pd.DataFrame) -> None:
        assert (panel["in_crz"] == panel["zone"].eq("crz")).all()

    def test_abs_distance_is_absolute(self, panel: pd.DataFrame) -> None:
        assert np.allclose(
            panel["abs_dist_to_crz_boundary_km"],
            panel["dist_to_crz_boundary_km"].abs(),
            rtol=0,
            atol=1e-9,
        )

    def test_control_min_distance(
        self, panel: pd.DataFrame, treatment: dict
    ) -> None:
        min_km = float(treatment["zone_bands"]["control_min_distance_km"])
        control = panel["zone"].eq("control")
        assert (panel.loc[control, "abs_dist_to_crz_boundary_km"] >= min_km - 1e-9).all()

    def test_near_within_buffer(self, panel: pd.DataFrame, treatment: dict) -> None:
        min_km = float(treatment["zone_bands"]["control_min_distance_km"])
        near = panel["zone"].eq("near")
        if near.any():
            assert (panel.loc[near, "abs_dist_to_crz_boundary_km"] < min_km + 1e-9).all()

    def test_crz_has_nonpositive_signed_distance(self, panel: pd.DataFrame) -> None:
        crz = panel["zone"].eq("crz")
        assert (panel.loc[crz, "dist_to_crz_boundary_km"] <= 1e-9).all()

    def test_zone_consistent_with_station_zones(
        self, panel: pd.DataFrame, zones: pd.DataFrame
    ) -> None:
        merged = panel[["station_complex_id", "zone", "in_crz"]].drop_duplicates(
            "station_complex_id"
        ).merge(
            zones[["station_complex_id", "zone", "in_crz"]],
            on="station_complex_id",
            suffixes=("_panel", "_zones"),
            how="left",
        )
        assert merged["zone_zones"].notna().all(), "Panel stations missing from zones"
        assert (merged["zone_panel"] == merged["zone_zones"]).all()
        assert (merged["in_crz_panel"] == merged["in_crz_zones"]).all()

    def test_zone_stable_within_station(self, panel: pd.DataFrame) -> None:
        n_zones = panel.groupby("station_complex_id")["zone"].nunique()
        assert (n_zones == 1).all(), "Zone assignment changed within a station over time"


class TestManifest:
    def test_manifest_matches_panel(
        self, panel: pd.DataFrame, manifest: dict
    ) -> None:
        entry = manifest["panel_station_week"]
        assert entry["n_rows"] == len(panel)
        assert entry["n_stations"] == panel["station_complex_id"].nunique()
        assert entry["n_weeks"] == panel["week_start"].nunique()
        assert entry["week_start_min"] == str(pd.to_datetime(panel["week_start"]).min().date())
        assert entry["week_start_max"] == str(pd.to_datetime(panel["week_start"]).max().date())
