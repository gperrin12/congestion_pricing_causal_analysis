"""Pre-period placebo DiD for S2 (fake T0 at midpoint of DiD pre window).

Splits the DiD estimation pre-window at its midpoint, assigns a fake treatment
date there, and runs static TWFE DiD on pre-period weeks only. Serializes both
the primary (station-trends) and unadjusted (no trends) placebos so the
comparison is visible. A non-zero adjusted placebo is a finding, not something
to tune away.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from cpca.estimators.base import EstimateResult, TreatmentConfig, serialize_estimate
from cpca.estimators.did import fit_static_did

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _placebo_config(config: TreatmentConfig) -> tuple[TreatmentConfig, pd.Timestamp]:
    """Rewrite windows so S2's sample is pre-period only with a fake T0."""
    windows = dict(config.sample_windows)
    did_pre_start = pd.Timestamp(windows["did_pre_start"])
    pre_end = pd.Timestamp(windows["pre_end"])
    if pre_end < did_pre_start:
        raise ValueError("pre_end before did_pre_start; cannot build placebo window")
    midpoint = did_pre_start + (pre_end - did_pre_start) / 2
    midpoint = pd.Timestamp(midpoint).normalize()

    # Out-of-range washout so build_did_sample keeps the full pre span.
    placebo_windows = {
        **windows,
        "did_pre_start": did_pre_start.strftime("%Y-%m-%d"),
        "washout_start": "2099-01-01",
        "washout_end": "2099-01-01",
        "post_start": midpoint.strftime("%Y-%m-%d"),
        "post_end": pre_end.strftime("%Y-%m-%d"),
    }

    placebo = TreatmentConfig(
        t0=midpoint.strftime("%Y-%m-%d"),
        fare_change_date=config.fare_change_date,
        sample_windows=placebo_windows,
        bsts=dict(config.bsts),
        did=dict(config.did),
    )
    return placebo, midpoint


def fit_preperiod_placebo_did(
    panel: pd.DataFrame,
    config: TreatmentConfig,
    *,
    outcome: str | None = None,
    treated_zone: str = "crz",
    station_trends: bool = True,
) -> EstimateResult:
    """Run S2 on DiD pre-window only with fake T0 at the midpoint."""
    outcome = outcome or config.did.get("outcome") or "log_weekly_entries"
    placebo_cfg, midpoint = _placebo_config(config)
    result = fit_static_did(
        panel,
        placebo_cfg,
        outcome=outcome,
        treated_zone=treated_zone,
        station_trends=station_trends,
    )
    result.estimator = "twfe_did_placebo"
    result.spec = {
        **result.spec,
        "placebo": True,
        "fake_t0": midpoint.strftime("%Y-%m-%d"),
        "real_t0": config.t0,
        "placebo_pre_start": config.sample_windows["did_pre_start"],
        "placebo_pre_end": config.sample_windows["pre_end"],
        "station_trends": station_trends,
    }
    result.diagnostics = {
        **result.diagnostics,
        "placebo": True,
        "fake_t0": midpoint.strftime("%Y-%m-%d"),
        "real_t0": config.t0,
        "station_trends": station_trends,
    }
    return result


def parse_args() -> argparse.Namespace:
    settings = load_yaml(ROOT / "config" / "settings.yaml")
    treatment = load_yaml(ROOT / "config" / "treatment.yaml")
    did = treatment.get("did") or {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel",
        default=settings["panel_station_week"]["output_path"],
    )
    p.add_argument(
        "--outcome",
        default=did.get("outcome", "log_weekly_entries"),
    )
    p.add_argument(
        "--treated-zone",
        default="crz",
        choices=["crz", "border"],
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(settings["paths"]["results"]) / "estimates"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    treatment = load_yaml(ROOT / "config" / "treatment.yaml")
    config = TreatmentConfig.from_yaml(treatment)

    panel_path = ROOT / args.panel
    if not panel_path.exists():
        print(f"Missing panel: {panel_path}. Run make panel first.", file=sys.stderr)
        return 1

    panel = pd.read_parquet(panel_path)
    out_dir = ROOT / args.out_dir

    for trends in (True, False):
        result = fit_preperiod_placebo_did(
            panel,
            config,
            outcome=args.outcome,
            treated_zone=args.treated_zone,
            station_trends=trends,
        )
        tag = "trends" if trends else "no_trends"
        stem = f"placebo_twfe_did_{args.treated_zone}_{args.outcome}_{tag}"
        path = serialize_estimate(result, out_dir, stem)
        print(
            f"Wrote {path}  ATT={result.att:.4f} "
            f"CI=[{result.ci_low:.4f}, {result.ci_high:.4f}] "
            f"p={result.diagnostics.get('pvalue')} "
            f"fake_t0={result.spec.get('fake_t0')} trends={trends}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
