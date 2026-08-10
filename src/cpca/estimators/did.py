"""S1 event study and S2 static TWFE DiD (Panel A).

FE absorption: pyfixest ``feols`` with station and week fixed effects. Station-
specific linear trends (primary per analysis_plan.md §12) enter as
``i(station_complex_id, t_index)`` on the RHS. Clustered SEs use CRV1 on
``station_complex_id``. pyfixest is preferred over manual demeaning because
sequential station-then-week demeaning is not exact for two-way FE once
varying slopes are present, and pyfixest handles the demeaning and DoF
adjustment internally.

Usage
-----
    PYTHONPATH=src python -m cpca.estimators.did --treated-zone crz
    PYTHONPATH=src python -m cpca.estimators.did --treated-zone border --no-station-trends
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from cpca.estimators.base import EstimateResult, TreatmentConfig, serialize_estimate
from cpca.plots import style as plot_style

ROOT = Path(__file__).resolve().parents[3]

_REL_COEF_RE = re.compile(r"^rel_month::(-?\d+):treated$")


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _require_pyfixest():
    try:
        import pyfixest as pf
    except ImportError as exc:
        raise ImportError(
            "pyfixest is required for S1/S2. Install with: "
            "pip install pyfixest (or add to pyproject.toml / requirements.txt)."
        ) from exc
    return pf


def add_rel_month(df: pd.DataFrame, t0: str | pd.Timestamp) -> pd.DataFrame:
    """Calendar month relative to the month containing T0."""
    out = df.copy()
    weeks = pd.to_datetime(out["week_start"])
    out["year_month"] = weeks.dt.to_period("M")
    t0_month = pd.Timestamp(t0).to_period("M")
    out["rel_month"] = (out["year_month"] - t0_month).apply(lambda p: int(p.n))
    return out


def build_did_sample(
    panel: pd.DataFrame,
    config: TreatmentConfig,
    *,
    treated_zone: str,
    outcome: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Primary-sample DiD window: did_pre_start..post_end, washout dropped."""
    windows = config.sample_windows
    did_pre_start = pd.Timestamp(windows["did_pre_start"])
    washout_start = pd.Timestamp(windows["washout_start"])
    washout_end = pd.Timestamp(windows["washout_end"])
    post_end = pd.Timestamp(windows["post_end"])
    t0 = pd.Timestamp(config.t0)

    if treated_zone not in {"crz", "border"}:
        raise ValueError(f"treated_zone must be 'crz' or 'border', got {treated_zone!r}")
    if outcome not in panel.columns:
        raise KeyError(f"Outcome {outcome} not in panel")

    work = panel.copy()
    work["week_start"] = pd.to_datetime(work["week_start"]).dt.normalize()
    work["station_complex_id"] = work["station_complex_id"].astype(str)
    work["zone"] = work["zone"].astype(str)

    if "primary_sample" not in work.columns:
        raise KeyError("Panel missing primary_sample flag")

    keep_zones = {treated_zone, "control"}
    mask = (
        work["primary_sample"].astype(bool)
        & work["zone"].isin(keep_zones)
        & (work["week_start"] >= did_pre_start)
        & (work["week_start"] <= post_end)
        & ~(
            (work["week_start"] >= washout_start) & (work["week_start"] <= washout_end)
        )
    )
    sample = work.loc[mask].copy()
    if sample.empty:
        raise ValueError("Empty DiD sample after window and zone filters")

    sample = add_rel_month(sample, t0)
    sample["treated"] = (sample["zone"] == treated_zone).astype(float)
    sample["post"] = (sample["week_start"] >= t0).astype(float)
    sample["did"] = sample["treated"] * sample["post"]
    sample["y"] = sample[outcome].astype(float)
    # Calendar week index (not rank among remaining weeks): compressing the
    # washout gap would distort station-specific linear trends.
    origin = did_pre_start
    sample["t_index"] = (sample["week_start"] - origin).dt.days / 7.0
    sample["week_fe"] = sample["week_start"].dt.strftime("%Y-%m-%d")

    zone_counts = {
        z: int(sample.loc[sample["zone"] == z, "station_complex_id"].nunique())
        for z in sorted(sample["zone"].unique())
    }
    meta = {
        "did_pre_start": str(did_pre_start.date()),
        "washout_start": str(washout_start.date()),
        "washout_end": str(washout_end.date()),
        "post_end": str(post_end.date()),
        "window_start": str(sample["week_start"].min().date()),
        "window_end": str(sample["week_start"].max().date()),
        "n_weeks": int(sample["week_start"].nunique()),
        "n_obs": int(len(sample)),
        "station_counts_by_zone": zone_counts,
        "treated_zone": treated_zone,
        "outcome": outcome,
        "t0": str(t0.date()),
    }
    return sample, meta


def _reference_rel_month(sample: pd.DataFrame) -> int:
    """Last pre-period relative month present after window cuts."""
    pre_months = sample.loc[sample["rel_month"] < 0, "rel_month"].unique()
    if len(pre_months) == 0:
        raise ValueError("No pre-period relative months in DiD sample")
    return int(max(pre_months))


def _cluster_var(config: TreatmentConfig) -> str:
    return str(config.did.get("cluster") or "station_complex_id")


def _joint_alpha(config: TreatmentConfig) -> float:
    return float(config.did.get("joint_test_alpha", 0.10))


def _parse_rel_month_coef(name: str) -> int | None:
    m = _REL_COEF_RE.match(str(name))
    return int(m.group(1)) if m else None


def _fit_feols(
    sample: pd.DataFrame,
    *,
    fml: str,
    cluster: str,
):
    pf = _require_pyfixest()
    return pf.feols(fml, data=sample, vcov={"CRV1": cluster})


def _static_formula(*, station_trends: bool) -> str:
    if station_trends:
        return "y ~ did + i(station_complex_id, t_index) | station_complex_id + week_fe"
    return "y ~ did | station_complex_id + week_fe"


def _event_formula(*, ref: int, station_trends: bool) -> str:
    rhs = f"i(rel_month, treated, ref={ref})"
    if station_trends:
        rhs = f"{rhs} + i(station_complex_id, t_index)"
    return f"y ~ {rhs} | station_complex_id + week_fe"


def _wald_pre_leads(fit, coef_names: list[str]) -> tuple[float, int, float]:
    """Joint Wald on all estimated pre-period lead coefficients."""
    pre_names = [
        n for n in coef_names if (rm := _parse_rel_month_coef(n)) is not None and rm < 0
    ]
    if not pre_names:
        return float("nan"), 0, float("nan")
    name_to_i = {n: i for i, n in enumerate(coef_names)}
    R = np.zeros((len(pre_names), len(coef_names)))
    for i, n in enumerate(pre_names):
        R[i, name_to_i[n]] = 1.0
    series = fit.wald_test(R=R)
    wald = float(series["statistic"])
    p_joint = float(series["pvalue"])
    return wald, len(pre_names), p_joint


def _average_post_lags(
    fit,
    coef_names: list[str],
) -> tuple[float, float, float]:
    """Average post-T0 lag coefficients with covariance-aware CI."""
    post_names = [
        n for n in coef_names if (rm := _parse_rel_month_coef(n)) is not None and rm >= 0
    ]
    if not post_names:
        raise ValueError("No post-T0 lag coefficients to average for S1 ATT")
    name_to_i = {n: i for i, n in enumerate(coef_names)}
    idx = [name_to_i[n] for n in post_names]
    b = np.asarray(fit.coef().values, dtype=float)
    V = np.asarray(fit._vcov, dtype=float)
    beta = b[idx]
    Vsub = V[np.ix_(idx, idx)]
    k = len(idx)
    att = float(beta.mean())
    ones = np.ones(k)
    var = float(ones @ Vsub @ ones) / (k**2)
    se = float(np.sqrt(max(var, 0.0)))
    return att, att - 1.96 * se, att + 1.96 * se


def fit_static_did(
    panel: pd.DataFrame,
    config: TreatmentConfig,
    *,
    outcome: str | None = None,
    treated_zone: str = "crz",
    station_trends: bool | None = None,
) -> EstimateResult:
    """S2: single treated x post coefficient with station and week FE."""
    did_cfg = config.did
    outcome = outcome or did_cfg.get("outcome") or "log_weekly_entries"
    if station_trends is None:
        station_trends = bool(did_cfg.get("station_trends", True))
    cluster = _cluster_var(config)

    sample, meta = build_did_sample(
        panel, config, treated_zone=treated_zone, outcome=outcome
    )
    fml = _static_formula(station_trends=station_trends)
    fit = _fit_feols(sample, fml=fml, cluster=cluster)

    if "did" not in fit.coef().index:
        raise RuntimeError("Static DiD fit missing 'did' coefficient")
    tidy = fit.tidy()
    att = float(tidy.loc["did", "Estimate"])
    ci_low = float(tidy.loc["did", "2.5%"])
    ci_high = float(tidy.loc["did", "97.5%"])
    se = float(tidy.loc["did", "Std. Error"])
    pvalue = float(tidy.loc["did", "Pr(>|t|)"])

    diagnostics = {
        **meta,
        "se": se,
        "pvalue": pvalue,
        "station_trends": station_trends,
        "formula": fml,
        "n_coefficients": int(len(fit.coef())),
    }
    spec = {
        "estimator": "twfe_did",
        "station_trends": station_trends,
        "did_pre_start": meta["did_pre_start"],
        "treated_zone": treated_zone,
        "cluster": cluster,
        "outcome": outcome,
        "formula": fml,
        "washout_excluded": True,
    }
    return EstimateResult(
        estimator="twfe_did",
        outcome=outcome,
        spec=spec,
        att=att,
        ci_low=ci_low,
        ci_high=ci_high,
        inference="cluster_se",
        dynamic_effects=None,
        diagnostics=diagnostics,
    )


def fit_event_study(
    panel: pd.DataFrame,
    config: TreatmentConfig,
    *,
    outcome: str | None = None,
    treated_zone: str = "crz",
    station_trends: bool | None = None,
) -> EstimateResult:
    """S1: dynamic DiD with monthly treated x rel_month coefficients."""
    did_cfg = config.did
    outcome = outcome or did_cfg.get("outcome") or "log_weekly_entries"
    if station_trends is None:
        station_trends = bool(did_cfg.get("station_trends", True))
    cluster = _cluster_var(config)
    alpha = _joint_alpha(config)

    sample, meta = build_did_sample(
        panel, config, treated_zone=treated_zone, outcome=outcome
    )
    ref = _reference_rel_month(sample)
    fml = _event_formula(ref=ref, station_trends=station_trends)
    fit = _fit_feols(sample, fml=fml, cluster=cluster)

    coef_names = [str(n) for n in fit.coef().index]
    tidy = fit.tidy()
    rows: list[dict[str, Any]] = []
    n_by_rm = sample.groupby("rel_month").size().to_dict()

    for name in coef_names:
        rm = _parse_rel_month_coef(name)
        if rm is None:
            continue
        rows.append(
            {
                "rel_month": rm,
                "coef": float(tidy.loc[name, "Estimate"]),
                "ci_low": float(tidy.loc[name, "2.5%"]),
                "ci_high": float(tidy.loc[name, "97.5%"]),
                "is_pre": rm < 0,
                "n_obs": int(n_by_rm.get(rm, 0)),
            }
        )
    rows.append(
        {
            "rel_month": ref,
            "coef": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "is_pre": True,
            "n_obs": int(n_by_rm.get(ref, 0)),
        }
    )
    dynamic = (
        pd.DataFrame(rows).sort_values("rel_month").drop_duplicates("rel_month").reset_index(drop=True)
    )

    att, ci_low, ci_high = _average_post_lags(fit, coef_names)
    wald, df_w, p_joint = _wald_pre_leads(fit, coef_names)
    clears = bool(np.isfinite(p_joint) and p_joint > alpha)

    diagnostics = {
        **meta,
        "reference_rel_month": ref,
        "joint_wald": wald,
        "joint_wald_df": df_w,
        "joint_wald_pvalue": p_joint,
        "joint_test_alpha": alpha,
        "joint_test_clears_alpha": clears,
        "station_trends": station_trends,
        "formula": fml,
        "n_coefficients": int(len(fit.coef())),
        "n_post_lags_averaged": int(
            sum(1 for n in coef_names if (rm := _parse_rel_month_coef(n)) is not None and rm >= 0)
        ),
    }
    spec = {
        "estimator": "event_study_did",
        "station_trends": station_trends,
        "did_pre_start": meta["did_pre_start"],
        "treated_zone": treated_zone,
        "cluster": cluster,
        "outcome": outcome,
        "reference_rel_month": ref,
        "formula": fml,
        "washout_excluded": True,
    }
    return EstimateResult(
        estimator="event_study_did",
        outcome=outcome,
        spec=spec,
        att=att,
        ci_low=ci_low,
        ci_high=ci_high,
        inference="cluster_se",
        dynamic_effects=dynamic,
        diagnostics=diagnostics,
    )


def plot_event_study(result: EstimateResult, out_path: Path) -> Path:
    """Event-study coefficients with CI ribbon vs calendar month of rel_month."""
    plot_style.apply_theme()
    dyn = result.dynamic_effects
    if dyn is None or dyn.empty:
        raise ValueError("No dynamic effects to plot")

    d = dyn.copy().sort_values("rel_month")
    t0 = pd.Timestamp(result.diagnostics.get("t0") or "2025-01-05")
    t0_month = t0.to_period("M")
    d["month_start"] = d["rel_month"].map(
        lambda rm: (t0_month + int(rm)).to_timestamp()
    )
    ref = int(result.diagnostics.get("reference_rel_month", result.spec.get("reference_rel_month", -1)))
    ref_date = (t0_month + ref).to_timestamp()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.fill_between(
        d["month_start"],
        d["ci_low"],
        d["ci_high"],
        color=plot_style.COLORS["ci"],
        alpha=0.55,
        label="95% CI",
    )
    ax.plot(
        d["month_start"],
        d["coef"],
        color=plot_style.COLORS["effect"],
        linewidth=1.2,
        marker="o",
        markersize=4,
        label="Treated x rel_month",
    )
    ax.axvline(ref_date, color="0.55", linestyle=":", linewidth=1.0, label=f"ref={ref}")
    plot_style.annotate_policy_dates(ax)
    zone = result.spec.get("treated_zone", "")
    trends = result.spec.get("station_trends")
    ax.set_title(
        f"S1 event study: {zone} vs control"
        f" (station trends={'on' if trends else 'off'})"
    )
    ax.set_ylabel(result.outcome)
    ax.set_xlabel("Month (relative to T0)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _contrast_to_zone(contrast: str) -> str:
    if contrast.startswith("crz"):
        return "crz"
    if contrast.startswith("border"):
        return "border"
    raise ValueError(f"Unrecognized contrast {contrast!r}")


def parse_args() -> argparse.Namespace:
    settings = load_yaml(ROOT / "config" / "settings.yaml")
    treatment = load_yaml(ROOT / "config" / "treatment.yaml")
    did = treatment.get("did") or {}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel",
        default=settings["panel_station_week"]["output_path"],
        help="Processed station-week panel path",
    )
    p.add_argument(
        "--outcome",
        default=did.get("outcome", "log_weekly_entries"),
    )
    p.add_argument(
        "--treated-zone",
        default=None,
        choices=["crz", "border"],
        help="Treated zone (default: all contrasts in treatment.yaml)",
    )
    p.add_argument(
        "--no-station-trends",
        action="store_true",
        help="Robustness: omit station-specific linear trends",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(settings["paths"]["results"]) / "estimates"),
    )
    p.add_argument(
        "--figure",
        default=None,
        help="Event-study figure path (default under results/figures/)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    treatment = load_yaml(ROOT / "config" / "treatment.yaml")
    config = TreatmentConfig.from_yaml(treatment)
    station_trends = not args.no_station_trends

    panel_path = ROOT / args.panel
    if not panel_path.exists():
        print(f"Missing panel: {panel_path}. Run make panel first.", file=sys.stderr)
        return 1

    panel = pd.read_parquet(panel_path)
    if args.treated_zone:
        zones = [args.treated_zone]
    else:
        contrasts = config.did.get("contrasts") or ["crz_vs_control", "border_vs_control"]
        zones = [_contrast_to_zone(c) for c in contrasts]

    out_dir = ROOT / args.out_dir
    for zone in zones:
        es = fit_event_study(
            panel,
            config,
            outcome=args.outcome,
            treated_zone=zone,
            station_trends=station_trends,
        )
        tw = fit_static_did(
            panel,
            config,
            outcome=args.outcome,
            treated_zone=zone,
            station_trends=station_trends,
        )
        es_stem = f"event_study_did_{zone}_{args.outcome}"
        tw_stem = f"twfe_did_{zone}_{args.outcome}"
        if not station_trends:
            es_stem += "_no_trends"
            tw_stem += "_no_trends"
        es_path = serialize_estimate(es, out_dir, es_stem)
        tw_path = serialize_estimate(tw, out_dir, tw_stem)

        fig_default = (
            Path(load_yaml(ROOT / "config" / "settings.yaml")["paths"]["results"])
            / "figures"
            / f"{es_stem}.png"
        )
        fig_path = plot_event_study(es, ROOT / (args.figure or fig_default))

        print(f"Wrote {es_path}")
        print(f"Wrote {tw_path}")
        print(f"Wrote {fig_path}")
        print(
            f"S1 {zone}: ATT={es.att:.4f} CI=[{es.ci_low:.4f}, {es.ci_high:.4f}] "
            f"ref={es.diagnostics.get('reference_rel_month')} "
            f"joint_p={es.diagnostics.get('joint_wald_pvalue')}"
        )
        print(
            f"S2 {zone}: ATT={tw.att:.4f} CI=[{tw.ci_low:.4f}, {tw.ci_high:.4f}] "
            f"p={tw.diagnostics.get('pvalue')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
