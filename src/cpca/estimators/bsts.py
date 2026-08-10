"""S6 Bayesian structural time series (CausalImpact-style) first-stage check.

Primary outcome: into-Manhattan B&T daily crossings (`panel_bt_day.parquet`).
CRZ vehicle entries lack pre-T0 coverage; see analysis_plan.md §7 and Deviations Log.

Usage
-----
    PYTHONPATH=src python -m cpca.estimators.bsts
    PYTHONPATH=src python -m cpca.estimators.bsts --outcome bt_manhattan_entries
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from cpca.estimators.base import EstimateResult, TreatmentConfig, serialize_estimate
from cpca.panel.holidays import (
    att_exclusion_mask,
    holiday_cfg,
    winter_trough_mask,
)
from cpca.plots import style as plot_style

ROOT = Path(__file__).resolve().parents[3]


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_ci_frame(
    panel: pd.DataFrame,
    *,
    outcome: str,
    config: TreatmentConfig,
) -> pd.DataFrame:
    """Wide frame indexed by date: y + covariates for CausalImpact."""
    bsts = config.bsts
    hcfg = holiday_cfg(config)
    control_cols = [s["series"] for s in bsts["bt_controls"]]
    if outcome.startswith("log_"):
        covars = [f"log_{c}" for c in control_cols]
    else:
        covars = list(control_cols)
    covars = covars + ["precip_mm", "tavg_c"]

    holiday_covars = list(hcfg.get("bsts_covariates") or [])
    trough_start = str(hcfg.get("trough_start_md") or "12-24")
    trough_end = str(hcfg.get("trough_end_md") or "01-02")

    windows = config.sample_windows
    pre_start = pd.Timestamp(windows["pre_start"])
    pre_end = pd.Timestamp(windows["pre_end"])
    post_end = pd.Timestamp(bsts.get("post_end_primary") or windows["post_end"])

    work = panel.copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()
    # Keep a continuous calendar through the primary post end (includes washout
    # days between pre_end and post_start). ATT is averaged on the post window only.
    keep = (work["date"] >= pre_start) & (work["date"] <= post_end)
    work = work.loc[keep].copy()

    if "winter_trough" in holiday_covars:
        work["winter_trough"] = winter_trough_mask(
            work["date"], trough_start, trough_end
        ).astype(float)
    if "holiday" in holiday_covars:
        if "holiday" not in work.columns:
            raise KeyError("Panel missing holiday flag required by bsts_covariates")
        work["holiday"] = work["holiday"].astype(bool).astype(float)

    covars = covars + [c for c in holiday_covars if c not in covars]

    missing_cov = [c for c in covars if c not in work.columns]
    if missing_cov:
        raise KeyError(f"Missing covariate columns: {missing_cov}")
    if outcome not in work.columns:
        raise KeyError(f"Outcome {outcome} not in panel")

    for c in ("precip_mm", "tavg_c"):
        work[c] = work[c].fillna(work[c].median())

    cols = [outcome, *covars]
    frame = work.set_index("date")[cols].sort_index().dropna()
    return frame


def fit_bsts(
    panel: pd.DataFrame,
    config: TreatmentConfig,
    *,
    outcome: str | None = None,
    att_holiday_mode: str | None = None,
) -> EstimateResult:
    try:
        from causal_impact import CausalImpact
    except ImportError as exc:
        raise ImportError(
            "bsts-causalimpact is required. Install with: "
            "pip install bsts-causalimpact"
        ) from exc

    bsts = config.bsts
    hcfg = holiday_cfg(config)
    outcome = outcome or bsts.get("outcome") or "log_bt_manhattan_entries"
    frame = prepare_ci_frame(panel, outcome=outcome, config=config)

    primary_att = str(hcfg.get("bsts_primary_att") or "exclude_trough_days")
    if att_holiday_mode is None:
        att_mode = "exclude" if primary_att.startswith("exclude") else "include"
    else:
        att_mode = att_holiday_mode
    if att_mode not in {"exclude", "include"}:
        raise ValueError(
            f"att_holiday_mode must be 'exclude' or 'include', got {att_mode!r}"
        )

    trough_start = str(hcfg.get("trough_start_md") or "12-24")
    trough_end = str(hcfg.get("trough_end_md") or "01-02")
    use_fed = bool(hcfg.get("use_federal_holidays", True))

    windows = config.sample_windows
    t0 = pd.Timestamp(config.t0)
    post_end = pd.Timestamp(bsts.get("post_end_primary") or windows["post_end"])
    pre_start = pd.Timestamp(windows["pre_start"])
    pre_end = pd.Timestamp(windows["pre_end"])

    idx = frame.index
    pre_lo = max(pre_start, idx.min())
    pre_hi = min(pre_end, idx.max())
    # CausalImpact forecasts n_post steps immediately after pre_end. To keep a
    # continuous series, the library post window starts the day after pre_end
    # (includes washout). Reported ATT is averaged only from T0 onward.
    lib_post_lo = pre_hi + pd.Timedelta(days=1)
    lib_post_hi = min(post_end, idx.max())
    att_lo = max(t0, lib_post_lo)
    att_hi = lib_post_hi
    if pre_hi < pre_lo or lib_post_hi < lib_post_lo or att_hi < att_lo:
        raise ValueError("Empty pre or post window after aligning to panel dates")

    model_args = {
        "niter": int(bsts.get("niter", 2000)),
        "nwarmup": int(bsts.get("nwarmup", 1000)),
        "seed": int(bsts.get("seed", 20250105)),
        "state_model": bsts.get("state_model", "local_linear_trend"),
        "nseasons": int(bsts.get("nseasons", 7)),
        "standardize_data": True,
    }

    ci = CausalImpact(
        frame,
        [pre_lo.strftime("%Y-%m-%d"), pre_hi.strftime("%Y-%m-%d")],
        [lib_post_lo.strftime("%Y-%m-%d"), lib_post_hi.strftime("%Y-%m-%d")],
        model_args=model_args,
    )

    inf = ci.inferences.copy()
    if not isinstance(inf.index, pd.DatetimeIndex):
        inf.index = pd.to_datetime(inf.index)

    required = {
        "actual",
        "predicted_mean",
        "point_effect",
        "point_effect_lower",
        "point_effect_upper",
    }
    missing = required - set(inf.columns)
    if missing:
        raise RuntimeError(
            f"Unexpected inferences columns {list(inf.columns)}; missing {missing}"
        )

    # Reported ATT: T0 through primary post end (washout excluded from average).
    att_mask = (inf.index >= att_lo) & (inf.index <= att_hi)
    holiday_series = None
    if "holiday" in frame.columns:
        holiday_series = frame["holiday"].reindex(inf.index).fillna(0).astype(bool)
    elif "holiday" in panel.columns:
        hol = panel.copy()
        hol["date"] = pd.to_datetime(hol["date"]).dt.normalize()
        holiday_series = (
            hol.set_index("date")["holiday"].astype(bool).reindex(inf.index).fillna(False)
        )

    exclude_days = att_exclusion_mask(
        inf.index,
        holiday=holiday_series,
        start_md=trough_start,
        end_md=trough_end,
        use_federal_holidays=use_fed,
    )
    n_post_full = int(att_mask.sum())
    if att_mode == "exclude":
        att_mask = att_mask & ~exclude_days.to_numpy()
    n_post_att = int(att_mask.sum())
    n_excluded = n_post_full - n_post_att

    post_inf = inf.loc[att_mask]
    if post_inf.empty:
        raise RuntimeError("No inference rows in the T0+ ATT window after holiday filter")

    effects = post_inf["point_effect"].astype(float)
    att_abs = float(effects.mean())
    ci_low_abs = float(post_inf["point_effect_lower"].astype(float).mean())
    ci_high_abs = float(post_inf["point_effect_upper"].astype(float).mean())

    obs = post_inf["actual"].astype(float)
    pred = post_inf["predicted_mean"].astype(float)
    base = float(pred.mean())
    if base != 0:
        rel_att = float((obs.mean() - base) / abs(base))
        rel_lo = float(ci_low_abs / abs(base))
        rel_hi = float(ci_high_abs / abs(base))
    else:
        rel_att = float("nan")
        rel_lo = float("nan")
        rel_hi = float("nan")

    # Share of posterior draws with negative mean effect over the ATT window.
    # Approximate from pointwise CI coverage if draws are unavailable.
    p_value = None
    stats = getattr(ci, "summary_stats", None) or {}
    if isinstance(stats, dict) and "p_value" in stats:
        # Library p-value includes washout in its post window; keep as diagnostic only.
        p_value = float(stats["p_value"])

    inclusion_raw = getattr(ci, "posterior_inclusion_probs", None)
    if inclusion_raw is not None and hasattr(inclusion_raw, "tolist"):
        vals = inclusion_raw.tolist()
        names = list(frame.columns[1:])
        inclusion = {
            names[i]: float(vals[i]) for i in range(min(len(names), len(vals)))
        }
    elif inclusion_raw is not None and not isinstance(inclusion_raw, dict):
        inclusion = {"values": list(inclusion_raw)}
    else:
        inclusion = inclusion_raw

    dynamic = pd.DataFrame(
        {
            "date": inf.index,
            "observed": inf["actual"].astype(float).to_numpy(),
            "prediction": inf["predicted_mean"].astype(float).to_numpy(),
            "effect": inf["point_effect"].astype(float).to_numpy(),
            "effect_low": inf["point_effect_lower"].astype(float).to_numpy(),
            "effect_high": inf["point_effect_upper"].astype(float).to_numpy(),
        }
    )

    diagnostics = {
        "absolute_att": att_abs,
        "absolute_ci_low": ci_low_abs,
        "absolute_ci_high": ci_high_abs,
        "relative_att": rel_att,
        "relative_ci_low": rel_lo,
        "relative_ci_high": rel_hi,
        "p_value_library_post_includes_washout": p_value,
        "summary_stats_library": {k: float(v) for k, v in stats.items()}
        if isinstance(stats, dict)
        else {},
        "pre_start": str(pre_lo.date()),
        "pre_end": str(pre_hi.date()),
        "library_post_start": str(lib_post_lo.date()),
        "library_post_end": str(lib_post_hi.date()),
        "att_start": str(att_lo.date()),
        "att_end": str(att_hi.date()),
        "post_start": str(att_lo.date()),
        "post_end": str(att_hi.date()),
        "n_pre": int(((frame.index >= pre_lo) & (frame.index <= pre_hi)).sum()),
        "n_post": n_post_att,
        "n_post_att_days": n_post_att,
        "n_post_full_window": n_post_full,
        "n_post_excluded_holiday_trough": n_excluded,
        "att_holiday_mode": att_mode,
        "trough_start_md": trough_start,
        "trough_end_md": trough_end,
        "n_library_post": int(
            ((inf.index >= lib_post_lo) & (inf.index <= lib_post_hi)).sum()
        ),
        "covariates": list(frame.columns[1:]),
        "posterior_inclusion_probs": inclusion,
        "model_args": model_args,
        "proxy": "bt_manhattan_entries",
        "note": (
            "Primary first-stage uses into-Manhattan B&T crossings; "
            "CRZ vehicle entries lack pre-T0 coverage. CausalImpact library "
            "post window starts day after pre_end (includes washout for "
            "forecast continuity); reported ATT averages T0 onward, with "
            "holiday/winter-trough days excluded when att_holiday_mode=exclude."
        ),
    }
    try:
        diagnostics["summary_text"] = str(ci.summary())
    except Exception:
        pass

    return EstimateResult(
        estimator="bsts",
        outcome=outcome,
        spec={
            "pre_period": [str(pre_lo.date()), str(pre_hi.date())],
            "library_post_period": [
                str(lib_post_lo.date()),
                str(lib_post_hi.date()),
            ],
            "att_period": [str(att_lo.date()), str(att_hi.date())],
            "washout_excluded_from_att": True,
            "att_holiday_mode": att_mode,
            "trough_start_md": trough_start,
            "trough_end_md": trough_end,
            "model_args": model_args,
            "bt_treated": bsts.get("bt_treated"),
            "bt_controls": bsts.get("bt_controls"),
            "holiday_covariates": list(hcfg.get("bsts_covariates") or []),
        },
        att=float(rel_att),
        ci_low=float(rel_lo),
        ci_high=float(rel_hi),
        inference="bayesian_ci",
        dynamic_effects=dynamic,
        diagnostics=diagnostics,
    )


def plot_bsts(result: EstimateResult, out_path: Path) -> Path:
    plot_style.apply_theme()
    dyn = result.dynamic_effects
    if dyn is None or dyn.empty:
        raise ValueError("No dynamic effects to plot")

    d = dyn.copy()
    d["date"] = pd.to_datetime(d["date"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(
        d["date"],
        d["observed"],
        color=plot_style.COLORS["treated"],
        label="Observed",
        linewidth=1.2,
    )
    ax0.plot(
        d["date"],
        d["prediction"],
        color=plot_style.COLORS["counterfactual"],
        label="Counterfactual",
        linewidth=1.2,
    )
    plot_style.annotate_policy_dates(ax0)
    ax0.set_ylabel(result.outcome)
    ax0.set_title("S6 BSTS: into-Manhattan B&T crossings")
    ax0.legend(frameon=False, loc="upper left")

    ax1 = axes[1]
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.fill_between(
        d["date"],
        d["effect_low"],
        d["effect_high"],
        color=plot_style.COLORS["ci"],
        alpha=0.6,
        label="95% CI",
    )
    ax1.plot(
        d["date"],
        d["effect"],
        color=plot_style.COLORS["effect"],
        linewidth=1.0,
        label="Pointwise effect",
    )
    plot_style.annotate_policy_dates(ax1)
    ax1.set_ylabel("Effect")
    ax1.set_xlabel("Date")
    ax1.legend(frameon=False, loc="upper left")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    settings = load_yaml(ROOT / "config" / "settings.yaml")
    treatment = load_yaml(ROOT / "config" / "treatment.yaml")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--panel",
        default=settings["panel_bt_day"]["output_path"],
        help="Processed B&T daily panel path",
    )
    p.add_argument(
        "--outcome",
        default=treatment["bsts"].get("outcome", "log_bt_manhattan_entries"),
    )
    p.add_argument(
        "--att-holiday-mode",
        default=None,
        choices=["exclude", "include"],
        help="Primary exclude holiday/trough days from ATT average; include = full post window",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(settings["paths"]["results"]) / "estimates"),
    )
    p.add_argument(
        "--figure",
        default=str(
            Path(settings["paths"]["results"]) / "figures" / "bsts_bt_manhattan.png"
        ),
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
    result = fit_bsts(
        panel,
        config,
        outcome=args.outcome,
        att_holiday_mode=args.att_holiday_mode,
    )

    stem = f"bsts_{args.outcome}"
    att_mode = result.spec.get("att_holiday_mode") or "exclude"
    if att_mode == "include":
        stem += "_att_all_days"
    out_dir = ROOT / args.out_dir
    json_path = serialize_estimate(result, out_dir, stem)
    fig_path = plot_bsts(result, ROOT / args.figure)

    rel = result.diagnostics.get("relative_att")
    print(f"Wrote {json_path}")
    print(f"Wrote {fig_path}")
    print(
        f"ATT (reported)={result.att:.4f} "
        f"CI=[{result.ci_low:.4f}, {result.ci_high:.4f}] "
        f"relative_att={rel} att_holiday_mode={att_mode}"
    )
    print(
        f"post window {result.diagnostics['post_start']} to "
        f"{result.diagnostics['post_end']} "
        f"(n_post={result.diagnostics['n_post']}; "
        f"excluded_holiday_trough="
        f"{result.diagnostics.get('n_post_excluded_holiday_trough')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
