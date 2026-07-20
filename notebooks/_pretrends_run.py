"""Pretrends diagnostics for Panel A (run from repo root or notebooks/)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
EST_DIR = ROOT / "results" / "estimates"
EST_DIR.mkdir(parents=True, exist_ok=True)

with open(ROOT / "config" / "treatment.yaml") as f:
    TREATMENT = yaml.safe_load(f)

windows = TREATMENT["sample_windows"]
T0 = pd.Timestamp(TREATMENT["treatment"]["t0"])
PAUSE = pd.Timestamp("2024-06-05")
PRE_START = pd.Timestamp(windows["pre_start"])
PRE_END = pd.Timestamp(windows["pre_end"])
WASHOUT = (
    pd.Timestamp(windows["washout_start"]),
    pd.Timestamp(windows["washout_end"]),
)

ZONE_COLORS = {
    "crz": "#c0392b",
    "border": "#e67e22",
    "control": "#2980b9",
}

ALPHA = 0.10  # plan threshold: joint F-test p > 0.10 supports parallel trends


def load_main() -> pd.DataFrame:
    panel = pd.read_parquet(ROOT / "data/processed/panel_station_week.parquet")
    main = panel[
        panel["primary_sample"] & panel["zone"].isin(["crz", "border", "control"])
    ].copy()
    main["week_start"] = pd.to_datetime(main["week_start"])
    return main


def station_demean(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col] - df.groupby("station_complex_id")[col].transform("mean")


def differential_trend(
    pre: pd.DataFrame, treated_zone: str, outcome: str = "log_weekly_entries"
):
    """Station-FE linear trend: y = a_i + b t + c (t x treated) + e on pre-period."""
    d = pre[pre["zone"].isin([treated_zone, "control"])].copy()
    d["treated"] = (d["zone"] == treated_zone).astype(float)
    d["t"] = (d["week_start"] - PRE_START).dt.days / 7.0
    d["y"] = d[outcome]
    d["y_dm"] = station_demean(d, "y")
    d["t_dm"] = station_demean(d, "t")
    d["txtr"] = d["t"] * d["treated"]
    d["txtr_dm"] = station_demean(d, "txtr")
    X = d[["t_dm", "txtr_dm"]]
    model = sm.OLS(d["y_dm"], X).fit(
        cov_type="cluster", cov_kwds={"groups": d["station_complex_id"]}
    )
    return model


def add_rel_month(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar month relative to month containing T0 (2025-01)."""
    out = df.copy()
    out["year_month"] = out["week_start"].dt.to_period("M")
    t0_month = pd.Timestamp(T0).to_period("M")
    out["rel_month"] = (out["year_month"] - t0_month).apply(lambda p: p.n)
    return out


def event_study_leads(
    df: pd.DataFrame,
    treated_zone: str,
    outcome: str = "log_weekly_entries",
    ref: int = -1,
    include_washout_post: bool = False,
):
    """
    TWFE event study: station + week FE, treated x rel_month dummies.
    Default sample = pre-period only; ref = last pre month relative to T0
    (month of PRE_END, typically -7 for June 2024 vs Jan 2025).
    Plan S1 uses ref = -1 (month before T0); that falls in washout, so the
    pre-only diagnostic re-references to the last pre month.
    """
    if include_washout_post:
        d = df[df["zone"].isin([treated_zone, "control"])].copy()
    else:
        d = df[
            (df["period"] == "pre") & df["zone"].isin([treated_zone, "control"])
        ].copy()

    d = add_rel_month(d)
    d["treated"] = (d["zone"] == treated_zone).astype(float)
    d["y"] = d[outcome]

    # Choose reference: last pre-period month if ref not in sample
    months = sorted(d["rel_month"].unique())
    if ref not in months:
        ref = max(m for m in months if m < 0)

    leads = [m for m in months if m != ref]
    # Build interactions
    for m in leads:
        d[f"D{m}"] = ((d["rel_month"] == m) & (d["treated"] == 1)).astype(float)

    # Two-way demean (station + week) via within transformation iteratively
    # Equivalent to Frisch-Waugh for FE when balanced-ish: demean by station then week
    cols = [f"D{m}" for m in leads]
    work = d[["station_complex_id", "week_start", "y", *cols]].copy()

    def tw_demean(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = frame.copy()
        for c in columns:
            out[c] = out[c] - out.groupby("station_complex_id")[c].transform("mean")
            out[c] = out[c] - out.groupby("week_start")[c].transform("mean")
        return out

    dem = tw_demean(work, ["y", *cols])
    X = dem[cols]
    y = dem["y"]
    # Drop all-zero columns (can happen if a month has no treated obs)
    keep = [c for c in cols if X[c].abs().sum() > 1e-12]
    X = X[keep]
    model = sm.OLS(y, X).fit(
        cov_type="cluster", cov_kwds={"groups": work["station_complex_id"]}
    )

    # Map back to rel_month
    coefs = []
    for c in keep:
        m = int(c[1:])
        coefs.append(
            {
                "rel_month": m,
                "coef": float(model.params[c]),
                "se": float(model.bse[c]),
                "pvalue": float(model.pvalues[c]),
            }
        )
    coefs.append({"rel_month": ref, "coef": 0.0, "se": 0.0, "pvalue": np.nan})
    coef_df = pd.DataFrame(coefs).sort_values("rel_month").reset_index(drop=True)
    coef_df["ci_low"] = coef_df["coef"] - 1.96 * coef_df["se"]
    coef_df["ci_high"] = coef_df["coef"] + 1.96 * coef_df["se"]

    # Joint Wald test on all pre-treatment leads (rel_month < 0, excluding ref)
    pre_leads = [c for c in keep if int(c[1:]) < 0]
    if len(pre_leads) >= 1:
        R = np.zeros((len(pre_leads), len(model.params)))
        name_to_i = {n: i for i, n in enumerate(model.params.index)}
        for i, c in enumerate(pre_leads):
            R[i, name_to_i[c]] = 1.0
        # Wald: (Rb)' (R V R')^{-1} (Rb) ~ chi2(q)
        b = model.params.values
        V = model.cov_params().values
        Rb = R @ b
        RVRT = R @ V @ R.T
        try:
            wald = float(Rb.T @ np.linalg.solve(RVRT, Rb))
            df_w = len(pre_leads)
            p_joint = float(1 - stats.chi2.cdf(wald, df_w))
        except np.linalg.LinAlgError:
            wald, df_w, p_joint = np.nan, len(pre_leads), np.nan
    else:
        wald, df_w, p_joint = np.nan, 0, np.nan

    return {
        "model": model,
        "coefs": coef_df,
        "ref": ref,
        "wald": wald,
        "df": df_w,
        "p_joint": p_joint,
        "n_obs": int(model.nobs),
        "treated_zone": treated_zone,
        "outcome": outcome,
    }


def plot_zone_means(pre: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    weekly = (
        pre.groupby(["week_start", "zone"], as_index=False)["log_weekly_entries"]
        .mean()
        .sort_values("week_start")
    )
    for z, g in weekly.groupby("zone"):
        axes[0].plot(
            g["week_start"],
            g["log_weekly_entries"],
            color=ZONE_COLORS[z],
            label=z,
            lw=1.5,
        )
    axes[0].set_title("Pre-period mean log(1+entries) by zone")
    axes[0].set_ylabel("log(1 + weekly entries)")
    axes[0].legend(loc="upper left", ncol=3, fontsize=9)

    # Index to early-2022 baseline (first 8 weeks)
    base_end = PRE_START + pd.Timedelta(weeks=8)
    base = (
        pre[pre["week_start"] < base_end]
        .groupby("zone")["log_weekly_entries"]
        .mean()
    )
    for z, g in weekly.groupby("zone"):
        axes[1].plot(
            g["week_start"],
            g["log_weekly_entries"] - base[z],
            color=ZONE_COLORS[z],
            label=z,
            lw=1.5,
        )
    axes[1].axhline(0, color="0.5", lw=0.8)
    axes[1].set_title("Indexed to early-2022 (first 8 weeks)")
    axes[1].set_ylabel("Δ log entries vs baseline")
    axes[1].legend(loc="upper left", ncol=3, fontsize=9)
    for ax in axes:
        ax.axvline(PAUSE, color="black", lw=1, ls=":", label="_pause")
    fig.suptitle("Parallel-trends eyeball (primary sample, pre only)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_event_study(result: dict, path: Path) -> None:
    cdf = result["coefs"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pre = cdf[cdf["rel_month"] < 0]
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axvline(result["ref"], color="0.7", lw=1, ls="--", label=f"ref={result['ref']}")
    ax.errorbar(
        pre["rel_month"],
        pre["coef"],
        yerr=1.96 * pre["se"],
        fmt="o",
        color=ZONE_COLORS.get(result["treated_zone"], "#333"),
        capsize=3,
        ms=5,
    )
    ax.set_xlabel("Month relative to T0 (2025-01)")
    ax.set_ylabel("Treated − control (log entries)")
    p = result["p_joint"]
    ax.set_title(
        f"Pre-period event-study leads: {result['treated_zone']} vs control\n"
        f"joint Wald χ²({result['df']})={result['wald']:.2f}, p={p:.4f} "
        f"({'PASS' if p > ALPHA else 'FAIL'} at α={ALPHA})"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    main_df = load_main()
    pre = main_df[main_df["period"] == "pre"].copy()
    print("Primary sample pre-period:")
    print(pre.groupby("zone")["station_complex_id"].nunique())
    print("weeks", pre["week_start"].nunique(), "rows", len(pre))

    plot_zone_means(pre, OUT_DIR / "pretrends_zone_means.png")

    print("\n=== Differential linear trends (pre only, station FE, cluster SE) ===")
    trend_rows = []
    for z in ["crz", "border"]:
        for outcome in ["log_weekly_entries", "weekly_entries"]:
            m = differential_trend(pre, z, outcome)
            row = {
                "contrast": f"{z}_vs_control",
                "outcome": outcome,
                "common_slope": float(m.params["t_dm"]),
                "diff_slope": float(m.params["txtr_dm"]),
                "diff_se": float(m.bse["txtr_dm"]),
                "diff_pvalue": float(m.pvalues["txtr_dm"]),
                "n_obs": int(m.nobs),
            }
            trend_rows.append(row)
            print(
                f"{z} | {outcome}: diff_slope={row['diff_slope']:.6f}, "
                f"p={row['diff_pvalue']:.4f}"
            )
    trend_df = pd.DataFrame(trend_rows)
    trend_df.to_csv(EST_DIR / "pretrends_differential_slopes.csv", index=False)

    print("\n=== Event-study joint tests on pre leads (TWFE, cluster SE) ===")
    # Last pre month relative to T0
    pre_rm = add_rel_month(pre)
    last_pre = int(pre_rm["rel_month"].max())
    print(f"Last pre-period rel_month (used as ref): {last_pre}")

    es_rows = []
    for z in ["crz", "border"]:
        for outcome in ["log_weekly_entries"]:
            res = event_study_leads(main_df, z, outcome, ref=last_pre)
            plot_event_study(
                res, OUT_DIR / f"pretrends_event_study_{z}_{outcome}.png"
            )
            res["coefs"].to_csv(
                EST_DIR / f"pretrends_event_coefs_{z}.csv", index=False
            )
            row = {
                "contrast": f"{z}_vs_control",
                "outcome": outcome,
                "ref_rel_month": res["ref"],
                "wald": res["wald"],
                "df": res["df"],
                "p_joint": res["p_joint"],
                "pass_alpha_0_10": bool(res["p_joint"] > ALPHA)
                if np.isfinite(res["p_joint"])
                else False,
                "n_obs": res["n_obs"],
            }
            es_rows.append(row)
            print(
                f"{z}: Wald={res['wald']:.3f}, df={res['df']}, "
                f"p={res['p_joint']:.4f}, PASS={row['pass_alpha_0_10']}"
            )
            # print largest |coef| pre leads
            c = res["coefs"][res["coefs"]["rel_month"] < 0].copy()
            c["abs"] = c["coef"].abs()
            print(c.nlargest(3, "abs")[["rel_month", "coef", "pvalue"]].to_string(index=False))

    es_df = pd.DataFrame(es_rows)
    es_df.to_csv(EST_DIR / "pretrends_joint_tests.csv", index=False)

    # Also: early vs late pre split (placebo DiD within pre)
    print("\n=== Placebo DiD within pre (split at midpoint) ===")
    mid = PRE_START + (PRE_END - PRE_START) / 2
    print(f"midpoint: {mid.date()}")
    placebo_rows = []
    for z in ["crz", "border"]:
        d = pre[pre["zone"].isin([z, "control"])].copy()
        d["treated"] = (d["zone"] == z).astype(float)
        d["fake_post"] = (d["week_start"] >= mid).astype(float)
        d["DID"] = d["treated"] * d["fake_post"]
        # TW demean
        work = d[["station_complex_id", "week_start", "log_weekly_entries", "DID"]].copy()
        work.columns = ["station_complex_id", "week_start", "y", "DID"]
        for c in ["y", "DID"]:
            work[c] = work[c] - work.groupby("station_complex_id")[c].transform("mean")
            work[c] = work[c] - work.groupby("week_start")[c].transform("mean")
        m = sm.OLS(work["y"], work[["DID"]]).fit(
            cov_type="cluster", cov_kwds={"groups": d["station_complex_id"]}
        )
        row = {
            "contrast": f"{z}_vs_control",
            "att": float(m.params["DID"]),
            "se": float(m.bse["DID"]),
            "pvalue": float(m.pvalues["DID"]),
        }
        placebo_rows.append(row)
        print(f"{z}: placebo ATT={row['att']:.5f}, p={row['pvalue']:.4f}")
    pd.DataFrame(placebo_rows).to_csv(EST_DIR / "pretrends_placebo_did.csv", index=False)

    print("\n=== VERDICT SUMMARY ===")
    print(es_df.to_string(index=False))
    print(trend_df[trend_df["outcome"] == "log_weekly_entries"].to_string(index=False))


if __name__ == "__main__":
    main()
