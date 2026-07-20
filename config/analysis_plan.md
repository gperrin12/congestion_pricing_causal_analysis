# Analysis Plan: Causal Effects of NYC Congestion Pricing

**Status:** Pre-registered. Committed before any post-treatment outcome data was analyzed.
**Commit rule:** This document may only be amended via the Deviations Log (Section 12). The original text is never edited after the first estimate is produced.

---

## 1. Background and setting

On January 5, 2025, the MTA began tolling vehicles entering the Congestion Relief Zone (CRZ), defined as Manhattan local streets south of 60th Street (the FDR Drive, West Side Highway, and Hugh L. Carey Tunnel connections to West Street are excluded; the MTA made minor boundary corrections on 2025-01-08). The peak toll for passenger vehicles is $9 (weekdays 5am-9pm, weekends 9am-9pm), with a 75% overnight discount, crossing credits for the four tunnels entering the zone, and scheduled increases to $12 in 2028 and $15 in 2031. Taxis and for-hire vehicles are tolled per trip rather than once daily: $0.75 per yellow/boro taxi trip and $1.50 per high-volume FHV trip.

Zone geometry note: the boundary runs along 60th Street with some sources describing the zone as "south of 61st." Station assignment uses the official CRZ polygon, not a latitude rule; no station complex sits ambiguously on the boundary.

Relevant timeline events that structure the identification strategy (all dates verified against contemporaneous reporting before estimation):

| Date | Event |
|---|---|
| 2023-06-26 | FHWA issues final federal approval; toll infrastructure build-out begins |
| 2023-12-06 | MTA board approves TMRB-recommended toll rates ($15 peak) |
| 2024-03-27 | MTA board gives final approval; start date set for 2024-06-30 |
| 2024-06-05 | Gov. Hochul pauses the program indefinitely (story leaked 2024-06-04) |
| 2024-11-14 | Hochul announces revival at reduced $9 toll, start 2025-01-05 |
| 2024-11-18 | MTA board approves revised plan; federal approval follows 2024-11-21 |
| 2025-01-05 | Tolling begins (treatment date, T0) |
| 2025-02-19 | USDOT revokes federal approval; tolling continues under litigation |
| 2026-01-04 | MTA base fare rises $2.90 to $3.00; B&T tolls rise 7.5%; OMNY-only transition begins |
| 2026-03-03 | Federal court rules the USDOT revocation unlawful; program secure |

## 2. Research questions

- **RQ1 (primary):** What is the effect of CRZ tolling on subway ridership (station entries), overall and by geography (inside CRZ, border band, rest of system)?
- **RQ2:** What is the effect on vehicle entries into the CRZ? (First-stage / manipulation check.)
- **RQ3:** What are the effects on secondary outcomes: bus speeds within the zone, taxi/FHV trips into the zone, Citi Bike ridership, MTA Bridges & Tunnels crossings (diversion), and air quality in and around the zone?
- **RQ4 (methodological):** How do estimates and inference differ across DiD, synthetic control, synthetic DiD, and Bayesian structural time series applied to the same panel, and why?

## 3. Hypotheses (directional, stated before estimation)

- **H1:** Vehicle entries into the CRZ decline post-T0 (expected magnitude: 5-15% based on London/Stockholm precedents).
- **H2:** Systemwide subway ridership increases modestly post-T0 relative to counterfactual (expected: 1-4%).
- **H3:** The ridership effect is not concentrated at CRZ stations. Because turnstile data records entries only, mode-shifted commuters board at origin stations outside the zone; CRZ entries capture mainly return legs and CRZ-origin trips. We expect diffuse positive effects, with border-band (60th-96th St) stations showing effects at least as large as CRZ stations.
- **H4:** Weekday bus speeds inside the CRZ increase post-T0.
- **H5:** Taxi/FHV trips into the CRZ are flat to positive post-T0. Taxis pay a per-trip toll roughly one-tenth the private-car rate ($0.75-$1.50 vs. $9) while benefiting fully from faster streets, so the net incentive plausibly favors taxi use. The taxi outcome is interpreted as a dose-response contrast: same streets, far lower toll exposure than private vehicles.
- **H6:** No large diversion increase at untolled crossings (George Washington Bridge spillover is the main candidate; treated as exploratory).

## 4. Units, outcomes, and panels

Two canonical panels, built once and consumed by all estimators:

**Panel A: `panel_station_week.parquet`** - NYC subway station complexes x ISO week.
- Primary outcome: `log(1 + weekly_entries)`, where entries = `SUM(TRY_CAST(ridership AS DOUBLE))` from the MTA hourly ridership dataset (2020-present grain: one row per station_complex x hour x payment_method x fare_class).
- Secondary outcomes on the same grain: weekday-peak entries, weekend entries.
- Covariates: weekly precipitation and mean temperature (NOAA GHCN, Central Park), holiday-week indicator, fare-change indicator (weeks on/after 2026-01-04).

**Panel B: `panel_system_month.parquet`** - US transit systems x month, from FTA National Transit Database (heavy rail UPT by agency).
- Primary outcome: `log(monthly_unlinked_trips)`.
- Donor pool (fixed in advance): WMATA, CTA, MBTA, SEPTA, BART, LA Metro, PATH excluded (too exposed to treatment), plus other heavy-rail agencies with continuous 2018-2026 reporting. Final donor list frozen at panel build and recorded in the manifest.

**Zone assignment (Panel A), fixed in `config/treatment.yaml`:**
- `crz`: station complex centroid inside the official CRZ polygon.
- `border`: centroid between 60th and 96th St in Manhattan.
- `control`: all other complexes at least 1 km from the CRZ boundary.
- Stations that opened, closed, or were renamed/re-complexed during the sample are dropped (list recorded in manifest).

## 5. Sample windows

- **Pre-period:** 2022-01-03 through 2024-06-02 (post-Omicron recovery onward; avoids the worst pandemic instability).
- **Washout:** 2024-06-03 through 2025-01-04, chosen to bracket the pause announcement (leak 2024-06-04, announcement 2024-06-05) and the anticipation window through the eve of tolling. Excluded from primary specifications; included in robustness spec R3.
- **Post-period:** 2025-01-05 through 2026-06-30 (hard stop; later data ignored even if available at pull time). Note the post-period splits naturally at 2026-01-04 (fare/toll changes): months 0-11 are free of concurrent MTA pricing events.

## 6. Primary specifications

All specifications run on both raw and log outcomes; log is primary.

- **S1. Event study (primary for RQ1).** Dynamic DiD on Panel A: station and week fixed effects, monthly relative-time coefficients, reference period = month before T0. Estimated with pyfixest, SEs clustered by station complex. The pre-period coefficients constitute the parallel-trends diagnostic: joint F-test on pre-treatment leads, threshold p > 0.10 for the design to be considered supported.
- **S2. Static TWFE DiD.** Same panel, single post x treated coefficient, separately for `crz` vs. control and `border` vs. control.
- **S3. Cross-city synthetic control (primary for RQ2/systemwide RQ1).** Panel B, treated unit = NYCT subway, Abadie-style weights fit on 2018-01 through 2024-05 pre-period. Inference by in-space placebos: permutation of treatment across donors, effect judged significant if NYCT's post/pre RMSPE ratio exceeds the 90th percentile of the donor distribution.
- **S4. Within-NYC synthetic control.** Synthetic CRZ aggregate built from control-zone stations. Interpreted as a lower bound given H3 spillovers; divergence from S3 is itself a reported result.
- **S5. Synthetic DiD** (Arkhangelsky et al. 2021) on Panel B and on Panel A zone aggregates.
- **S6. Bayesian structural time series (CausalImpact-style)** for single treated series: CRZ vehicle entries, CRZ bus speeds, B&T crossings. Local-linear-trend + weekly seasonality model, covariates = donor-city series and weather; 95% credible intervals.

## 7. First-stage / manipulation checks

Before interpreting any ridership estimate: replicate the reduction in CRZ vehicle entries from the public CRZ Vehicle Entries dataset using S6. If we cannot detect a first-stage decline in vehicle entries, downstream ridership estimates are reported but flagged as lacking a mechanism. If the public entries dataset lacks a sufficient pre-T0 series (detection infrastructure predates tolling only briefly), substitute longer-running proxies (B&T crossings, DOT traffic speeds) for the pre-period model and record the substitution in the Deviations Log.

## 8. Robustness checks (enumerated in advance)

- **R1. Border-band sensitivity:** redefine border as 60th-86th and 60th-110th.
- **R2. Control distance sensitivity:** minimum control distance 0.5 km and 2 km.
- **R3. Washout handling:** include the washout period with a separate anticipation dummy.
- **R4. Placebo-in-time:** fake T0 at 2023-01-09; a significant "effect" fails the design.
- **R5. Placebo outcomes:** 311 complaint volume in categories with no plausible congestion channel.
- **R6. Leave-one-out donors** for S3 (drop each donor, re-fit, report weight and estimate stability).
- **R7. Weekend vs. weekday split** (toll differential implies larger weekday effects).
- **R8. Fare-change window exclusion:** drop the 4 weeks around 2026-01-04.
- **R9. Winsorization:** cap station-weeks at the 99.5th percentile to check outlier sensitivity.

## 9. Known threats to validity (acknowledged in advance)

- **SUTVA / spillovers:** control stations are partially treated via origin-station boarding of mode-shifted drivers. Within-NYC designs (S1, S2, S4) are biased toward zero for the systemwide effect; the cross-city design (S3) is primary for the aggregate.
- **Confounding recovery trends:** return-to-office growth through 2025 and Manhattan's broader economic recovery inflate naive before/after comparisons; the donor pool in S3 and control stations in S1/S2 are the mitigations.
- **Anticipation:** handled by the washout window and R3.
- **Concurrent events (2026-01-04 bundle):** MTA base fare increase, 7.5% B&T toll increase, and the start of the OMNY-only transition all take effect the same day. The fare increase affects all stations symmetrically and is absorbed by time fixed effects in S1/S2 but not in S3/S6; handled there as an explicit covariate and via R8. The B&T toll increase directly contaminates the B&T crossings outcome from 2026-01-04 onward; that series is interpreted primarily on 2025 data.
- **Post-period policy uncertainty:** federal approval was revoked 2025-02-19 and the revocation ruled unlawful 2026-03-03. Behavioral responses during 2025 may be damped if travelers expected the toll to be struck down; this biases toward smaller estimated effects and is noted in interpretation, not modeled.
- **Measurement:** entries-only turnstile data (see H3); OMNY adoption shifting fare-class composition (mitigated by summing across payment methods; OMNY-only transition begins 2026-01-04); MTA data restatements (mitigated by frozen snapshots and manifest hashes).

## 10. External benchmarks (recorded before estimation; to reconcile against, not to target)

Published figures this analysis should be able to reconcile with, noting that most are raw comparisons rather than causal estimates:

- MTA-reported vehicle-entry decline: roughly 11-13% (about 67,000-87,000 fewer daily entries depending on period). Comparison point for H1/S6.
- Reported MTA ridership increases of 4.4-13% across transit modes (May 2025 analyses), and a raw 7.7% year-over-year subway ridership increase 2024 to 2025. The raw YoY figure includes recovery trend; our causal estimates (H2) are expected to be smaller than the raw change.
- Early reports of increased taxi trips and travel speeds within the zone. Comparison point for H5.

Divergence from these benchmarks is a finding to explain, not an error to correct toward.

## 11. Reporting rules

- Every estimator run serializes an `EstimateResult` to `results/estimates/`; the comparison chapter renders all of them with no manual selection.
- All point estimates are reported with uncertainty intervals regardless of sign or significance. No specification is dropped for producing an unwelcome result.
- The headline figure is the S1 event study; the headline table is the cross-estimator comparison (S1-S6) for the systemwide ridership effect.
- Effects are reported as percentage changes with counterfactual baselines, not raw counts alone.

## 12. Deviations log

Any change to the above after first estimation is recorded here with date, reason, and whether results had been observed at the time of the change. Pre-estimation verifications are also logged for transparency.

| Date | Section | Change | Results observed? | Reason |
|---|---|---|---|---|
| 2026-07-16 | 1, 4, 5, 8, 9 | Verified fare-increase date as 2026-01-04 (MTA board approved 2025-09-30); replaced TBV placeholder; dated the fare-change covariate and R8 window; noted same-day B&T toll increase and OMNY transition | No | Verification from mta.info before panel build |
| 2026-07-16 | 1 | Timeline verified and refined: rate approval 2023-12-06, final approval 2024-03-27, revival board vote 2024-11-18, federal approval 2024-11-21; added revocation (2025-02-19) and court ruling (2026-03-03) | No | Verification from contemporaneous reporting |
| 2026-07-16 | 3 | H5 revised from "decline or flat" to "flat to positive" after confirming taxis/FHVs pay per-trip tolls of $0.75/$1.50 rather than the $9 daily rate | No | Toll structure fact-check; no outcome data examined |
| 2026-07-20 | 4 | Added frozen `exclude_stations` in `treatment.yaml` (Rockaway seasonal A/S stations; Flushing Line / 7 renewals with documented long-run skips). Panel keeps rows but flags `excluded` / `primary_sample` for primary specs | Yes (EDA only; no causal estimates) | Pre→post station ranks showed Rockaway seasonality and 7-line construction declines contaminating control pool |
| 2026-07-20 | 5, 6 | Pretrends rejected for Panel A DiD on registered pre window (2022-01-03 to 2024-06-02): joint Wald on monthly leads CRZ vs control χ²(28)=668, border vs control χ²(28)=349, both p≈0 (S1 threshold p>0.10). CRZ differential log slope ≈0.002/week (p≈0); violation concentrated in 2022 post-Omicron catch-up. Linear differential slope insignificant from 2023-07 (CRZ p=0.40). Adjustment: (1) S1/S2 primary specs include station-specific linear trends; unadjusted TWFE robustness-only; (2) DiD estimation pre-window starts 2023-07-01 via `sample_windows.did_pre_start` in `treatment.yaml` (registered `pre_start` kept for SC/SDID); (3) elevate interpretive weight on S3/S5 for systemwide RQ1 | Yes (pre-period diagnostics only; no post-treatment causal estimates) | Formal pretrend notebook `notebooks/02_pretrends.ipynb`; outputs in `results/estimates/pretrends_*.csv` |
