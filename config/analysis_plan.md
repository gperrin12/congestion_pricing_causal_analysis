# Analysis Plan: Causal Effects of NYC Congestion Pricing

**Status:** Pre-registered. Committed before any post-treatment outcome data was analyzed.
**Commit rule:** This document may only be amended via the Deviations Log (Section 11). The original text is never edited after the first estimate is produced.

---

## 1. Background and setting

On January 5, 2025, the MTA began tolling vehicles entering the Congestion Relief Zone (CRZ), defined as Manhattan local streets south of 60th Street (the FDR Drive, West Side Highway, and Hugh L. Carey Tunnel connections to West Street are excluded). The peak toll for passenger vehicles is $9 (weekdays 5am-9pm, weekends 9am-9pm), with a 75% overnight discount, crossing credits for the four tunnels entering the zone, and scheduled increases to $12 in 2028 and $15 in 2031.

Relevant timeline events that structure the identification strategy:

| Date | Event |
|---|---|
| 2024-03 | MTA board approves final toll structure |
| 2024-06-05 | Gov. Hochul pauses the program indefinitely (original start: 2024-06-30) |
| 2024-11-14 | Program un-paused; start date announced for 2025-01-05, toll reduced $15 to $9 |
| 2025-01-05 | Tolling begins (treatment date, T0) |
| TBV | MTA base fare increase (verify exact effective date before panel build; model as covariate/control event) |

"TBV" = to be verified during data collection; the verified value will be recorded in the Deviations Log before estimation.

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
- **H5:** Taxi/FHV trips with a pickup or dropoff in the CRZ decline or are flat (fare pass-through vs. faster travel times cut in opposite directions; direction not confidently predicted).
- **H6:** No large diversion increase at untolled crossings (George Washington Bridge spillover is the main candidate; treated as exploratory).

## 4. Units, outcomes, and panels

Two canonical panels, built once and consumed by all estimators:

**Panel A: `panel_station_week.parquet`** - NYC subway station complexes x ISO week.
- Primary outcome: `log(1 + weekly_entries)`, where entries = `SUM(TRY_CAST(ridership AS DOUBLE))` from the MTA hourly ridership dataset (2020-present grain: one row per station_complex x hour x payment_method x fare_class).
- Secondary outcomes on the same grain: weekday-peak entries, weekend entries.
- Covariates: weekly precipitation and mean temperature (NOAA GHCN, Central Park), holiday-week indicator, fare-change indicator.

**Panel B: `panel_system_month.parquet`** - US transit systems x month, from FTA National Transit Database (heavy rail UPT by agency).
- Primary outcome: `log(monthly_unlinked_trips)`.
- Donor pool (fixed in advance): WMATA, CTA, MBTA, SEPTA, BART, LA Metro, PATH excluded (too exposed to treatment), plus other heavy-rail agencies with continuous 2018-2026 reporting. Final donor list frozen at panel build and recorded in the manifest.

**Zone assignment (Panel A), fixed in `config/treatment.yaml`:**
- `crz`: station complex centroid inside the CRZ polygon (south of 60th St).
- `border`: centroid between 60th and 96th St in Manhattan.
- `control`: all other complexes at least 1 km from the CRZ boundary.
- Stations that opened, closed, or were renamed/re-complexed during the sample are dropped (list recorded in manifest).

## 5. Sample windows

- **Pre-period:** 2022-01-03 through 2024-06-02 (post-Omicron recovery onward; avoids the worst pandemic instability).
- **Washout:** 2024-06-03 through 2025-01-04 (pause saga; anticipation and announcement effects). Excluded from primary specifications; included in robustness spec R3.
- **Post-period:** 2025-01-05 through 2026-06-30 (hard stop; later data ignored even if available at pull time).

## 6. Primary specifications

All specifications run on both raw and log outcomes; log is primary.

- **S1. Event study (primary for RQ1).** Dynamic DiD on Panel A: station and week fixed effects, monthly relative-time coefficients, reference period = month before T0. Estimated with pyfixest, SEs clustered by station complex. The pre-period coefficients constitute the parallel-trends diagnostic: joint F-test on pre-treatment leads, threshold p > 0.10 for the design to be considered supported.
- **S2. Static TWFE DiD.** Same panel, single post x treated coefficient, separately for `crz` vs. control and `border` vs. control.
- **S3. Cross-city synthetic control (primary for RQ2/systemwide RQ1).** Panel B, treated unit = NYCT subway, Abadie-style weights fit on 2018-01 through 2024-05 pre-period. Inference by in-space placebos: permutation of treatment across donors, effect judged significant if NYCT's post/pre RMSPE ratio exceeds the 90th percentile of the donor distribution.
- **S4. Within-NYC synthetic control.** Synthetic CRZ aggregate built from control-zone stations. Interpreted as a lower bound given H3 spillovers; divergence from S3 is itself a reported result.
- **S5. Synthetic DiD** (Arkhangelsky et al. 2021) on Panel B and on Panel A zone aggregates.
- **S6. Bayesian structural time series (CausalImpact-style)** for single treated series: CRZ vehicle entries, CRZ bus speeds, B&T crossings. Local-linear-trend + weekly seasonality model, covariates = donor-city series and weather; 95% credible intervals.

## 7. First-stage / manipulation checks

Before interpreting any ridership estimate: replicate the reduction in CRZ vehicle entries from the public CRZ Vehicle Entries dataset using S6. If we cannot detect a first-stage decline in vehicle entries, downstream ridership estimates are reported but flagged as lacking a mechanism.

## 8. Robustness checks (enumerated in advance)

- **R1. Border-band sensitivity:** redefine border as 60th-86th and 60th-110th.
- **R2. Control distance sensitivity:** minimum control distance 0.5 km and 2 km.
- **R3. Washout handling:** include the washout period with a separate anticipation dummy.
- **R4. Placebo-in-time:** fake T0 at 2023-01-09; a significant "effect" fails the design.
- **R5. Placebo outcomes:** 311 complaint volume in categories with no plausible congestion channel.
- **R6. Leave-one-out donors** for S3 (drop each donor, re-fit, report weight and estimate stability).
- **R7. Weekend vs. weekday split** (toll differential implies larger weekday effects).
- **R8. Fare-change window exclusion:** drop 4 weeks around the verified fare-increase date.
- **R9. Winsorization:** cap station-weeks at the 99.5th percentile to check outlier sensitivity.

## 9. Known threats to validity (acknowledged in advance)

- **SUTVA / spillovers:** control stations are partially treated via origin-station boarding of mode-shifted drivers. Within-NYC designs (S1, S2, S4) are biased toward zero for the systemwide effect; the cross-city design (S3) is primary for the aggregate.
- **Confounding recovery trends:** return-to-office growth through 2025 and Manhattan's broader economic recovery inflate naive before/after comparisons; the donor pool in S3 and control stations in S1/S2 are the mitigations.
- **Anticipation:** handled by the washout window and R3.
- **Concurrent events:** MTA fare increase (date TBV), service changes, weather. Fare increase affects all stations symmetrically and is absorbed by time fixed effects in S1/S2 but not in S3/S6; handled there as an explicit covariate.
- **Measurement:** entries-only turnstile data (see H3); OMNY adoption shifting fare-class composition (mitigated by summing across payment methods); MTA data restatements (mitigated by frozen snapshots and manifest hashes).

## 10. Reporting rules

- Every estimator run serializes an `EstimateResult` to `results/estimates/`; the comparison chapter renders all of them with no manual selection.
- All point estimates are reported with uncertainty intervals regardless of sign or significance. No specification is dropped for producing an unwelcome result.
- The headline figure is the S1 event study; the headline table is the cross-estimator comparison (S1-S6) for the systemwide ridership effect.
- Effects are reported as percentage changes with counterfactual baselines, not raw counts alone.

## 11. Deviations log

Any change to the above after first estimation is recorded here with date, reason, and whether results had been observed at the time of the change.

| Date | Section | Change | Results observed? | Reason |
|---|---|---|---|---|
| | | | | |