# AGENTS.md

Guidance for AI coding agents working in this repo.

## What this project is

A causal inference analysis of NYC congestion pricing (CRZ tolling, began 2025-01-05) on MTA subway ridership and related outcomes. Portfolio piece. The scientific credibility of the final report depends on process discipline, so several rules below are about protecting the analysis design, not code style.

## Rules that must never be violated

1. **Never edit `config/analysis_plan.md`** except by appending rows to the Deviations Log (Section 12). The plan is pre-registered; its text is frozen. If a task seems to require changing it, stop and ask the human.
2. **Never hardcode treatment definitions.** Treatment date, zone definitions, distance bands, and sample windows live in `config/treatment.yaml` and are read from there. If code needs a threshold that isn't in config, add it to config.
3. **All estimators consume `data/processed/panel_*.parquet` and nothing else.** No estimator reads raw or interim data, does its own joins, or filters the panel in ways not driven by `treatment.yaml`. If two estimators disagree, it must be attributable to the method, not the data.
4. **Every estimator returns an `EstimateResult`** (see `src/cpca/estimators/base.py`) and serializes it to `results/estimates/`. Do not print results ad hoc or invent new output formats.
5. **Do not silence failing panel invariant tests** (`tests/test_panel_invariants.py`) by loosening assertions. A failing invariant means a data problem; fix the data or escalate.
6. **Respect the sample hard stop.** Post-period data ends 2026-06-30 even if newer data is available at pull time.

## Repo layout

- `config/` - settings, treatment definitions, and the frozen analysis plan
- `data/raw|interim|processed/` - gitignored; layered pipeline; `processed/manifest.json` records provenance (download date, row counts, content hashes)
- `src/cpca/ingest/` - one module per public data source; idempotent; resume-safe
- `src/cpca/geo/` - CRZ polygon and station zone assignment; the only place geometry libraries (shapely/geopandas) may be imported
- `src/cpca/panel/` - builds the canonical panels; the single convergence point of all joins
- `src/cpca/estimators/` - one module per causal method, all conforming to `base.py`
- `src/cpca/inference/` - placebo and permutation machinery
- `src/cpca/plots/` - one shared matplotlib theme; all figures use it
- `notebooks/` - exploration only; nothing imports from notebooks; no report figure comes from a notebook
- `results/estimates/` and `results/figures/` - generated outputs, reproducible from `make estimates`
- `report/` - Quarto chapters; figures and tables execute against `results/`, never inline recomputation of estimates

## Workflow

- `make data` - run all ingest (skips fresh downloads via manifest)
- `make panel` - rebuild panels, then run invariant tests; both must pass
- `make estimates` - run the estimator x outcome x spec grid
- `make report` - render Quarto
- Run `pytest` before considering any change to `src/` done. Estimators are validated against synthetic panels with known injected effects (`tests/test_estimators_on_synthetic_data.py`); a new or modified estimator must recover the known ATT within tolerance.

## Conventions

- Python 3.12+, managed with uv; add dependencies to `pyproject.toml`, never ad hoc pip installs
- Data pulls hit public sources only (Socrata, FTA NTD, NOAA); no private warehouses, no credentials beyond `SOCRATA_APP_TOKEN` in `.env` (never commit it)
- Ingest writes parquet with explicit dtypes; write to a temp path then rename, so interrupted runs never leave partial files that look complete
- Panel keys: `station_complex_id` (string, as published by MTA) and ISO week start date; do not invent alternate keys
- Dates in configs and code are ISO strings (YYYY-MM-DD)
- Figures: policy dates (2024-06-05 pause, 2025-01-05 T0, 2026-01-04 fare bundle) are annotated on every time series plot; treated/control/synthetic use the same colors across all figures (defined in `plots/style.py`)
- In all prose (docstrings, report text, READMEs): use a plain dash "-", never an em dash
- Prefer clarity over cleverness; this repo will be read by reviewers evaluating the analysis, and the code is part of the exhibit

## Domain context an agent needs

- CRZ = Congestion Relief Zone, Manhattan south of 60th St. Zone assignment comes from the official polygon via `src/cpca/geo/`, never a latitude cutoff.
- Subway data records entries only (no tap-out). Effects are expected to be diffuse across the system, not concentrated at CRZ stations. Do not "fix" analyses that show small CRZ-station effects; that pattern is hypothesis H3.
- The washout window (2024-06-03 to 2025-01-04) exists because the program was paused and revived; it is excluded from primary specs by design, not by accident.
- Estimates from different methods are expected to disagree; the disagreement is a documented result (RQ4). Do not tune specs to make methods agree.