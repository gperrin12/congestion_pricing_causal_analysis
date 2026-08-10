"""Shared estimator contract: EstimateResult + serialization helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


@dataclass
class TreatmentConfig:
    """Subset of treatment.yaml fields an estimator may need."""

    t0: str
    fare_change_date: str
    sample_windows: dict[str, str]
    bsts: dict[str, Any] = field(default_factory=dict)
    did: dict[str, Any] = field(default_factory=dict)
    holiday_adjustment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, treatment: dict) -> TreatmentConfig:
        return cls(
            t0=treatment["treatment"]["t0"],
            fare_change_date=treatment["treatment"]["fare_change_date"],
            sample_windows=dict(treatment["sample_windows"]),
            bsts=dict(treatment.get("bsts") or {}),
            did=dict(treatment.get("did") or {}),
            holiday_adjustment=dict(treatment.get("holiday_adjustment") or {}),
        )


@dataclass
class EstimateResult:
    estimator: str  # "twfe_did", "bsts", ...
    outcome: str  # "log_entries", "log_bt_manhattan_entries", ...
    spec: dict
    att: float
    ci_low: float
    ci_high: float
    inference: str  # "cluster_se", "permutation", "bayesian_ci", ...
    dynamic_effects: pd.DataFrame | None  # period-relative effects for event plots
    diagnostics: dict


class Estimator(Protocol):
    def fit(self, panel: pd.DataFrame, config: TreatmentConfig) -> EstimateResult: ...


def serialize_estimate(result: EstimateResult, out_dir: Path, stem: str) -> Path:
    """Write EstimateResult to results/estimates/{stem}.json (+ dynamic parquet)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "estimator": result.estimator,
        "outcome": result.outcome,
        "spec": result.spec,
        "att": result.att,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "inference": result.inference,
        "diagnostics": result.diagnostics,
        "dynamic_effects_path": None,
    }

    if result.dynamic_effects is not None and len(result.dynamic_effects):
        dyn_path = out_dir / f"{stem}_dynamic.parquet"
        tmp = dyn_path.with_suffix(".parquet.tmp")
        result.dynamic_effects.to_parquet(tmp, index=False)
        tmp.rename(dyn_path)
        payload["dynamic_effects_path"] = str(dyn_path.name)

    json_path = out_dir / f"{stem}.json"
    tmp_json = json_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(payload, indent=2, default=str))
    tmp_json.rename(json_path)
    return json_path


def load_estimate(path: Path) -> EstimateResult:
    payload = json.loads(path.read_text())
    dyn = None
    dyn_name = payload.get("dynamic_effects_path")
    if dyn_name:
        dyn_path = path.parent / dyn_name
        if dyn_path.exists():
            dyn = pd.read_parquet(dyn_path)
    return EstimateResult(
        estimator=payload["estimator"],
        outcome=payload["outcome"],
        spec=payload["spec"],
        att=float(payload["att"]),
        ci_low=float(payload["ci_low"]),
        ci_high=float(payload["ci_high"]),
        inference=payload["inference"],
        dynamic_effects=dyn,
        diagnostics=payload.get("diagnostics") or {},
    )
