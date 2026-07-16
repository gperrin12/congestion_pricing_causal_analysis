### base.py is the contract file - it contains almost no logic, just the shared definitions that force every estimator to speak the same language

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

@dataclass
class EstimateResult:
    estimator: str            # "twfe_did", "synth_control_ntd", ...
    outcome: str              # "log_entries", "vehicle_entries", ...
    spec: dict                # full config that produced this
    att: float                # average treatment effect on treated
    ci_low: float
    ci_high: float
    inference: str            # "cluster_se", "permutation", "conformal"
    dynamic_effects: pd.DataFrame | None   # period-relative effects for event plots
    diagnostics: dict         # pretrend p-value, donor weights, RMSPE ratio...

class Estimator(Protocol):
    def fit(self, panel: pd.DataFrame, config: TreatmentConfig) -> EstimateResult: ...