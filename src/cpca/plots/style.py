"""Shared matplotlib theme for report figures."""

from __future__ import annotations

from datetime import date

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# Treated / control / synthetic colors (stable across figures).
COLORS = {
    "treated": "#C0392B",
    "control": "#2C3E50",
    "synthetic": "#2980B9",
    "counterfactual": "#2980B9",
    "effect": "#8E44AD",
    "ci": "#AED6F1",
}

POLICY_DATES = {
    "pause": date(2024, 6, 5),
    "t0": date(2025, 1, 5),
    "fare_bundle": date(2026, 1, 4),
}


def apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "figure.dpi": 120,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "font.size": 11,
        }
    )


def annotate_policy_dates(ax: Axes, *, ymin: float | None = None, ymax: float | None = None) -> None:
    """Vertical lines for pause, T0, and 2026-01-04 fare bundle."""
    labels = {
        "pause": "Pause",
        "t0": "T0",
        "fare_bundle": "Fare bundle",
    }
    colors = {
        "pause": "#7F8C8D",
        "t0": COLORS["treated"],
        "fare_bundle": "#D35400",
    }
    for key, d in POLICY_DATES.items():
        ax.axvline(d, color=colors[key], linestyle="--", linewidth=1.0, alpha=0.8)
        y = ymax if ymax is not None else ax.get_ylim()[1]
        ax.text(
            d,
            y,
            labels[key],
            rotation=90,
            va="top",
            ha="right",
            fontsize=8,
            color=colors[key],
        )
