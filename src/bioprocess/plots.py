"""Figures for factor effects and cross-validated predictions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioprocess.data import TARGET_COLUMN
from bioprocess.modeling import MODELS

MODEL_COLORS = {
    "mean_baseline": "#6b6b6b",
    "linear_regression": "#3b6ea5",
    "quadratic_response_surface": "#d17a22",
    "random_forest": "#2f7d4a",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
        }
    )


def save_factor_figures(dataset: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Box plots of red pigment yield against temperature and fermentation time."""
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        (
            "temperature_c",
            "Temperature (°C)",
            "Red pigment yield by temperature",
            "temperature_vs_yield.png",
        ),
        (
            "fermentation_time_h",
            "Fermentation time (h)",
            "Red pigment yield by fermentation time",
            "fermentation_time_vs_yield.png",
        ),
    )
    written = []
    for column, xlabel, title, filename in specs:
        path = figures_dir / filename
        _factor_boxplot(dataset, column, xlabel, title, path)
        written.append(path)
    return written


def _factor_boxplot(
    dataset: pd.DataFrame,
    column: str,
    xlabel: str,
    title: str,
    path: Path,
) -> None:
    levels = sorted(dataset[column].unique())
    groups = [dataset.loc[dataset[column] == level, TARGET_COLUMN].to_numpy() for level in levels]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    axis.boxplot(
        groups,
        tick_labels=[_tick(level) for level in levels],
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#1f4e79", "linewidth": 1.6},
        boxprops={"color": "#1f4e79"},
        whiskerprops={"color": "#1f4e79"},
        capprops={"color": "#1f4e79"},
    )
    rng = np.random.default_rng(0)
    for index, values in enumerate(groups, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=28,
            color="#c45c26",
            zorder=3,
            edgecolors="white",
            linewidths=0.4,
        )
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Red pigment yield (g/L RYrp eq.)")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_prediction_figure(
    predictions: pd.DataFrame,
    figures_dir: Path,
    filename: str = "loo_predicted_vs_observed.png",
) -> Path:
    """Leave-one-out predicted yield against the observed yield, plus error spread."""
    _style()
    figures_dir.mkdir(parents=True, exist_ok=True)
    observed = predictions["observed_red_pigment_yield_g_l"].to_numpy()
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))

    low = min(observed.min(), min(predictions[slug].min() for slug, _, _ in MODELS))
    high = max(observed.max(), max(predictions[slug].max() for slug, _, _ in MODELS))
    pad = 0.08 * (high - low)
    limits = (low - pad, high + pad)
    axes[0].plot(limits, limits, color="#b0b0b0", linewidth=1, zorder=0)
    for slug, label, _ in MODELS:
        axes[0].scatter(
            observed,
            predictions[slug],
            s=26,
            alpha=0.85,
            label=label,
            color=MODEL_COLORS[slug],
            edgecolors="white",
            linewidths=0.3,
        )
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("Observed red pigment yield (g/L)")
    axes[0].set_ylabel("Leave-one-out prediction (g/L)")
    axes[0].set_title("Predicted against observed")

    errors = [np.abs(observed - predictions[slug].to_numpy()) for slug, _, _ in MODELS]
    axes[1].boxplot(
        errors,
        tick_labels=[label.replace(" ", "\n") for _, label, _ in MODELS],
        medianprops={"color": "#1f4e79"},
    )
    axes[1].set_ylabel("Absolute leave-one-out error (g/L)")
    axes[1].set_title("Error spread across the 29 runs")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )
    figure.tight_layout()
    path = figures_dir / filename
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return path


def _tick(level: float) -> str:
    if float(level).is_integer():
        return str(int(level))
    return str(level)
