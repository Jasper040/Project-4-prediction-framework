"""Plotting helpers — keeps notebooks visually consistent.

We deliberately keep this thin: matplotlib is the workhorse, and we use a
single style applied at import time so charts across notebooks look like
they came from the same project.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from . import config

# House style — apply once on import.
sns.set_theme(context="notebook", style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "figure.figsize": (8, 5),
        "figure.dpi": 110,
        "axes.titleweight": "bold",
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    }
)


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save a figure to outputs/figures/<name>.png. Returns the path."""
    path = config.FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    return path


# ---------------------------------------------------------------------------
# Standard model-eval plots
# ---------------------------------------------------------------------------
def plot_roc(y_true, y_proba, label: str, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_proba, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
    ax.set_title("ROC curve")
    return ax


def plot_pr(y_true, y_proba, label: str, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    PrecisionRecallDisplay.from_predictions(y_true, y_proba, name=label, ax=ax)
    base_rate = np.mean(y_true)
    ax.axhline(base_rate, linestyle="--", color="grey", linewidth=1, label=f"Base rate ({base_rate:.2%})")
    ax.legend(loc="lower left")
    ax.set_title("Precision-Recall curve")
    return ax


def plot_confusion(y_true, y_pred, ax=None, title: str = "Confusion matrix"):
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def plot_calibration(calibration_table: pd.DataFrame, ax=None):
    """Reliability diagram — predicted vs. observed by bin."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, label="Perfect")
    ax.plot(
        calibration_table["mean_predicted"],
        calibration_table["observed_rate"],
        marker="o",
        label="Model",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed conversion rate")
    ax.set_title("Calibration (reliability diagram)")
    ax.legend()
    return ax


# ---------------------------------------------------------------------------
# Decision-framework plots
# ---------------------------------------------------------------------------
def plot_ev_curve(ev_df: pd.DataFrame, ax=None, mark_threshold: float | None = None):
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(ev_df["threshold"], ev_df["expected_value_eur"])
    if mark_threshold is not None:
        ax.axvline(mark_threshold, linestyle="--", color="red", linewidth=1,
                   label=f"Chosen threshold ({mark_threshold:.2f})")
        ax.legend()
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Expected value (€)")
    ax.set_title("Expected value vs. contact threshold")
    return ax


def plot_decile_lift(lift_df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots()
    sns.barplot(data=lift_df, x="decile", y="lift_vs_base", ax=ax, color="steelblue")
    ax.axhline(1.0, linestyle="--", color="grey", linewidth=1, label="Base rate")
    ax.set_xlabel("Decile (1 = highest predicted probability)")
    ax.set_ylabel("Lift vs. base rate")
    ax.set_title("Lift by decile")
    ax.legend()
    return ax
