"""Evaluation metrics: standard ones plus the business-aware ones the
decision framework actually cares about.

The "headline" metrics for the rubric are PR-AUC (handles class imbalance
better than ROC-AUC), Expected Value at the chosen threshold, and lift at
the top-K decile. These are *all* in here so the notebooks stay clean.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from . import config


# ---------------------------------------------------------------------------
# Classification metric summary
# ---------------------------------------------------------------------------
def classification_summary(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """One-shot summary: ROC-AUC, PR-AUC, F1 at threshold, Brier, confusion."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred),
        "brier": brier_score_loss(y_true, y_proba),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


# ---------------------------------------------------------------------------
# Expected value (the business-aware metric)
# ---------------------------------------------------------------------------
def expected_value(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    cost_per_contact: float = config.COST_PER_CONTACT_EUR,
    value_per_conversion: float = config.VALUE_PER_CONVERSION_EUR,
) -> dict:
    """Expected value of the contact policy at a given probability threshold.

    Policy: contact every customer with predicted probability >= threshold.
        - True positive  -> +value_per_conversion - cost_per_contact
        - False positive -> -cost_per_contact
        - False negative -> 0 (we didn't contact them, no cost; lost value
                                 is implicit — surfaced via the recall metric)
        - True negative  -> 0 (we didn't contact them)

    Returns total EV plus a per-contact breakdown.
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    contacts = tp + fp
    ev = tp * (value_per_conversion - cost_per_contact) - fp * cost_per_contact
    return {
        "threshold": float(threshold),
        "contacts": int(contacts),
        "successful": int(tp),
        "wasted": int(fp),
        "missed": int(fn),
        "expected_value_eur": float(ev),
        "ev_per_contact_eur": float(ev / contacts) if contacts else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) else 0.0,
    }


def ev_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Iterable[float] | None = None,
    cost_per_contact: float = config.COST_PER_CONTACT_EUR,
    value_per_conversion: float = config.VALUE_PER_CONVERSION_EUR,
) -> pd.DataFrame:
    """EV across a range of thresholds. Useful for the decision-framework chart."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = [
        expected_value(
            y_true,
            y_proba,
            t,
            cost_per_contact=cost_per_contact,
            value_per_conversion=value_per_conversion,
        )
        for t in thresholds
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Lift / gains by decile (for the "top-K targeting" story)
# ---------------------------------------------------------------------------
def decile_lift(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """Sort customers by predicted probability, bucket into deciles,
    and report the conversion rate per decile vs. the overall base rate.

    Decile 1 = highest predicted probability.
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["decile"] = pd.qcut(
        df["p"].rank(method="first", ascending=False),
        q=n_deciles,
        labels=range(1, n_deciles + 1),
    )
    base_rate = df["y"].mean()
    out = (
        df.groupby("decile", observed=True)
        .agg(
            n_customers=("y", "size"),
            successful=("y", "sum"),
            conversion_rate=("y", "mean"),
        )
        .reset_index()
    )
    out["lift_vs_base"] = out["conversion_rate"] / base_rate
    out["cum_successful"] = out["successful"].cumsum()
    out["cum_pct_of_successful"] = out["cum_successful"] / df["y"].sum()
    return out


# ---------------------------------------------------------------------------
# Calibration assessment
# ---------------------------------------------------------------------------
def calibration_table(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Bin predictions and compare predicted vs. observed conversion.

    A well-calibrated model has predicted ~= observed in each bin.
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba})
    df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    out = (
        df.groupby("bin", observed=True)
        .agg(
            n=("y", "size"),
            mean_predicted=("p", "mean"),
            observed_rate=("y", "mean"),
        )
        .reset_index()
    )
    out["calibration_gap"] = out["mean_predicted"] - out["observed_rate"]
    return out
