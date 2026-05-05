"""Threshold tuning and decision-rule construction.

Notebook 07 is the primary consumer. Notebook 08 sweeps the assumptions
this module exposes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .metrics import expected_value, ev_curve


@dataclass
class DecisionRule:
    """A concrete contact policy expressed in plain language."""

    threshold: float
    contact_budget_fraction: float
    cost_per_contact: float
    value_per_conversion: float
    expected_value_eur: float
    contacts: int
    successful: int
    wasted: int
    missed: int

    def describe(self) -> str:
        return (
            f"Contact every customer with predicted subscription probability "
            f">= {self.threshold:.2f}.\n"
            f"  Budget: top {self.contact_budget_fraction:.0%} of the customer base.\n"
            f"  Assumed unit costs: €{self.cost_per_contact:.2f} per call, "
            f"€{self.value_per_conversion:.2f} per conversion.\n"
            f"  Expected value on the test set: €{self.expected_value_eur:,.0f} "
            f"across {self.contacts:,} contacts "
            f"({self.successful:,} successful / {self.wasted:,} wasted).\n"
            f"  Missed positives (lost engagement): {self.missed:,}."
        )


def threshold_for_top_k(
    y_proba: np.ndarray,
    contact_budget_fraction: float = config.CONTACT_BUDGET_FRACTION,
) -> float:
    """Pick the probability threshold that contacts the top K% of customers."""
    return float(np.quantile(y_proba, 1 - contact_budget_fraction))


def threshold_max_ev(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_per_contact: float = config.COST_PER_CONTACT_EUR,
    value_per_conversion: float = config.VALUE_PER_CONVERSION_EUR,
) -> float:
    """Pick the unconstrained EV-maximising threshold."""
    curve = ev_curve(
        y_true,
        y_proba,
        cost_per_contact=cost_per_contact,
        value_per_conversion=value_per_conversion,
    )
    return float(curve.loc[curve["expected_value_eur"].idxmax(), "threshold"])


def build_decision_rule(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    contact_budget_fraction: float = config.CONTACT_BUDGET_FRACTION,
    cost_per_contact: float = config.COST_PER_CONTACT_EUR,
    value_per_conversion: float = config.VALUE_PER_CONVERSION_EUR,
) -> DecisionRule:
    """Default policy: contact the top K% by predicted probability.

    This is a budget-constrained rule (top K%), not the unconstrained
    EV-maximiser. Real ops teams have fixed campaign budgets, so the
    constrained version is the more defensible recommendation.
    """
    threshold = threshold_for_top_k(y_proba, contact_budget_fraction)
    ev = expected_value(
        y_true,
        y_proba,
        threshold,
        cost_per_contact=cost_per_contact,
        value_per_conversion=value_per_conversion,
    )
    return DecisionRule(
        threshold=threshold,
        contact_budget_fraction=contact_budget_fraction,
        cost_per_contact=cost_per_contact,
        value_per_conversion=value_per_conversion,
        expected_value_eur=ev["expected_value_eur"],
        contacts=ev["contacts"],
        successful=ev["successful"],
        wasted=ev["wasted"],
        missed=ev["missed"],
    )


# ---------------------------------------------------------------------------
# Sensitivity sweep — the engine behind notebook 08
# ---------------------------------------------------------------------------
def sensitivity_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_per_contact_grid: list[float],
    value_per_conversion_grid: list[float],
    contact_budget_fraction: float = config.CONTACT_BUDGET_FRACTION,
) -> pd.DataFrame:
    """Vary cost-per-contact and value-per-conversion. Report EV at each cell.

    Output is a long DataFrame ready to pivot or heatmap.
    """
    rows = []
    for c in cost_per_contact_grid:
        for v in value_per_conversion_grid:
            rule = build_decision_rule(
                y_true,
                y_proba,
                contact_budget_fraction=contact_budget_fraction,
                cost_per_contact=c,
                value_per_conversion=v,
            )
            rows.append(
                {
                    "cost_per_contact": c,
                    "value_per_conversion": v,
                    "threshold": rule.threshold,
                    "expected_value_eur": rule.expected_value_eur,
                    "contacts": rule.contacts,
                    "successful": rule.successful,
                    "wasted": rule.wasted,
                    "missed": rule.missed,
                }
            )
    return pd.DataFrame(rows)
