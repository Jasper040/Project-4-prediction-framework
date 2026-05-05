"""Model factory + training/evaluation helpers.

Wraps the scikit-learn / LightGBM API into a single consistent surface so
notebooks 04 and 05 stay short.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from . import config
from .features import build_preprocessor


# ---------------------------------------------------------------------------
# Lightweight container for evaluation results
# ---------------------------------------------------------------------------
@dataclass
class ModelResult:
    name: str
    model: Any
    feature_names: list[str]
    y_test: np.ndarray
    y_proba: np.ndarray  # calibrated probabilities for the positive class
    y_pred: np.ndarray   # default 0.5-threshold predictions

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "ModelResult":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# Logistic regression baseline
# ---------------------------------------------------------------------------
def make_logreg_pipeline() -> Pipeline:
    """Logistic regression with full preprocessing — our baseline model."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", LogisticRegression(**config.LOGREG_DEFAULTS)),
        ]
    )


# ---------------------------------------------------------------------------
# LightGBM improved model
# ---------------------------------------------------------------------------
def make_lgbm_pipeline(params: dict | None = None) -> Pipeline:
    """LightGBM in a Pipeline with the same preprocessing surface.

    Trees don't need scaling, so we turn it off for speed.
    """
    # Imported lazily so the module loads even if lightgbm isn't installed yet.
    from lightgbm import LGBMClassifier

    params = {**config.LIGHTGBM_DEFAULTS, **(params or {})}
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            ("classifier", LGBMClassifier(**params)),
        ]
    )


# ---------------------------------------------------------------------------
# Calibration wrapper
# ---------------------------------------------------------------------------
def calibrate(pipeline: Pipeline, X, y, method: str = "isotonic") -> CalibratedClassifierCV:
    """Wrap a fitted pipeline in CalibratedClassifierCV.

    Important: pass an UNFITTED pipeline. Calibration uses cross-validation
    internally and re-fits the underlying estimator on each fold.
    """
    calibrated = CalibratedClassifierCV(pipeline, method=method, cv=5)
    calibrated.fit(X, y)
    return calibrated


# ---------------------------------------------------------------------------
# Class imbalance helper
# ---------------------------------------------------------------------------
def positive_class_weight(y: pd.Series | np.ndarray) -> float:
    """Compute scale_pos_weight = n_negative / n_positive for LightGBM."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0:
        raise ValueError("No positive examples — can't compute scale_pos_weight.")
    return n_neg / n_pos
