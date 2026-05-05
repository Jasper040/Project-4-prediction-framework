"""Feature engineering and encoding.

Everything that *transforms* features (one-hot, target encoding, scaling,
imputation) lives here. The functions return scikit-learn compatible
preprocessors so the same pipeline can be applied identically to train and
test.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Build a ColumnTransformer that handles numeric + categorical features.

    Numeric pipeline:
      - median imputation (handles pdays NaNs after sentinel removal)
      - optional standard scaling (on for logistic, off for tree models)

    Categorical pipeline:
      - 'unknown' is left as its own level (it's informative, not random missing)
      - one-hot encoded with handle_unknown='ignore' so test-set surprises
        don't break the pipeline.

    Set scale_numeric=False for tree-based models (LightGBM/XGBoost), which
    don't need scaling and run faster without it.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
            ),
        ]
    )

    numeric_cols = config.NUMERIC_COLS + ["was_previously_contacted"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, config.CATEGORICAL_COLS),
        ],
        remainder="drop",
    )
    return preprocessor


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a handful of engineered features.

    Kept lightweight on purpose — heavy feature engineering tends to leak
    domain assumptions. Add features here only if the EDA in notebook 03
    motivates them.
    """
    df = df.copy()

    # Age buckets — useful for the fairness analysis in notebook 09.
    df["age_band"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 120],
        labels=["under_30", "30_44", "45_59", "60_plus"],
    )

    # Macro indicator: is the economy contracting? (emp.var.rate < 0)
    df["economy_contracting"] = (df["emp.var.rate"] < 0).astype(int)

    # Campaign intensity: high-touch contact in this campaign.
    df["high_contact_campaign"] = (df["campaign"] >= 3).astype(int)

    return df
