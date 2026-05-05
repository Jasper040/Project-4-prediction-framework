"""Data loading, cleaning, and train/test splitting.

Everything that reads the raw CSV and turns it into a stable train/test pair
lives here. Notebook 02 calls these functions; downstream notebooks re-load
the saved interim files for speed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_raw() -> pd.DataFrame:
    """Read the UCI Bank Marketing CSV exactly as shipped (semicolon-delimited).

    Returns a DataFrame with the original column names.
    """
    df = pd.read_csv(config.RAW_CSV, sep=";")
    return df


def light_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the bare minimum cleaning that's safe to do BEFORE the split.

    Rules:
      - Strip whitespace from string cells.
      - Lower-case the target and convert to 0/1.
      - Replace pdays sentinel (999) with NaN AND add a was_contacted flag.
        This is honest preprocessing, not feature engineering — it just
        encodes the sentinel correctly.
    """
    df = df.copy()

    # Strip whitespace on every object column.
    obj_cols = df.select_dtypes(include="object").columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()

    # Encode target.
    df[config.TARGET_COL] = (df[config.TARGET_COL] == "yes").astype(int)

    # pdays = 999 means the customer was not previously contacted.
    df["was_previously_contacted"] = (
        df["pdays"] != config.PDAYS_NOT_CONTACTED_SENTINEL
    ).astype(int)
    df.loc[
        df["pdays"] == config.PDAYS_NOT_CONTACTED_SENTINEL, "pdays"
    ] = pd.NA

    return df


def split_train_test(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split on the target. Returns (train_df, test_df)."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[config.TARGET_COL],
        random_state=random_state,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_interim(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Persist the split so downstream notebooks don't re-split."""
    train_df.to_parquet(config.INTERIM_DIR / "train.parquet", index=False)
    test_df.to_parquet(config.INTERIM_DIR / "test.parquet", index=False)


def load_interim() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load the saved train/test split. Use this in notebooks 03+."""
    train_df = pd.read_parquet(config.INTERIM_DIR / "train.parquet")
    test_df = pd.read_parquet(config.INTERIM_DIR / "test.parquet")
    return train_df, test_df


def feature_columns(include_leaky: bool = False) -> list[str]:
    """Return the list of feature columns for modelling.

    By default, drops `duration` (and any other LEAKY_COLS) so the model is
    deployable. Set include_leaky=True only for the leaky-benchmark in
    notebook 04.
    """
    cols = (
        config.NUMERIC_COLS
        + config.CATEGORICAL_COLS
        + ["was_previously_contacted"]
    )
    if include_leaky:
        cols = config.LEAKY_COLS + cols
    return cols
