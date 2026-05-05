"""Project-wide configuration: paths, seeds, business assumptions.

Anything that gets reused across notebooks lives here so we change it once.
Everything in this file is sensitivity-tested in notebook 08.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolves to the project root (novabank_retention/) regardless of where the
# notebook or script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
MODELS_DIR = OUTPUTS_DIR / "models"
TABLES_DIR = OUTPUTS_DIR / "tables"

RAW_CSV = RAW_DIR / "bank-additional-full.csv"

# Make sure output folders exist whenever this module is imported.
for _d in (INTERIM_DIR, PROCESSED_DIR, FIGURES_DIR, MODELS_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
TARGET_COL = "y"

# Categorical columns in the raw dataset.
CATEGORICAL_COLS = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

# Numeric columns. `duration` is intentionally NOT in the deployable feature set
# (see LEAKY_COLS) but is kept here for the leaky benchmark notebook.
NUMERIC_COLS = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

# Columns that leak the outcome (only known after the call ends).
# These are excluded from the deployable model. We keep them only for the
# leaky-benchmark contrast in notebook 04.
LEAKY_COLS = ["duration"]

# Sentinel values that mean "not previously contacted" — convert to a feature
# rather than treating as a real numeric value.
PDAYS_NOT_CONTACTED_SENTINEL = 999

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20  # 80/20 stratified split on the target

# ---------------------------------------------------------------------------
# Business assumptions for the decision framework (notebook 07).
#
# These are *assumptions* — not facts. They are sensitivity-tested in
# notebook 08. Document any change to them in ai_usage_log.md.
# ---------------------------------------------------------------------------

# Cost of contacting one customer (operations cost: agent time + telecom).
COST_PER_CONTACT_EUR = 5.0

# Expected gross margin from one successful subscription (retention value
# captured: deeper relationship, reduced future churn, deposit margin).
VALUE_PER_CONVERSION_EUR = 120.0

# Contact budget as a fraction of the customer base we can call per cycle.
# The decision rule will target the top-K by predicted probability where
# K = CONTACT_BUDGET_FRACTION * |population|.
CONTACT_BUDGET_FRACTION = 0.20

# ---------------------------------------------------------------------------
# Modelling defaults
# ---------------------------------------------------------------------------

LOGREG_DEFAULTS = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "solver": "liblinear",
    "random_state": RANDOM_SEED,
}

LIGHTGBM_DEFAULTS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "n_estimators": 500,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 50,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

# Number of Optuna trials for hyperparameter tuning (notebook 05).
N_TUNING_TRIALS = 50

# k-means k-search range (notebook 06).
KMEANS_K_RANGE = range(2, 11)
