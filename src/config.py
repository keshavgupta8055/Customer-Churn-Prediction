"""
config.py
---------
Central configuration: paths, constants, and model settings.
All other modules import from here — change one value and it updates everywhere.
"""

from pathlib import Path

# ── Project root (two levels up from this file: src/ → project root) ──────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ─────────────────────────────────────────────────────────────────
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_DATA_FILE      = DATA_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_FILE = DATA_PROCESSED_DIR / "telco_processed.csv"

# ── Output paths ───────────────────────────────────────────────────────────────
OUTPUTS_DIR    = ROOT_DIR / "outputs"
FIGURES_DIR    = OUTPUTS_DIR / "figures"
MODELS_DIR     = OUTPUTS_DIR / "models"
REPORTS_DIR    = OUTPUTS_DIR / "reports"
METRICS_DIR    = OUTPUTS_DIR / "metrics"

# ── Target column ──────────────────────────────────────────────────────────────
TARGET_COLUMN = "Churn"

# ── Column that should be dropped before modelling ────────────────────────────
DROP_COLUMNS = ["customerID"]

# ── Columns with known data-type issues in the Telco dataset ──────────────────
NUMERIC_COERCE_COLUMNS = ["TotalCharges"]

# ── Train / test split ────────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── Class imbalance handling ──────────────────────────────────────────────────
USE_SMOTE = True          # set to False to skip SMOTE

# ── Model hyperparameter grids (used by tune_models.py) ──────────────────────
TUNING_PARAM_GRIDS = {
    "RandomForest": {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "class_weight":      ["balanced"],
    },
    "XGBoost": {
        "n_estimators":  [100, 200],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample":     [0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators":  [100, 200],
        "max_depth":     [5, 10, -1],
        "learning_rate": [0.05, 0.1],
        "num_leaves":    [31, 63],
    },
}

TUNING_CV_FOLDS = 5
TUNING_SCORING  = "roc_auc"
