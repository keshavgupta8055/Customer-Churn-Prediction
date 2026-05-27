"""
train_models.py
---------------
Train five ML models on the prepared training data:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - XGBoost
  - LightGBM

Each trained model is saved to outputs/models/ as a .joblib file.
Returns a dict of {model_name: fitted_model}.
"""

import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[WARN] xgboost not installed — XGBoost model will be skipped.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("[WARN] lightgbm not installed — LightGBM model will be skipped.")

from src.config import RANDOM_STATE, MODELS_DIR
from src.utils import print_section


def get_model_definitions() -> dict:
    """
    Return a dict of {name: unfitted_model_instance} for all models to train.

    Returns
    -------
    dict
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            solver="lbfgs",
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )

    return models


def train_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Train all models defined in get_model_definitions() and save them to disk.

    Parameters
    ----------
    X_train : np.ndarray
        Preprocessed training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    dict
        {model_name: fitted_model}
    """
    print_section("MODEL TRAINING")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_definitions = get_model_definitions()
    trained_models = {}

    for name, model in model_definitions.items():
        print(f"  Training {name}…", end=" ", flush=True)
        model.fit(X_train, y_train)
        trained_models[name] = model

        # Save to disk
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(model, model_path)
        print(f"saved → {model_path.name}")

    print(f"\n  {len(trained_models)} model(s) trained and saved to {MODELS_DIR}")
    return trained_models


def load_model(name: str):
    """
    Load a previously saved model by name.

    Parameters
    ----------
    name : str
        Model name (e.g. "RandomForest").

    Returns
    -------
    Fitted sklearn-compatible model.
    """
    model_path = MODELS_DIR / f"{name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DATA_FILE
    from src.feature_engineering import engineer_features

    df = pd.read_csv(PROCESSED_DATA_FILE)
    X_train, X_test, y_train, y_test, _, _ = engineer_features(df)
    trained_models = train_all_models(X_train, y_train)
