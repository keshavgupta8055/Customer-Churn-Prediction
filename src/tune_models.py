"""
tune_models.py
--------------
Hyperparameter tuning using GridSearchCV on the top candidate models
(Random Forest, XGBoost, LightGBM).

Tuned models are saved over the original .joblib files in outputs/models/.
Returns a dict of {model_name: best_fitted_model}.
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from src.config import (
    TUNING_PARAM_GRIDS,
    TUNING_CV_FOLDS,
    TUNING_SCORING,
    RANDOM_STATE,
    MODELS_DIR,
)
from src.utils import print_section


def _get_base_model(name: str):
    """Instantiate a fresh base model for tuning by name."""
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    if name == "XGBoost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    if name == "LightGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    raise ValueError(f"Unknown model name for tuning: {name}")


def tune_model(
    name: str,
    param_grid: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> object:
    """
    Run GridSearchCV on one model and return the best estimator.

    Parameters
    ----------
    name : str
        Model name (must match a key in TUNING_PARAM_GRIDS).
    param_grid : dict
        Hyperparameter grid.
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    Best fitted estimator.
    """
    print(f"\n  Tuning {name}…")
    print(f"    Grid: {param_grid}")

    base_model = _get_base_model(name)
    cv = StratifiedKFold(n_splits=TUNING_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring=TUNING_SCORING,
        cv=cv,
        n_jobs=-1,
        verbose=0,
        refit=True,
    )

    grid_search.fit(X_train, y_train)
    best = grid_search.best_estimator_

    print(f"    Best params  : {grid_search.best_params_}")
    print(f"    Best CV {TUNING_SCORING}: {grid_search.best_score_:.4f}")

    # Save best model (overwrites original)
    path = MODELS_DIR / f"{name}_tuned.joblib"
    joblib.dump(best, path)
    print(f"    Saved tuned model → {path.name}")

    return best


def tune_all_models(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """
    Tune all models listed in TUNING_PARAM_GRIDS.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    dict
        {model_name: best_fitted_model}
    """
    print_section("HYPERPARAMETER TUNING")
    print(f"  Strategy : GridSearchCV, {TUNING_CV_FOLDS}-fold stratified CV")
    print(f"  Scoring  : {TUNING_SCORING}")

    tuned_models = {}

    for name, param_grid in TUNING_PARAM_GRIDS.items():
        try:
            tuned_models[name] = tune_model(name, param_grid, X_train, y_train)
        except Exception as e:
            print(f"  [SKIP] {name} — {e}")

    print(f"\n  Tuning complete. {len(tuned_models)} model(s) tuned.")
    return tuned_models


if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DATA_FILE
    from src.feature_engineering import engineer_features

    df = pd.read_csv(PROCESSED_DATA_FILE)
    X_train, X_test, y_train, y_test, _, _ = engineer_features(df)
    tuned = tune_all_models(X_train, y_train)
