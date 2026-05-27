"""
explain.py
----------
Feature importance and model interpretability.

For tree-based models: built-in feature_importances_ bar chart.
For all models: SHAP summary plot and beeswarm plot.

Plots saved to outputs/figures/:
  - feature_importance_{model_name}.png
  - shap_summary_{model_name}.png
  - shap_beeswarm_{model_name}.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import FIGURES_DIR
from src.utils import print_section, save_figure


def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 20) -> None:
    """
    Plot the top-N built-in feature importances for tree-based models.

    Parameters
    ----------
    model : fitted model with feature_importances_ attribute
    feature_names : list of str
    model_name : str
    top_n : int
    """
    if not hasattr(model, "feature_importances_"):
        print(f"    {model_name} has no feature_importances_ — skipping bar chart.")
        return

    importances = model.feature_importances_
    n_features = min(top_n, len(feature_names), len(importances))

    # Align lengths (SMOTE doesn't change feature count, but guard anyway)
    names = feature_names[:len(importances)]

    indices = np.argsort(importances)[::-1][:n_features]
    sorted_names = [names[i] for i in indices]
    sorted_vals = importances[indices]

    fig, ax = plt.subplots(figsize=(9, max(5, n_features * 0.35)))
    bars = ax.barh(range(n_features), sorted_vals[::-1], color="#4C72B0", edgecolor="black")
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(sorted_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {n_features} Feature Importances — {model_name}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure(f"feature_importance_{model_name}.png")


def plot_shap_summary(model, X_test: np.ndarray, feature_names: list, model_name: str) -> None:
    """
    Generate a SHAP summary (dot) plot for the given model.

    Parameters
    ----------
    model : fitted model
    X_test : np.ndarray
    feature_names : list of str
    model_name : str
    """
    try:
        import shap
    except ImportError:
        print("    [SKIP] shap not installed — run: pip install shap")
        return

    print(f"    Computing SHAP values for {model_name} (may take a moment)…")

    # Use TreeExplainer for tree models; fallback to KernelExplainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # For binary classifiers, TreeExplainer may return a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        background = shap.kmeans(X_test, 50)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_test, nsamples=100)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    # Align feature names to the number of features in X_test
    names = feature_names[:X_test.shape[1]]

    # Summary dot plot
    fig, ax = plt.subplots(figsize=(9, 7))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=names,
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure(f"shap_summary_{model_name}.png")

    # Bar plot (mean |SHAP|)
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=names,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Feature Importance (Mean |SHAP|) — {model_name}",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure(f"shap_bar_{model_name}.png")


def explain_models(
    trained_models: dict,
    X_test: np.ndarray,
    feature_names: list,
    top_n: int = 20,
) -> None:
    """
    Run feature importance and SHAP analysis for all trained models.

    Parameters
    ----------
    trained_models : dict
        {model_name: fitted_model}
    X_test : np.ndarray
    feature_names : list of str
    top_n : int
        Number of top features to show.
    """
    print_section("FEATURE IMPORTANCE & EXPLANATION")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Prioritise tree-based models for SHAP (faster)
    shap_candidates = ["RandomForest", "XGBoost", "LightGBM"]

    for name, model in trained_models.items():
        print(f"\n  ── {name} ──")
        plot_feature_importance(model, feature_names, name, top_n)

        if name in shap_candidates:
            plot_shap_summary(model, X_test, feature_names, name)
        else:
            print(f"    Skipping SHAP for {name} (non-tree model).")

    print(f"\n  Explanation plots saved → {FIGURES_DIR}")


if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DATA_FILE
    from src.feature_engineering import engineer_features
    from src.train_models import train_all_models

    df = pd.read_csv(PROCESSED_DATA_FILE)
    X_train, X_test, y_train, y_test, _, feature_names = engineer_features(df)
    trained = train_all_models(X_train, y_train)
    explain_models(trained, X_test, feature_names)
