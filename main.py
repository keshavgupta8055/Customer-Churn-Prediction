"""
main.py
-------
Customer Churn Prediction — Full Pipeline Runner

Runs every stage of the project in order:
  1. Load raw data
  2. Preprocess
  3. EDA
  4. Feature engineering
  5. Train models
  6. Evaluate models
  7. Hyperparameter tuning
  8. Feature importance & SHAP explanation

Run from the project root with:
    python main.py
"""

import sys
from src.utils import print_section, timer, get_best_model_name
from src.config import (
    FIGURES_DIR, MODELS_DIR, REPORTS_DIR, METRICS_DIR, DATA_PROCESSED_DIR
)


def ensure_output_dirs() -> None:
    """Create all required output directories if they don't already exist."""
    for d in [FIGURES_DIR, MODELS_DIR, REPORTS_DIR, METRICS_DIR, DATA_PROCESSED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


@timer
def run_pipeline() -> None:
    """Execute the full ML pipeline end-to-end."""

    print_section("CUSTOMER CHURN PREDICTION — ML PIPELINE")
    print("  Starting full pipeline…\n")

    ensure_output_dirs()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    from src.data_loader import load_raw_data, inspect_data
    df_raw = load_raw_data()
    inspect_data(df_raw)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    from src.preprocessing import preprocess
    df_clean = preprocess(df_raw)

    # ── 3. EDA ────────────────────────────────────────────────────────────────
    from src.eda import run_eda
    run_eda(df_clean)

    # ── 4. Feature engineering ────────────────────────────────────────────────
    from src.feature_engineering import engineer_features
    X_train, X_test, y_train, y_test, preprocessor, feature_names = engineer_features(df_clean)

    # ── 5. Train models ───────────────────────────────────────────────────────
    from src.train_models import train_all_models
    trained_models = train_all_models(X_train, y_train)

    # ── 6. Evaluate ───────────────────────────────────────────────────────────
    from src.evaluate import evaluate_all_models
    metrics = evaluate_all_models(trained_models, X_test, y_test)

    best_model_name = get_best_model_name(metrics, metric="roc_auc")
    print(f"\n  ★  Best model by ROC-AUC: {best_model_name} "
          f"({metrics[best_model_name]['roc_auc']:.4f})")

    # ── 7. Hyperparameter tuning ──────────────────────────────────────────────
    from src.tune_models import tune_all_models
    tuned_models = tune_all_models(X_train, y_train)

    if tuned_models:
        print_section("EVALUATING TUNED MODELS")
        tuned_metrics = evaluate_all_models(tuned_models, X_test, y_test)
        best_tuned = get_best_model_name(tuned_metrics, metric="roc_auc")
        print(f"\n  ★  Best tuned model by ROC-AUC: {best_tuned} "
              f"({tuned_metrics[best_tuned]['roc_auc']:.4f})")

    # ── 8. Explanation ────────────────────────────────────────────────────────
    from src.explain import explain_models
    explain_models(trained_models, X_test, feature_names)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_section("PIPELINE COMPLETE")
    print(f"  Figures   → {FIGURES_DIR}")
    print(f"  Models    → {MODELS_DIR}")
    print(f"  Reports   → {REPORTS_DIR}")
    print(f"  Metrics   → {METRICS_DIR}")


if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
        sys.exit(0)
