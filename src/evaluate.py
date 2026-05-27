"""
evaluate.py
-----------
Evaluate all trained models on the test set.

Outputs per model:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Confusion matrix plot  → outputs/figures/
  - ROC curve plot         → outputs/figures/
  - Classification report  → outputs/reports/

Summary table saved to:
  outputs/metrics/metrics_summary.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

from src.config import FIGURES_DIR, REPORTS_DIR, METRICS_DIR
from src.utils import print_section, save_figure


def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Compute classification metrics for a single fitted model.

    Parameters
    ----------
    model : fitted sklearn-compatible model
    X_test : np.ndarray
    y_test : np.ndarray

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, roc_auc
    """
    y_pred = model.predict(X_test)
    y_proba = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )

    return {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
        "y_pred":    y_pred,
        "y_proba":   y_proba,
    }


def plot_confusion_matrix(model_name: str, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Save a confusion matrix heatmap for one model."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12)
    plt.tight_layout()
    save_figure(f"cm_{model_name}.png")


def plot_all_roc_curves(models_results: dict, y_test: np.ndarray) -> None:
    """
    Plot all models' ROC curves on a single figure.

    Parameters
    ----------
    models_results : dict
        {model_name: metrics_dict} from evaluate_all_models.
    y_test : np.ndarray
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, result in models_results.items():
        fpr, tpr, _ = roc_curve(y_test, result["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_figure("roc_curves_all_models.png")


def save_classification_report(model_name: str, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """Write the sklearn classification_report to a text file."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
    report_path = REPORTS_DIR / f"report_{model_name}.txt"
    with open(report_path, "w") as f:
        f.write(f"=== Classification Report — {model_name} ===\n\n")
        f.write(report)
    print(f"    Report saved → {report_path.name}")


def save_metrics_summary(all_metrics: dict) -> None:
    """Save a CSV table comparing all models."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, m in all_metrics.items():
        rows.append({
            "Model":     name,
            "Accuracy":  m["accuracy"],
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1-Score":  m["f1"],
            "ROC-AUC":   m["roc_auc"],
        })
    summary_df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    path = METRICS_DIR / "metrics_summary.csv"
    summary_df.to_csv(path, index=False)
    print(f"\n  Metrics summary saved → {path}")
    return summary_df


def evaluate_all_models(trained_models: dict, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate every trained model; produce plots and reports.

    Parameters
    ----------
    trained_models : dict
        {model_name: fitted_model}
    X_test : np.ndarray
    y_test : np.ndarray

    Returns
    -------
    dict
        {model_name: metrics_dict}
    """
    print_section("MODEL EVALUATION")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for name, model in trained_models.items():
        print(f"\n  ── {name} ──")
        result = compute_metrics(model, X_test, y_test)
        all_results[name] = result

        print(f"    Accuracy  : {result['accuracy']:.4f}")
        print(f"    Precision : {result['precision']:.4f}")
        print(f"    Recall    : {result['recall']:.4f}")
        print(f"    F1-Score  : {result['f1']:.4f}")
        print(f"    ROC-AUC   : {result['roc_auc']:.4f}")

        plot_confusion_matrix(name, y_test, result["y_pred"])
        save_classification_report(name, y_test, result["y_pred"])

    print("\n  Plotting combined ROC curves…")
    plot_all_roc_curves(all_results, y_test)

    summary_df = save_metrics_summary(all_results)
    print("\n  ── Metrics Summary ──────────────────────────────────────────────")
    print(summary_df.to_string(index=False))

    return all_results


if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DATA_FILE
    from src.feature_engineering import engineer_features
    from src.train_models import train_all_models

    df = pd.read_csv(PROCESSED_DATA_FILE)
    X_train, X_test, y_train, y_test, _, _ = engineer_features(df)
    trained = train_all_models(X_train, y_train)
    evaluate_all_models(trained, X_test, y_test)
