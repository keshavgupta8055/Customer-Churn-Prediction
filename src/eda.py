"""
eda.py
------
Exploratory Data Analysis: generate and save diagnostic plots.

Plots saved to outputs/figures/:
  - churn_distribution.png       : class balance bar chart
  - numeric_distributions.png    : histograms of all numeric features
  - churn_by_categorical.png     : churn rate per categorical feature
  - churn_by_numeric.png         : boxplots of numeric features vs churn
  - correlation_heatmap.png      : Pearson correlation heatmap
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.config import FIGURES_DIR, TARGET_COLUMN
from src.utils import print_section, save_figure


def plot_churn_distribution(df: pd.DataFrame) -> None:
    """Bar chart of the target class balance."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df[TARGET_COLUMN].value_counts()
    labels = ["No Churn (0)", "Churn (1)"]
    colors = ["#4C72B0", "#DD8452"]

    axes[0].bar(labels, counts.values, color=colors, edgecolor="black", width=0.5)
    axes[0].set_title("Churn Count", fontsize=13)
    axes[0].set_ylabel("Number of customers")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, f"{v:,}", ha="center", fontsize=10)

    axes[1].pie(
        counts.values,
        labels=labels,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("Churn Proportion", fontsize=13)

    plt.suptitle("Target Variable Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure("churn_distribution.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Histograms for all numeric columns, coloured by churn."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COLUMN]

    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        for churn_val, label, color in [(0, "No Churn", "#4C72B0"), (1, "Churn", "#DD8452")]:
            subset = df[df[TARGET_COLUMN] == churn_val][col]
            axes[i].hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="none")
        axes[i].set_title(col, fontsize=11)
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Feature Distributions by Churn", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure("numeric_distributions.png")


def plot_churn_by_categorical(df: pd.DataFrame) -> None:
    """Churn rate per category for all object/binary columns."""
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    # Some binary columns are encoded but still look categorical
    binary_cols = [c for c in df.columns if df[c].nunique() == 2 and c != TARGET_COLUMN]
    cat_cols = list(set(cat_cols + binary_cols))
    cat_cols = [c for c in cat_cols if c != TARGET_COLUMN]

    if not cat_cols:
        print("  No categorical columns found for plot_churn_by_categorical.")
        return

    n = len(cat_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        churn_rate = df.groupby(col)[TARGET_COLUMN].mean().reset_index()
        churn_rate.columns = [col, "Churn Rate"]
        churn_rate = churn_rate.sort_values("Churn Rate", ascending=False)

        axes[i].bar(
            churn_rate[col].astype(str),
            churn_rate["Churn Rate"],
            color="#DD8452",
            edgecolor="black",
        )
        axes[i].set_title(f"Churn Rate by {col}", fontsize=10)
        axes[i].set_ylabel("Churn Rate")
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis="x", rotation=30)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Churn Rate by Categorical Feature", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure("churn_by_categorical.png")


def plot_churn_by_numeric(df: pd.DataFrame) -> None:
    """Boxplots comparing numeric feature distributions for churned vs non-churned."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_COLUMN]

    n = len(numeric_cols)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 4))
    axes = axes.flatten()

    plot_df = df.copy()
    plot_df[TARGET_COLUMN] = plot_df[TARGET_COLUMN].map({0: "No Churn", 1: "Churn"})

    for i, col in enumerate(numeric_cols):
        sns.boxplot(
            data=plot_df,
            x=TARGET_COLUMN,
            y=col,
            palette={"No Churn": "#4C72B0", "Churn": "#DD8452"},
            ax=axes[i],
        )
        axes[i].set_title(col, fontsize=11)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Numeric Features vs Churn (Boxplots)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure("churn_by_numeric.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Pearson correlation heatmap of all numeric columns."""
    numeric_df = df.select_dtypes(include="number")

    fig, ax = plt.subplots(figsize=(10, 8))
    corr = numeric_df.corr()
    mask = pd.DataFrame(False, index=corr.index, columns=corr.columns)
    # Mask the upper triangle to reduce visual noise
    import numpy as np
    mask_arr = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask_arr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_figure("correlation_heatmap.png")


def run_eda(df: pd.DataFrame) -> None:
    """
    Run all EDA plots in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned (preprocessed) dataframe. Target should already be 0/1.
    """
    print_section("EXPLORATORY DATA ANALYSIS")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("  Plotting churn distribution…")
    plot_churn_distribution(df)

    print("  Plotting numeric distributions…")
    plot_numeric_distributions(df)

    print("  Plotting churn by categorical features…")
    plot_churn_by_categorical(df)

    print("  Plotting boxplots (numeric vs churn)…")
    plot_churn_by_numeric(df)

    print("  Plotting correlation heatmap…")
    plot_correlation_heatmap(df)

    print(f"\n  All EDA plots saved → {FIGURES_DIR}")


if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DATA_FILE
    df = pd.read_csv(PROCESSED_DATA_FILE)
    run_eda(df)
