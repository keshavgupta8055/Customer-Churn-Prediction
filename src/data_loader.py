"""
data_loader.py
--------------
Load the raw Telco Customer Churn CSV and print an initial inspection report.
"""

import sys
import pandas as pd

from src.config import RAW_DATA_FILE
from src.utils import print_section


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw CSV dataset.

    Returns
    -------
    pd.DataFrame
        The full raw dataset as loaded from disk.

    Raises
    ------
    FileNotFoundError
        If the CSV is not present at the path defined in config.py.
    """
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found at: {RAW_DATA_FILE}"
            "\nPlease download it from Kaggle and place it in data/raw/\n"
            "  https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n"
        )

    df = pd.read_csv(RAW_DATA_FILE)
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """
    Print a structured inspection of the raw dataframe:
    shape, dtypes, missing values, and class balance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.
    """
    print_section("DATA INSPECTION")

    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n")

    print("── Data types ──────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing values ──────────────────────────────")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("  No null values found (check TotalCharges for whitespace blanks).")
    else:
        print(missing.to_string())

    print("\n── First 3 rows ────────────────────────────────")
    print(df.head(3).to_string())

    print("\n── Target distribution (raw) ───────────────────")
    if "Churn" in df.columns:
        vc = df["Churn"].value_counts()
        pct = df["Churn"].value_counts(normalize=True) * 100
        summary = pd.DataFrame({"Count": vc, "Percent": pct.round(1)})
        print(summary.to_string())


if __name__ == "__main__":
    df = load_raw_data()
    inspect_data(df)
