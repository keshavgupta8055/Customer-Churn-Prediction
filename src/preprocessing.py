"""
preprocessing.py
----------------
Clean the raw Telco dataset:
  - Fix known data-type issues (TotalCharges whitespace → NaN → float)
  - Drop unnecessary columns (customerID)
  - Handle missing values
  - Binary-encode the target column (Churn: Yes/No → 1/0)
  - Save the cleaned dataframe to data/processed/
"""

import pandas as pd

from src.config import (
    DROP_COLUMNS,
    NUMERIC_COERCE_COLUMNS,
    PROCESSED_DATA_FILE,
    TARGET_COLUMN,
    DATA_PROCESSED_DIR,
)
from src.utils import print_section


def fix_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce columns that should be numeric but may contain whitespace or empty
    strings (a known issue with TotalCharges in the Telco dataset).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected numeric columns.
    """
    for col in NUMERIC_COERCE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute or drop missing values after type coercion.
    TotalCharges NaNs (≈11 rows) are imputed with the median.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    for col in NUMERIC_COERCE_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            n_missing = df[col].isnull().sum()
            df[col] = df[col].fillna(median_val)
            print(f"  Imputed {n_missing} missing value(s) in '{col}' with median ({median_val:.2f})")

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the target column from Yes/No strings to binary integers (1/0).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].replace({"Yes": 1, "No": 0})
        df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="raise")
    return df


def drop_unneeded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove columns that carry no predictive value (e.g. customerID).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    cols_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Steps
    -----
    1. Fix numeric column types.
    2. Handle missing values.
    3. Encode the target.
    4. Drop unneeded columns.
    5. Save processed CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned, ready-to-use dataframe.
    """
    print_section("PREPROCESSING")

    df = fix_numeric_columns(df)
    df = handle_missing_values(df)
    df = encode_target(df)
    df = drop_unneeded_columns(df)

    # Verify no remaining nulls
    remaining_nulls = df.isnull().sum().sum()
    print(f"  Remaining null values after preprocessing: {remaining_nulls}")
    print(f"  Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Target balance - Churn=1: {df[TARGET_COLUMN].sum():,}  "
          f"({df[TARGET_COLUMN].mean()*100:.1f}%)")

    # Save
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"  Saved processed data -> {PROCESSED_DATA_FILE}")

    return df


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    raw_df = load_raw_data()
    clean_df = preprocess(raw_df)
