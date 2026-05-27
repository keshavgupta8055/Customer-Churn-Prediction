"""
feature_engineering.py
-----------------------
Prepare features for model training:
  1. Separate features (X) and target (y).
  2. Identify numeric and categorical columns automatically.
  3. Build a sklearn Pipeline: OrdinalEncoder for categoricals, StandardScaler
     for numerics.
  4. Train / test split.
  5. Optionally apply SMOTE to the training set to handle class imbalance.

Returns X_train, X_test, y_train, y_test plus the fitted preprocessor.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

from src.config import TARGET_COLUMN, TEST_SIZE, RANDOM_STATE, USE_SMOTE
from src.utils import print_section


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Construct a ColumnTransformer that:
      - Scales numeric columns with StandardScaler.
      - Encodes categorical/object columns with OrdinalEncoder.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (no target column).

    Returns
    -------
    ColumnTransformer
        Unfitted sklearn transformer.
    """
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

    return preprocessor


def apply_smote(X_train: np.ndarray, y_train: np.ndarray):
    """
    Oversample the minority class using SMOTE.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    X_resampled, y_resampled : np.ndarray
    """
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  SMOTE applied → training set size: {X_res.shape[0]:,} "
          f"(original: {X_train.shape[0]:,})")
    return X_res, y_res


def engineer_features(df: pd.DataFrame):
    """
    Full feature engineering pipeline.

    Steps
    -----
    1. Split X / y.
    2. Train-test split.
    3. Fit preprocessor on training set; transform both sets.
    4. Optionally apply SMOTE.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe from preprocessing.py.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, preprocessor, feature_names)
    """
    print_section("FEATURE ENGINEERING")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    # Capture feature names before transformation (for SHAP labels later)
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    feature_names = numeric_cols + categorical_cols

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  Train size : {X_train_raw.shape[0]:,} samples")
    print(f"  Test size  : {X_test_raw.shape[0]:,} samples")
    print(f"  Features   : {X_train_raw.shape[1]}")

    preprocessor = build_preprocessor(X_train_raw)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    if USE_SMOTE:
        X_train, y_train = apply_smote(X_train, y_train)
    else:
        churn_pct = y_train.mean() * 100
        print(f"  SMOTE disabled. Training churn rate: {churn_pct:.1f}%")

    print(f"  Final training shape : {X_train.shape}")
    print(f"  Final test shape     : {X_test.shape}")

    return X_train, X_test, y_train, y_test, preprocessor, feature_names


if __name__ == "__main__":
    import pandas as pd
    from src.config import PROCESSED_DATA_FILE
    df = pd.read_csv(PROCESSED_DATA_FILE)
    X_train, X_test, y_train, y_test, preprocessor, features = engineer_features(df)
