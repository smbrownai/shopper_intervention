"""
features.py — Shared preprocessing pipeline for online shopper intention dataset.
Used by both train.py and the FastAPI inference server.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "Weekend",
]

TARGET = "Revenue"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Month ordering for ordinal encoding (not used here — OHE handles it fine)
MONTH_ORDER = ["Feb", "Mar", "May", "Jun", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def validate_data(df: pd.DataFrame) -> dict:
    """
    Validate the raw dataset before training.

    Returns a dict with keys:
      - "passed": bool — False if any critical check failed
      - "errors": list of str — critical failures that block training
      - "warnings": list of str — non-blocking issues
      - "stats": dict — numeric summary logged to MLflow
    """
    errors = []
    warnings = []
    stats = {}

    # --- Critical: required columns present ---
    expected_cols = ALL_FEATURES + [TARGET]
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    # --- Critical: minimum row count ---
    stats["row_count"] = len(df)
    if len(df) < 500:
        errors.append(f"Too few rows: {len(df)} (minimum 500)")

    # If columns are missing we can't safely run the remaining checks
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings, "stats": stats}

    # --- Critical: target must be binary ---
    target_vals = set(df[TARGET].dropna().unique())
    if not target_vals.issubset({0, 1, True, False}):
        errors.append(f"Target '{TARGET}' contains non-binary values: {target_vals}")

    # --- Critical: no fully-null column ---
    fully_null = [c for c in expected_cols if df[c].isna().all()]
    if fully_null:
        errors.append(f"Columns are entirely null: {fully_null}")

    # --- Warning: purchase rate sanity ---
    purchase_rate = df[TARGET].astype(int).mean()
    stats["purchase_rate"] = round(float(purchase_rate), 4)
    if purchase_rate < 0.01:
        warnings.append(f"Purchase rate is very low: {purchase_rate:.1%} (expected ≥ 1%)")
    elif purchase_rate > 0.70:
        warnings.append(f"Purchase rate is unusually high: {purchase_rate:.1%} (expected ≤ 70%)")

    # --- Warning: BounceRates and ExitRates should be in [0, 1] ---
    for col in ["BounceRates", "ExitRates"]:
        if col in df.columns:
            out_of_range = ((df[col] < 0) | (df[col] > 1)).sum()
            if out_of_range > 0:
                warnings.append(f"{col} has {out_of_range} values outside [0, 1]")

    # --- Warning: duplicate rows ---
    duplicate_count = int(df.duplicated().sum())
    stats["duplicate_rows"] = duplicate_count
    if duplicate_count > 0:
        warnings.append(f"{duplicate_count} duplicate rows detected")

    # --- Info: per-column null rates ---
    null_rates = (df[expected_cols].isna().mean() * 100).round(2)
    cols_with_nulls = null_rates[null_rates > 0]
    stats["columns_with_nulls"] = len(cols_with_nulls)
    for col, pct in cols_with_nulls.items():
        stats[f"null_pct_{col}"] = float(pct)

    return {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "stats": stats,
    }


def load_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load CSV and return X, y."""
    df = pd.read_csv(path)
    # Normalize Weekend column to string so OHE is consistent
    df["Weekend"] = df["Weekend"].astype(str)
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].astype(int)  # True→1 (purchase), False→0 (no purchase)
    return X, y


def build_preprocessor(
    numeric_imputer_strategy: str = "median",
    excluded_features: list = None
) -> ColumnTransformer:
    
    excluded_features = excluded_features or []
    
    active_numeric = [f for f in NUMERIC_FEATURES if f not in excluded_features]
    active_categorical = [f for f in CATEGORICAL_FEATURES if f not in excluded_features]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=numeric_imputer_strategy)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, active_numeric),
            ("cat", categorical_pipeline, active_categorical),
        ],
        remainder="drop",
    )
    return preprocessor


def session_dict_to_dataframe(session: dict) -> pd.DataFrame:
    """Convert a single API session dict → single-row DataFrame."""
    row = {col: session.get(col, None) for col in ALL_FEATURES}
    df = pd.DataFrame([row])
    df["Weekend"] = df["Weekend"].astype(str)
    return df
