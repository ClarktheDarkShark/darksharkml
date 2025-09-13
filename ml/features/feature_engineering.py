"""Simple feature engineering utilities for the ml package."""
from __future__ import annotations

from typing import Tuple
import pandas as pd


def build_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix ``X`` and target vector ``y``.

    Parameters
    ----------
    df:
        Raw dataframe loaded from disk.
    target:
        Name of the target column.  The column is removed from ``df`` and
        returned separately as ``y``.

    Returns
    -------
    (X, y):
        ``X`` is a dataframe containing only feature columns and ``y`` is the
        corresponding target series.
    """
    df = df.copy()

    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in dataframe")

    y = df[target]
    X = df.drop(columns=[target])

    # Basic handling for categorical data: convert to category codes so that a
    # downstream model receives numeric values.  This keeps the function
    # dependency‑free and ensures deterministic, reproducible encoding.
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        X[col] = X[col].astype("category").cat.codes

    # Propagate original column order for reproducibility.
    X = X.loc[:, sorted(X.columns)]

    return X, y
