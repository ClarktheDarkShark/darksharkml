"""Utilities for loading data used in training models.

The functions here provide a small abstraction around reading raw data from the
``ml/data`` directory and returning train/test splits that can be consumed by
scikit‑learn models.  The implementation is intentionally light‑weight so it can
be reused in notebooks or scripts without pulling in heavy dependencies.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .features.feature_engineering import build_features

# Directory containing raw data files.  By default we expect a CSV file but the
# path can be overridden when calling :func:`load_raw_data`.
DATA_DIR = Path(__file__).resolve().parent / "data"


def load_raw_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load a raw dataset from ``path`` or the default data directory.

    Parameters
    ----------
    path:
        Optional path to a CSV file.  If omitted, the first ``*.csv`` file in
        :data:`DATA_DIR` is used.
    """
    if path is not None:
        csv_path = Path(path)
    else:
        csv_files = sorted(DATA_DIR.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in data directory: {DATA_DIR!r}"
            )
        csv_path = csv_files[0]

    return pd.read_csv(csv_path)


def get_train_test_data(
    target: str,
    *,
    data_path: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    shuffle: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return train/test splits for the dataset.

    The function loads the raw data, applies basic feature engineering and then
    splits the resulting feature matrix and target vector into training and test
    sets using :func:`sklearn.model_selection.train_test_split`.
    """
    df = load_raw_data(data_path)

    X, y = build_features(df, target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=shuffle
    )

    return X_train, X_test, y_train, y_test
