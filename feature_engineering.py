import logging
import warnings
from pandas.errors import PerformanceWarning

warnings.filterwarnings(
    "ignore",
    message=".*highly fragmented.*",
    category=PerformanceWarning
)

import numpy as np
import pandas as pd
from typing import Optional, List
# import matplotlib.pyplot as plt

ROLL_WINDOWS = [1, 3, 7, 14]

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def _drop_unused_columns(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing sentiment scores and drop columns that are entirely null,
    constant (nunique ≤ 1), or explicitly excluded. Any column that raises
    an error during .nunique() (e.g. list‑valued) is simply left in place.
    """
    # 1) Fill missing sentiment scores
    df_daily['min_sentiment_score'] = (
        df_daily['min_sentiment_score']
        .fillna(df_daily['avg_sentiment_score'])
    )
    df_daily['max_sentiment_score'] = (
        df_daily['max_sentiment_score']
        .fillna(df_daily['avg_sentiment_score'])
    )

    # 2) Identify columns to drop
    drop_cols = []
    for col in df_daily.columns:
        try:
            all_null   = df_daily[col].isnull().all()
            is_constant = df_daily[col].nunique(dropna=False) <= 1
        except (TypeError, ValueError):
            # Skip unhashable or problematic columns
            continue
        if all_null or is_constant:
            drop_cols.append(col)

    # 3) Add explicit exclusions
    explicit = [
        'created_at', 'new_subscriptions_t1', 'resubscriptions', 'title_length',
        'subs_per_avg_viewer', 'subs_7d_moving_avg', 'subs_3d_moving_avg',
        'day_over_day_peak_change', 'followers_start', 'followers_end',
    ]
    drop_cols += explicit

    # 4) Deduplicate and drop
    to_drop = list(dict.fromkeys(drop_cols))
    logging.debug("Dropping columns: %s", to_drop)
    return df_daily.drop(columns=to_drop, errors="ignore")

def _round_to_nearest_hour(t) -> int:
    if pd.isna(t):
        return np.nan
    if isinstance(t, pd.Timestamp):
        minute, hour = t.minute, t.hour
    else:
        minute, hour = getattr(t, "minute", 0), getattr(t, "hour", 0)
    return (hour + 1) % 24 if minute >= 30 else hour

def _compute_days_since_prev(group: pd.DataFrame) -> pd.Series:
    return group["stream_date"].diff().dt.days.fillna(0).astype(int)

def _compute_is_weekend(series: pd.Series) -> pd.Series:
    return series.isin(["Saturday","Sunday"]).astype(bool)

def _add_historical_rollups(df: pd.DataFrame):
    df = df.copy()
    grouped = df.groupby('stream_name', group_keys=False)
    def roll(col, n):
        return grouped[col].apply(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    def roll_mean(col, n):
        return grouped[col].apply(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
    def roll_std(col, n):
        return grouped[col].apply(lambda x: x.shift(1).rolling(n, min_periods=1).std())
    def roll_min(col, n):
        return grouped[col].apply(lambda x: x.shift(1).rolling(n, min_periods=1).min())
    def roll_max(col, n):
        return grouped[col].apply(lambda x: x.shift(1).rolling(n, min_periods=1).max())

    cols = [
        'total_subscriptions',
        'net_follower_change',
        'unique_viewers',
        'peak_concurrent_viewers',
        'stream_duration',
        'total_num_chats',
        'total_emotes_used',
        'bits_donated',
        'raids_received',
        'avg_sentiment_score', 
        'min_sentiment_score', 
        'max_sentiment_score',
        'category_changes'
    ]
    hist_cols = []
    for col in cols:
        for n in ROLL_WINDOWS:
            mean_col = f"avg_{col}_last_{n}"
            std_col  = f"std_{col}_last_{n}"
            min_col  = f"min_{col}_last_{n}"
            max_col  = f"max_{col}_last_{n}"

            df[mean_col] = roll_mean(col, n).fillna(0)
            df[std_col]  = roll_std(col, n).fillna(0)
            df[min_col]  = roll_min(col, n).fillna(0)
            df[max_col]  = roll_max(col, n).fillna(0)

            hist_cols.extend([mean_col, std_col, min_col, max_col])

    # forward-fill and ensure the very first value is zero
    for c in hist_cols:
        df[c] = grouped[c].apply(lambda x: x.ffill().fillna(0) if len(x) > 1 else x.fillna(0))
        first_idx = grouped[c].head(1).index
        df.loc[first_idx, c] = 0

    return df, hist_cols





def _prepare_training_frame(df_daily: pd.DataFrame):
    df = _drop_unused_columns(df_daily)
    df = df[df['stream_duration'] >= 1]
    # df = df.dropna()

    df["stream_date"] = pd.to_datetime(df["stream_date"])
    if "stream_start_time" in df:
        df["stream_start_time"] = pd.to_datetime(
            df["stream_start_time"].astype(str), errors="coerce"
        )

    df = df.sort_values(['stream_name','stream_date','stream_start_time'])
    df['start_time_hour'] = df['stream_start_time'].apply(_round_to_nearest_hour)

    # >>> ADD the cyclic features early
    df['start_hour_sin'] = np.sin(2 * np.pi * df['start_time_hour'] / 24)
    df['start_hour_cos'] = np.cos(2 * np.pi * df['start_time_hour'] / 24)

    if 'day_of_week' not in df:
        df['day_of_week'] = df['stream_date'].dt.day_name()
    df['is_weekend'] = _compute_is_weekend(df['day_of_week'])
    df['days_since_previous_stream'] = (
        df.groupby('stream_name', group_keys=False)
          .apply(_compute_days_since_prev)
          .reset_index(level=0, drop=True)
    )
    df, hist_cols = _add_historical_rollups(df)

    df = df.dropna(subset=['total_subscriptions', 'net_follower_change'] + hist_cols)
    
    base_feats = [
        'day_of_week',
        'start_hour_sin', 'start_hour_cos',
        'is_weekend',
        'days_since_previous_stream',
        'game_category',
        'stream_duration'
    ]

    features = base_feats + hist_cols

    if 'raw_tags' in df.columns:
        features = base_feats + hist_cols + ['raw_tags']
    else:
        features = base_feats + hist_cols

    df['game_category'] = df['game_category'].str.lower()

    # print('Features:')
    # for f in features:
    #     print(f)
    return df, features, hist_cols



def drop_outliers(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    method: str = 'iqr',
    factor: float = 1.5
) -> pd.DataFrame:
    """
    Returns a copy of df with outlier rows removed.
    
    Args:
      df: original DataFrame.
      cols: list of numeric columns to test. If None, uses all numeric dtype columns.
      method: 'iqr' or 'zscore'.
      factor:
        - for 'iqr': multiply IQR by this factor to set the bounds (default 1.5)
        - for 'zscore': drop rows where abs(z) > factor (so factor=3 drops >3σ)
    """
    df = df.copy()
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        cols = [c for c in cols if c in df.columns]

    if method == 'iqr':
        Q1 = df[cols].quantile(0.25)
        Q3 = df[cols].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        # keep rows where for ALL cols: lower <= val <= upper
        mask = ~((df[cols] < lower) | (df[cols] > upper)).any(axis=1)

    elif method == 'zscore':
        # note: requires scipy
        from scipy.stats import zscore
        zs = df[cols].apply(zscore)
        mask = (zs.abs() <= factor).all(axis=1)

    else:
        raise ValueError(f"Unknown method={method!r}, use 'iqr' or 'zscore'")

    return df.loc[mask].reset_index(drop=True)
