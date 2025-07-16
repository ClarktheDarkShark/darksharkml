"""
Predictor (no Twitch)

Loads DailyStats from DB, engineers features, trains a tuned RandomForestRegressor,
and provides prediction utilities for start-time/duration planning.

Exports:
    train_predictor(app, log_metrics=True)
    get_predictor_artifacts()
    predgame(stream_name, ...)
    predhour(stream_name, hour, ...)
    format_pred_rows(df, include_hour=True)
    _infer_grid_for_game(...)   # intentionally exported for dashboard use

The `dashboard_predictions.py` blueprint expects:
    from predictor import get_predictor_artifacts, _infer_grid_for_game

So do **not** rename these without updating the dashboard.
"""

import itertools
import logging
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from db import db
from models import DailyStats, TimeSeries  # TimeSeries kept for possible future extension


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROLL_N = 5

# Default inference grids (override after training if desired)
DEFAULT_START_TIMES   = list(range(24))     # 0..23 hours
DEFAULT_DURATIONS_HRS = list(range(2, 13))  # 2..12 hours


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL STORAGE (populated after training)
# ─────────────────────────────────────────────────────────────────────────────
_predictor_state = {
    "pipeline": None,                      # best_estimator_ (fitted Pipeline)
    "model": None,                         # full GridSearchCV wrapper
    "df_for_inf": None,                    # cleaned feature frame incl stream_name
    "features": None,                      # feature column list
    "stream_category_options_inf": None,   # sorted list of known categories
    "optional_start_times": DEFAULT_START_TIMES,
    "stream_duration_opts": DEFAULT_DURATIONS_HRS,
    "trained_on": None,                    # UTC datetime
    "metrics": {},                         # dict of training metrics
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _load_daily_stats_df(app):
    """Read the entire DailyStats table into a DataFrame."""
    with app.app_context():
        df_daily = pd.read_sql_table(DailyStats.__tablename__, con=db.engine)
    df_daily.drop(columns=['tags'], errors='ignore', inplace=True)
    return df_daily


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def _drop_unused_columns(df_daily: pd.DataFrame) -> pd.DataFrame:
    to_drop = df_daily.columns[
        df_daily.isnull().all() |
        (df_daily.nunique(dropna=False) == 1)
    ].union([
        'created_at',
        'new_subscriptions_t1',
        'resubscriptions',
        'title_length',
        'subs_per_avg_viewer',
        'subs_7d_moving_avg',
        'subs_3d_moving_avg',
        'day_over_day_peak_change',
        'followers_start',
        'followers_end',
    ])
    logging.debug("Dropping columns: %s", list(to_drop))
    return df_daily.drop(columns=to_drop, errors="ignore")


def _round_to_nearest_hour(t) -> int:
    """
    t: pandas-compatible time-like value (datetime.time or Timestamp).
    Returns hour int 0..23; rounds up if minutes >= 30.
    """
    if pd.isna(t):
        return np.nan
    if isinstance(t, pd.Timestamp):
        minutes = t.minute
        hour = t.hour
    else:
        minutes = getattr(t, "minute", 0)
        hour = getattr(t, "hour", 0)
    return (hour + 1) % 24 if minutes >= 30 else hour


def _compute_days_since_prev(group: pd.DataFrame) -> pd.Series:
    # group sorted by stream_date already
    return group["stream_date"].diff().dt.days.fillna(0).astype(int)


def _compute_is_weekend(series: pd.Series) -> pd.Series:
    return series.isin(["Saturday", "Sunday"]).astype(bool)


def _add_historical_rollups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds rolling mean (shifted by 1 to exclude current row) for last ROLL_N streams per stream_name.
    Fills forward then zeroes first obs per group.
    """
    df = df.copy()
    grouped = df.groupby('stream_name', group_keys=False)

    def roll_mean(col):
        return grouped[col].apply(lambda x: x.shift(1).rolling(ROLL_N, min_periods=1).mean())

    df['avg_subs_last_5']                    = roll_mean('total_subscriptions')
    df['avg_follower_gain_last_5']           = roll_mean('net_follower_change')
    df['avg_unique_viewers_last_5']          = roll_mean('unique_viewers')
    df['avg_peak_concurrent_viewers_last_5'] = roll_mean('peak_concurrent_viewers')
    df['avg_stream_duration_last_5']         = roll_mean('stream_duration')
    df['avg_total_num_chats_last_5']         = roll_mean('total_num_chats')
    df['avg_total_emotes_used_last_5']       = roll_mean('total_emotes_used')
    df['avg_bits_donated_last_5']            = roll_mean('bits_donated')
    df['avg_raids_received_last_5']          = roll_mean('raids_received')
    df['avg_avg_sentiment_score_last_5']     = roll_mean('avg_sentiment_score')

    historical_cols = [
        'avg_subs_last_5',
        'avg_follower_gain_last_5',
        'avg_unique_viewers_last_5',
        'avg_peak_concurrent_viewers_last_5',
        'avg_stream_duration_last_5',
        'avg_total_num_chats_last_5',
        'avg_total_emotes_used_last_5',
        'avg_bits_donated_last_5',
        'avg_raids_received_last_5',
        'avg_avg_sentiment_score_last_5',
    ]

    # forward-fill within each group, then set first row to 0
    for col in historical_cols:
        df[col] = grouped[col].apply(lambda x: x.ffill().fillna(0) if len(x) > 1 else x.fillna(0))
        first_idx = grouped[col].head(1).index
        df.loc[first_idx, col] = 0

    return df, historical_cols


def _prepare_training_frame(df_daily: pd.DataFrame):
    """
    Full cleaning + feature engineering; returns:
      df_clean_sorted, features(list), historical_cols(list)
    """
    df = _drop_unused_columns(df_daily)

    # drop rows with any NaN (after drop_unused) – adjust if you want imputation
    df = df.dropna()

    # ensure datetime
    df["stream_date"] = pd.to_datetime(df["stream_date"])

    # ensure time column as Timestamp (combine w/ date if needed)
    if "stream_start_time" in df.columns:
        df["stream_start_time"] = pd.to_datetime(df["stream_start_time"].astype(str), errors="coerce")

    # sort for time ops
    df = df.sort_values(by=['stream_name', 'stream_date', 'stream_start_time'])

    # compute features
    df['start_time_hour'] = df['stream_start_time'].apply(_round_to_nearest_hour)

    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['stream_date'].dt.day_name()

    df['is_weekend'] = _compute_is_weekend(df['day_of_week'])
    df['days_since_previous_stream'] = (
        df.groupby('stream_name', group_keys=False)
          .apply(_compute_days_since_prev)
          .reset_index(level=0, drop=True)
    )

    # add rolling history
    df, historical_cols = _add_historical_rollups(df)

    # features used for training
    base_feats = [
        'day_of_week',
        'start_time_hour',
        'is_weekend',
        'days_since_previous_stream',
        'game_category',      # must exist in table
        'stream_duration',
    ]
    features = base_feats + historical_cols

    return df, features, historical_cols


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BUILD
# ─────────────────────────────────────────────────────────────────────────────
def _build_pipeline(X: pd.DataFrame):
    bool_cols        = X.select_dtypes(include=['bool']).columns.tolist()
    numeric_cols_all = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols     = [c for c in numeric_cols_all if c not in bool_cols]
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # day_of_week ordinal
    ordered_days = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
    from sklearn.ensemble import RandomForestRegressor

    preprocessor = ColumnTransformer(transformers=[
        ('num',   MinMaxScaler(), numeric_cols),
        ('bool',  'passthrough',  bool_cols),
        ('dow',   OrdinalEncoder(
                    categories=[ordered_days],
                    dtype=int,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1),
          ['day_of_week']),
        ('cat', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1),
          [c for c in categorical_cols if c != 'day_of_week']),
    ])

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        max_features=0.8,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ('pre', preprocessor),
        ('reg', rf),
    ])
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def _train_model(df_daily: pd.DataFrame):
    """
    Full train sequence. Returns:
      (gridsearch_model, best_estimator_pipeline, df_for_inf, features, metrics_dict)
    """
    df_clean_sorted, features, _ = _prepare_training_frame(df_daily)

    # training target
    y = df_clean_sorted['total_subscriptions'].copy()
    X = df_clean_sorted[features].copy()

    # temporal split (80/20 by date)
    cutoff = df_clean_sorted["stream_date"].quantile(0.8)
    train_mask = df_clean_sorted["stream_date"] < cutoff
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    pipeline = _build_pipeline(X)

    # grid search
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
    tscv = TimeSeriesSplit(n_splits=5)

    params = {
        "reg__n_estimators": [10, 50],
        "reg__max_depth":    [1, 3, 5],
        "reg__max_features": [1, 3, 7],
    }

    model = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=tscv,
        scoring="r2",
        n_jobs=-1,
        refit=True,
    )

    model.fit(X_train, y_train)

    # metrics
    from sklearn.metrics import mean_absolute_error
    test_r2 = model.score(X_test, y_test)
    y_pred_test = model.predict(X_test)
    const_baseline = np.full_like(y_test, y_train.mean())
    metrics = {
        "best_params": model.best_params_,
        "cv_r2": model.best_score_,
        "test_r2": test_r2,
        "const_mae": float(mean_absolute_error(y_test, const_baseline)),
        "model_mae": float(mean_absolute_error(y_test, y_pred_test)),
    }

    # inference frame (keep stream_name to locate last rows)
    df_for_inf = df_clean_sorted[['stream_name'] + features].copy()

    return model, model.best_estimator_, df_for_inf, features, metrics


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_last_row_for_stream(df_for_inf: pd.DataFrame, stream_name: str):
    """Return the most recent feature row for a given stream_name."""
    rows = df_for_inf[df_for_inf["stream_name"] == stream_name]
    if rows.empty:
        raise ValueError(f"No rows found for stream_name={stream_name!r}.")
    return rows.iloc[-1]


def _infer_grid_for_game(
    pipeline,
    df_for_inf: pd.DataFrame,
    features: List[str],
    stream_name: str,
    start_times: Optional[Iterable[int]] = None,
    durations: Optional[Iterable[int]] = None,
    category_options: Optional[Iterable[str]] = None,
    today_name: Optional[str] = None,
    top_n: int = 10,
    unique_scores: bool = True,
    start_hour_filter: Optional[int] = None,
):
    """
    Build an inference grid anchored on the last row for the specified stream_name.
    Returns a DataFrame sorted by predicted subs desc.

    IMPORTANT CHANGE vs. old Cog behavior:
    - If `category_options` is provided, **do not** auto-filter back to the stream's last
      recorded game_category. This lets the dashboard override to a specific game.
    - If `category_options` is None, we preserve original behavior: restrict to the
      stream's most recent game_category.
    """
    if pipeline is None:
        raise RuntimeError("Predictor pipeline is not trained.")

    if today_name is None:
        today_name = datetime.now().strftime("%A")

    last_row = _get_last_row_for_stream(df_for_inf, stream_name)
    base = last_row.to_frame().T  # single-row DataFrame

    if category_options is None:
        category_options = sorted(df_for_inf["game_category"].dropna().unique().tolist())
        restrict_to_stream_game = True
    else:
        category_options = list(category_options)
        restrict_to_stream_game = False  # caller is choosing categories

    if start_times is None:
        start_times = DEFAULT_START_TIMES
    if durations is None:
        durations = DEFAULT_DURATIONS_HRS

    dyn_cols = ["game_category", "start_time_hour", "stream_duration"]
    combos = list(itertools.product(category_options, start_times, durations))
    grid = pd.DataFrame(combos, columns=dyn_cols)

    # Broadcast base row across the grid, overwrite dynamic cols
    base_rep = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
    for col in dyn_cols:
        base_rep[col] = grid[col]

    # Force "today" day_of_week so user asks "If I stream this game today..."
    base_rep["day_of_week"] = today_name

    X_inf = base_rep[features]
    preds = pipeline.predict(X_inf)

    results = X_inf.copy()
    results['y_pred'] = preds

    # Preserve legacy behavior when caller didn't specify categories
    if restrict_to_stream_game:
        game_cat = last_row["game_category"]
        results = results[results['game_category'] == game_cat]

    if start_hour_filter is not None:
        results = results[results['start_time_hour'] == start_hour_filter]

    results = results.sort_values('y_pred', ascending=False)

    if unique_scores:
        results = results.drop_duplicates(subset=['y_pred'], keep='first')

    top_df = results.head(top_n).reset_index(drop=True)
    return top_df


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ACCESSORS (for dashboard / notebook)
# ─────────────────────────────────────────────────────────────────────────────
def get_predictor_artifacts():
    """Return trained artifacts (may be None if training not run yet)."""
    return (
        _predictor_state["pipeline"],
        _predictor_state["df_for_inf"],
        _predictor_state["features"],
        _predictor_state["stream_category_options_inf"],
        _predictor_state["optional_start_times"],
        _predictor_state["stream_duration_opts"],
        _predictor_state["metrics"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def train_predictor(app, *, log_metrics: bool = True):
    """
    Load data, train the model, and cache artifacts in _predictor_state.
    Call once at startup (or whenever you want to retrain).
    """
    df_daily = _load_daily_stats_df(app)
    model, pipe, df_for_inf, features, metrics = _train_model(df_daily)

    _predictor_state["pipeline"] = pipe
    _predictor_state["model"] = model
    _predictor_state["df_for_inf"] = df_for_inf
    _predictor_state["features"] = features
    _predictor_state["stream_category_options_inf"] = (
        sorted(df_for_inf["game_category"].dropna().unique().tolist())
    )
    _predictor_state["trained_on"] = datetime.utcnow()
    _predictor_state["metrics"] = metrics

    if log_metrics:
        logging.info("Predictor trained: %s", metrics)
    return metrics


def _ensure_trained(app=None):
    """
    Ensure the predictor is trained.
    If untrained and an app is supplied, trains immediately.
    If untrained and no app, raises RuntimeError.
    """
    if _predictor_state["pipeline"] is not None:
        return
    if app is None:
        raise RuntimeError("Predictor not trained and no app supplied to train.")
    train_predictor(app)


# ─────────────────────────────────────────────────────────────────────────────
# USER-FACING PREDICTION HELPERS (replacement for old Twitch commands)
# ─────────────────────────────────────────────────────────────────────────────
def predgame(
    stream_name: str,
    *,
    topn: int = 5,
    app=None,
    today_name: Optional[str] = None,
    start_times: Optional[Iterable[int]] = None,
    durations: Optional[Iterable[int]] = None,
    category_options: Optional[Iterable[str]] = None,
    unique_scores: bool = True,
):
    """
    Return topN start-time/duration combos (like old !predgame).
    Trains lazily if needed and app is provided.
    """
    _ensure_trained(app)
    pipe   = _predictor_state["pipeline"]
    df_inf = _predictor_state["df_for_inf"]
    feats  = _predictor_state["features"]
    return _infer_grid_for_game(
        pipe,
        df_inf,
        feats,
        stream_name=stream_name,
        start_times=start_times or _predictor_state["optional_start_times"],
        durations=durations or _predictor_state["stream_duration_opts"],
        category_options=category_options,  # None => use legacy stream-game behavior
        today_name=today_name,
        top_n=topn,
        unique_scores=unique_scores,
        start_hour_filter=None,
    )


def predhour(
    stream_name: str,
    hour: int,
    *,
    topn: int = 5,
    app=None,
    today_name: Optional[str] = None,
    start_times: Optional[Iterable[int]] = None,
    durations: Optional[Iterable[int]] = None,
    category_options: Optional[Iterable[str]] = None,
    unique_scores: bool = True,
):
    """
    Return topN duration combos for a specific start hour (like old !predhour).
    """
    _ensure_trained(app)
    pipe   = _predictor_state["pipeline"]
    df_inf = _predictor_state["df_for_inf"]
    feats  = _predictor_state["features"]
    return _infer_grid_for_game(
        pipe,
        df_inf,
        feats,
        stream_name=stream_name,
        start_times=start_times or _predictor_state["optional_start_times"],
        durations=durations or _predictor_state["stream_duration_opts"],
        category_options=category_options,  # None => legacy behavior
        today_name=today_name,
        top_n=topn,
        unique_scores=unique_scores,
        start_hour_filter=hour,
    )


def format_pred_rows(df: pd.DataFrame, include_hour: bool = True) -> str:
    """
    Compact string formatter approximating old Twitch chat output.
    """
    parts = []
    for _, r in df.iterrows():
        if include_hour:
            parts.append(f"{int(r.start_time_hour):02d}:00 h{int(r.stream_duration)} → {r.y_pred:.1f}")
        else:
            parts.append(f"h{int(r.stream_duration)} → {r.y_pred:.1f}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: run `python predictor.py` after your Flask app is importable.
    try:
        from main import app
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Could not import Flask app: {e}") from e

    metrics = train_predictor(app)
    print("Training complete.")
    print("Metrics:", metrics)

    # demo (replace with a real stream_name present in your DB)
    example_stream = None
    if _predictor_state["df_for_inf"] is not None and not _predictor_state["df_for_inf"].empty:
        example_stream = _predictor_state["df_for_inf"]["stream_name"].iloc[-1]

    if example_stream:
        df_demo = predgame(example_stream, topn=5)
        print(f"Top 5 for {example_stream}:")
        print(df_demo)
        df_demo_hour = predhour(example_stream, hour=20, topn=5)
        print(f"Top 5 for {example_stream} @20:00:")
        print(df_demo_hour)
    else:
        print("No data available to demo predictions.")
