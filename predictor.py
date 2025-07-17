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
import os
from datetime import datetime
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.compose import TransformedTargetRegressor

from db import db
from models import DailyStats, TimeSeries  # TimeSeries kept for possible future extension
from pipeline import _build_pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROLL_N = 5
DEFAULT_START_TIMES   = list(range(24))     # 0..23 hours
DEFAULT_DURATIONS_HRS = list(range(2, 13))  # 2..12 hours
_ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "predictor_artifacts.joblib")

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL STATE
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
# ATTEMPT TO LOAD PRE‐TRAINED ARTIFACTS (skip on‐dyno training)
# ─────────────────────────────────────────────────────────────────────────────
if os.path.exists(_ARTIFACT_PATH):
    try:
        data = joblib.load(_ARTIFACT_PATH)
        df_inf = data.get("df_for_inf")
        df_inf['game_category'] = df_inf['game_category'].str.lower()
        if isinstance(df_inf, pd.DataFrame):
            df_inf.columns = df_inf.columns.map(str)
        _predictor_state.update({
            "pipeline":                    data.get("pipeline"),
            "model":                       None,  # drop full GridSearchCV at runtime
            "df_for_inf":                  df_inf,
            "features":                    data.get("features"),
            "stream_category_options_inf": data.get("stream_category_options_inf"),
            "optional_start_times":        data.get("optional_start_times", DEFAULT_START_TIMES),
            "stream_duration_opts":        data.get("stream_duration_opts", DEFAULT_DURATIONS_HRS),
            "trained_on":                  data.get("trained_on", None),
            "metrics":                     data.get("metrics", {}),
        })
        logging.info("Loaded predictor artifacts from %s", _ARTIFACT_PATH)
    except Exception as e:
        logging.exception("Failed to load artifacts; will train on‐dyno when invoked: %s", e)
else:
    logging.info("No predictor_artifacts.joblib found; on‐dyno training available when called.")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _load_daily_stats_df(app):
    """Read the entire daily_stats table into a DataFrame, with string column names."""
    with app.app_context():
        df_daily = pd.read_sql_table(DailyStats.__tablename__, con=db.engine)
    df_daily.columns = df_daily.columns.map(str)
    df_daily.drop(columns=['tags'], errors='ignore', inplace=True)
    df_daily['game_category'] = df_daily['game_category'].str.lower()
    return df_daily

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def _drop_unused_columns(df_daily: pd.DataFrame) -> pd.DataFrame:
    to_drop = df_daily.columns[
        df_daily.isnull().all() |
        (df_daily.nunique(dropna=False) == 1)
    ].union([
        'created_at','new_subscriptions_t1','resubscriptions','title_length',
        'subs_per_avg_viewer','subs_7d_moving_avg','subs_3d_moving_avg',
        'day_over_day_peak_change','followers_start','followers_end',
    ])
    logging.debug("Dropping columns: %s", list(to_drop))
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
    def roll(col):
        return grouped[col].apply(lambda x: x.shift(1).rolling(ROLL_N, min_periods=1).mean())
    cols = [
        'total_subscriptions','net_follower_change','unique_viewers',
        'peak_concurrent_viewers','stream_duration','total_num_chats',
        'total_emotes_used','bits_donated','raids_received',
        'avg_sentiment_score'
    ]
    for col in cols:
        df[f"avg_{col}_last_5"] = roll(col)
    hist_cols = [c for c in df.columns if c.endswith("_last_5")]
    for c in hist_cols:
        df[c] = grouped[c].apply(lambda x: x.ffill().fillna(0) if len(x)>1 else x.fillna(0))
        first = grouped[c].head(1).index
        df.loc[first, c] = 0
    return df, hist_cols

def _prepare_training_frame(df_daily: pd.DataFrame):
    df = _drop_unused_columns(df_daily)
    df = df[df['stream_duration'] >= 1]
    df = df.dropna()

    df["stream_date"] = pd.to_datetime(df["stream_date"])
    if "stream_start_time" in df:
        df["stream_start_time"] = pd.to_datetime(
            df["stream_start_time"].astype(str), errors="coerce"
        )
    df = df.sort_values(['stream_name','stream_date','stream_start_time'])
    df['start_time_hour'] = df['stream_start_time'].apply(_round_to_nearest_hour)
    if 'day_of_week' not in df:
        df['day_of_week'] = df['stream_date'].dt.day_name()
    df['is_weekend'] = _compute_is_weekend(df['day_of_week'])
    df['days_since_previous_stream'] = (
        df.groupby('stream_name', group_keys=False)
          .apply(_compute_days_since_prev)
          .reset_index(level=0, drop=True)
    )
    df, hist_cols = _add_historical_rollups(df)
    base_feats = [
        'day_of_week','start_time_hour','is_weekend',
        'days_since_previous_stream','game_category','stream_duration'
    ]
    features = base_feats + hist_cols
    df['game_category'] = df['game_category'].str.lower()

    return df, features, hist_cols



# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────
def _train_model(df_daily: pd.DataFrame):
    df_clean, feats, _ = _prepare_training_frame(df_daily)
    y = df_clean['total_subscriptions']
    # y = df_clean['net_follower_change']
    X = df_clean[feats]
    cutoff = df_clean["stream_date"].quantile(0.8)
    train_mask = df_clean["stream_date"] < cutoff
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    # import matplotlib.pyplot as plt
    # plt.hist(y_train, bins=20, alpha=0.6, label='train')
    # plt.hist(y_test,  bins=20, alpha=0.6, label='test')
    # plt.legend(); plt.title("Subscriptions: train vs test")
    # plt.show()

    # plot_features(feats, X_train, y_train, train_mask, df_clean)

    print('Training sample size:',X_train.shape)

    pipeline, mod = _build_pipeline(X_train)
    
    tscv = TimeSeriesSplit(n_splits=5, gap=0)  # increase gap if leakage via lagged feats

    if mod == 'rf':
        params = {
            "reg__n_estimators":    [200, 600, 1000],
            "reg__min_samples_leaf":[3, 5, 10, 50],
            "reg__min_samples_split":[5, 10, 50],
            "reg__max_features":    ['sqrt', 0.5, 0.8, 1.0],
            "reg__max_depth": [None, 5, 10]
        }
    elif mod == 'hgb':
        params = [
            # Poisson‐loss grid
            {
            'reg__regressor__loss': ['poisson'],
            'reg__regressor__learning_rate':    [0.05, 0.1, 0.2],
            'reg__regressor__max_leaf_nodes':   [31, 63, 80],
            'reg__regressor__l2_regularization':[0.1, 1.0, 10.0],
            }
            # Tweedie‐loss grid (if your sklearn version supports it)
            # {
            # 'reg__regressor__loss': ['tweedie'],
            # 'reg__regressor__power':          [1.1, 1.5, 1.9],
            # 'reg__regressor__learning_rate':  [0.01, 0.05, 0.1],
            # 'reg__regressor__max_leaf_nodes': [15, 31, 63],
            # 'reg__regressor__l2_regularization':[0.0, 0.1, 1.0, 10.0],
            # },
        ]


    scoring = {
        'R2': 'r2',
        'MAE': make_scorer(mean_absolute_error, greater_is_better=False),
        'RMSE': 'neg_root_mean_squared_error',
    }
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=params,
        cv=tscv,
        scoring=scoring,
        refit='MAE',        # pick the metric you care about; MAE more stable
        n_jobs=-1,
        verbose=1,
        error_score='raise',  # fail fast if something breaks
    )
    model.fit(X_train, y_train)

    
    test_r2      = model.score(X_test, y_test)
    y_pred_test  = model.predict(X_test)
    const_base   = np.full_like(y_test, y_train.mean())
    metrics = {
        "best_params": model.best_params_,
        "cv_r2":       model.best_score_,
        "test_r2":     test_r2,
        "const_mae":   float(mean_absolute_error(y_test, const_base)),
        "model_mae":   float(mean_absolute_error(y_test, y_pred_test)),
    }
    print()
    print()
    for m in metrics:
        print(m, metrics[m])

    df_for_inf = df_clean[['stream_name'] + feats].copy()
    return model, model.best_estimator_, df_for_inf, feats, metrics

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC TRAIN WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def train_predictor(app, *, log_metrics: bool = True):
    """
    Explicitly train (on-dyno or offline), updating _predictor_state.
    """
    df_daily = _load_daily_stats_df(app)
    model, pipe, df_inf, feats, metrics = _train_model(df_daily)
    _predictor_state.update({
        "pipeline":                    pipe,
        "model":                       model,
        "df_for_inf":                  df_inf,
        "features":                    feats,
        "stream_category_options_inf": sorted(df_inf["game_category"].unique().tolist()),
        "optional_start_times":        DEFAULT_START_TIMES,
        "stream_duration_opts":        DEFAULT_DURATIONS_HRS,
        "trained_on":                  datetime.utcnow(),
        "metrics":                     metrics,
    })

    if log_metrics:
        logging.info("Predictor trained: %s", metrics)
    return metrics

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _get_last_row_for_stream(df_for_inf: pd.DataFrame, stream_name: str):
    # print(df_for_inf)
    rows = df_for_inf[df_for_inf["stream_name"] == stream_name]
    # print(rows)
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
    Returns a DataFrame sorted by predicted subs desc, with a 'conf' column.
    """
    if pipeline is None:
        raise RuntimeError("Predictor pipeline is not trained.")

    if today_name is None:
        today_name = datetime.now().strftime("%A")

    # grab the most recent features for this stream
    last_row = _get_last_row_for_stream(df_for_inf, stream_name)
    base = last_row.to_frame().T

    # determine which categories to try
    if category_options is None:
        category_options = sorted(df_for_inf["game_category"].dropna().unique().tolist())
        restrict_to_stream_game = True
    else:
        category_options = list(category_options)
        restrict_to_stream_game = False

    if start_times is None:
        start_times = DEFAULT_START_TIMES
    if durations is None:
        durations = DEFAULT_DURATIONS_HRS

    # build the grid of (game, hour, duration)
    combos = list(itertools.product(category_options, start_times, durations))
    grid = pd.DataFrame(combos, columns=['game_category','start_time_hour','stream_duration'])

    # replicate base row and overwrite dynamic columns
    base_rep = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
    for col in ['game_category','start_time_hour','stream_duration']:
        base_rep[col] = grid[col]
    base_rep["day_of_week"] = today_name

    # extract features and predict
    X_inf = base_rep[features]
    preds = pipeline.predict(X_inf)

    # --- compute per‐tree std‐dev as a confidence score ---
    pre = pipeline.named_steps['pre']
    rf  = pipeline.named_steps['reg']
    X_pre = pre.transform(X_inf)

    model = pipeline.named_steps['reg']
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    else:
        model = model

    # compute per‐tree std‐dev for forests; else fallback to NaN
    if hasattr(model, 'estimators_'):
        all_tree_preds = np.stack([t.predict(X_pre) for t in model.estimators_], axis=1)
        conf = np.std(all_tree_preds, axis=1)
    else:
        conf = np.full(len(X_pre), np.nan)
    


    # sigma = all_tree_preds.std(axis=1)
    # conf = 1.0 / (1.0 + sigma)

    # assemble results
    results = X_inf.copy()
    results['y_pred'] = preds
    results['conf']  = conf
    print('Conf', conf)

    # legacy: if user didn’t supply category_options, restrict back to last game
    if restrict_to_stream_game:
        game_cat = last_row["game_category"]
        results = results[results['game_category'] == game_cat]

    if start_hour_filter is not None:
        results = results[results['start_time_hour'] == start_hour_filter]

    results = results.sort_values('y_pred', ascending=False)

    if unique_scores:
        results = results.drop_duplicates(subset=['y_pred'], keep='first')

    return results.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# DASHBOARD EXPORTS
# ─────────────────────────────────────────────────────────────────────────────
def get_predictor_artifacts():
    """
    Return (pipeline, df_for_inf, features,
            stream_category_options_inf,
            optional_start_times,
            stream_duration_opts,
            metrics)
    """
    
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
# CONVENIENCE PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_trained(app=None):
    if _predictor_state["pipeline"] is None:
        if app is None:
            raise RuntimeError("Model not trained; pass `app` to train_predictor.")
        train_predictor(app)

def predgame(
    stream_name: str,
    *,
    topn: int = 5,
    app=None,
    **kwargs
):
    _ensure_trained(app)
    return _infer_grid_for_game(
        _predictor_state["pipeline"],
        _predictor_state["df_for_inf"],
        _predictor_state["features"],
        stream_name,
        top_n=topn,
        **kwargs
    )

def predhour(
    stream_name: str,
    hour: int,
    *,
    topn: int = 5,
    app=None,
    **kwargs
):
    _ensure_trained(app)
    return _infer_grid_for_game(
        _predictor_state["pipeline"],
        _predictor_state["df_for_inf"],
        _predictor_state["features"],
        stream_name,
        start_hour_filter=hour,
        top_n=topn,
        **kwargs
    )

def format_pred_rows(df: pd.DataFrame, include_hour: bool = True) -> str:
    parts = []
    for _, r in df.iterrows():
        if include_hour:
            parts.append(f"{int(r.start_time_hour):02d}:00 h{int(r.stream_duration)} → {r.y_pred:.1f}")
        else:
            parts.append(f"h{int(r.stream_duration)} → {r.y_pred:.1f}")
    return " | ".join(parts)


def plot_features(feats, X_train, y_train, train_mask, df_clean):
    feature = feats[0]  # ← replace with whichever column you want
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train[feature], y_train, alpha=0.6)
    plt.xlabel(feature)
    plt.ylabel("total_subscriptions")
    plt.title(f"total_subscriptions vs. {feature}")
    plt.tight_layout()
    plt.show()

    # 2) Plot subscriptions over time:
    dates = df_clean.loc[train_mask, "stream_date"]
    plt.figure(figsize=(10, 4))
    plt.plot(dates, y_train, marker="o", linestyle="-")
    plt.xlabel("stream_date")
    plt.ylabel("total_subscriptions")
    plt.title("Subscriptions over Time (Training Set)")
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# SCRIPT ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        from main import app
    except ImportError as e:
        raise SystemExit(f"Cannot import Flask app: {e}") from e

    m = train_predictor(app)
    print("Training complete:", m)
    last_stream = _predictor_state["df_for_inf"]["stream_name"].iloc[-1]
    print("Example predgame:", predgame(last_stream, topn=3))
    print("Example predhour:", predhour(last_stream, hour=20, topn=3))
