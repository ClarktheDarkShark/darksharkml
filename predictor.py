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
import pytz

from db import db
from models import DailyStats, TimeSeries  # TimeSeries kept for possible future extension
from pipeline import _build_pipeline
from feature_engineering import _prepare_training_frame, drop_outliers

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
    # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
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

_TWO_PI = 2 * np.pi
EST = pytz.timezone("US/Eastern")

def add_time_features(df: pd.DataFrame) -> None:
    # 1) Ensure the hour column exists
    if 'start_time_hour' not in df.columns:
        df['start_time_hour'] = df['stream_start_time'].apply(lambda t: t.hour)

    # 2) **Cast to numeric** so we don’t end up with object‑dtype
    df['start_time_hour'] = pd.to_numeric(df['start_time_hour'], errors='coerce')

    # 3) Now compute the cyclic features
    df['start_hour_sin'] = np.sin(_TWO_PI * df['start_time_hour'] / 24)
    df['start_hour_cos'] = np.cos(_TWO_PI * df['start_time_hour'] / 24)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _load_daily_stats_df(app):
    """Read the entire daily_stats table into a DataFrame, with string column names,
    and one-hot encode the list-of-tags into separate columns."""
    with app.app_context():
        df_daily = pd.read_sql_table(DailyStats.__tablename__, con=db.engine)
    # ensure columns are strings
    df_daily.columns = df_daily.columns.map(str)
    add_time_features(df_daily)


    df_daily['raw_tags'] = df_daily['tags'].apply(lambda x: x if isinstance(x, list) else [])
    
    # lowercase your game_category as before
    df_daily['game_category'] = df_daily['game_category'].str.lower()

    print(df_daily)
    return df_daily




# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────
def _train_model(df_daily: pd.DataFrame):
    df_clean, feats, _ = _prepare_training_frame(df_daily)
    df_daily = drop_outliers(df_daily, method='iqr', factor=1.5)

    tag_cols = [c for c in df_clean.columns if c.startswith('tag_')]
    feats = feats + tag_cols

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

    print()
    # print(X_train)
    print('Train sample size:',X_train['start_time_hour'].tail())
    print('Test sample size:',X_test['start_time_hour'].tail())
    print()
    # print(y_train.describe())
    # print(X_train.nunique())

    pipeline, mod = _build_pipeline(X_train)
    
    tscv = TimeSeriesSplit(n_splits=5, gap=0)  # increase gap if leakage via lagged feats

    if mod == 'rf':

        # # BaggingRegressor wraps the TTR->HGB, so target inner params under estimator__regressor
        # pipeline, _ = _build_pipeline(X_train)
        # for p in sorted(pipeline.get_params().keys()):
        #     if p.startswith("reg__"):
        #         print(p)
        params = {
            "reg__regressor__n_estimators":    [200, 600, 1000],
            "reg__regressor__min_samples_leaf":[3, 5, 10, 50],
            "reg__regressor__min_samples_split":[5, 10, 50],
            "reg__regressor__max_features":    ['sqrt', 0.5, .8, 1.0],
            "reg__regressor__max_depth": [None, 5, 10]
        }
    elif mod == 'hgb':
        # # BaggingRegressor wraps the TTR->HGB, so target inner params under estimator__regressor
        # pipeline, _ = _build_pipeline(X_train)
        # for p in sorted(pipeline.get_params().keys()):
        #     if p.startswith("reg__"):
        #         print(p)
        params = [
            {
                'reg__loss': ['poisson'],
                'reg__learning_rate': [0.05, 0.1, 0.2],
                'reg__max_leaf_nodes': [31, 63, 80],
                'reg__l2_regularization': [0.1, 1.0, 10.0]
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

    preds = model.predict(X_test)
    print(preds)
    return model, model.best_estimator_, df_for_inf, feats, metrics

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC TRAIN WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def train_predictor(app, *, log_metrics=True):
    # 1) load
    df_daily = _load_daily_stats_df(app)

    # 2) **add hour + cyclic features**
    add_time_features(df_daily)

    # 3) train on that enriched frame
    model, pipe, df_inf, feats, metrics = _train_model(df_daily)

    # 4) update state (no more DEFAULT_START_TIMES)
    _predictor_state.update({
        "pipeline":                 pipe,
        "model":                    model,
        "df_for_inf":               df_inf,
        "features":                 feats,
        "stream_category_options_inf": sorted(df_inf["game_category"].unique().tolist()),
        "stream_duration_opts":     DEFAULT_DURATIONS_HRS,
        "trained_on":               datetime.utcnow(),
        "metrics":                  metrics,
    })

    # 5) capture **only** the hours we actually saw
    observed_hours = sorted(
        df_daily['start_time_hour']
                .dropna()
                .astype(int)
                .unique()
                .tolist()
    )
    _predictor_state["optional_start_times"] = observed_hours

    print()
    print('Optional Start Times:')
    print(_predictor_state["optional_start_times"])

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
    override_tags: Optional[list[str]] = None,
    start_times: Optional[Iterable[int]] = None,
    durations: Optional[Iterable[int]] = None,
    category_options: Optional[Iterable[str]] = None,
    today_name: Optional[str] = None,
    top_n: int = 10,
    unique_scores: bool = True,
    start_hour_filter: Optional[int] = None,
    vary_tags: bool = False,           # ← new flag
):
    if pipeline is None:
        raise RuntimeError("Predictor pipeline is not trained.")

    # 1) get the last-known feature row
    last_row = _get_last_row_for_stream(df_for_inf, stream_name)
    base = last_row[features].to_frame().T
    add_time_features(base)

    # apply any override_tags (for manual tuning)
    tag_cols = [c for c in features if c.startswith('tag_')]
    if override_tags is not None and not vary_tags:
        for col in tag_cols:
            tag = col[len("tag_"):]
            base[col] = 1 if tag in override_tags else 0

    # ————————————— TAG EFFECT EXPERIMENT (vary_tags) —————————————
    if vary_tags:
        y_base = pipeline.predict(base)[0]
        records = []

        # 1) if you have raw_tags lists in df_for_inf, use those
        if 'raw_tags' in df_for_inf.columns:
            # collect every tag ever seen
            all_tags = sorted({
                t
                for tags in df_for_inf['raw_tags'].fillna([])
                for t in tags
            })
            for tag in all_tags:
                mod = base.copy()
                tags_list = list(mod.at[0, 'raw_tags'] or [])
                if tag in tags_list:
                    tags_list.remove(tag)
                else:
                    tags_list.append(tag)
                mod.at[0, 'raw_tags'] = tags_list
                y_mod = pipeline.predict(mod)[0]
                records.append({
                    'tag': tag,
                    'y_pred': y_mod,
                    'delta_from_baseline': y_mod - y_base
                })

        # 2) fallback to one-hot tag_* if raw_tags isn’t available
        elif tag_cols:
            for col in tag_cols:
                mod = base.copy()
                mod[col] = 1 - mod[col].iloc[0]
                y_mod = pipeline.predict(mod)[0]
                records.append({
                    'tag': col[len('tag_'):],
                    'y_pred': y_mod,
                    'delta_from_baseline': y_mod - y_base
                })

        # 3) if no tags at all, return an empty frame with the right columns
        else:
            return pd.DataFrame(columns=['tag', 'y_pred', 'delta_from_baseline'])

        return (
            pd.DataFrame(records)
              .sort_values('delta_from_baseline', ascending=False)
              .reset_index(drop=True)
        )
    # ————————————————————————————————————————————————————————————————

    # --- ORIGINAL GRID-BASED INFERENCE (unchanged) ---
    if today_name is None:
        est = pytz.timezone("US/Eastern")
        today_name = datetime.now(est).strftime("%A")

    if category_options is None:
        category_options = sorted(df_for_inf["game_category"].dropna().unique().tolist())
        restrict_to_stream_game = True
    else:
        category_options = list(category_options)
        restrict_to_stream_game = False

    if start_times is None:
        start_times = _predictor_state['optional_start_times']
    if durations is None:
        durations = DEFAULT_DURATIONS_HRS

    combos = list(itertools.product(category_options, start_times, durations))
    grid = pd.DataFrame(combos, columns=['game_category','start_time_hour','stream_duration'])

    base_rep = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
    for col in ['game_category','start_time_hour','stream_duration']:
        base_rep[col] = grid[col]
    add_time_features(base_rep)

    X_inf = base_rep[features]
    preds  = pipeline.predict(X_inf)

    pre    = pipeline.named_steps['pre']
    X_pre  = pre.transform(X_inf)
    model  = pipeline.named_steps['reg']
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if hasattr(model, 'estimators_'):
        all_tree_preds = np.stack([t.predict(X_pre) for t in model.estimators_], axis=1)
        sigma = all_tree_preds.std(axis=1)
    elif hasattr(model, 'staged_predict'):
        staged  = np.stack(list(model.staged_predict(X_pre)), axis=1)
        sigma   = np.diff(staged, axis=1).std(axis=1)
    else:
        sigma = np.full(len(X_pre), fill_value=np.mean(preds)*0.01)

    conf = 1.0 / (1.0 + sigma)

    results = X_inf.copy()
    results['y_pred'] = preds
    results['conf']   = conf

    if tag_cols:
        results['tags'] = results[tag_cols].apply(
            lambda row: [c[len('tag_'):] for c,v in row.items() if v==1],
            axis=1
        )
    else:
        results['tags'] = [[]] * len(results)

    if restrict_to_stream_game:
        game_cat = last_row["game_category"]
        results = results[results['game_category'] == game_cat]
    if start_hour_filter is not None:
        results = results[results['start_time_hour'] == start_hour_filter]

    results = results.sort_values('y_pred', ascending=False)
    results = results.drop_duplicates(
        subset=['y_pred', 'start_time_hour'],
        keep='first'
    )

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


def debugging_code(pipeline, X_train):
    # ── DEBUG: inspect the preprocessor output ────────────────────────────
    pre = pipeline.named_steps['pre']

    # tell it to produce a DataFrame with column names (sklearn ≥1.2)
    pre.set_output(transform="pandas")

    # fit & transform on the training set
    X_debug = pre.fit_transform(X_train)

    # print a sample and the feature names
    with pd.option_context('display.max_rows', None):
        print("\n>>> [DEBUG] Transformed features (row 0, transposed):")
        print(X_debug.tail(5).T.round(4))

    # revert to default (so GridSearchCV still sees an ndarray)
    pre.set_output(transform="default")