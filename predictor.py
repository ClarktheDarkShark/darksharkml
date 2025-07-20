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
from feature_engineering import _prepare_training_frame

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

    # normalize missing or non-list tags to empty list
    # df_daily['tags'] = df_daily['tags'].apply(lambda x: x if isinstance(x, list) else [])
    df_daily['raw_tags'] = df_daily.pop('tags').apply(lambda x: x if isinstance(x, list) else [])


    # explode to one tag per row, get dummies, then sum back into one row per original index
    df_tags = (
        df_daily['raw_tags']
        .explode()                       # one row per tag
        .str.get_dummies()               # one-hot encode
        .groupby(level=0)                # group back by original row index
        .sum()                           # 1 if tag was present, else 0
        .add_prefix('tag_')
    )

    df_daily = pd.concat([df_daily, df_tags], axis=1)
    # if you no longer need the list version:
    

    # 4) Lowercase game_category, etc.
    df_daily['game_category'] = df_daily['game_category'].str.lower()

    # DEBUG: confirm
    print("raw_tags sample:", df_daily['raw_tags'].head())
    df_daily.drop(columns=['raw_tags'], inplace=True)
    print("dummy‐tag columns:", [c for c in df_daily.columns if c.startswith('tag_')])


    # lowercase your game_category as before
    df_daily['game_category'] = df_daily['game_category'].str.lower()

    print(df_daily)
    return df_daily




# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────
def _train_model(df_daily: pd.DataFrame):
    df_clean, feats, _ = _prepare_training_frame(df_daily)

    tag_cols = [c for c in df_clean.columns if c.startswith('tag_')]
    feats = feats + tag_cols

    y = df_clean['total_subscriptions']
    # y = df_clean['net_follower_change']
    X = df_clean[feats]
    cutoff = df_clean["stream_date"].quantile(0.9)
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
    print('Train sample size:',X_train.shape)
    print('Test sample size:',X_test.shape)
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
            "reg__regressor__max_features":    ['sqrt', 0.5, 0.8, 1.0],
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
    vary_tags: bool = False,           # ← new flag
):
    """
    If vary_tags=False (default), does the usual game/hour/duration grid and returns
    a DataFrame of top_n rows with y_pred and conf, plus a 'tags' column pulled
    from the one-hot tag_* features.

    If vary_tags=True, ignores start_times/durations/category_options and instead
    flips each tag_* on/off in turn (holding everything else at its last-known
    values), predicts y_pred, and returns a DataFrame of tag effects sorted
    by delta_from_baseline.
    """
    if pipeline is None:
        raise RuntimeError("Predictor pipeline is not trained.")

    # 1) get the last-known feature row
    last_row = _get_last_row_for_stream(df_for_inf, stream_name)
    # collect your tag columns
    tag_cols = [c for c in features if c.startswith('tag_')]
    # base feature vector
    base = last_row[features].to_frame().T

    if vary_tags:
        # --- TAG EFFECT EXPERIMENT ---
        # baseline prediction
        y_base = pipeline.predict(base)[0]

        records = []
        for tag in tag_cols:
            mod = base.copy()
            # flip this tag
            mod[tag] = 1 if base[tag].iloc[0] == 0 else 0
            y_mod   = pipeline.predict(mod)[0]
            records.append({
                'tag': tag[len('tag_'):],               # strip prefix
                'y_pred': y_mod,
                'delta_from_baseline': y_mod - y_base
            })

        return (
            pd.DataFrame(records)
              .sort_values('delta_from_baseline', ascending=False)
              .reset_index(drop=True)
        )

    # --- ORIGINAL GRID-BASED INFERENCE ---
    if today_name is None:
        today_name = datetime.now().strftime("%A")

    # determine categories
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

    # build grid of (game, hour, duration)
    combos = list(itertools.product(category_options, start_times, durations))
    grid = pd.DataFrame(combos, columns=['game_category','start_time_hour','stream_duration'])

    # replicate base row and overwrite dynamic cols
    base_rep = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
    for col in ['game_category','start_time_hour','stream_duration']:
        base_rep[col] = grid[col]
    base_rep["day_of_week"] = today_name

    # predict
    X_inf = base_rep[features]
    preds  = pipeline.predict(X_inf)

    # confidence as before
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

    # assemble
    results = X_inf.copy()
    results['y_pred'] = preds
    results['conf']   = conf

    # pull tags from one-hot
    if tag_cols:
        results['tags'] = results[tag_cols].apply(
            lambda row: [c[len('tag_'):] for c,v in row.items() if v==1],
            axis=1
        )
    else:
        results['tags'] = [[]] * len(results)

    # legacy game restriction
    if restrict_to_stream_game:
        game_cat = last_row["game_category"]
        results = results[results['game_category'] == game_cat]

    if start_hour_filter is not None:
        results = results[results['start_time_hour'] == start_hour_filter]

    # sort & dedupe
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
