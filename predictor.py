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
from sklearn.pipeline import Pipeline
import pytz

from db import db
from models import DailyStats, TimeSeries  # TimeSeries kept for possible future extension
from pipeline import _build_pipeline
from feature_engineering import _prepare_training_frame, drop_outliers

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROLL_WINDOWS = [1, 3, 10]
DEFAULT_START_TIMES   = list(range(24))     # 0..23 hours
DEFAULT_DURATIONS_HRS = list(range(2, 13))  # 2..12 hours
_ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "predictor_artifacts.joblib")

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL STATE
# ─────────────────────────────────────────────────────────────────────────────
_predictor_state = {
    "pipelines": [],          # will hold N best_estimator_ pipelines
    "metrics_list": [],       # will hold N corresponding metrics dicts
    "model": None,                         # full GridSearchCV wrapper
    "df_for_inf": None,                    # cleaned feature frame incl stream_name
    "features": None,                      # feature column list
    "stream_category_options_inf": None,   # sorted list of known categories
    "optional_start_times": DEFAULT_START_TIMES,
    "stream_duration_opts": DEFAULT_DURATIONS_HRS,
    "trained_on": None                    # UTC datetime
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
            "pipelines":                   data.get("pipelines", []),
            "metrics_list":                data.get("metrics_list", []),
            "model":                       None,  # drop full GridSearchCV at runtime
            "df_for_inf":                  df_inf,
            "features":                    data.get("features"),
            "stream_category_options_inf": data.get("stream_category_options_inf"),
            "optional_start_times":        data.get("optional_start_times", DEFAULT_START_TIMES),
            "stream_duration_opts":        data.get("stream_duration_opts", DEFAULT_DURATIONS_HRS),
            "trained_on":                  data.get("trained_on", None),
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

    return df

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
    df_daily = add_time_features(df_daily)


    df_daily['raw_tags'] = df_daily['tags'].apply(lambda x: x if isinstance(x, list) else [])
    
    # lowercase your game_category as before
    df_daily['game_category'] = df_daily['game_category'].str.lower()

    # print(df_daily)
    return df_daily




# ─────────────────────────────────────────────────────────────────────────────
# TRAINING (GridSearchCV)
# ─────────────────────────────────────────────────────────────────────────────
def _train_model(df_daily: pd.DataFrame):
    df_clean, feats, hist_cols = _prepare_training_frame(df_daily)
    df_clean = df_clean.dropna()
    # df_clean = drop_outliers(df_clean, cols=['total_subscriptions', 'avg_concurrent_viewers', 'net_follower_change'] ,method='iqr', factor=1.5)


    # target_list = ['total_subscriptions', 'avg_concurrent_viewers', 'net_follower_change']

    target_list = ['total_subscriptions', 'net_follower_change', 'avg_concurrent_viewers']
    pipe_list = []
    # print(df_clean["net_follower_change"])
    for t in target_list:
        y = df_clean[t]
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


        pipeline, mod = _build_pipeline(X_train)
        # if t == "net_follower_change":
        #     # pipeline.named_steps['reg'] is the TransformedTargetRegressor
        #     ttr = pipeline.named_steps["reg"]
        #     raw_reg = ttr.regressor_
        #     # rebuild exactly the same pipeline but swap in the raw regressor
        #     pipeline = Pipeline([*pipeline.steps[:-1], ("reg", raw_reg)])
        
        # tscv = TimeSeriesSplit(n_splits=5, gap=0)  # increase gap if leakage via lagged feats
        tscv = TimeSeriesSplit(n_splits=5, gap=0)

        if mod == 'rf':

            # # BaggingRegressor wraps the TTR->HGB, so target inner params under estimator__regressor
            # pipeline, _ = _build_pipeline(X_train)
            # for p in sorted(pipeline.get_params().keys()):
            #     if p.startswith("reg__"):
            #         print(p)
            params = {
                'reg__regressor__n_estimators':    [500, 1000, 5000],
                'reg__regressor__max_depth':       [5, 10, 15],
                'reg__regressor__max_features':    ['sqrt', 'log2', 0.8],
                'reg__regressor__min_samples_split': [5, 10],
                'reg__regressor__min_samples_leaf':  [2, 5, 10],
                'reg__regressor__bootstrap':       [True, False]
                }
        elif mod == 'hgb':
            # # BaggingRegressor wraps the TTR->HGB, so target inner params under estimator__regressor
            # pipeline, _ = _build_pipeline(X_train)
            # for p in sorted(pipeline.get_params().keys()):
            #     if p.startswith("reg__"):
            #         print(p)
            params = [
                # {
                #     'reg__loss': ['poisson'],
                #     'reg__learning_rate': [0.05, 0.1, 0.2],
                #     'reg__max_leaf_nodes': [31, 63, 80],
                #     'reg__l2_regularization': [0.1, 1.0, 10.0]
                # }
                # Tweedie‐loss grid (if your sklearn version supports it)
                {
                # 'reg__regressor__loss': [
                #     'squared_error',    # classic MSE
                #     'absolute_error',   # MAE
                # ],
                'reg__regressor__loss': ['poisson'],
                # 'reg__regressor__loss': ['tweedie'],
                # 'reg__regressor__power':          [1.1, 1.5, 1.9],
                'reg__regressor__learning_rate':  [0.01, 0.05, 0.1],
                'reg__regressor__max_leaf_nodes': [15, 31, 63],
                'reg__regressor__l2_regularization':[0.0, 0.1, 1.0, 10.0],
                },
            ]
        elif mod == 'svr':
            # tune the SVM’s C, epsilon and kernel
            params = {
                'reg__regressor__kernel':  ['rbf', 'poly', 'sigmoid'],
                'reg__regressor__C':       [0.1, 1, 10, 100],
                'reg__regressor__gamma':   ['scale', 'auto', 1e-3, 1e-2, 1e-1],
                'reg__regressor__epsilon': [0.01, 0.1, 0.5],
                'reg__regressor__degree':  [2, 3]          # only used if kernel='poly'
            }


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

        
        test_r2      = model.best_estimator_.score(X_test, y_test)
        y_pred_test  = model.best_estimator_.predict(X_test)
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
        print('For target:', t)
        for m in metrics:
            print(m, metrics[m])

        import matplotlib.pyplot as plt
        from sklearn.metrics import r2_score

        y_true = y_test
        y_pred = model.best_estimator_.predict(X_test)

        plt.scatter(y_true, y_pred, alpha=0.3)
        plt.plot([y_true.min(), y_true.max()],
                [y_true.min(), y_true.max()],
                'k--', lw=2)
        plt.xlabel("Actual subscriptions")
        plt.ylabel("Predicted subscriptions")
        plt.title(f"R² = {r2_score(y_true, y_pred):.2f}")
        plt.show()

        df_for_inf = df_clean[['stream_name'] + feats].copy()

        preds = model.predict(X_test)
        # print(preds)
        pipe_list.append((model.best_estimator_, metrics))


    return model, pipe_list, df_for_inf, feats


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC TRAIN WRAPPER
# ─────────────────────────────────────────────────────────────────────────────
def train_predictor(app, *, log_metrics=True):
    # 1) load
    df_daily = _load_daily_stats_df(app)

    # 3) train on that enriched frame
    model, pipe_list, df_inf, feats = _train_model(df_daily)

    # 4) update state (no more DEFAULT_START_TIMES)
    pipelines, metrics_list = zip(*pipe_list)
    _predictor_state.update({
        "pipelines":     list(pipelines),
        "metrics_list":  list(metrics_list),
        "model":                    model,
        "df_for_inf":               df_inf,
        "features":                 feats,
        "stream_category_options_inf": sorted(df_inf["game_category"].unique().tolist()),
        "stream_duration_opts":     DEFAULT_DURATIONS_HRS,
        "trained_on":               datetime.utcnow(),
        "metrics":                  pipe_list[0][1],
        "metrics2":                 pipe_list[1][1]
    })

    # 5) capture **only** the hours we actually saw
    observed_hours = sorted(
        df_daily['start_time_hour']
                .dropna()
                .astype(int)
                .unique()
                .tolist()
    )
    # _predictor_state["optional_start_times"] = observed_hours
    _predictor_state["optional_start_times"] = DEFAULT_START_TIMES

    # print()
    # print('Optional Start Times:')
    # print(_predictor_state["optional_start_times"])

    if log_metrics:
        logging.info("Predictor trained for subs: %s", pipe_list[0][1])
        logging.info("Predictor trained for follows: %s", pipe_list[1][1])
    return pipe_list[0][1]





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
    override_tags: Optional[List[str]] = None,
    start_times: Optional[Iterable[int]] = None,
    durations: Optional[Iterable[int]] = None,
    category_options: Optional[Iterable[str]] = None,
    today_name: Optional[str] = None,
    top_n: int = 10,
    unique_scores: bool = True,
    start_hour_filter: Optional[int] = None,
    vary_tags: bool = False,
):
    """
    If vary_tags=False, returns top_n rows over the (game, hour, duration) grid,
    with predictions + confidence + tags list.
    If vary_tags=True, flips each tag in turn on/off in the raw_tags list,
    measures delta_from_baseline, and returns those effects.
    """
    if pipeline is None:
        raise RuntimeError("Predictor pipeline is not trained.")

    # 1) grab the last-known feature row for this stream
    last = _get_last_row_for_stream(df_for_inf, stream_name)

    # 2) build a 1-row DataFrame and insert raw_tags as a single-cell list
    non_tag_feats = [f for f in features if f != "raw_tags"]
    base = last[non_tag_feats].to_frame().T

    # wrap in [ … ] so pandas sees one element
    base["raw_tags"] = [ last["raw_tags"] ]

    # 4) if override_tags is provided, replace the list
    if override_tags is not None and not vary_tags:
        # again, wrap in a list so we get a single‐cell column
        base["raw_tags"] = [ override_tags ]

    # 5) TAG‐FLIPPING mode?
    if vary_tags:
        # baseline prediction
        y_base = pipeline.predict(base)[0]

        # universe of all tags ever seen
        all_tags = sorted({t for tags in df_for_inf["raw_tags"] for t in tags})

        records = []
        for tag in all_tags:
            mod = base.copy()
            current = set(mod.at[mod.index[0], "raw_tags"])
            # flip this tag’s presence
            if tag in current:
                current.remove(tag)
            else:
                current.add(tag)
            mod.at[mod.index[0], "raw_tags"] = list(current)

            y_mod = pipeline.predict(mod)[0]
            records.append({
                "tag": tag,
                "y_pred": y_mod,
                "delta_from_baseline": y_mod - y_base
            })

        return (
            pd.DataFrame(records)
              .sort_values("delta_from_baseline", ascending=False)
              .reset_index(drop=True)
        )

    # 6) GRID‐BASED mode: build (game, hour, duration) grid
    if today_name is None:
        today_name = datetime.now(EST).strftime("%A")

    base["day_of_week"] = today_name  
    base["is_weekend"] = today_name in ("Saturday", "Sunday")

    if category_options is None:
        category_options = sorted(df_for_inf["game_category"].dropna().unique().tolist())
        restrict_to_stream_game = True
    else:
        category_options = list(category_options)
        restrict_to_stream_game = False

    if start_times is None:
        start_times = _predictor_state["optional_start_times"]
    if durations is None:
        durations = DEFAULT_DURATIONS_HRS

    import itertools
    combos = list(itertools.product(category_options, start_times, durations))
    grid = pd.DataFrame(combos, columns=["game_category","start_time_hour","stream_duration"])

    # repeat base row for each combo, then overwrite the 3 dynamic columns
    base_rep = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
    for col in ["game_category","start_time_hour","stream_duration"]:
        base_rep[col] = grid[col]
    
    print()
    base_rep = add_time_features(base_rep)

    # predict
    X_inf = base_rep[features]
    print(X_inf[['game_category','stream_duration',
             'avg_total_subscriptions_last_3']].head())

    preds  = pipeline.predict(X_inf)

    # approximate confidence via tree‑ensemble σ
    pre  = pipeline.named_steps["pre"]
    X_pre = pre.transform(X_inf)
    model = pipeline.named_steps["reg"]
    from sklearn.compose import TransformedTargetRegressor
    if isinstance(model, TransformedTargetRegressor):
        model = model.regressor_
    if hasattr(model, "estimators_"):
        all_tree_preds = np.stack([t.predict(X_pre) for t in model.estimators_], axis=1)
        sigma = all_tree_preds.std(axis=1)
    elif hasattr(model, "staged_predict"):
        staged = np.stack(list(model.staged_predict(X_pre)), axis=1)
        sigma  = np.diff(staged, axis=1).std(axis=1)
    else:
        sigma = np.full(len(X_pre), fill_value=np.mean(preds) * 0.01)
    conf = 1.0 / (1.0 + sigma)

    # assemble results
    results = X_inf.copy()
    results["y_pred"] = preds
    results["conf"]   = conf

    # pull tags back out of the pipeline’s encoded features
    # by re-using raw_tags column
    results["tags"] = base_rep["raw_tags"]
    results["start_time_hour"] = base_rep["start_time_hour"].values

    # optionally restrict to the stream’s own game category
    if restrict_to_stream_game:
        this_game = last["game_category"]
        results = results[results["game_category"] == this_game]

    # filter a single hour if requested
    if start_hour_filter is not None:
        results = results[results["start_time_hour"] == start_hour_filter]

    # sort & drop duplicate scores for the same hour
    results = results.sort_values("y_pred", ascending=False)
    # results = results.drop_duplicates(
    #     subset=["y_pred"],
    #     keep="first"
    # )
    if unique_scores:
        results = results.drop_duplicates(
            subset=["start_time_hour"],
            keep="first",
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
_predictor_state["pipelines"],
_predictor_state["df_for_inf"],
_predictor_state["features"],
_predictor_state["stream_category_options_inf"],
_predictor_state["optional_start_times"],
_predictor_state["stream_duration_opts"],
_predictor_state["metrics_list"],
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_trained(app=None):
    if _predictor_state["pipelines"] is None:
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
        _predictor_state["pipeline2"],
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
        _predictor_state["pipeline2"],
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