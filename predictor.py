"""
Predictor (Inference Only – no Twitch, no on-dyno training)

This module loads a pre-trained prediction pipeline and supporting artifacts
from ``predictor_artifacts.joblib`` and exposes lightweight inference
utilities used by the dashboard blueprint.

Expected artifact dict keys in predictor_artifacts.joblib:
    {
        "pipeline": <sklearn Pipeline>,
        "df_for_inf": <pandas DataFrame with stream_name + feature cols>,
        "features": <list[str]>,
        "stream_category_options_inf": <list[str]>,
        "optional_start_times": <iterable[int]>,
        "stream_duration_opts": <iterable[int]>,
        "metrics": <dict>  # optional
    }

Dashboard import contract (do not rename without updating dashboard_predictions.py):
    from predictor import get_predictor_artifacts, _infer_grid_for_game
"""

from __future__ import annotations

import itertools
import logging
import os
from datetime import datetime
from typing import Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS (used if not supplied in artifact bundle)
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_START_TIMES   = list(range(24))     # 0..23 hours
DEFAULT_DURATIONS_HRS = list(range(2, 13))  # 2..12 hours


# ─────────────────────────────────────────────────────────────────────────────
# IN-MEMORY STATE
# ─────────────────────────────────────────────────────────────────────────────
_predictor_state = {
    "pipeline": None,                      # fitted sklearn Pipeline
    "model": None,                         # legacy (unused)
    "df_for_inf": None,                    # feature DataFrame incl stream_name
    "features": None,                      # list of feature column names
    "stream_category_options_inf": None,   # inferred game categories
    "optional_start_times": DEFAULT_START_TIMES,
    "stream_duration_opts": DEFAULT_DURATIONS_HRS,
    "trained_on": None,                    # timestamp (optional)
    "metrics": {},                         # training metrics (optional)
}


# ─────────────────────────────────────────────────────────────────────────────
# LOAD PRE-TRAINED ARTIFACTS (at import)
# ─────────────────────────────────────────────────────────────────────────────
_ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "predictor_artifacts.joblib")

def _load_artifacts_from_disk(path: str = _ARTIFACT_PATH) -> bool:
    """Internal: load artifact bundle; return True if loaded."""
    if not os.path.exists(path):
        logging.warning("Predictor artifact not found at %s; running without model.", path)
        return False

    try:
        data = joblib.load(path)
    except Exception as exc:  # pragma: no cover
        logging.exception("Failed to load predictor artifact %s: %s", path, exc)
        return False

    # Defensive: coerce columns to string names
    df_for_inf = data.get("df_for_inf")
    if isinstance(df_for_inf, pd.DataFrame):
        df_for_inf.columns = df_for_inf.columns.map(str)

    _predictor_state.update({
        "pipeline":                    data.get("pipeline"),
        "model":                       None,  # no grid object at runtime
        "df_for_inf":                  df_for_inf,
        "features":                    data.get("features"),
        "stream_category_options_inf": data.get("stream_category_options_inf"),
        "optional_start_times":        data.get("optional_start_times", DEFAULT_START_TIMES),
        "stream_duration_opts":        data.get("stream_duration_opts", DEFAULT_DURATIONS_HRS),
        "trained_on":                  data.get("trained_on"),
        "metrics":                     data.get("metrics", {}),
    })
    logging.info("Loaded predictor artifacts from %s.", path)
    return True

# load immediately
_load_artifacts_from_disk()


# ─────────────────────────────────────────────────────────────────────────────
# BASIC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _get_last_row_for_stream(df_for_inf: pd.DataFrame, stream_name: str):
    """Return the most recent feature row for ``stream_name``."""
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
    Build an inference grid anchored on the *last* row for the specified stream.

    Behavior:
    • If ``category_options`` is provided, we evaluate those categories only.
    • If ``category_options`` is None, we use *all* categories in df_for_inf
      **and then** restrict to the stream's most recent game_category
      (legacy behavior preserved for callers that don't specify a game).
    • If ``start_hour_filter`` is provided, we filter the results to that hour.
    • Results sorted descending by predicted subs; top_n returned.

    Returns
    -------
    pandas.DataFrame
        Columns include the feature columns + 'y_pred' (float).
    """
    if pipeline is None:
        raise RuntimeError("Predictor pipeline is not loaded. Did you bundle predictor_artifacts.joblib?")

    if df_for_inf is None or features is None:
        raise RuntimeError("Predictor artifacts incomplete (df_for_inf or features missing).")

    if today_name is None:
        today_name = datetime.now().strftime("%A")

    last_row = _get_last_row_for_stream(df_for_inf, stream_name)
    base = last_row.to_frame().T  # single-row DataFrame

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

    dyn_cols = ["game_category", "start_time_hour", "stream_duration"]
    combos = list(itertools.product(category_options, start_times, durations))
    grid = pd.DataFrame(combos, columns=dyn_cols)

    # replicate base row across grid; overwrite dynamic columns
    base_rep = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
    for col in dyn_cols:
        base_rep[col] = grid[col]

    # set the "what if I stream today?" day-of-week
    base_rep["day_of_week"] = today_name

    # slice to feature order expected by the trained pipeline
    X_inf = base_rep[features]
    preds = pipeline.predict(X_inf)

    results = X_inf.copy()
    results["y_pred"] = preds

    # Legacy: restrict to stream's last game unless caller specified categories
    if restrict_to_stream_game:
        game_cat = last_row["game_category"]
        results = results[results["game_category"] == game_cat]

    if start_hour_filter is not None:
        results = results[results["start_time_hour"] == start_hour_filter]

    results = results.sort_values("y_pred", ascending=False)

    if unique_scores:
        results = results.drop_duplicates(subset=["y_pred"], keep="first")

    return results.head(top_n).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────
def get_predictor_artifacts():
    """
    Return the loaded artifacts tuple expected by the dashboard blueprint.

    (pipeline, df_for_inf, features, category_opts, start_opts, dur_opts, metrics)
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
# NO-OP TRAINING STUB (for backward compatibility)
# ─────────────────────────────────────────────────────────────────────────────
def train_predictor(*_, **__):  # pylint: disable=unused-argument
    """
    Training is disabled in this inference-only build.

    This stub exists to avoid import errors if legacy code tries to call it.
    It simply returns the current metrics (if any) and logs a warning.
    """
    logging.warning("train_predictor() called but training is disabled; using pre-trained artifacts only.")
    return _predictor_state.get("metrics", {})


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE HELPERS (replacement for old Twitch commands)
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_loaded():
    if _predictor_state["pipeline"] is None:
        raise RuntimeError(
            "Predictor artifacts not loaded. Ensure predictor_artifacts.joblib "
            "is present and readable at import time."
        )

def predgame(
    stream_name: str,
    *,
    topn: int = 5,
    today_name: Optional[str] = None,
    start_times: Optional[Iterable[int]] = None,
    durations: Optional[Iterable[int]] = None,
    category_options: Optional[Iterable[str]] = None,
    unique_scores: bool = True,
):
    """Return top-N start-time/duration combos (legacy !predgame)."""
    _ensure_loaded()
    return _infer_grid_for_game(
        _predictor_state["pipeline"],
        _predictor_state["df_for_inf"],
        _predictor_state["features"],
        stream_name=stream_name,
        start_times=start_times or _predictor_state["optional_start_times"],
        durations=durations or _predictor_state["stream_duration_opts"],
        category_options=category_options,  # None => legacy stream-game behavior
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
    today_name: Optional[str] = None,
    start_times: Optional[Iterable[int]] = None,
    durations: Optional[Iterable[int]] = None,
    category_options: Optional[Iterable[str]] = None,
    unique_scores: bool = True,
):
    """Return top-N duration combos at a fixed start hour (legacy !predhour)."""
    _ensure_loaded()
    return _infer_grid_for_game(
        _predictor_state["pipeline"],
        _predictor_state["df_for_inf"],
        _predictor_state["features"],
        stream_name=stream_name,
        start_times=start_times or _predictor_state["optional_start_times"],
        durations=durations or _predictor_state["stream_duration_opts"],
        category_options=category_options,
        today_name=today_name,
        top_n=topn,
        unique_scores=unique_scores,
        start_hour_filter=hour,
    )


def format_pred_rows(df: pd.DataFrame, include_hour: bool = True) -> str:
    """Compact string formatter approximating old Twitch chat output."""
    parts = []
    for _, r in df.iterrows():
        if include_hour:
            parts.append(f"{int(r.start_time_hour):02d}:00 h{int(r.stream_duration)} → {r.y_pred:.1f}")
        else:
            parts.append(f"h{int(r.stream_duration)} → {r.y_pred:.1f}")
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CLI DEMO
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _ensure_loaded()
    df_inf = _predictor_state["df_for_inf"]
    if df_inf is None or df_inf.empty:
        print("Predictor loaded but df_for_inf is empty; nothing to demo.")
    else:
        example_stream = df_inf["stream_name"].iloc[-1]
        print(f"Demo: top 5 predictions for {example_stream!r}")
        print(predgame(example_stream, topn=5))
        print(f"\nDemo: top 5 @20:00 for {example_stream!r}")
        print(predhour(example_stream, hour=20, topn=5))
