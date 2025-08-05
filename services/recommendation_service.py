# app/services/recommendation_service.py
from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from threading import Thread, Event
from typing import Dict, List

import pytz

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
    _get_last_row_for_stream,  # still handy elsewhere
)

# ────────────────────────────── Globals ──────────────────────────────
TZ = pytz.timezone("US/Eastern")

# In-memory store for pre-computed recommendations
_recommendation_cache: Dict[str, dict] = {}


# ─────────────────────── Artifact loading (once) ─────────────────────
@lru_cache(maxsize=1)
def _load_artifacts():
    """
    Load pipelines, inference grid, and feature metadata only once.
    The LRU cache keeps them in RAM for the life of the dyno.
    """
    return get_predictor_artifacts()  # -> pipes, df_inf, feats, …


# Warm the model artefacts as soon as the module is imported
_load_artifacts()


# ───────────────────────── Core helpers ──────────────────────────────
def _top_n(
    pipe,
    df_inf,
    feats,
    stream_name: str,
    legend_tags: List[str],
    start_opts: List[int],
    dur_opts: List[int],
    legend_games: List[str],
    n: int = 3,
):
    """Return `n` best rows (as dicts) for a single metric/model."""
    df = _infer_grid_for_game(
        pipe,
        df_inf,
        feats,
        stream_name=stream_name,
        override_tags=legend_tags,
        start_times=start_opts,
        durations=dur_opts,
        category_options=legend_games,
        top_n=n,
        unique_scores=True,
    )
    return df.to_dict("records")


def _compute_recommendations_for_stream(stream_name: str) -> dict:
    """
    Heavy: run the three pipelines for a single streamer
    and package the ‘top-3’ tables.
    """
    pipes, df_inf, feats, _, start_opts, dur_opts, _ = _load_artifacts()

    # Guard: artefacts missing
    if not pipes or df_inf is None:
        raise RuntimeError("Model artefacts not loaded.")

    # Default to most frequent streamer if caller passed ""
    if stream_name not in df_inf["stream_name"].unique():
        stream_name = df_inf["stream_name"].mode()[0]

    legend_games = (
        df_inf.loc[df_inf.stream_name == stream_name, "game_category"].unique().tolist()
    )
    legend_tags = sorted(
        {
            t
            for tags in df_inf.loc[df_inf.stream_name == stream_name, "raw_tags"]
            .dropna()
            for t in tags
        }
    )

    # Select the prediction pipelines
    pipe_sub = pipes[0]
    pipe_fol = pipes[1] if len(pipes) > 1 else pipes[0]
    pipe_view = pipes[2] if len(pipes) > 2 else pipes[0]

    # Bundle it up
    return {
        "selected_stream": stream_name,
        "today_name": datetime.now(TZ).strftime("%A, %B %d, %Y"),
        "top3_subs": _top_n(
            pipe_sub,
            df_inf,
            feats,
            stream_name,
            legend_tags,
            start_opts,
            dur_opts,
            legend_games,
        ),
        "top3_followers": _top_n(
            pipe_fol,
            df_inf,
            feats,
            stream_name,
            legend_tags,
            start_opts,
            dur_opts,
            legend_games,
        ),
        "top3_viewers": _top_n(
            pipe_view,
            df_inf,
            feats,
            stream_name,
            legend_tags,
            start_opts,
            dur_opts,
            legend_games,
        ),
    }


# ───────────────────── Pre-compute & refresh thread ──────────────────
def _compute_all_recommendations() -> None:
    """Fill the global cache for *every* streamer in the data‐set."""
    _, df_inf, *_ = _load_artifacts()
    for s in df_inf["stream_name"].unique():
        _recommendation_cache[s] = _compute_recommendations_for_stream(s)


def _refresh_loop(interval_seconds: int, stop_event: Event) -> None:
    """Background loop to keep the cache fresh."""
    _compute_all_recommendations()  # initial fill
    while not stop_event.wait(interval_seconds):
        _compute_all_recommendations()


def _start_refresher(interval_seconds: int = 3600) -> Event:
    """
    Launch a daemon thread that refreshes the cache every `interval_seconds`.
    Returns the `Event` you can set to stop the loop (rarely needed on Heroku).
    """
    stop = Event()
    Thread(
        target=_refresh_loop,
        args=(interval_seconds, stop),
        daemon=True,
        name="recommendation-refresher",
    ).start()
    return stop


# Kick-off the refresher as soon as the module is imported
_stop_refresher = _start_refresher()  # default 3600 s = 1 hour


# ───────────────────────── Public API ────────────────────────────────
def get_stream_recommendations(stream_name: str | None = None) -> dict:
    """
    Return cached recommendations for `stream_name` (or most-popular if None).
    Falls back to on-demand computation on a cache miss.
    Runs in **milliseconds**.
    """
    if not _recommendation_cache:
        # Should only happen on first dyno boot before refresher fills cache
        _compute_all_recommendations()

    # Pick a streamer if caller left it blank
    if not stream_name:
        _, df_inf, *_ = _load_artifacts()
        stream_name = df_inf["stream_name"].mode()[0]

    data = _recommendation_cache.get(stream_name)
    if data is None:
        # Rare: new streamer or cache evicted; compute once and store
        data = _compute_recommendations_for_stream(stream_name)
        _recommendation_cache[stream_name] = data

    return data
