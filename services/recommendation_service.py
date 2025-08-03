# app/services/recommendation_service.py
from __future__ import annotations

from functools import lru_cache
from datetime import datetime

import pytz
from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
    _get_last_row_for_stream,        # still useful elsewhere
)

TZ = pytz.timezone("US/Eastern")


@lru_cache(maxsize=1)
def _load_artifacts():
    """
    One-time load of pipelines + inference grid options.
    Using an LRU cache means the objects stay in memory
    but can be re-loaded after a code reload or manual clear().
    """
    return get_predictor_artifacts()           # -> pipes, df_inf, feats, …


def _top_n(
    pipe,
    df_inf,
    feats,
    stream_name: str,
    legend_tags: list[str],
    start_opts: list[int],
    dur_opts: list[int],
    legend_games: list[str],
    n: int = 3,
):
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


def get_stream_recommendations(stream_name: str | None = None) -> dict:
    """
    Compute the ‘top-3’ recommendation tables for a given streamer
    and return everything the front-end needs in a single dict.
    """
    pipes, df_inf, feats, _, start_opts, dur_opts, _ = _load_artifacts()

    if not pipes or df_inf is None:
        raise RuntimeError("Model artefacts not loaded.")

    # -------- pick a streamer --------
    if not stream_name or stream_name not in df_inf["stream_name"].unique():
        stream_name = df_inf["stream_name"].mode()[0]

    legend_games = (
        df_inf.loc[df_inf.stream_name == stream_name, "game_category"]
        .unique()
        .tolist()
    )
    legend_tags = sorted(
        {
            t
            for tags in df_inf.loc[df_inf.stream_name == stream_name, "raw_tags"]
            .dropna()
            for t in tags
        }
    )

    # -------- choose the three models --------
    pipe_sub = pipes[0]
    pipe_fol = pipes[1] if len(pipes) > 1 else pipes[0]
    pipe_view = pipes[2] if len(pipes) > 2 else pipes[0]

    # -------- assemble the payload --------
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
