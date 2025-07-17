# dashboard_predictions.py
from flask import Blueprint, render_template_string, request
import pandas as pd
import numpy as np

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,   # internal helper; used for dashboard inference
)

# Mount at root so this handles '/'
dash_preds = Blueprint('dash_preds', __name__, url_prefix='/')

TEMPLATE = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Stream Predictions</title>
    <style>
      table { border-collapse: collapse; width: 60%; max-width: 600px; }
      th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: center; }
      th { background: #f5f5f5; }
      body { font-family: sans-serif; margin: 2rem; }
      form { margin-bottom: 1rem; }
      input, select { padding: 4px; }
      .note { margin-top: 1rem; font-size: 0.9em; color: #666; }
      .warn { color: #b00; font-weight: bold; }
    </style>
  </head>
  <body>
    <h1>Top {{ top_n }} Predictions{% if game %} for “{{ game }}”{% endif %}</h1>
    <form method="get">
      <label>Stream (channel): <input name="stream" value="{{ stream }}"></label>
      <label>Game:
        <select name="game">
          <option value="">All</option>
          {% for g in cat_opts %}
            <option value="{{ g }}" {% if g == game %}selected{% endif %}>{{ g }}</option>
          {% endfor %}
        </select>
      </label>
      <label>Top N: <input name="top_n" type="number" value="{{ top_n }}" min="1" max="50" style="width:4em;"></label>
      <button type="submit">Go</button>
    </form>
    {% if message %}<p class="warn">{{ message }}</p>{% endif %}
    {% if not ready %}
      <p>Model not trained yet. Try again soon.</p>
    {% else %}
    <table>
      <thead>
        <tr>
          <th>Start Time</th>
          <th>Duration (hrs)</th>
          <th>Expected Subs</th>
          <th>Conf </th>
        </tr>
      </thead>
      <tbody>
      {% for row in predictions %}
        <tr>
          <td>{{ row.Time }}</td>
          <td>{{ row.Duration }}</td>
          <td>{{ row.Expected_Subs }}</td>
          <td>{{ row.Confidence }}</td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    <p class="note">
      Confidence is the 1/1σ spread across the RandomForest's trees (higher = more certain).
    </p>
    {% endif %}
  </body>
</html>
"""


@dash_preds.route('/', methods=['GET'])
def show_predictions():
    # --- load artifacts ---
    pipe, df_for_inf, features, cat_opts, start_opts, dur_opts, metrics = get_predictor_artifacts()
    ready = pipe is not None and df_for_inf is not None

    # --- query params ---
    raw_stream = request.args.get('stream', 'thelegendyagami')
    raw_game   = request.args.get('game', '')
    raw_top_n  = request.args.get('top_n', '10')

    # sanitize
    stream = (raw_stream or '').strip()
    game   = (raw_game or '').strip()
    try:
        top_n = max(1, min(50, int(raw_top_n)))
    except ValueError:
        top_n = 10

    # If model not ready, still pass cat_opts for dropdown
    if not ready:
        return render_template_string(
            TEMPLATE,
            ready=False,
            stream=stream,
            game=game,
            top_n=top_n,
            cat_opts=cat_opts or [],
            predictions=[],
            message="",
        )

    # --- work off a *copy* so we don't mutate global predictor state ---
    df = df_for_inf.copy()

    # normalize to lowercase for matching
    df['stream_name_lc']   = df['stream_name'].str.lower()
    df['game_category_lc'] = df['game_category'].str.lower()

    # mapping back to display case
    stream_display_map = df.groupby('stream_name_lc')['stream_name'].last().to_dict()
    game_display_map   = df.groupby('game_category_lc')['game_category'].last().to_dict()

    stream_lc = stream.lower()
    game_lc   = game.lower()

    message = ""

    # validate stream
    if stream_lc not in stream_display_map:
        message = f"Unknown stream '{stream}'."
        return render_template_string(
            TEMPLATE,
            ready=True,
            stream=stream,
            game=game,
            top_n=top_n,
            cat_opts=cat_opts or [],
            predictions=[],
            message=message,
        )

    stream_disp = stream_display_map[stream_lc]

    # case-insensitive category list
    cat_opts_lc = [c.lower() for c in (cat_opts or [])]

    # choose game or fallback
    if game_lc and game_lc in cat_opts_lc:
        sel_game_lc = game_lc
    else:
        if game_lc and game_lc not in cat_opts_lc:
            message = f"Game '{game}' not found. Using last recorded game for stream."
        sel_game_lc = (
            df.loc[df['stream_name_lc'] == stream_lc, 'game_category_lc'].iloc[-1]
        )

    sel_game_disp = game_display_map.get(sel_game_lc, sel_game_lc)

    # --- call predictor ---
    top_df = _infer_grid_for_game(
        pipe,
        df_for_inf,
        features,
        stream_name=stream_disp,
        start_times=start_opts,
        durations=dur_opts,
        category_options=[sel_game_lc],
        top_n=top_n,
        unique_scores=True,
    )

    if 'conf' not in top_df.columns:
        top_df['conf'] = np.nan

    disp = top_df.copy()
    disp['Time']          = disp['start_time_hour'].astype(int).map(lambda h: f"{h:02d}:00")
    disp['Duration']      = disp['stream_duration'].astype(int)
    disp['Expected_Subs'] = disp['y_pred'].round().astype(int)
    disp['Confidence']    = disp['conf'].apply(
        lambda v: "?" if pd.isna(v) else f"±{float(v):.2f}"
    )

    return render_template_string(
        TEMPLATE,
        ready=True,
        stream=stream_disp,
        game=sel_game_disp,
        top_n=top_n,
        cat_opts=cat_opts or [],
        predictions=disp[['Time', 'Duration', 'Expected_Subs', 'Confidence']].to_dict(orient='records'),
        message=message,
    )
