# app.py
import os
from flask import Flask, Blueprint, render_template_string, request, redirect, url_for
import pandas as pd
import numpy as np

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game  # internal helper; used for dashboard inference
)

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP & BLUEPRINT SETUP
# ─────────────────────────────────────────────────────────────────────────────
# app = Flask(__name__)

dash = Blueprint('dash', __name__, url_prefix='')  # mount at root

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE: Dark‐Mode, Modern Styling
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stream Predictions</title>
  <style>
    :root {
      --bg: #121212;
      --fg: #e0e0e0;
      --card: #1e1e1e;
      --accent: #1e88e5;
      --muted: #757575;
      --radius: 8px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 2rem;
      background: var(--bg);
      color: var(--fg);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1 { margin-bottom: 1rem; }
    form {
      display: flex; flex-wrap: wrap; gap: 0.5rem;
      margin-bottom: 1.5rem;
    }
    label { display: flex; align-items: center; gap: 0.5rem; }
    input, select, button {
      background: var(--card);
      color: var(--fg);
      border: 1px solid #333;
      border-radius: var(--radius);
      padding: 0.5rem 1rem;
      font-size: 1rem;
    }
    select { min-width: 10rem; }
    button {
      background: var(--accent);
      border: none;
      cursor: pointer;
      transition: background 0.2s;
    }
    button:hover { background: #1565c0; }
    .warn { color: #ff5252; margin-bottom: 1rem; }
    table {
      width: 100%; border-collapse: separate; border-spacing: 0;
      background: var(--card);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.5);
    }
    th, td {
      padding: 0.75rem 1rem;
      text-align: center;
    }
    th {
      background: #1f1f1f;
      font-weight: 600;
      border-bottom: 1px solid #333;
    }
    tbody tr:nth-child(even) { background: #1a1a1a; }
    tbody tr:hover { background: #2a2a2a; }
    .note {
      margin-top: 1rem;
      color: var(--muted);
      font-size: 0.9rem;
    }
  </style>
</head>
<body>
  <h1>Top {{ top_n }} Predictions{% if game %} for “{{ game }}”{% endif %}</h1>
  <form method="get">
    <label>Stream:
      <input name="stream" value="{{ stream }}" placeholder="Channel name">
    </label>
    <label>Game:
      <select name="game">
        <option value="">(all)</option>
        {% for g in cat_opts %}
        <option value="{{ g }}" {% if g==game %}selected{% endif %}>{{ g }}</option>
        {% endfor %}
      </select>
    </label>
    <label>Top N:
      <input name="top_n" type="number" value="{{ top_n }}" min="1" max="50" style="width:5rem;">
    </label>
    <button type="submit">Go</button>
  </form>
  {% if message %}<div class="warn">{{ message }}</div>{% endif %}
  {% if not ready %}
    <div>Model not trained yet. Try again soon.</div>
  {% else %}
  <table>
    <thead>
      <tr>
        <th>Start Time</th>
        <th>Duration (hrs)</th>
        <th>Expected Subs</th>
        <th>Confidence</th>
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
  <div class="note">
    Confidence is the 1σ spread across the RandomForest's trees (larger = more uncertain).
  </div>
  {% endif %}
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE HANDLER
# ─────────────────────────────────────────────────────────────────────────────
@dash.route('/', methods=['GET'])
def show_predictions():
    pipe, df_for_inf, features, cat_opts, start_opts, dur_opts, metrics = get_predictor_artifacts()
    ready = pipe is not None and df_for_inf is not None

    # Query params
    stream = (request.args.get('stream','thelegendyagami') or '').strip()
    game   = (request.args.get('game','') or '').strip()
    try:
        top_n = max(1, min(50, int(request.args.get('top_n','10'))))
    except ValueError:
        top_n = 10

    # Render early if not ready
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

    # Prepare data copy
    df = df_for_inf.copy()
    df['stream_name_lc']   = df['stream_name'].str.lower()
    df['game_category_lc'] = df['game_category'].str.lower()

    # Display maps
    stream_map = df.groupby('stream_name_lc')['stream_name'].last().to_dict()
    game_map   = df.groupby('game_category_lc')['game_category'].last().to_dict()

    stream_lc = stream.lower()
    game_lc   = game.lower()
    message = ""

    if stream_lc not in stream_map:
        return render_template_string(
            TEMPLATE,
            ready=True,
            stream=stream,
            game=game,
            top_n=top_n,
            cat_opts=cat_opts or [],
            predictions=[],
            message=f"Unknown stream '{stream}'.",
        )

    stream_disp = stream_map[stream_lc]
    cat_opts_lc = [c.lower() for c in (cat_opts or [])]

    if game_lc and game_lc in cat_opts_lc:
        sel_game_lc = game_lc
    else:
        if game_lc and game_lc not in cat_opts_lc:
            message = f"Game '{game}' not found. Using last recorded for stream."
        sel_game_lc = df.loc[df['stream_name_lc']==stream_lc, 'game_category_lc'].iloc[-1]

    top_df = _infer_grid_for_game(
        pipe, df_for_inf, features,
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
    disp['Confidence']    = disp['conf'].apply(lambda v: "?" if pd.isna(v) else f"±{float(v):.2f}")

    return render_template_string(
        TEMPLATE,
        ready=True,
        stream=stream_disp,
        game=game_map.get(sel_game_lc, sel_game_lc),
        top_n=top_n,
        cat_opts=cat_opts or [],
        predictions=disp[['Time','Duration','Expected_Subs','Confidence']].to_dict('records'),
        message=message,
    )

