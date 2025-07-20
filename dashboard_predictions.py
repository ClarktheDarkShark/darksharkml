# app.py
import os
from flask import Flask, Blueprint, render_template_string, request, redirect, url_for
import pandas as pd
import numpy as np
from datetime import datetime

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game  # internal helper; used for dashboard inference
)

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP & BLUEPRINT SETUP
# ─────────────────────────────────────────────────────────────────────────────
# app = Flask(__name__)

dash_preds = Blueprint('dash_preds', __name__, url_prefix='')  # mount at root

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
  <h2>Predictions are for streaming on date: {{today_name}} </h2>
  
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
    <label>
      <input type="checkbox" name="vary_tags" {% if vary_tags %}checked{% endif %}>
      Vary Tags
    </label>
    <button type="submit">Go</button>
  </form>
  <p class="note">Showing {{ predictions|length }} of top {{ top_n }} items.</p>
  {% if message %}<div class="warn">{{ message }}</div>{% endif %}
  {% if not ready %}
    <div>Model not trained yet. Try again soon.</div>
  {% else %}
     {% if vary_tags %}
        <h2>Tag Effects</h2>
        <table>
          <thead>
            <tr><th>Tag</th><th>Δ Predicted Subs</th><th>Predicted Subs</th></tr>
          </thead>
          <tbody>
            {% for row in tag_effects %}
            <tr>
              <td>{{ row.tag }}</td>
              <td>{{ row.delta_from_baseline|round(1) }}</td>
              <td>{{ row.y_pred|round(1) }}</td>
            </tr>
            {% endfor %}
          </tbody>
        </table>
        <div class="note">Δ Predicted Subs = change from baseline when flipping that tag.</div>
      {% else %}
      {% if best_tags %}
        <p><strong>Tags driving top prediction:</strong>
          {{ best_tags | join(', ') }}
        </p>
    {% endif %}
  {% endif %}

  
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
    Confidence is the 1/1σ spread across the RandomForest's trees (larger = more certain).
  </div>
  {% endif %}
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE HANDLER
# ─────────────────────────────────────────────────────────────────────────────
@dash_preds.route('/', methods=['GET'])
def show_predictions():
    pipe, df_for_inf, features, cat_opts, start_opts, dur_opts, metrics = get_predictor_artifacts()
    ready = pipe is not None and df_for_inf is not None

    # Should we flip tags to see tag effects?
    vary_tags = (request.args.get('vary_tags') == 'on')

    # Read query params
    stream = (request.args.get('stream', 'thelegendyagami') or '').strip()
    game   = (request.args.get('game', '') or '').strip()
    try:
        top_n = max(1, min(50, int(request.args.get('top_n', '10'))))
    except ValueError:
        top_n = 10
    today_name = datetime.now().strftime("%A")

    # initialize message early so we can always reference it
    message = ""

    # 1) If model not ready, bail out
    if not ready:
        return render_template_string(
            TEMPLATE,
            ready=False,
            stream=stream,
            game=game,
            top_n=top_n,
            today_name=today_name,
            cat_opts=cat_opts or [],
            predictions=[],
            message=message,
            best_tags=[],
            vary_tags=vary_tags,
            tag_effects=[]
        )

    # 2) Build lowercase lookup maps
    df = df_for_inf.copy()
    df['stream_name_lc']   = df['stream_name'].str.lower()
    df['game_category_lc'] = df['game_category'].str.lower()
    stream_map = df.groupby('stream_name_lc')['stream_name'].last().to_dict()
    game_map   = df.groupby('game_category_lc')['game_category'].last().to_dict()

    stream_lc = stream.lower()
    if stream_lc not in stream_map:
        message = f"Unknown stream '{stream}'"
        return render_template_string(
            TEMPLATE,
            ready=True,
            stream=stream,
            game=game,
            top_n=top_n,
            today_name=today_name,
            cat_opts=cat_opts or [],
            predictions=[],
            message=message,
            best_tags=[],
            vary_tags=vary_tags,
            tag_effects=[]
        )

    stream_disp = stream_map[stream_lc]
    cat_opts_lc = [c.lower() for c in (cat_opts or [])]
    game_lc     = game.lower()

    if game and game_lc in cat_opts_lc:
        sel_game_lc = game_lc
    else:
        if game and game_lc not in cat_opts_lc:
            message = f"Game '{game}' not found; using last recorded for stream."
        sel_game_lc = df.loc[df['stream_name_lc'] == stream_lc, 'game_category_lc'].iloc[-1]

    # 2.5) Find typical start times for this streamer
    streamer_rows = df[df['stream_name_lc'] == stream_lc]
    preferred_hours = set()
    if not streamer_rows.empty:
        common_hours = (
            streamer_rows['start_time_hour']
            .astype(int)
            .value_counts()
            .index[:3]
            .tolist()
        )
        for h in common_hours:
            preferred_hours.update([h-1, h, h+1])

    # 3) Run inference on all start_opts
    top_df = _infer_grid_for_game(
        pipe,
        df_for_inf,
        features,
        stream_name=stream_disp,
        start_times=start_opts,  # use all possible times
        durations=dur_opts,
        category_options=[sel_game_lc],
        top_n=100,  # get enough rows to filter/sort
        unique_scores=True,
        vary_tags=vary_tags,
    )

    # Add a column: is_preferred_time
    top_df['is_preferred'] = top_df['start_time_hour'].astype(int).isin(preferred_hours)

    # Sort: preferred times first, then by predicted subs
    top_df = top_df.sort_values(['is_preferred', 'y_pred'], ascending=[False, False]).head(top_n)


    # ensure we always have a conf column
    if 'conf' not in top_df.columns:
        top_df['conf'] = np.nan

    # 4) If we're in vary-tags mode, show that table and return early
    if vary_tags:
        tag_effects = top_df.to_dict('records')
        return render_template_string(
            TEMPLATE,
            ready=True,
            stream=stream_disp,
            game=game_map.get(sel_game_lc, sel_game_lc),
            top_n=top_n,
            today_name=today_name,
            cat_opts=cat_opts or [],
            message=message,
            best_tags=[],
            vary_tags=True,
            tag_effects=tag_effects
        )

    # 5) Otherwise, fall back to the normal predictions grid
    best_tags = top_df.loc[0, 'tags'] if not top_df.empty else []
    disp = top_df.copy()
    disp['Time']           = disp['start_time_hour'].astype(int).map(lambda h: f"{h:02d}:00")
    disp['Duration']       = disp['stream_duration'].astype(int)
    disp['Expected_Subs']  = disp['y_pred'].round(1)
    disp['Confidence']     = disp['conf'].apply(lambda v: "?" if pd.isna(v) else f"{float(v):.2f}")

    return render_template_string(
        TEMPLATE,
        ready=True,
        stream=stream_disp,
        game=game_map.get(sel_game_lc, sel_game_lc),
        top_n=top_n,
        today_name=today_name,
        cat_opts=cat_opts or [],
        predictions=disp[['Time','Duration','Expected_Subs','Confidence']].to_dict('records'),
        message=message,
        best_tags=best_tags,
        vary_tags=False,
        tag_effects=[]
    )
