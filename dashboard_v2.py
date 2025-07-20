import os
from flask import Flask, Blueprint, render_template_string
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
dash_v2 = Blueprint('dash_v2', __name__, url_prefix='/v2')  # mount at /v2

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE: Dark‐Mode, Modern Styling
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE_V2 = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Feature Insights Dashboard</title>
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
  <h1>Feature Insights for “thelegendyagami”</h1>
  <h2>Predictions for streaming on date: {{today_name}}</h2>
  
  <h2>Game Category Comparison</h2>
  <table>
    <thead>
      <tr>
        <th>Game Category</th>
        <th>Average Predicted Subs</th>
        <th>Confidence</th>
      </tr>
    </thead>
    <tbody>
      {% for row in game_insights %}
      <tr>
        <td>{{ row.game }}</td>
        <td>{{ row.avg_subs }}</td>
        <td>{{ row.confidence }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Tag Combination Effects</h2>
  <table>
    <thead>
      <tr>
        <th>Tag Combination</th>
        <th>Δ Predicted Subs</th>
        <th>Predicted Subs</th>
      </tr>
    </thead>
    <tbody>
      {% for row in tag_insights %}
      <tr>
        <td>{{ row.tags }}</td>
        <td>{{ row.delta }}</td>
        <td>{{ row.subs }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>Start Time Analysis</h2>
  <table>
    <thead>
      <tr>
        <th>Start Time</th>
        <th>Average Predicted Subs</th>
        <th>Confidence</th>
      </tr>
    </thead>
    <tbody>
      {% for row in time_insights %}
      <tr>
        <td>{{ row.time }}</td>
        <td>{{ row.avg_subs }}</td>
        <td>{{ row.confidence }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE HANDLER
# ─────────────────────────────────────────────────────────────────────────────
@dash_v2.route('/v2', methods=['GET'])
def show_feature_insights():
    pipe, df_for_inf, features, cat_opts, start_opts, dur_opts, metrics = get_predictor_artifacts()
    ready = pipe is not None and df_for_inf is not None

    today_name = datetime.now().strftime("%A")
    stream_name = "thelegendyagami"

    if not ready:
        return render_template_string(
            TEMPLATE_V2,
            today_name=today_name,
            game_insights=[],
            tag_insights=[],
            time_insights=[]
        )

    # 1) Game category insights
    game_insights = (
        df_for_inf.groupby('game_category')
        .agg(avg_subs=('y_pred', 'mean'), confidence=('conf', 'mean'))
        .reset_index()
        .rename(columns={'game_category': 'game'})
        .to_dict('records')
    )

    # 2) Tag combination effects
    tag_effects = _infer_grid_for_game(
        pipe,
        df_for_inf,
        features,
        stream_name=stream_name,
        start_times=start_opts,
        durations=dur_opts,
        category_options=cat_opts,
        top_n=100,
        unique_scores=True,
        vary_tags=True,
    )
    tag_insights = tag_effects[['tags', 'delta_from_baseline', 'y_pred']].rename(
        columns={'delta_from_baseline': 'delta', 'y_pred': 'subs'}
    ).to_dict('records')

    # 3) Start time analysis
    time_insights = (
        df_for_inf.groupby('start_time_hour')
        .agg(avg_subs=('y_pred', 'mean'), confidence=('conf', 'mean'))
        .reset_index()
        .rename(columns={'start_time_hour': 'time'})
        .to_dict('records')
    )

    return render_template_string(
        TEMPLATE_V2,
        today_name=today_name,
        game_insights=game_insights,
        tag_insights=tag_insights,
        time_insights=time_insights
    )
