import itertools
from flask import Blueprint, render_template_string, request
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
    _get_last_row_for_stream,
)

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP & BLUEPRINT SETUP
# ─────────────────────────────────────────────────────────────────────────────
dash_v3 = Blueprint('dash_v3', __name__, url_prefix='')  # mount at /

# simple cache for expensive SHAP plots
_shap_cache = {"pipe_id": None, "plots": None}

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE: Dark-Mode, Modern Styling (extended)
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE_V3 = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" width="device-width, initial-scale=1">
  <title>Feature Insights Dashboard</title>
  <style>
    /* same CSS as your TEMPLATE_V2… */
  </style>
  <script>
    /* same selectFeature + Plotly scripts as TEMPLATE_V2… */
  </script>

</head>
<body>
  <h1>Feature Insights for: {{ selected_stream }}</h1>
  <h3>Predictions for: {{ today_name }}</h3>
  <p class="note"><strong>Features used:</strong> {{ features|join(', ') }}</p>

  {% if pred_result %}
    <div class="pred-result">
      <strong>PREDICTION FOR SELECTED FEATURES</strong>
      <div class="pred-value">
        {{ pred_result.y_pred }} subscribers
      </div>
      <div class="pred-details">
        <b>Game:</b> {{ selected_game }}<br>
        <b>Start Time:</b> {{ selected_start_time }}:00<br>
        <b>Tags:</b> {{ selected_tags|join(', ') }}<br>
        <b>Confidence:</b> {{ pred_result.conf }}
      </div>
    </div>
  {% endif %}

  <!-- ─── Feature Selection Form ─────────────────────────────────────────────── -->
  <!-- unchanged from TEMPLATE_V2… -->

  <!-- ─── TOP-3 RECOMMENDATIONS ─────────────────────────────────────────────── -->
  <h2>Top Recommendations</h2>

  <h3>Subscriptions</h3>
  <table>
    <thead>
      <tr><th>Game</th><th>Start</th><th>Duration</th><th>Subs</th><th>Conf</th></tr>
    </thead>
    <tbody>
      {% for r in top3_subs %}
      <tr>
        <td>{{ r.game_category }}</td>
        <td>{{ "%02d:00"|format(r.start_time_hour) }}</td>
        <td>{{ r.stream_duration }}h</td>
        <td>{{ r.y_pred|round(2) }}</td>
        <td>{{ r.conf|round(2) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h3>Follower Growth</h3>
  <table>
    <thead>
      <tr><th>Game</th><th>Start</th><th>Duration</th><th>Followers</th><th>Conf</th></tr>
    </thead>
    <tbody>
      {% for r in top3_followers %}
      <tr>
        <td>{{ r.game_category }}</td>
        <td>{{ "%02d:00"|format(r.start_time_hour) }}</td>
        <td>{{ r.stream_duration }}h</td>
        <td>{{ r.y_pred|round(2) }}</td>
        <td>{{ r.conf|round(2) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h3>Viewers</h3>
  <table>
    <thead>
      <tr><th>Game</th><th>Start</th><th>Duration</th><th>Viewers</th><th>Conf</th></tr>
    </thead>
    <tbody>
      {% for r in top3_viewers %}
      <tr>
        <td>{{ r.game_category }}</td>
        <td>{{ "%02d:00"|format(r.start_time_hour) }}</td>
        <td>{{ r.stream_duration }}h</td>
        <td>{{ r.y_pred|round(2) }}</td>
        <td>{{ r.conf|round(2) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <!-- ─── Start Time Heatmap, Feature Scores, Game Comparison,
         Tag Effects, and SHAP Sections ──────────────────────────────────── -->
  <!-- reuse the HTML from TEMPLATE_V2 for those… -->

</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# Route
# ─────────────────────────────────────────────────────────────────────────────
@dash_v3.route('/v3', methods=['GET'])
def show_feature_insights_v3():
    # 1) load artifacts
    pipelines, df_inf, features, cat_opts, start_opts, dur_opts, metrics_list = get_predictor_artifacts()
    ready = bool(pipelines and df_inf is not None)

    # select stream
    selected_stream = request.args.get('stream', df_inf['stream_name'].mode()[0])
    selected_stream = 'thelegendyagami'
    if selected_stream not in df_inf['stream_name'].unique():
        selected_stream = df_inf['stream_name'].mode()[0]

    # pick pipelines
    pipe_sub = pipelines[0]
    pipe_fol = pipelines[1] if len(pipelines)>1 else pipelines[0]
    pipe_view= pipelines[2] if len(pipelines)>2 else pipelines[0]

    # date & tz
    tz = pytz.timezone("US/Eastern")
    today_dt = datetime.now(tz)
    today_name = today_dt.strftime("%A, %B %d, %Y")

    # baseline & legend history
    baseline    = _get_last_row_for_stream(df_inf, selected_stream)
    legend_games= df_inf.loc[df_inf['stream_name']==selected_stream,'game_category'].unique().tolist()
    legend_tags = sorted({
        tag
        for tags in df_inf.loc[df_inf['stream_name']==selected_stream,'raw_tags'].dropna()
        for tag in tags
    })

    # 2) Top-3 combos for each metric via _infer_grid_for_game
    def top3(pipe):
        df_top = _infer_grid_for_game(
            pipe, df_inf, features,
            stream_name=selected_stream,
            override_tags=legend_tags,
            start_times=start_opts,
            durations=dur_opts,
            category_options=legend_games,
            top_n=3,
            unique_scores=True
        )
        return df_top.to_dict('records')

    top3_subs      = top3(pipe_sub)
    top3_followers = top3(pipe_fol)
    top3_viewers   = top3(pipe_view)

    # 3) Existing sections
    # … heatmap_cells, feature_scores, game_insights, tag_insights, shap_plots
    # copy your compute logic from v2 here unchanged …

    return render_template_string(
        TEMPLATE_V3,
        selected_stream=selected_stream,
        today_name=today_name,
        features=features,
        # form options
        stream_opts=sorted(df_inf['stream_name'].unique()),
        game_opts=legend_games,
        start_opts=start_opts,
        all_tags=legend_tags,
        legend_tag_opts=legend_tags,
        # prediction inputs
        selected_game=request.args.get('game', legend_games[0] if legend_games else ''),
        selected_start_time=int(request.args.get('start_time', start_opts[0])),
        selected_tags=request.args.getlist('tags'),
        pred_result={},                # reuse your manual_prediction result if you want
        # top-3 recommendations
        top3_subs=top3_subs,
        top3_followers=top3_followers,
        top3_viewers=top3_viewers,
    )
