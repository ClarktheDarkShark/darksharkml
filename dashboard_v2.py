import os
from flask import Flask, Blueprint, render_template_string, request
import pandas as pd
import numpy as np
from datetime import datetime
from shap_utils import generate_shap_plots
import pytz

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
      _get_last_row_for_stream  # internal helper; used for dashboard inference
)

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP & BLUEPRINT SETUP
# ─────────────────────────────────────────────────────────────────────────────
dash_v2 = Blueprint('dash_v2', __name__, url_prefix='')  # Remove prefix; route will be accessible at /v2

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATE: Dark‐Mode, Modern Styling
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE_V2 = '''
<!doctype html>
<html lang="en">
<head>

  <meta charset="utf-8">
  <meta name="viewport" width="device-width, initial-scale=1">
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
      margin: 0; padding: 1.2rem;
      background: var(--bg);
      color: var(--fg);
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      font-size: 0.95rem;
    }
    h1 { margin-bottom: 0.7rem; font-size: 1.5rem; }
    h2 { font-size: 1.15rem; margin-top: 1.2rem; margin-bottom: 0.7rem; }
    table {
      width: 100%; border-collapse: separate; border-spacing: 0;
      background: var(--card);
      border-radius: var(--radius);
      overflow: hidden;
      box-shadow: 0 2px 8px rgba(0,0,0,0.5);
      font-size: 0.92rem;
    }
    th, td {
      padding: 0.5rem 0.7rem;
      text-align: center;
    }
    th {
      background: #1f1f1f;
      font-weight: 600;
      border-bottom: 1px solid #333;
      font-size: 0.95rem;
    }
    tbody tr:nth-child(even) { background: #1a1a1a; }
    tbody tr:hover { background: #2a2a2a; }
    .note {
      margin-top: 0.7rem;
      color: var(--muted);
      font-size: 0.85rem;
    }
    .heatmap {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      grid-template-rows: repeat(4, 1fr);
      gap: 4px 4px;
      margin-top: 1rem;
      margin-bottom: 1rem;
      max-width: 700px;
    }
    .heatcell-wrap {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: flex-start;
    }
    .heatcell-label {
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .heatcell {
      height: 48px;
      min-width: 120px;
      font-size: 0.95rem;
      border-radius: 6px;
      color: #fff;
      font-weight: bold;
      transition: background 0.2s;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .heatcell:hover {
      outline: 2px solid #fff;
      z-index: 2;
    }
    .date-form {
      margin-bottom: 1rem;
    }
    .date-form input[type="date"] {
      background: var(--card);
      color: var(--fg);
      border: 1px solid #333;
      border-radius: var(--radius);
      padding: 0.3rem 0.7rem;
      font-size: 0.95rem;
    }
    .date-form button {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: var(--radius);
      padding: 0.3rem 0.7rem;
      font-size: 0.95rem;
      cursor: pointer;
      margin-left: 0.4rem;
    }
    .date-form button:hover {
      background: #1565c0;
    }
    .feature-select { margin-bottom: 1.2rem; }
    .feature-group { margin-bottom: 0.7rem; }
    .feature-label { font-weight: bold; margin-right: 0.7rem; font-size: 0.95rem; display: block; margin-bottom: 0.3rem;}
    .feature-btn {
      background: var(--card);
      color: var(--fg);
      border: 1px solid #333;
      border-radius: var(--radius);
      padding: 0.25rem 0.7rem;
      margin-right: 0.3rem;
      margin-bottom: 0.2rem;
      cursor: pointer;
      font-size: 0.92rem;
      transition: background 0.2s;
    }
    .feature-btn.selected, .feature-btn:active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
    }
    .feature-btn:hover { background: #1565c0; color: #fff; }
    .tag-btn { font-size: 0.9rem; }
    .update-btn {
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: var(--radius);
      padding: 0.4rem 1.1rem;
      font-size: 1rem;
      cursor: pointer;
      margin-top: 0.7rem;
      margin-bottom: 1.2rem;
    }
    .update-btn:hover { background: #1565c0; }
    .pred-result {
      background: var(--card);
      color: var(--fg);
      border-radius: var(--radius);
      padding: 1.2rem;
      margin: 0.7rem 0 1.2rem 0;
      font-size: 1.08rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4);
    }
    .pred-value {
      font-size: 1.5rem;
      color: var(--accent);
      text-align: center;
      padding: 0.7rem 0;
      margin: 0.3rem 0;
      border-top: 1px solid #333;
      border-bottom: 1px solid #333;
    }
    .pred-details {
      color: var(--muted);
      font-size: 0.85rem;
      margin-top: 0.7rem;
    }
    /* legend‐used tags get a colored border / background when NOT selected */
    .feature-btn.legend-used {
        border-color: var(--accent) !important;
        background: rgba(30,136,229,0.15) !important;
    }


  </style>
  <script>
    function selectFeature(name, value, multi=false) {
      if (multi) {
        // Toggle tag selection without submitting
        var input = document.getElementById('input_' + name + '_' + value);
        input.checked = !input.checked;
        var btn = document.querySelector(`button[onclick="selectFeature('${name}','${value}',true)"]`);
        btn.classList.toggle('selected');
      } else {
        // Radio button behavior without submitting
        var inputs = document.querySelectorAll('input[name="' + name + '"]');
        inputs.forEach(i => {
          i.checked = (i.value === value);
          var btn = document.querySelector(`button[onclick="selectFeature('${name}','${i.value}')"]`);
          btn.classList.toggle('selected', i.value === value);
        });
      }
    }
  </script>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  {{ shap_plots.js | safe }}
</head>

<body>
  <h1>Feature Insights for: TheLegendYagami</h1>
  <h3> Predictions are for: {{today_name}} </h3>
  
  {% if pred_result %}
    <div class="pred-result">
      <strong>PREDICTION FOR SELECTED FEATURES</strong>
      <div class="pred-value">
        {{ pred_result.y_pred }} subscribers
      </div>
      <div class="pred-details">
        <b>Game:</b> {{selected_game}}<br>
        <b>Start Time:</b> {{selected_start_time|default('')}}:00<br>
        <b>Tags:</b> {{selected_tags|join(', ')}}<br>
        <b>Confidence:</b> {{pred_result.conf}}
      </div>
    </div>
  {% endif %}


  <form id="feature-form" class="feature-select" method="get">
    <div class="feature-group">
      <div class="feature-label">Game:</div>
      {% for g in game_opts %}
        <label>
          <input type="radio" name="game" id="input_game_{{g}}" value="{{g}}" {% if g==selected_game %}checked{% endif %} style="display:none;">
          <button type="button" class="feature-btn {% if g==selected_game %}selected{% endif %}" onclick="selectFeature('game','{{g}}')">{{g}}</button>
        </label>
      {% endfor %}
    </div>
    <div class="feature-group">
      <div class="feature-label">Start Time:</div>
      {% for h in start_opts %}
        <label>
          <input type="radio" name="start_time" id="input_start_time_{{h}}" value="{{h}}" {% if h==selected_start_time %}checked{% endif %} style="display:none;">
          <button type="button" class="feature-btn {% if h==selected_start_time %}selected{% endif %}" onclick="selectFeature('start_time','{{h}}')">{{"%02d:00"|format(h)}}</button>
        </label>
      {% endfor %}
    </div>
    <div class="feature-group">
    <div class="feature-label">Tags:</div><br>
    {% for t in all_tags[:20] %}
        <label>
        <input type="checkbox"
                name="tags"
                id="input_tags_{{t}}"
                value="{{t}}"
                {% if t in selected_tags %}checked{% endif %}
                style="display:none;">
        <button type="button"
                class="feature-btn tag-btn
                        {% if t in selected_tags %} selected{% endif %}
                        {% if t in legend_tag_opts %} legend-used{% endif %}"
                onclick="selectFeature('tags','{{t}}',true)">
            {{ t }}
        </button>
        </label>
    {% endfor %}
    </div>

    <input type="hidden" name="manual" value="1">
    <button type="submit" class="update-btn">Update Prediction</button>
  </form>

  <!-- 1) Start Time Analysis (Heat Map) -->
  <h2>Start Time Analysis (Heat Map)</h2>
  <div class="heatmap">
    {% for cell in heatmap_cells %}
      <div class="heatcell-wrap">
        <div class="heatcell-label">{{ cell.label }}</div>
        <div class="heatcell" style="background: {{ cell.bg }};"
             title="Subs: {{ cell.avg_subs }}, Confidence: {{ cell.confidence }}">
          {{ cell.avg_subs }}
        </div>
      </div>
    {% endfor %}
  </div>
  <div class="note">
    Red = lowest, Blue = highest predicted subs. Hover for details.
  </div>

  <!-- 2) Game Category Comparison -->
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

  <!-- 3) Tag Combination Effects -->
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

      {% for row in tag_insights[:20] %}
      <tr>
        <td>{{ row.tags }}</td>
        <td>{{ row.delta }}</td>
        <td>{{ row.subs }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <!-- 4) Your SHAP section -->
  <h2>Feature Impact Analysis (SHAP)</h2>
  <div class="shap-container">
    <div id="shap-summary">
      <h3>Summary Plot</h3>
      {{ shap_plots.summary  | safe }}
    </div>
    <div id="shap-dependence">
      <h3>Dependence Plot</h3>
      {{ shap_plots.dependence| safe }}
    </div>
    <div id="shap-bar">
      <h3>Bar Plot</h3>
      {{ shap_plots.bar       | safe }}
    </div>
    <div id="shap-decision">
      <h3>Decision Plot</h3>
      {{ shap_plots.decision  | safe }}
    </div>
    <div id="shap-force">
      <h3>Force Plot</h3>
      {{ shap_plots.force     | safe }}
    </div>
  </div>

</body>
</html>
'''

# ─────────────────────────────────────────────────────────────────────────────
# ROUTE HANDLER
# ─────────────────────────────────────────────────────────────────────────────
@dash_v2.route('/v2', methods=['GET'])
def show_feature_insights():
    est = pytz.timezone("US/Eastern")
    pipe, df_for_inf, features, cat_opts, start_opts, dur_opts, metrics = get_predictor_artifacts()
    ready = pipe is not None and df_for_inf is not None

    all_tags: list[str] = []
    if ready:
        pre = pipe.named_steps["pre"]
        tag_pipe = pre.named_transformers_["tags"]
        vectorizer = tag_pipe.named_steps["vectorize"]
        all_tags = vectorizer.get_feature_names_out().tolist()

    stream_name = "thelegendyagami"
    today_name = datetime.now(est).strftime("%A") 

    # --- Limit games to top 10 by predicted subs + all games played by thelegendyagami ---
    df = df_for_inf.copy()
    df['y_pred'] = pipe.predict(df[features]) if ready else 0
    # Get all games played by thelegendyagami
    legend_games = df[df['stream_name'] == stream_name]['game_category'].unique().tolist()
    # Get top 10 games by predicted subs
    game_scores = (
        df.groupby('game_category')
        .agg(avg_subs=('y_pred', 'mean'))
        .reset_index()
        .sort_values('avg_subs', ascending=False)
    )
    top_games = game_scores.head(10)['game_category'].tolist()
    # Union and preserve order: legend_games first, then top_games not already included
    game_opts = legend_games + [g for g in top_games if g not in legend_games]

    # --- Limit tags to top 10 by effect ---
    tag_effects_full = _infer_grid_for_game(
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
        today_name=today_name,
    )
    tag_effects_full = tag_effects_full[abs(tag_effects_full['delta_from_baseline']) > 0.01]
    top_tags = tag_effects_full.sort_values('delta_from_baseline', ascending=False).head(10)['tag'].tolist()
    # Get all tags ever used by thelegendyagami

    # print(df.columns)
    legend_rows = df[df["stream_name"] == stream_name]
    legend_tag_opts: list[str] = []
    for tags in legend_rows["raw_tags"].dropna():
        # raw_tags is a list of strings
        for t in tags:
            if t not in legend_tag_opts:
                legend_tag_opts.append(t)

    # now union with the top_tags
    tag_opts = legend_tag_opts + [t for t in top_tags if t not in legend_tag_opts]
    all_tags = legend_tag_opts + [t for t in all_tags if t not in legend_tag_opts]

    # --- Feature selection from query params ---
    selected_game = request.args.get('game', game_opts[0] if game_opts else '')
    try:
        selected_start_time = int(request.args.get('start_time', start_opts[0] if start_opts else 0))
    except Exception:
        selected_start_time = start_opts[0] if start_opts else 0
    selected_tags = request.args.getlist('tags')
    manual = request.args.get('manual', None)

    pred_result = None
    if manual and ready:
        last_row = df[df["stream_name"] == stream_name].iloc[-1].copy()
        last_row['game_category'] = selected_game
        last_row['start_time_hour'] = selected_start_time
        for t in tag_opts:
            last_row['tag_' + t] = 1 if t in selected_tags else 0
        X = last_row[features].to_frame().T
        y_pred = pipe.predict(X)[0]
        try:
            pre = pipe.named_steps['pre']
            X_pre = pre.transform(X)
            model = pipe.named_steps['reg']
            from sklearn.compose import TransformedTargetRegressor
            if isinstance(model, TransformedTargetRegressor):
                model = model.regressor_
            if hasattr(model, 'estimators_'):
                all_tree_preds = np.stack([t.predict(X_pre) for t in model.estimators_], axis=1)
                sigma = all_tree_preds.std(axis=1)
            else:
                sigma = np.full(len(X_pre), fill_value=np.mean(y_pred)*0.01)
            conf = float(1.0 / (1.0 + sigma[0]))
        except Exception:
            conf = float('nan')
        pred_result = {
            'y_pred': round(y_pred, 2),
            'conf': round(conf, 2) if not np.isnan(conf) else '?'
        }

    # Generate predictions for each row in df_for_inf
    df['y_pred'] = pipe.predict(df[features]) if ready else 0
    # Confidence: 1/(1+std) across trees if available
    try:
        pre = pipe.named_steps['pre']
        X_pre = pre.transform(df[features]) if ready else np.zeros((len(df), len(features)))
        model = pipe.named_steps['reg']
        from sklearn.compose import TransformedTargetRegressor
        if isinstance(model, TransformedTargetRegressor):
            model = model.regressor_
        if hasattr(model, 'estimators_'):
            all_tree_preds = np.stack([t.predict(X_pre) for t in model.estimators_], axis=1)
            sigma = all_tree_preds.std(axis=1)
        else:
            sigma = np.full(len(X_pre), fill_value=np.mean(df['y_pred'])*0.01)
        df['conf'] = 1.0 / (1.0 + sigma)
    except Exception:
        df['conf'] = np.nan

    # 1) Game category insights (top 10 by avg_subs) -- ONLY for thelegendyagami
    legend_df = df[df['stream_name'] == stream_name].copy()
    game_insights = (
        legend_df.groupby('game_category')
        .agg(avg_subs=('y_pred', 'mean'), confidence=('conf', 'mean'))
        .reset_index()
        .rename(columns={'game_category': 'game'})
    )
    game_insights = game_insights.sort_values('avg_subs', ascending=False).head(10)
    game_insights['avg_subs'] = game_insights['avg_subs'].round(2)
    game_insights['confidence'] = game_insights['confidence'].round(2)
    game_insights = game_insights.to_dict('records')

    # 2) Tag combination effects (limit to abs(delta) > 0.01, round values)
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
        today_name=today_name,  # always "Saturday"
    )
    if 'tag' in tag_effects.columns:
        tag_effects = tag_effects[abs(tag_effects['delta_from_baseline']) > 0.01]
        tag_effects['delta_from_baseline'] = tag_effects['delta_from_baseline'].round(2)
        tag_effects['y_pred'] = tag_effects['y_pred'].round(2)
        tag_insights = tag_effects[['tag', 'delta_from_baseline', 'y_pred']].rename(
            columns={'tag': 'tags', 'delta_from_baseline': 'delta', 'y_pred': 'subs'}
        ).to_dict('records')
    else:
        tag_effects = tag_effects[abs(tag_effects['delta_from_baseline']) > 0.01]
        tag_effects['delta_from_baseline'] = tag_effects['delta_from_baseline'].round(2)
        tag_effects['y_pred'] = tag_effects['y_pred'].round(2)
        tag_insights = tag_effects[['tags', 'delta_from_baseline', 'y_pred']].rename(
            columns={'delta_from_baseline': 'delta', 'y_pred': 'subs'}
        ).to_dict('records')


    # 3) Start time analysis (heatmap)
    # Generate predictions for ALL possible start times

    time_predictions = _infer_grid_for_game(
        pipe,
        df_for_inf,
        features,
        stream_name=stream_name,
        override_tags=top_tags,
        start_times=list(range(24)),  # all 24 hours
        durations=dur_opts,
        category_options=[selected_game],  # use currently selected game
        top_n=1000,  # get all predictions
        unique_scores=False,  # we want all times
        vary_tags=False
    )
    
    # Average predictions for each start time
    time_df = (
        time_predictions
        .groupby('start_time_hour')
        .agg(
            avg_subs=('y_pred', 'mean'),
            confidence=('conf', 'mean')
        )
        .reset_index()
        .rename(columns={'start_time_hour': 'time'})
    )
    
    time_df['avg_subs'] = time_df['avg_subs'].round(2)
    time_df['confidence'] = time_df['confidence'].round(2)

    # Color normalization (no need to handle empty slots anymore)
    subs_vals = time_df['avg_subs']
    min_subs = np.percentile(subs_vals, 5)
    max_subs = np.percentile(subs_vals, 95)
    def interp_color(val):
        # Clamp value to [min_subs, max_subs]
        val = max(min_subs, min(val, max_subs))
        if max_subs == min_subs:
            return "#1e88e5"  # blue
        ratio = (val - min_subs) / (max_subs - min_subs)
        # Red: (222, 45, 38), Blue: (30, 136, 229)
        r = int(222 + (30 - 222) * ratio)
        g = int(45 + (136 - 45) * ratio)
        b = int(38 + (229 - 38) * ratio)
        return f"rgb({r},{g},{b})"

    # Build heatmap cells
    heatmap_cells = [
        {
            'label': f"{int(row['time']):02d}:00",
            'avg_subs': f"{row['avg_subs']:.2f}",
            'confidence': f"{row['confidence']:.2f}",
            'bg': interp_color(row['avg_subs'])
        }
        for _, row in time_df.iterrows()
    ]

    # Generate SHAP plots
    if ready:
        shap_plots = generate_shap_plots(pipe, df, features)
    else:
        shap_plots = {'summary': '{}', 'dependence': '{}'}

    print("DEBUG: passing shap_plots to template:")
    for name, blob in shap_plots.items():
        print(f"  {name}: {type(blob)} length={len(blob)}")

    return render_template_string(
        TEMPLATE_V2,
        today_name=today_name,
        selected_date=datetime.now().strftime("%Y-%m-%d"),
        game_opts=game_opts,
        start_opts=start_opts,
        tag_opts=tag_opts,
        selected_game=selected_game,
        selected_start_time=selected_start_time,
        selected_tags=selected_tags,
        pred_result=pred_result,
        game_insights=game_insights,
        tag_insights=tag_insights,
        heatmap_cells=heatmap_cells,
        all_tags=all_tags,
        shap_plots=shap_plots,
        legend_tag_opts=legend_tag_opts
    )
