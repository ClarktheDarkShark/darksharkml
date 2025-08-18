from flask import Blueprint, render_template_string, request
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import TransformedTargetRegressor
from shap_utils import generate_shap_plots
import pytz
import itertools
from threading import Lock
from typing import List, Dict, Optional, Any
_shap_lock = Lock()


from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
      _get_last_row_for_stream  # internal helper; used for dashboard inference
)
from extensions import cache               # ✔ no longer imports main

# ─────────────────────────────────────────────────────────────────────────────
# FLASK APP & BLUEPRINT SETUP
# ─────────────────────────────────────────────────────────────────────────────
dash_v2 = Blueprint('dash_v2', __name__, url_prefix='')  # Remove prefix; route will be accessible at /v2

# simple cache for expensive SHAP plots
_shap_cache = {"pipe_id": None, "plots": None}

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
        background: var(--card);
        border-color: #ffa726;
        color: #ffa726;
    }


  </style>
  <script>
    function selectFeature(name, value, multi=false) {
        if (multi) {
        // Toggle tag selection without submitting
        var input = document.getElementById('input_' + name + '_' + value);
        input.checked = !input.checked;

        var btn = document.querySelector(
            `button[onclick="selectFeature('${name}','${value}',true)"]`
        );
        // toggle the selected class
        var isNowSelected = btn.classList.toggle('selected');

        if (isNowSelected) {
            // once selected, remove the legend‐used styling
            btn.classList.remove('legend-used');
        } else {
            // if un‐selecting, re‐apply legend if it truly is a legend‐tag
            if (btn.dataset.legend === 'true') {
            btn.classList.add('legend-used');
            }
        }
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
  <h1>Feature Insights for: {{ selected_stream }}</h1>
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
      <div class="feature-label">Streamer:</div>
      {% for s in stream_opts %}
        <label>
          <input type="radio" name="stream" id="input_stream_{{s}}" value="{{s}}" {% if s==selected_stream %}checked{% endif %} style="display:none;">
          <button type="button" class="feature-btn {% if s==selected_stream %}selected{% endif %}" onclick="selectFeature('stream','{{s}}')">{{s}}</button>
        </label>
      {% endfor %}
    </div>
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
                        {% if t in selected_tags %}selected{% endif %}
                        {% if t in legend_tag_opts and t not in selected_tags %}legend-used{% endif %}"
                    data-legend="{{ 'true' if t in legend_tag_opts else 'false' }}"
                    onclick="selectFeature('tags','{{t}}',true)">
            {{ t }}
            </button>
        </label>
        {% endfor %}
    </div>


    <input type="hidden" name="manual" value="1">
    <button type="submit" class="update-btn">Update Prediction</button>
  </form>

  <!-- 1) Start Time Analysis (Heat Map) Subs -->
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

    <!-- 1) Start Time Analysis (Heat Map) Viewers -->
  <h2>Start Time Analysis (Heat Map)</h2>
  <div class="heatmap">
    {% for cell in heatmap_cells_viewers %}
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

def load_artifacts():
    """Load model + inference DataFrame + metadata."""
    return get_predictor_artifacts()

def predict_time_grid(
    baseline_row: pd.Series,
    start_times: List[int],
    game_category: str,
    duration: int,
    selected_tags: List[str],
    tag_opts: List[str],
    pipeline,
    features: List[str],
    tz: str = "US/Eastern",
) -> pd.DataFrame:
    """Return one row per hour with the same feature schema as manual_prediction."""
    grid = pd.DataFrame({"start_time_hour": start_times}, dtype="int64")

    template = baseline_row.copy(deep=True)
    template["game_category"] = game_category
    template["stream_duration"] = duration * 60

    base_rep = pd.DataFrame(
        np.repeat(template.values[None, :], len(grid), axis=0),
        columns=template.index
    )
    base_rep["start_time_hour"] = grid["start_time_hour"].astype("int64")

    now = datetime.now(pytz.timezone(tz))
    dow = now.strftime("%A")

    base_rep["day_of_week"] = dow
    base_rep["is_weekend"] = dow in ("Saturday", "Sunday")

    for t in tag_opts:
        base_rep[f"tag_{t}"] = int(t in selected_tags)

    X = base_rep[features]
    base_rep["y_pred"] = pipeline.predict(X)
    base_rep["conf"] = compute_confidence(base_rep, pipeline, features)
    return base_rep

def extract_all_tags(pipelines) -> List[str]:
    """Pull the vectorizer’s vocabulary out of the FIRST pipeline in the list."""
    first_pipe = pipelines[0]
    pre = first_pipe.named_steps['pre']
    tag_pipe = pre.named_transformers_['tags']
    vect = tag_pipe.named_steps['vectorize']
    return vect.get_feature_names_out().tolist()

def select_stream(request, df_inf: pd.DataFrame, default: str = 'thelegendyagami') -> str:
    """Pick a streamer from the URL params or fall back to default."""
    top5 = df_inf['stream_name'].value_counts().head(5).index.tolist()
    if default not in top5:
        top5.append(default)
    choice = request.args.get('stream', default)
    return choice if choice in df_inf['stream_name'].unique() else default

def compute_baseline_row(df_inf: pd.DataFrame, stream: str):
    """The “last row” for manual override defaults."""
    return _get_last_row_for_stream(df_inf, stream)

def predict_df(df_inf: pd.DataFrame, pipeline, features: List[str]) -> pd.DataFrame:
    """Add y_pred column to entire inference frame."""
    df = df_inf.copy()
    df['y_pred'] = pipeline.predict(df[features])
    return df

def compute_confidence(df: pd.DataFrame, pipeline, features: List[str]) -> pd.Series:
    """1 / (1 + σ) across tree preds, or constant fallback."""
    try:
        pre = pipeline.named_steps['pre']
        X_pre = pre.transform(df[features])
        model = pipeline.named_steps['reg']
        if isinstance(model, TransformedTargetRegressor):
            model = model.regressor_
        if hasattr(model, 'estimators_'):
            all_preds = np.stack([t.predict(X_pre) for t in model.estimators_], axis=1)
            sigma = all_preds.std(axis=1)
        else:
            sigma = np.full(len(df), fill_value=np.mean(df['y_pred']) * 0.01)
        conf = 1.0 / (1.0 + sigma)
        return pd.Series(conf, index=df.index)
    except Exception:
        return pd.Series(np.nan, index=df.index)

def compute_game_opts(df_pred: pd.DataFrame, stream: str) -> List[str]:
    """Legend’s own games first, then top‐10 by mean y_pred."""
    legend = df_pred.loc[df_pred.stream_name == stream, 'game_category'].unique().tolist()
    top10 = (
        df_pred.groupby('game_category')['y_pred'].mean()
               .nlargest(10)
               .index
               .tolist()
    )
    return legend + [g for g in top10 if g not in legend]

def compute_legend_tags(df_pred: pd.DataFrame, stream: str) -> List[str]:
    """All raw_tags ever used by this streamer."""
    rows = df_pred.loc[df_pred.stream_name == stream, 'raw_tags'].dropna()
    seen: set[str] = set()
    for tag_list in rows:
        for tag in tag_list:
            seen.add(tag)
    return list(seen)

def compute_tag_insights(
    pipeline, df_inf, features, stream, cat_opts, start_opts, dur_opts, today_name
) -> List[Dict]:
    te = _infer_grid_for_game(
        pipeline, df_inf, features,
        stream_name=stream,
        start_times=start_opts,
        durations=dur_opts,
        category_options=cat_opts,
        top_n=100,
        unique_scores=True,
        vary_tags=True,
        today_name=today_name
    )
    te = te.loc[te.delta_from_baseline.abs() > 0.01]
    te[['delta_from_baseline','y_pred']] = te[['delta_from_baseline','y_pred']].round(2)
    if 'tag' in te.columns:
        return (te[['tag','delta_from_baseline','y_pred']]
                .rename(columns={'tag':'tags','delta_from_baseline':'delta','y_pred':'subs'})
                .to_dict('records'))
    return (te[['tags','delta_from_baseline','y_pred']]
            .rename(columns={'delta_from_baseline':'delta','y_pred':'subs'})
            .to_dict('records'))

def union_preserve(first: List[str], second: List[str]) -> List[str]:
    """Return first + [s for s in second if s not in first]."""
    return first + [s for s in second if s not in first]

def parse_int(val, default=0) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

def manual_prediction(
    baseline_row: pd.Series,
    selected_game: str,
    selected_start_time: int,
    selected_tags: List[str],
    tag_opts: List[str],
    pipeline,
    features: List[str],
    tz='US/Eastern'
) -> Dict[str, float]:
    """Override baseline_row with the form selections and re‐predict."""
    row = baseline_row.copy()
    row['game_category'] = selected_game
    row['start_time_hour'] = selected_start_time
    now = datetime.now(pytz.timezone(tz))
    dow = now.strftime('%A')
    row.update({
        'day_of_week': dow,
        'is_weekend': dow in ('Saturday','Sunday')
    })
    for t in tag_opts:
        row[f'tag_{t}'] = int(t in selected_tags)
    X = row[features].to_frame().T
    y_pred = float(pipeline.predict(X)[0])
    conf = compute_confidence(pd.DataFrame([row]), pipeline, features).iloc[0]
    return {'y_pred': round(y_pred, 2), 'conf': round(conf, 2) if not np.isnan(conf) else '?'}

def compute_game_insights(df_pred: pd.DataFrame, stream: str) -> List[Dict]:
    df = df_pred.query("stream_name == @stream")
    out = (
        df.groupby('game_category')
          .agg(avg_subs=('y_pred','mean'), confidence=('conf','mean'))
          .round(2)
          .reset_index()
          .rename(columns={'game_category':'game'})
          .nlargest(10, 'avg_subs')
    )
    return out.to_dict('records')

def compute_heatmap_cells(time_df: pd.DataFrame) -> List[Dict]:
    """Build the 24‐cell heatmap with color interpolation."""
    subs = time_df['avg_subs']
    lo, hi = np.percentile(subs, 5), np.percentile(subs, 95)

    def interp(val):
        v = max(lo, min(val, hi))
        if hi == lo:
            return "#1e88e5"
        r = int(222 + (30-222)*(v-lo)/(hi-lo))
        g = int(45  + (136-45)*(v-lo)/(hi-lo))
        b = int(38  + (229-38)*(v-lo)/(hi-lo))
        return f"rgb({r},{g},{b})"

    cells = []
    for _, r in time_df.iterrows():
        cells.append({
            'label': f"{int(r.time):02d}:00",
            'avg_subs': f"{r.avg_subs:.2f}",
            'confidence': f"{r.confidence:.2f}",
            'bg': interp(r.avg_subs)
        })
    return cells

def compute_feature_scores(time_preds: pd.DataFrame, selected_game: str, top_n_tags: int = 3) -> List[Dict]:
    feature_scores: List[Dict] = []

    duration_scores = (
        time_preds.groupby('stream_duration')['y_pred']
                  .mean()
                  .round(2)
                  .reset_index()
    )
    for _, r in duration_scores.iterrows():
        feature_scores.append({
            'feature': 'Duration',
            'value': f"{int(r.stream_duration)}h",
            'score': f"{r.y_pred:.2f}"
        })

    feature_scores.append({
        'feature': 'Category',
        'value': selected_game,
        'score': f"{time_preds['y_pred'].mean():.2f}"
    })

    if 'delta_from_baseline' in time_preds.columns:
        top = time_preds.nlargest(top_n_tags, 'delta_from_baseline')
        for _, r in top.iterrows():
            feature_scores.append({
                'feature': 'Tag',
                'value': r.get('tag', '<unknown>'),
                'score': f"{r['delta_from_baseline']:.2f}"
            })

    return feature_scores

# SHAP cache/lock assumed defined elsewhere
def get_shap_blocks(pipe, df_pred, features):
    pid = id(pipe)
    with _shap_lock:
        if _shap_cache.get("pipe_id") != pid:
            try:
                _shap_cache["pipe_id"] = pid
                _shap_cache["plots"] = generate_shap_plots(pipe, df_pred, features)
            except Exception:
                _shap_cache["plots"] = {"summary": "{}", "dependence": "{}"}
    return _shap_cache["plots"]

# ─────────────────────────────────────────────────────────────────────────────
# Load artifacts ONCE at module import
# ─────────────────────────────────────────────────────────────────────────────
pipelines, df_inf, features, cat_opts, start_opts, dur_opts, metrics_list = load_artifacts()
ALL_TAGS = extract_all_tags(pipelines) if pipelines else []
TZ = pytz.timezone("US/Eastern")

# ─────────────────────────────────────────────────────────────────────────────
# Cached computations — KEY BY model_idx (stable), not id(pipe)
# ─────────────────────────────────────────────────────────────────────────────

@cache.memoize(timeout=15*60)
def cached_df_pred(model_idx: int):
    model_idx = max(0, min(model_idx, len(pipelines) - 1))
    pipe = pipelines[model_idx]
    dfp = predict_df(df_inf, pipe, features).copy()
    dfp['conf'] = compute_confidence(dfp, pipe, features)
    return dfp

@cache.memoize(timeout=15*60)
def cached_shap_blocks(model_idx: int):
    model_idx = max(0, min(model_idx, len(pipelines) - 1))
    pipe = pipelines[model_idx]
    dfp  = cached_df_pred(model_idx)
    return get_shap_blocks(pipe, dfp, features)

@cache.memoize(timeout=15*60)
def cached_infer_grid(model_idx: int, stream: str, game: str, tag_key: str, today_name: str):
    model_idx = max(0, min(model_idx, len(pipelines) - 1))
    pipe = pipelines[model_idx]
    tags = tag_key.split(",") if tag_key else None
    return _infer_grid_for_game(
        pipe, df_inf, features,
        stream_name=stream,
        override_tags=tags,
        start_times=start_opts,
        durations=dur_opts,
        category_options=[game],
        top_n=1000,
        unique_scores=False,
        vary_tags=False,
        today_name=today_name,
    )

@cache.memoize(timeout=15*60)
def cached_tag_insights(model_idx: int, stream: str, today_name: str):
    model_idx = max(0, min(model_idx, len(pipelines) - 1))
    pipe = pipelines[model_idx]
    return compute_tag_insights(
        pipe, df_inf, features, stream,
        cat_opts, start_opts, dur_opts, today_name
    )

@cache.memoize(timeout=15*60)
def cached_top_effect_tags(model_idx: int, stream: str, today_name: str, selected_game: str, n: int = 10) -> List[str]:
    """Lightweight tag effects to avoid combinatorial blowups."""
    model_idx = max(0, min(model_idx, len(pipelines) - 1))
    pipe = pipelines[model_idx]
    eff = _infer_grid_for_game(
        pipe, df_inf, features,
        stream_name=stream,
        start_times=[0, 6, 12, 18],  # probe subset
        durations=dur_opts[:2],      # first two durations
        category_options=[selected_game],
        top_n=100,
        unique_scores=True,
        vary_tags=True,
        today_name=today_name
    )
    eff = eff.loc[eff.delta_from_baseline.abs() > 0.01]
    return eff.nlargest(n, 'delta_from_baseline')['tag'].tolist()

# ─────────────────────────────────────────────────────────────────────────────
# Route
# ─────────────────────────────────────────────────────────────────────────────

@dash_v2.route('/v2', methods=['GET'])
def show_feature_insights():
    # readiness
    ready = bool(pipelines and df_inf is not None)
    if not ready:
        return "Server warming up", 503

    today = datetime.now(TZ).strftime("%A")

    # model selection
    model_idx = int(request.args.get('model', 0))
    model_idx = max(0, min(model_idx, len(pipelines)-1))
    pipe      = pipelines[model_idx]
    metrics   = metrics_list[model_idx]

    # stream/baseline
    selected_stream = select_stream(request, df_inf)
    baseline        = compute_baseline_row(df_inf, selected_stream)

    # frame predictions (CACHED)
    df_pred = cached_df_pred(model_idx)

    # options & tags
    game_opts     = compute_game_opts(df_pred, selected_stream)
    selected_game = request.args.get('game', game_opts[0] if game_opts else '')
    legend_tags   = compute_legend_tags(df_pred, selected_stream)
    top_tags      = cached_top_effect_tags(model_idx, selected_stream, today, selected_game)
    tag_opts      = union_preserve(legend_tags, top_tags)
    all_tags      = union_preserve(legend_tags, ALL_TAGS)

    # inputs
    selected_start_time = parse_int(request.args.get('start_time'), default=start_opts[0])
    selected_tags       = request.args.getlist('tags')
    manual              = request.args.get('manual') in ('1','true','True','yes')

    # FAST PATH for manual requests — skip heavy blocks
    if manual:
        pred_result = manual_prediction(baseline, selected_game, selected_start_time,
                                        selected_tags, tag_opts, pipe, features)
        return render_template_string(
            TEMPLATE_V2,
            today_name=today,
            stream_opts=sorted(df_inf['stream_name'].unique()),
            game_opts=game_opts,
            start_opts=start_opts,
            tag_opts=tag_opts,
            selected_stream=selected_stream,
            selected_game=selected_game,
            selected_start_time=selected_start_time,
            selected_tags=selected_tags,
            pred_result=pred_result,
            game_insights=[],
            tag_insights=[],
            heatmap_cells=[],
            feature_scores=[],
            heatmap_cells_viewers=[],
            feature_scores_viewers=[],
            all_tags=all_tags,
            shap_plots={'summary':'{}','dependence':'{}'},
            legend_tag_opts=legend_tags,
        )

    # time grid (CACHED) — single computation, do NOT overwrite
    tag_key    = ",".join(sorted(tag_opts))
    time_preds = cached_infer_grid(model_idx, selected_stream, selected_game, tag_key, today)

    time_df = (
        time_preds.groupby('start_time_hour')
                  .agg(avg_subs=('y_pred','max'), confidence=('conf','mean'))
                  .reset_index()
                  .rename(columns={'start_time_hour':'time'})
                  .round(2)
    )
    heatmap_cells  = compute_heatmap_cells(time_df)
    feature_scores = compute_feature_scores(time_preds, selected_game)

    # viewers model (guarded + CACHED)
    if len(pipelines) > 2:
        time_preds_viewers = cached_infer_grid(2, selected_stream, selected_game, tag_key, today)
        time_df_viewers = (
            time_preds_viewers.groupby('start_time_hour')
                              .agg(avg_subs=('y_pred','max'), confidence=('conf','mean'))
                              .reset_index()
                              .rename(columns={'start_time_hour':'time'})
                              .round(2)
        )
        heatmap_cells_viewers  = compute_heatmap_cells(time_df_viewers)
        feature_scores_viewers = compute_feature_scores(time_preds_viewers, selected_game)
    else:
        heatmap_cells_viewers, feature_scores_viewers = [], []

    # insights (cached)
    game_insights = compute_game_insights(df_pred, selected_stream)
    tag_insights  = cached_tag_insights(model_idx, selected_stream, today)

    # SHAP (cached)
    shap_plots = cached_shap_blocks(model_idx)

    # render
    return render_template_string(
        TEMPLATE_V2,
        today_name=today,
        stream_opts=sorted(df_inf['stream_name'].unique()),
        game_opts=game_opts,
        start_opts=start_opts,
        tag_opts=tag_opts,
        selected_stream=selected_stream,
        selected_game=selected_game,
        selected_start_time=selected_start_time,
        selected_tags=selected_tags,
        pred_result=None,
        game_insights=game_insights,
        tag_insights=tag_insights,
        heatmap_cells=heatmap_cells,
        feature_scores=feature_scores,
        heatmap_cells_viewers=heatmap_cells_viewers,
        feature_scores_viewers=feature_scores_viewers,
        all_tags=all_tags,
        shap_plots=shap_plots,
        legend_tag_opts=legend_tags,
    )

# Optional: avoid timeouts for iOS touch icon requests tying up the sole worker
@app.route('/apple-touch-icon-precomposed.png', methods=['GET'])
@app.route('/apple-touch-icon.png', methods=['GET'])
def touch_icon():
    return ("", 204)