# dashboard_v3.py  – single-file, dark-mode, modern

import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
from flask import Blueprint, render_template_string, request

from predictor import (
    get_predictor_artifacts,
    _get_last_row_for_stream,
    _infer_grid_for_game,
)

# ─────────────────────────────────────────────────────────────────────────────
dash_v3 = Blueprint("dash_v3", __name__, url_prefix="")  # route at /v3
_shap_cache = {"pipe_id": None, "plots": None}
tz_EST = pytz.timezone("US/Eastern")

# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Stream AI Dashboard</title>
<style>
/* ---------- LIGHTWEIGHT DARK THEME ---------- */
:root{
  --bg:#111;--card:#1c1c1c;--fg:#e6e6e6;--muted:#888;
  --accent:#1e88e5;--accent-dark:#1565c0;--radius:9px;
  --good:#26c6da;--bad:#ef5350;
}
html,body{margin:0;padding:0;font-family:system-ui,Segoe UI,Roboto,"Helvetica Neue",Arial,sans-serif;background:var(--bg);color:var(--fg);font-size:15px}
h1{margin:.4rem 0;font-size:1.35rem}
h2{margin:1rem 0 .6rem;font-size:1.15rem}
.card{background:var(--card);border-radius:var(--radius);padding:.8rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,.45)}
.grid{display:grid;gap:12px}
@media(min-width:750px){.grid-cols-3{grid-template-columns:repeat(3,1fr)}}
table{width:100%;border-collapse:collapse;font-size:.92rem}
th,td{padding:.45rem .6rem;text-align:center}
th{background:#1a1a1a;font-weight:600}
tbody tr:nth-child(even){background:#181818}
tbody tr:hover{background:#222}
.badge{display:inline-block;padding:.15rem .55rem;font-size:.75rem;border-radius:6px;background:var(--accent);color:#fff}
button{cursor:pointer;background:var(--card);color:var(--fg);border:1px solid #333;border-radius:7px;font-size:.88rem;padding:.3rem .7rem;margin:.15rem .25rem}
button.selected{background:var(--accent);border-color:var(--accent);color:#fff}
button:hover{background:var(--accent-dark);color:#fff}
</style>
<script src="https://cdn.plot.ly/plotly-latest.min.js" defer></script>
{{ shap_plots.js | safe }}
</head>
<body>
<h1>AI Stream Insights – {{ today_name }}</h1>

<!-- SUMMARY CARDS -->
<div class="grid {% if viewport_large %}grid-cols-3{% endif %}">
  <div class="card">
    <h2>Best Combo (Subs)</h2>
    {% if best_sub_combo %}
      <p><span class="badge">{{ best_sub_combo.score }}</span> predicted subs</p>
      <p>{{ best_sub_combo.game }} • {{ "%02d:00"|format(best_sub_combo.hour) }} • {{ best_sub_combo.dur }}h</p>
      <p>Tags: {{ best_sub_combo.tags|join(", ") if best_sub_combo.tags else "None" }}</p>
    {% else %}<p>No data</p>{% endif %}
  </div>
  <div class="card">
    <h2>Model Quality</h2>
    <p>MAE (Subs): {{ metrics_sub.get("model_mae","–")|round(2) }}</p>
    <p>MAE (Follows): {{ metrics_fol.get("model_mae","–")|round(2) }}</p>
  </div>
  <div class="card">
    <h2>Context</h2>
    <p>Streamer: <strong>{{ selected_stream }}</strong></p>
    <p>Features: {{ features|length }}</p>
  </div>
</div>

<!-- TOP-3 tables -->
<h2>Top-3 Combos</h2>
{% for label, rows in [("Subs",top3_subs),("Followers",top3_follows),("Viewers",top3_views)] %}
<h3>{{ label }}</h3>
<table class="card">
<thead><tr><th>Game</th><th>Start</th><th>Dur</th><th>{{ label }}</th><th>Conf.</th></tr></thead>
<tbody>
 {% for r in rows %}
  <tr><td>{{ r.game_category }}</td><td>{{ "%02d:00"|format(r.start_time_hour) }}</td>
      <td>{{ r.stream_duration }}h</td><td>{{ r.y_pred|round(2) }}</td><td>{{ r.conf|round(2) }}</td></tr>
 {% endfor %}
</tbody>
</table>
{% endfor %}

<!-- SHAP or other sections here ... -->
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def best_overall(pipe, baseline, games, hours, durs, tags, features, k_tags=3):
    """Return the single best (game,hour,dur,tags) by predicted subs."""
    best = None
    # start with 0-tag baseline
    base_pred = pipe.predict(_row(baseline, games[0], hours[0], [], features))[0]
    best = {"score": base_pred, "tags": (), "game": games[0], "hour": hours[0], "dur": durs[0]}
    for game in games:
        for hour in hours:
            for dur in durs:
                # cheap heuristic: test single tags, keep top m, then comb
                tag_scores = []
                for t in tags:
                    y = pipe.predict(_row(baseline, game, hour, [t], features))[0]
                    tag_scores.append((t, y))
                tag_scores.sort(key=lambda x: x[1], reverse=True)
                top_tags = [t for t, _ in tag_scores[:6]]
                # brute up to k_tags on reduced set
                from itertools import combinations
                for k in range(0, k_tags + 1):
                    for combo in combinations(top_tags, k):
                        y = pipe.predict(_row(baseline, game, hour, list(combo), features))[0]
                        if y > best["score"]:
                            best = {"score": y, "tags": combo, "game": game, "hour": hour, "dur": dur}
    return best


def _row(baseline, game, hour, tags, features):
    r = baseline.copy()
    r["game_category"] = game
    r["start_time_hour"] = hour
    r["day_of_week"] = datetime.now(tz_EST).strftime("%A")
    r["start_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    r["start_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    r["is_weekend"] = r["day_of_week"] in ("Saturday", "Sunday")
    for t in tags:  # minimal tag set
        r[f"tag_{t}"] = 1
    return pd.DataFrame([r])[features]


def _top3(pipe, df_inf, feats, stream, games, hours, durs, legend_tags):
    df = _infer_grid_for_game(
        pipe,
        df_inf,
        feats,
        stream_name=stream,
        override_tags=legend_tags,
        start_times=hours,
        durations=durs,
        category_options=games,
        top_n=3,
        unique_scores=True,
    )
    return df.to_dict("records")


# ─────────────────────────────────────────────────────────────────────────────
# route
# ─────────────────────────────────────────────────────────────────────────────
@dash_v3.route("/v3", methods=["GET"])
def insights():
    # artifacts
    pipes, df_inf, feats, cat_opts, start_opts, dur_opts, metrics_list = get_predictor_artifacts()
    if not (pipes and df_inf is not None):
        return "Predictor not ready", 503

    # choose streamer
    sel_stream = request.args.get("stream") or df_inf["stream_name"].mode()[0]
    if sel_stream not in df_inf["stream_name"].unique():
        sel_stream = df_inf["stream_name"].mode()[0]

    baseline = _get_last_row_for_stream(df_inf, sel_stream)
    legend_games = df_inf.loc[df_inf["stream_name"] == sel_stream, "game_category"].unique().tolist()
    legend_tags = sorted(
        {t for tags in df_inf.loc[df_inf["stream_name"] == sel_stream, "raw_tags"].dropna() for t in tags}
    )

    # models
    pipe_sub, pipe_fol, pipe_view = pipes[0], pipes[min(1, len(pipes) - 1)], pipes[min(2, len(pipes) - 1)]
    metrics_sub, metrics_fol = metrics_list[0], metrics_list[min(1, len(metrics_list) - 1)]

    # top-3 grids
    top3_subs = _top3(pipe_sub, df_inf, feats, sel_stream, legend_games, start_opts, dur_opts, legend_tags)
    top3_fols = _top3(pipe_fol, df_inf, feats, sel_stream, legend_games, start_opts, dur_opts, legend_tags)
    top3_views = _top3(pipe_view, df_inf, feats, sel_stream, legend_games, start_opts, dur_opts, legend_tags)

    # best overall tag combo (subs)
    best_combo = best_overall(
        pipe_sub,
        baseline,
        legend_games,
        start_opts,
        dur_opts,
        legend_tags,
        feats,
        k_tags=3,
    )

    # simple shape: adapt SHAP cache from v2 if desired
    shap_plots = {"js": ""}

    return render_template_string(
        TEMPLATE,
        today_name=datetime.now(tz_EST).strftime("%A · %d %b %Y"),
        selected_stream=sel_stream,
        features=feats,
        viewport_large=True,
        metrics_sub=metrics_sub,
        metrics_fol=metrics_fol,
        best_sub_combo=best_combo,
        top3_subs=top3_subs,
        top3_follows=top3_fols,
        top3_views=top3_views,
        shap_plots=shap_plots,
    )
