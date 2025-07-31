# dashboard_v3.py — drop-in blueprint

import itertools
from datetime import datetime
from flask import Blueprint, render_template_string, request
import numpy as np, pandas as pd, pytz

from predictor import (
    get_predictor_artifacts,
    _get_last_row_for_stream,
    _infer_grid_for_game,
)

dash_v3 = Blueprint("dash_v3", __name__, url_prefix="")   # route at /v3
tz_EST  = pytz.timezone("US/Eastern")

# ─────────────────────────────────────────────────────────────────────────────
# HTML template
# ─────────────────────────────────────────────────────────────────────────────
TEMPLATE = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stream AI · {{ selected_stream }}</title>
<style>
:root{--bg:#111;--card:#1d1d1d;--fg:#eaeaea;--muted:#888;--acc:#1e88e5;--acc-d:#1565c0;--radius:8px}
html,body{margin:0;padding:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;font-size:15px}
h1{font-size:1.4rem;margin:.6rem 0 .2rem}
h2{font-size:1.1rem;margin:1.1rem 0 .4rem}
.card{background:var(--card);border-radius:var(--radius);padding:.75rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,.45)}
.grid{display:grid;gap:12px}
@media(min-width:700px){.grid-3{grid-template-columns:repeat(3,1fr)}}
table{width:100%;border-collapse:collapse;font-size:.9rem}
th,td{padding:.45rem .6rem;text-align:center}
th{background:#1a1a1a;font-weight:600}
tbody tr:nth-child(even){background:#181818}
tbody tr:hover{background:#222}
.badge{display:inline-block;padding:.15rem .55rem;font-size:.72rem;border-radius:5px;background:var(--acc);color:#fff}
</style>
</head><body>
<h1>AI Stream Insights · {{ today_long }}</h1>

<!-- summary -->
<div class="grid grid-3">
  <div class="card">
    <h2>Best Combo<br><span style="font-weight:400">(Subs)</span></h2>
    {% if best_combo %}
      <p><span class="badge">{{ best_combo.score }}</span> predicted subs</p>
      <p>{{ best_combo.game }} • {{ "%02d:00"|format(best_combo.hour) }} • {{ best_combo.dur }}h</p>
      <p style="font-size:.8rem">Tags: {{ best_combo.tags|join(", ") or "None" }}</p>
    {% else %}No data{% endif %}
  </div>
  <div class="card">
    <h2>Model MAE</h2>
    <p>Subs: {{ metrics_sub.model_mae|round(2) }}</p>
    <p>Follows: {{ metrics_fol.model_mae|round(2) }}</p>
  </div>
  <div class="card">
    <h2>Context</h2>
    <p>Streamer: <strong>{{ selected_stream }}</strong></p>
    <p>#Features: {{ features|length }}</p>
  </div>
</div>

<h2>Top-3 Combos</h2>
{% for lbl, rows in [("Subs", top3_subs), ("Followers", top3_fol), ("Viewers", top3_view)] %}
<h3>{{ lbl }}</h3>
<table class="card">
  <thead><tr><th>Game</th><th>Start</th><th>Dur</th><th>{{ lbl }}</th><th>Conf.</th></tr></thead>
  <tbody>{% for r in rows %}
    <tr><td>{{ r.game_category }}</td><td>{{ "%02d:00"|format(r.start_time_hour) }}</td>
        <td>{{ r.stream_duration }}h</td><td>{{ r.y_pred|round(2) }}</td><td>{{ r.conf|round(2) }}</td></tr>
  {% endfor %}</tbody>
</table>
{% endfor %}
</body></html>
"""

# ─────────────────────────────────────────────────────────────────────────────
# helpers 
# ─────────────────────────────────────────────────────────────────────────────
def _row(baseline, game, hour, dur, tags, feats):
    r = baseline.copy()
    r["game_category"], r["start_time_hour"], r["stream_duration"] = game, hour, dur
    dow = datetime.now(tz_EST).strftime("%A")
    r["day_of_week"] = dow
    r["start_hour_sin"], r["start_hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    r["is_weekend"] = dow in ("Saturday","Sunday")
    for t in tags: r[f"tag_{t}"] = 1
    return pd.DataFrame([r])[feats]

def best_combo(pipe, baseline, games, hours, durs, tags, feats):
    best = None
    for game in games:
        for hour in hours:
            for dur in durs:
                y0 = pipe.predict(_row(baseline, game, hour, dur, [], feats))[0]
                if best is None or y0 > best["score"]:
                    best = dict(score=y0, game=game, hour=hour, dur=dur, tags=())
                # quick heuristic: test adding each tag individually
                for t in tags:
                    y = pipe.predict(_row(baseline, game, hour, dur, [t], feats))[0]
                    if y > best["score"]:
                        best = dict(score=y, game=game, hour=hour, dur=dur, tags=(t,))
    return best

def top3_grid(pipe, df_inf, feats, stream, games, hours, durs, base_tags):
    df = _infer_grid_for_game(
        pipe, df_inf, feats,
        stream_name=stream,
        override_tags=base_tags,
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
def v3():
    pipes, df_inf, feats, _, hours, durs, metrics = get_predictor_artifacts()
    if not pipes: return "Model not loaded", 503

    stream = request.args.get("stream") or df_inf["stream_name"].mode()[0]
    if stream not in df_inf["stream_name"].unique():
        stream = df_inf["stream_name"].mode()[0]

    baseline = _get_last_row_for_stream(df_inf, stream)
    games    = df_inf.loc[df_inf.stream_name==stream, "game_category"].unique().tolist()
    tags     = sorted({t for lst in df_inf.loc[df_inf.stream_name==stream, "raw_tags"].dropna() for t in lst})

    pipe_sub, pipe_fol, pipe_view = pipes[0], pipes[min(1,len(pipes)-1)], pipes[min(2,len(pipes)-1)]
    metrics_sub, metrics_fol      = metrics[0], metrics[min(1,len(metrics)-1)]

    best_sub_combo = best_combo(pipe_sub, baseline, games, hours, durs, tags, feats)
    top3_sub = top3_grid(pipe_sub, df_inf, feats, stream, games, hours, durs, tags)
    top3_fol = top3_grid(pipe_fol, df_inf, feats, stream, games, hours, durs, tags)
    top3_view= top3_grid(pipe_view, df_inf, feats, stream, games, hours, durs, tags)

    return render_template_string(
        TEMPLATE,
        today_long=datetime.now(tz_EST).strftime("%A · %d %b %Y"),
        selected_stream=stream,
        features=feats,
        metrics_sub=metrics_sub,
        metrics_fol=metrics_fol,
        best_combo=best_sub_combo,
        top3_subs = top3_sub,
        top3_fol  = top3_fol,
        top3_view = top3_view,
    )
