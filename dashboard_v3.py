# dashboard_v3.py
# --------------------------------------------------------------------------
import numpy as np, pandas as pd, pytz
from datetime import datetime
from flask import Blueprint, render_template_string, request

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
    _get_last_row_for_stream,
)

dash_v3 = Blueprint("dash_v3", __name__, url_prefix="")  # route at /v3
TZ = pytz.timezone("US/Eastern")

# ───────────────────────────────────────── HTML (dark-mode) ────────────────
TEMPLATE_V3 = """
<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stream AI · {{ selected_stream }}</title>
<style>
:root{--bg:#121212;--card:#1e1e1e;--fg:#e0e0e0;--muted:#8e8e8e;--acc:#1e88e5;--radius:8px}
html,body{margin:0;background:var(--bg);color:var(--fg);font-family:system-ui,Segoe UI,Roboto,Arial,sans-serif;font-size:15px}
h1{font-size:1.4rem;margin:.6rem 0 .3rem}
h2{font-size:1.1rem;margin:1rem 0 .4rem}
.card{background:var(--card);border-radius:var(--radius);padding:.8rem 1rem;box-shadow:0 2px 8px rgba(0,0,0,.45)}
.grid{display:grid;gap:12px}
@media(min-width:720px){.grid-3{grid-template-columns:repeat(3,1fr)}}
table{width:100%;border-collapse:collapse;font-size:.9rem;margin-bottom:.8rem}
th,td{padding:.45rem .6rem;text-align:center}
th{background:#1a1a1a;font-weight:600}
tbody tr:nth-child(even){background:#181818}
tbody tr:hover{background:#222}
.badge{display:inline-block;background:var(--acc);padding:.15rem .55rem;border-radius:6px;font-size:.72rem;color:#fff}
.note{color:var(--muted);font-size:.82rem;margin:.2rem 0 .8rem}
</style>
</head><body>

<h1>AI Stream Insights</h1>
<div class="note">{{ today_name }} &nbsp;|&nbsp; Streamer: <strong>{{ selected_stream }}</strong></div>

<!-- Top-3 recommendation tables -->
<h2>Top Recommendations</h2>

{% for title, rows in [("Subscriptions", top3_subs),
                       ("Follower Growth", top3_followers),
                       ("Viewers", top3_viewers)] %}
<h3>{{ title }}</h3>
<table class="card">
  <thead><tr><th>Game</th><th>Start</th><th>Dur</th><th>{{ title }}</th><th>Conf.</th></tr></thead>
  <tbody>
  {% for r in rows %}
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
{% endfor %}

<!-- where you can paste heat-map / SHAP blocks from v2 -->
</body></html>
"""

# ───────────────────────────────────────── route ────────────────────────────
@dash_v3.route("/v3", methods=["GET"])
def show_feature_insights_v3():
    # 1) load artefacts
    pipes, df_inf, feats, _, start_opts, dur_opts, metrics = get_predictor_artifacts()
    if not pipes or df_inf is None:
        return "Model not ready", 503

    # 2) choose streamer (query param or default to most frequent)
    sel_stream = request.args.get("stream") or df_inf["stream_name"].mode()[0]
    if sel_stream not in df_inf["stream_name"].unique():
        sel_stream = df_inf["stream_name"].mode()[0]

    # 3) streamer-specific history
    baseline = _get_last_row_for_stream(df_inf, sel_stream)
    legend_games = df_inf.loc[df_inf.stream_name == sel_stream, "game_category"].unique().tolist()
    legend_tags = sorted(
        {t for tags in df_inf.loc[df_inf.stream_name == sel_stream, "raw_tags"].dropna() for t in tags}
    )

    # 4) pick three pipelines
    pipe_sub = pipes[0]
    pipe_fol = pipes[1] if len(pipes) > 1 else pipes[0]
    pipe_view= pipes[2] if len(pipes) > 2 else pipes[0]

    # 5) helper: top-3 grid for one metric
    def _top3(pipe):
        df = _infer_grid_for_game(
            pipe, df_inf, feats,
            stream_name=sel_stream,
            override_tags=legend_tags,
            start_times=start_opts,
            durations=dur_opts,
            category_options=legend_games,
            top_n=3,
            unique_scores=True,
        )
        return df.to_dict("records")

    top3_subs      = _top3(pipe_sub)
    top3_followers = _top3(pipe_fol)
    top3_viewers   = _top3(pipe_view)

    # 6) render
    today_name = datetime.now(TZ).strftime("%A, %B %d, %Y")
    return render_template_string(
        TEMPLATE_V3,
        selected_stream=sel_stream,
        today_name=today_name,
        features=feats,
        top3_subs=top3_subs,
        top3_followers=top3_followers,
        top3_viewers=top3_viewers,
        pred_result={},  # keep placeholder so template variables exist
    )
