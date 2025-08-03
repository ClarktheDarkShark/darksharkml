# dashboard_v3.py
# --------------------------------------------------------------------------
import numpy as np, pandas as pd, pytz
from datetime import datetime
from flask import Blueprint, render_template_string, request, Response, stream_with_context

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
    _get_last_row_for_stream,
)

from services.recommendation_service import get_stream_recommendations

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

# app/routes/dashboard_v3.py

# ───────────────────────── HTML template (unchanged) ─────────────────────────
TEMPLATE_V3 = """<!doctype html> ... (exact same HTML you pasted) ... """

@dash_v3.route("/v3", methods=["GET"])
def show_feature_insights_v3():
    stream = request.args.get("stream")

    @stream_with_context
    def generate():
        yield " " * 2048  # 2 KiB HTTP body = router sees data ⇒ timer resets
        ctx = get_stream_recommendations(stream)   # heavy bit
        html = render_template_string(TEMPLATE_V3, **ctx, pred_result={})
        yield html

    return Response(generate(), mimetype="text/html")
