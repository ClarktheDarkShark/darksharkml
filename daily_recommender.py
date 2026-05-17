from __future__ import annotations

import itertools
import os
from datetime import datetime
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import pytz
from flask import Blueprint, render_template_string, request

from recommender_modeling import TARGET_STREAM, normalize_tags, tags_to_text


daily_recommender = Blueprint("daily_recommender", __name__)

ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "recommender_artifacts.joblib")
TZ = pytz.timezone("US/Eastern")


@lru_cache(maxsize=1)
def load_recommender_artifact():
    if not os.path.exists(ARTIFACT_PATH):
        return None
    return joblib.load(ARTIFACT_PATH)


def _parse_date(value: str | None) -> datetime:
    if value:
        try:
            return TZ.localize(datetime.strptime(value, "%Y-%m-%d"))
        except ValueError:
            pass
    return datetime.now(TZ)


def _last_row(df: pd.DataFrame, stream_name: str) -> pd.Series:
    rows = df[df["stream_name_lc"] == stream_name.lower()].sort_values(["stream_date", "stream_start_time"])
    if rows.empty:
        raise ValueError(f"No rows found for stream={stream_name!r}")
    return rows.iloc[-1]


def _base_row(artifact: dict, stream_name: str, target_date: datetime) -> pd.DataFrame:
    df = artifact["df_for_inf"].copy()
    last = _last_row(df, stream_name)
    base = last.to_frame().T

    day_name = target_date.strftime("%A")
    base["day_of_week"] = day_name
    base["is_weekend"] = day_name in {"Saturday", "Sunday"}
    last_date = pd.Timestamp(last["stream_date"]).tz_localize(None).normalize()
    today = pd.Timestamp(target_date.date())
    base["days_since_previous_stream"] = max(int((today - last_date).days), 0)
    base["raw_tags"] = [normalize_tags(last.get("raw_tags", []))]
    base["raw_tags_text"] = base["raw_tags"].apply(tags_to_text)
    return base


def _predict(model_info: dict, frame: pd.DataFrame) -> np.ndarray:
    preds = model_info["pipeline"].predict(frame[model_info["features"]])
    return np.asarray(preds, dtype=float)


def _build_grid(artifact: dict, stream_name: str, target_date: datetime) -> pd.DataFrame:
    base = _base_row(artifact, stream_name, target_date)
    categories = artifact.get("category_options") or sorted(
        artifact["df_for_inf"]["game_category"].dropna().astype(str).unique().tolist()
    )
    combos = list(itertools.product(categories, artifact["start_times"], artifact["durations"]))
    grid = pd.DataFrame(combos, columns=["game_category", "start_time_hour", "stream_duration"])
    rows = base.loc[base.index.repeat(len(grid))].reset_index(drop=True)
    rows["game_category"] = grid["game_category"].values
    rows["start_time_hour"] = grid["start_time_hour"].astype(float).values
    rows["stream_duration"] = grid["stream_duration"].astype(float).values
    return rows


def _score_grid(artifact: dict, stream_name: str, target_date: datetime) -> pd.DataFrame:
    grid = _build_grid(artifact, stream_name, target_date)
    models = artifact["models"]
    grid["expected_subs"] = np.maximum(_predict(models["total_subscriptions"], grid), 0.0)
    grid["expected_followers"] = _predict(models["net_follower_change"], grid)
    grid["combined_score"] = grid["expected_subs"] + np.maximum(grid["expected_followers"], 0.0)
    return grid


def _best_by(grid: pd.DataFrame, group_col: str, metric: str, n: int = 10) -> list[dict]:
    idx = grid.groupby(group_col)[metric].idxmax()
    cols = list(
        dict.fromkeys(
            [group_col, "game_category", "start_time_hour", "stream_duration", "expected_subs", "expected_followers"]
        )
    )
    rows = grid.loc[idx, cols].sort_values(metric, ascending=False).head(n).copy()
    rows["start_time"] = rows["start_time_hour"].astype(int).map(lambda h: f"{h:02d}:00")
    rows["duration_hours"] = (rows["stream_duration"].astype(float) / 60.0).round(1)
    rows["expected_subs"] = rows["expected_subs"].round(2)
    rows["expected_followers"] = rows["expected_followers"].round(2)
    return rows.to_dict("records")


def _top_rows(grid: pd.DataFrame, metric: str, n: int) -> list[dict]:
    rows = grid.sort_values(metric, ascending=False).head(n).copy()
    rows["start_time"] = rows["start_time_hour"].astype(int).map(lambda h: f"{h:02d}:00")
    rows["duration_hours"] = (rows["stream_duration"].astype(float) / 60.0).round(1)
    rows["expected_subs"] = rows["expected_subs"].round(2)
    rows["expected_followers"] = rows["expected_followers"].round(2)
    return rows[
        ["game_category", "start_time", "duration_hours", "expected_subs", "expected_followers", "combined_score"]
    ].to_dict("records")


def _tag_effects(artifact: dict, stream_name: str, target_date: datetime, top_row: pd.Series, n: int = 10) -> list[dict]:
    tags = artifact.get("tag_options") or []
    if not tags:
        return []

    base = _base_row(artifact, stream_name, target_date)
    for col in ["game_category", "start_time_hour", "stream_duration"]:
        base[col] = top_row[col]

    current_tags = set(normalize_tags(base.at[base.index[0], "raw_tags"]))
    baseline_subs = float(np.maximum(_predict(artifact["models"]["total_subscriptions"], base)[0], 0.0))
    rows = []
    for tag in tags:
        mod = base.copy()
        mod_tags = set(current_tags)
        mod_tags.add(tag)
        mod.at[mod.index[0], "raw_tags"] = sorted(mod_tags)
        mod.at[mod.index[0], "raw_tags_text"] = tags_to_text(sorted(mod_tags))
        subs = float(np.maximum(_predict(artifact["models"]["total_subscriptions"], mod)[0], 0.0))
        rows.append({"tag": tag, "expected_subs": round(subs, 2), "delta_subs": round(subs - baseline_subs, 2)})
    return sorted(rows, key=lambda r: r["delta_subs"], reverse=True)[:n]


TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Daily Stream Recommender</title>
  <style>
    :root { --bg:#101214; --panel:#181c20; --line:#2b3238; --text:#e9eef2; --muted:#9aa7b2; --accent:#4aa3ff; --good:#6fd18b; --warn:#f2b84b; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font-family:Arial, Helvetica, sans-serif; }
    main { max-width:1180px; margin:0 auto; padding:24px; }
    h1 { font-size:28px; margin:0 0 8px; letter-spacing:0; }
    h2 { font-size:18px; margin:28px 0 10px; letter-spacing:0; }
    .meta { color:var(--muted); margin-bottom:18px; }
    form { display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin:16px 0 22px; }
    label { display:flex; flex-direction:column; gap:6px; color:var(--muted); font-size:13px; }
    input, button { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:9px 11px; font-size:14px; }
    button { background:var(--accent); border-color:var(--accent); color:#06111d; font-weight:700; cursor:pointer; }
    .notice { border:1px solid var(--line); border-left:4px solid var(--warn); background:var(--panel); padding:12px 14px; border-radius:6px; color:var(--muted); margin:16px 0; }
    .grid { display:grid; grid-template-columns:repeat(2, minmax(0, 1fr)); gap:18px; }
    table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    th, td { padding:9px 10px; border-bottom:1px solid var(--line); text-align:left; font-size:14px; }
    th { color:var(--muted); font-weight:700; background:#15191d; }
    tr:last-child td { border-bottom:0; }
    .num { text-align:right; font-variant-numeric:tabular-nums; }
    .pill { display:inline-block; color:#06140b; background:var(--good); border-radius:999px; padding:2px 7px; font-size:12px; font-weight:700; }
    @media (max-width: 820px) { main { padding:16px; } .grid { grid-template-columns:1fr; } table { font-size:13px; } }
  </style>
</head>
<body>
<main>
  <h1>Daily Stream Recommender</h1>
  <div class="meta">{{ stream }} · {{ date_label }} · artifact {{ created_at }}</div>

  <form method="get" action="/recommend">
    <label>Stream
      <input name="stream" value="{{ stream }}">
    </label>
    <label>Date
      <input type="date" name="date" value="{{ date_value }}">
    </label>
    <label>Rows
      <input type="number" name="top_n" min="3" max="30" value="{{ top_n }}">
    </label>
    <button type="submit">Rank</button>
  </form>

  {% if message %}<div class="notice">{{ message }}</div>{% endif %}

  <h2>Top Recommendations By Subscriptions <span class="pill">promoted</span></h2>
  <table>
    <thead><tr><th>Game / Category</th><th>Start</th><th>Hours</th><th class="num">Subs</th><th class="num">Followers</th></tr></thead>
    <tbody>
    {% for row in top_subs %}
      <tr><td>{{ row.game_category }}</td><td>{{ row.start_time }}</td><td>{{ row.duration_hours }}</td><td class="num">{{ row.expected_subs }}</td><td class="num">{{ row.expected_followers }}</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>Top Recommendations By Followers <span class="meta">experimental</span></h2>
  <table>
    <thead><tr><th>Game / Category</th><th>Start</th><th>Hours</th><th class="num">Subs</th><th class="num">Followers</th></tr></thead>
    <tbody>
    {% for row in top_followers %}
      <tr><td>{{ row.game_category }}</td><td>{{ row.start_time }}</td><td>{{ row.duration_hours }}</td><td class="num">{{ row.expected_subs }}</td><td class="num">{{ row.expected_followers }}</td></tr>
    {% endfor %}
    </tbody>
  </table>

  <div class="grid">
    <section>
      <h2>Best Categories For Subs</h2>
      <table><thead><tr><th>Category</th><th>Start</th><th>Hours</th><th class="num">Subs</th></tr></thead><tbody>
      {% for row in category_subs %}
        <tr><td>{{ row.game_category }}</td><td>{{ row.start_time }}</td><td>{{ row.duration_hours }}</td><td class="num">{{ row.expected_subs }}</td></tr>
      {% endfor %}
      </tbody></table>
    </section>
    <section>
      <h2>Best Start Times For Subs</h2>
      <table><thead><tr><th>Start</th><th>Category</th><th>Hours</th><th class="num">Subs</th></tr></thead><tbody>
      {% for row in hour_subs %}
        <tr><td>{{ row.start_time }}</td><td>{{ row.game_category }}</td><td>{{ row.duration_hours }}</td><td class="num">{{ row.expected_subs }}</td></tr>
      {% endfor %}
      </tbody></table>
    </section>
    <section>
      <h2>Best Durations For Subs</h2>
      <table><thead><tr><th>Hours</th><th>Category</th><th>Start</th><th class="num">Subs</th></tr></thead><tbody>
      {% for row in duration_subs %}
        <tr><td>{{ row.duration_hours }}</td><td>{{ row.game_category }}</td><td>{{ row.start_time }}</td><td class="num">{{ row.expected_subs }}</td></tr>
      {% endfor %}
      </tbody></table>
    </section>
    <section>
      <h2>Tag Effects For Top Sub Row</h2>
      <table><thead><tr><th>Tag</th><th class="num">Subs</th><th class="num">Delta</th></tr></thead><tbody>
      {% for row in tag_effects %}
        <tr><td>{{ row.tag }}</td><td class="num">{{ row.expected_subs }}</td><td class="num">{{ row.delta_subs }}</td></tr>
      {% endfor %}
      </tbody></table>
    </section>
  </div>
</main>
</body>
</html>
"""


@daily_recommender.route("/recommend", methods=["GET"])
def recommend_today():
    artifact = load_recommender_artifact()
    if artifact is None:
        return render_template_string(
            TEMPLATE,
            stream=TARGET_STREAM,
            date_label=datetime.now(TZ).strftime("%A, %B %d, %Y"),
            date_value=datetime.now(TZ).strftime("%Y-%m-%d"),
            created_at="missing",
            top_n=10,
            message="recommender_artifacts.joblib is missing. Run python train_recommender.py first.",
            top_subs=[],
            top_followers=[],
            category_subs=[],
            hour_subs=[],
            duration_subs=[],
            tag_effects=[],
        )

    stream = (request.args.get("stream") or artifact.get("target_stream") or TARGET_STREAM).strip()
    target_date = _parse_date(request.args.get("date"))
    try:
        top_n = max(3, min(30, int(request.args.get("top_n", "10"))))
    except ValueError:
        top_n = 10

    message = artifact.get("notes", {}).get("follower_model", "")
    try:
        grid = _score_grid(artifact, stream, target_date)
    except Exception as exc:
        message = str(exc)
        grid = pd.DataFrame()

    if grid.empty:
        top_subs = top_followers = category_subs = hour_subs = duration_subs = tag_effects = []
    else:
        top_subs = _top_rows(grid, "expected_subs", top_n)
        top_followers = _top_rows(grid, "expected_followers", top_n)
        category_subs = _best_by(grid, "game_category", "expected_subs", top_n)
        hour_subs = _best_by(grid, "start_time_hour", "expected_subs", top_n)
        duration_subs = _best_by(grid, "stream_duration", "expected_subs", top_n)
        tag_effects = _tag_effects(artifact, stream, target_date, grid.sort_values("expected_subs", ascending=False).iloc[0])

    return render_template_string(
        TEMPLATE,
        stream=stream,
        date_label=target_date.strftime("%A, %B %d, %Y"),
        date_value=target_date.strftime("%Y-%m-%d"),
        created_at=str(artifact.get("created_at_utc", ""))[:19],
        top_n=top_n,
        message=message,
        top_subs=top_subs,
        top_followers=top_followers,
        category_subs=category_subs,
        hour_subs=hour_subs,
        duration_subs=duration_subs,
        tag_effects=tag_effects,
    )
