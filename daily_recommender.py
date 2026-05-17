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
DEFAULT_START_HOUR = 19
ALT_TIME_MIN_RELATIVE_GAIN = 0.25
ALT_TIME_MIN_SCORE_GAIN = 0.35


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
    opportunity = artifact.get("opportunity_stats", {})
    categories = opportunity.get("candidate_categories") or artifact.get("category_options") or []
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
    grid["model_subs"] = np.maximum(_predict(models["total_subscriptions"], grid), 0.0)
    grid["model_followers"] = _predict(models["net_follower_change"], grid)

    opportunity = artifact.get("opportunity_stats", {})
    category_stats = opportunity.get("category_stats", {})
    hour_factors = opportunity.get("hour_factors", {})
    duration_factors = opportunity.get("duration_factors", {})
    global_rates = opportunity.get("global_rates", {})
    profile = opportunity.get("stream_profile", {})

    base_subs = max(float(profile.get("recent_subs_per_stream", 1.0)), 0.25)
    base_followers = max(float(profile.get("recent_positive_followers_per_stream", 0.25)), 0.1)
    global_sub_rate = max(float(global_rates.get("fighting_subs_per_100_viewers", 1.0)), 1e-6)
    global_follow_rate = max(float(global_rates.get("fighting_followers_per_100_viewers", 0.25)), 1e-6)

    def _cat_value(category, key, fallback):
        return float(category_stats.get(str(category), {}).get(key, fallback))

    def _factor(mapping, key, metric):
        item = mapping.get(int(key), mapping.get(str(int(key)), {}))
        return float(item.get(metric, 1.0))

    grid["category_rows"] = grid["game_category"].map(lambda c: int(category_stats.get(str(c), {}).get("rows", 0)))
    grid["category_source"] = grid["game_category"].map(lambda c: category_stats.get(str(c), {}).get("source", "curated"))
    grid["confidence"] = grid["game_category"].map(lambda c: _cat_value(c, "confidence", 0.12))
    grid["category_sub_rate"] = grid["game_category"].map(
        lambda c: _cat_value(c, "subs_per_100_viewers", global_sub_rate)
    )
    grid["category_follower_rate"] = grid["game_category"].map(
        lambda c: _cat_value(c, "followers_per_100_viewers", global_follow_rate)
    )
    grid["hour_sub_factor"] = grid["start_time_hour"].map(lambda h: _factor(hour_factors, h, "subs"))
    grid["hour_follower_factor"] = grid["start_time_hour"].map(lambda h: _factor(hour_factors, h, "followers"))
    grid["duration_bucket"] = ((grid["stream_duration"].astype(float) / 60.0).round().clip(2, 7) * 60).astype(int)
    grid["duration_sub_factor"] = grid["duration_bucket"].map(lambda d: _factor(duration_factors, d, "subs"))
    grid["duration_follower_factor"] = grid["duration_bucket"].map(lambda d: _factor(duration_factors, d, "followers"))
    grid["evidence_weight"] = 0.65 + 0.35 * grid["confidence"]

    relative_sub = grid["category_sub_rate"] / global_sub_rate
    relative_follower = grid["category_follower_rate"] / global_follow_rate
    grid["opportunity_subs"] = (
        base_subs * relative_sub * grid["hour_sub_factor"] * grid["duration_sub_factor"] * grid["evidence_weight"]
    )
    grid["opportunity_followers"] = (
        base_followers
        * relative_follower
        * grid["hour_follower_factor"]
        * grid["duration_follower_factor"]
        * grid["evidence_weight"]
    )
    grid["expected_subs"] = 0.35 * grid["model_subs"] + 0.65 * grid["opportunity_subs"]
    grid["expected_followers"] = 0.25 * np.maximum(grid["model_followers"], 0.0) + 0.75 * grid["opportunity_followers"]
    grid["opportunity_index"] = 100.0 * relative_sub * grid["hour_sub_factor"] * grid["duration_sub_factor"]
    grid["growth_score"] = grid["expected_subs"] + 0.6 * grid["expected_followers"] + 0.1 * grid["confidence"]
    return grid


def _format_rows(rows: pd.DataFrame) -> list[dict]:
    if rows.empty:
        return []
    rows = rows.copy()
    rows["start_time"] = rows["start_time_hour"].astype(int).map(lambda h: f"{h:02d}:00")
    rows["duration_hours"] = (rows["stream_duration"].astype(float) / 60.0).round(1)
    rows["expected_subs"] = rows["expected_subs"].round(2)
    rows["expected_followers"] = rows["expected_followers"].round(2)
    rows["growth_score"] = rows["growth_score"].round(2)
    rows["opportunity_index"] = rows["opportunity_index"].round(0).astype(int)
    rows["confidence"] = rows["confidence"].round(2)
    rows["signal"] = rows.apply(lambda row: f"{row['category_source']} · {int(row['category_rows'])}", axis=1)
    if "score_gain" in rows.columns:
        rows["score_gain"] = rows["score_gain"].round(2)
        rows["score_gain_pct"] = (100.0 * rows["score_gain_pct"]).round(0).astype(int)
        rows["sub_gain"] = rows["sub_gain"].round(2)
        rows["default_score"] = rows["default_score"].round(2)
    return rows.to_dict("records")


def _default_time_rows(grid: pd.DataFrame) -> pd.DataFrame:
    default_rows = grid[grid["start_time_hour"].astype(int) == DEFAULT_START_HOUR]
    return default_rows if not default_rows.empty else grid


def _top_default_plan(grid: pd.DataFrame, n: int) -> list[dict]:
    rows = (
        _default_time_rows(grid)
        .sort_values("growth_score", ascending=False)
        .drop_duplicates("game_category")
        .head(n)
    )
    return _format_rows(rows)


def _expansion_picks(grid: pd.DataFrame, n: int = 3) -> list[dict]:
    default_rows = _default_time_rows(grid)
    rows = (
        default_rows[default_rows["category_source"].eq("curated")]
        .sort_values("growth_score", ascending=False)
        .drop_duplicates("game_category")
        .head(n)
    )
    return _format_rows(rows)


def _strong_time_tests(grid: pd.DataFrame, n: int = 5) -> list[dict]:
    records = []
    for _, group in grid.groupby("game_category"):
        default_rows = group[group["start_time_hour"].astype(int) == DEFAULT_START_HOUR]
        alternate_rows = group[group["start_time_hour"].astype(int) != DEFAULT_START_HOUR]
        if default_rows.empty or alternate_rows.empty:
            continue

        default_row = default_rows.sort_values("growth_score", ascending=False).iloc[0]
        alternate_row = alternate_rows.sort_values("growth_score", ascending=False).iloc[0].copy()
        score_gain = float(alternate_row["growth_score"] - default_row["growth_score"])
        score_gain_pct = score_gain / max(float(default_row["growth_score"]), 0.01)
        if score_gain < ALT_TIME_MIN_SCORE_GAIN or score_gain_pct < ALT_TIME_MIN_RELATIVE_GAIN:
            continue

        alternate_row["default_score"] = float(default_row["growth_score"])
        alternate_row["score_gain"] = score_gain
        alternate_row["score_gain_pct"] = score_gain_pct
        alternate_row["sub_gain"] = float(alternate_row["expected_subs"] - default_row["expected_subs"])
        records.append(alternate_row)

    if not records:
        return []
    rows = pd.DataFrame(records).sort_values(["score_gain", "growth_score"], ascending=False).head(n)
    return _format_rows(rows)


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
    main { max-width:980px; margin:0 auto; padding:24px; }
    h1 { font-size:28px; margin:0 0 8px; letter-spacing:0; }
    h2 { font-size:17px; margin:22px 0 10px; letter-spacing:0; }
    .meta { color:var(--muted); margin-bottom:16px; }
    form { display:flex; flex-wrap:wrap; gap:10px; align-items:end; margin:14px 0 18px; }
    label { display:flex; flex-direction:column; gap:6px; color:var(--muted); font-size:13px; }
    input, button { background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:6px; padding:9px 11px; font-size:14px; }
    button { background:var(--accent); border-color:var(--accent); color:#06111d; font-weight:700; cursor:pointer; }
    .notice, .empty { border:1px solid var(--line); background:var(--panel); padding:12px 14px; border-radius:6px; color:var(--muted); margin:14px 0; }
    .summary { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:12px; margin:16px 0 6px; }
    .metric { background:var(--panel); border:1px solid var(--line); border-radius:6px; padding:14px; min-height:104px; }
    .metric span { display:block; color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:0; margin-bottom:8px; }
    .metric strong { display:block; font-size:20px; line-height:1.2; margin-bottom:8px; }
    .metric small { color:var(--muted); line-height:1.35; }
    .table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:6px; }
    table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); border-radius:6px; overflow:hidden; }
    .table-wrap table { border:0; border-radius:0; }
    th, td { padding:10px 11px; border-bottom:1px solid var(--line); text-align:left; font-size:14px; white-space:nowrap; }
    th { color:var(--muted); font-weight:700; background:#15191d; }
    tr:last-child td { border-bottom:0; }
    .num { text-align:right; font-variant-numeric:tabular-nums; }
    .pill { display:inline-block; color:#06140b; background:var(--good); border-radius:999px; padding:2px 7px; font-size:12px; font-weight:700; }
    @media (max-width: 820px) { main { padding:16px; } .summary { grid-template-columns:1fr; } th, td { font-size:13px; } }
  </style>
</head>
<body>
<main>
  <h1>Daily Stream Recommender</h1>
  <div class="meta">{{ stream }} &middot; {{ date_label }} &middot; default start {{ default_start }} &middot; artifact {{ created_at }}</div>

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

  {% if top_plan %}
    <div class="summary">
      <div class="metric">
        <span>Best {{ default_start }} Play</span>
        <strong>{{ top_plan[0].game_category }}</strong>
        <small>{{ top_plan[0].duration_hours }}h &middot; {{ top_plan[0].expected_subs }} sub upside &middot; {{ top_plan[0].signal }}</small>
      </div>
      <div class="metric">
        <span>Start Time</span>
        {% if time_tests %}
          <strong>{{ time_tests[0].start_time }}</strong>
          <small>{{ time_tests[0].game_category }} gains {{ time_tests[0].score_gain_pct }}% vs {{ default_start }}</small>
        {% else %}
          <strong>{{ default_start }}</strong>
          <small>No alternate start cleared the test threshold.</small>
        {% endif %}
      </div>
      <div class="metric">
        <span>Best New Option</span>
        {% if expansion_picks %}
          <strong>{{ expansion_picks[0].game_category }}</strong>
          <small>{{ expansion_picks[0].duration_hours }}h &middot; low evidence &middot; score {{ expansion_picks[0].growth_score }}</small>
        {% else %}
          <strong>None</strong>
          <small>No curated expansion category is available.</small>
        {% endif %}
      </div>
    </div>
  {% endif %}

  <section>
    <h2>Recommended {{ default_start }} Plan <span class="pill">default</span></h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Game / Category</th><th>Hours</th><th class="num">Sub Upside</th><th class="num">Follower Upside</th><th class="num">Score</th><th>Signal</th></tr></thead>
        <tbody>
        {% for row in top_plan %}
          <tr><td>{{ row.game_category }}</td><td>{{ row.duration_hours }}</td><td class="num">{{ row.expected_subs }}</td><td class="num">{{ row.expected_followers }}</td><td class="num">{{ row.growth_score }}</td><td>{{ row.signal }}</td></tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>Strong Start-Time Tests <span class="meta">vs {{ default_start }}</span></h2>
    {% if time_tests %}
      <div class="table-wrap">
        <table>
          <thead><tr><th>Game / Category</th><th>Test Start</th><th>Hours</th><th class="num">Score</th><th class="num">Gain</th><th class="num">Sub Gain</th><th>Signal</th></tr></thead>
          <tbody>
          {% for row in time_tests %}
            <tr><td>{{ row.game_category }}</td><td>{{ row.start_time }}</td><td>{{ row.duration_hours }}</td><td class="num">{{ row.growth_score }}</td><td class="num">+{{ row.score_gain }} / {{ row.score_gain_pct }}%</td><td class="num">+{{ row.sub_gain }}</td><td>{{ row.signal }}</td></tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    {% else %}
      <div class="empty">No alternate start cleared the gain threshold.</div>
    {% endif %}
  </section>

  <section>
    <h2>Expansion Picks <span class="meta">low evidence</span></h2>
    {% if expansion_picks %}
      <div class="table-wrap">
        <table>
          <thead><tr><th>Game / Category</th><th>Hours</th><th class="num">Sub Upside</th><th class="num">Score</th><th>Signal</th></tr></thead>
          <tbody>
          {% for row in expansion_picks %}
            <tr><td>{{ row.game_category }}</td><td>{{ row.duration_hours }}</td><td class="num">{{ row.expected_subs }}</td><td class="num">{{ row.growth_score }}</td><td>{{ row.signal }}</td></tr>
          {% endfor %}
          </tbody>
        </table>
      </div>
    {% else %}
      <div class="empty">No low-evidence expansion picks are available.</div>
    {% endif %}
  </section>
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
            default_start=f"{DEFAULT_START_HOUR:02d}:00",
            top_n=10,
            message="recommender_artifacts.joblib is missing. Run python train_recommender.py first.",
            top_plan=[],
            time_tests=[],
            expansion_picks=[],
        )

    stream = (request.args.get("stream") or artifact.get("target_stream") or TARGET_STREAM).strip()
    target_date = _parse_date(request.args.get("date"))
    try:
        top_n = max(3, min(30, int(request.args.get("top_n", "10"))))
    except ValueError:
        top_n = 10

    message = (
        "Fighting-game pool only. "
        f"{DEFAULT_START_HOUR:02d}:00 is the default start; alternate times require at least "
        f"{int(ALT_TIME_MIN_RELATIVE_GAIN * 100)}% and {ALT_TIME_MIN_SCORE_GAIN:.2f} score gain."
    )
    try:
        grid = _score_grid(artifact, stream, target_date)
    except Exception as exc:
        message = str(exc)
        grid = pd.DataFrame()

    if grid.empty:
        top_plan = time_tests = expansion_picks = []
    else:
        top_plan = _top_default_plan(grid, top_n)
        time_tests = _strong_time_tests(grid, min(top_n, 5))
        expansion_picks = _expansion_picks(grid, min(top_n, 5))

    return render_template_string(
        TEMPLATE,
        stream=stream,
        date_label=target_date.strftime("%A, %B %d, %Y"),
        date_value=target_date.strftime("%Y-%m-%d"),
        created_at=str(artifact.get("created_at_utc", ""))[:19],
        default_start=f"{DEFAULT_START_HOUR:02d}:00",
        top_n=top_n,
        message=message,
        top_plan=top_plan,
        time_tests=time_tests,
        expansion_picks=expansion_picks,
    )
