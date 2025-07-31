import os
from flask import Flask, Blueprint, render_template_string, request
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
      _get_last_row_for_stream  # internal helper; used for dashboard inference
)

est = pytz.timezone("US/Eastern")
pipe, pipe2, df_for_inf, features, cat_opts, start_opts, dur_opts, metrics, metrics2 = get_predictor_artifacts()
ready = pipe is not None and df_for_inf is not None
baseline_row = None


all_tags: list[str] = []
if ready:
    pre = pipe.named_steps["pre"]
    tag_pipe = pre.named_transformers_["tags"]
    vectorizer = tag_pipe.named_steps["vectorize"]
    all_tags = vectorizer.get_feature_names_out().tolist()

default_stream = "thelegendyagami"
stream_opts = (
    df_for_inf["stream_name"].value_counts().head(5).index.tolist()
)
if default_stream not in stream_opts:
    stream_opts.append(default_stream)
selected_stream = default_stream
if selected_stream not in df_for_inf["stream_name"].unique():
    selected_stream = default_stream
stream_name = selected_stream
today_name = datetime.now(est).strftime("%A")

if ready:
    baseline_row = _get_last_row_for_stream(df_for_inf, stream_name)

# --- Limit games to top 10 by predicted subs + all games played by the selected stream ---
df = df_for_inf.copy()
df['y_pred'] = pipe.predict(df[features]) if ready else 0
df['y_pred2'] = pipe2.predict(df[features]) if ready else 0

# print('Printing preds...:')
# print(np.max(df['y_pred2']))
# Get all games played by this streamer
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
# Get all tags ever used by this streamer

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
selected_game = game_opts[0]
try:
    selected_start_time = int(request.args.get('start_time', start_opts[0] if start_opts else 0))
except Exception:
    selected_start_time = start_opts[0] if start_opts else 0
selected_tags = legend_tag_opts
manual = True

pred_result = None
if manual and ready:
    last_row = baseline_row.copy()
    last_row['game_category'] = selected_game
    last_row['start_time_hour'] = selected_start_time
    today = datetime.now(est)
    dow   = today.strftime("%A")
    last_row['day_of_week']   = dow
    h = selected_start_time
    last_row['start_hour_sin'] = np.sin(2 * np.pi * h / 24)
    last_row['start_hour_cos'] = np.cos(2 * np.pi * h / 24)
    last_row['is_weekend']     = dow in ("Saturday", "Sunday")
    # prev_dates = df[df["stream_name"] == stream_name]["stream_date"].sort_values()
    # if len(prev_dates) >= 2:
    #     last = prev_dates.iloc[-1].date()
    #     prev = prev_dates.iloc[-2].date()
    #     last_row["days_since_previous_stream"] = (last - prev).days

    for t in tag_opts:
        last_row['tag_' + t] = 1 if t in selected_tags else 0
    X = last_row[features].to_frame().T
    print("X for selected features:", X)
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

# Feature score table for the heat map section
duration_scores_df = (
    time_predictions.groupby('stream_duration')['y_pred']
    .mean()
    .round(2)
    .reset_index()
)
feature_scores = [
    {
        'feature': 'Duration',
        'value': f"{int(r.stream_duration)}h",
        'score': f"{r.y_pred:.2f}"
    }
    for _, r in duration_scores_df.iterrows()
]
feature_scores.append({
    'feature': 'Category',
    'value': selected_game,
    'score': f"{time_predictions['y_pred'].mean():.2f}"
})
top_tag_rows = (
    tag_effects_full.sort_values('delta_from_baseline', ascending=False)
    .head(3)
)
for _, r in top_tag_rows.iterrows():
    feature_scores.append({
        'feature': 'Tag',
        'value': r['tag'],
        'score': f"{r['delta_from_baseline']:.2f}"
    })
