# 1) Imports & load your artifacts
import pytz
from datetime import datetime
import pandas as pd
import numpy as np

from predictor import (
    get_predictor_artifacts,
    _infer_grid_for_game,
    _get_last_row_for_stream,
)

# load pipelines (you said you have three), plus data & metadata
pipe1, pipe2, pipe3, df, features, cat_opts, start_opts, dur_opts, metrics, metrics2 = get_predictor_artifacts()
pipes = [pipe1, pipe2, pipe3]
ready = all(p is not None for p in pipes) and df is not None

# extract the full tag vocabulary from your first pipeline
pre = pipes[0].named_steps["pre"]
vectorizer = pre.named_transformers_["tags"].named_steps["vectorize"]
all_tags = vectorizer.get_feature_names_out().tolist()


# 2) User‐adjustable parameters
stream_name        = "thelegendyagami"
selected_game      = "SomeGameCategory"      # e.g. "Fortnite"
selected_start_time = 19                      # hour in 0–23
selected_tags      = ["shooter","fun"]       # list of tags you want to test


# 3) Helper: build a feature‐row for a given stream/game/start/tags
def make_feature_row(baseline, game, hour, tags, features):
    r = baseline.copy()
    # set categorical & time features
    r["game_category"]    = game
    r["start_time_hour"]  = hour
    now_est               = datetime.now(pytz.timezone("US/Eastern"))
    r["day_of_week"]      = now_est.strftime("%A")
    r["start_hour_sin"]   = np.sin(2*np.pi*hour/24)
    r["start_hour_cos"]   = np.cos(2*np.pi*hour/24)
    r["is_weekend"]       = r["day_of_week"].isin(["Saturday","Sunday"])
    # set tags
    for t in all_tags:
        r[f"tag_{t}"] = int(t in tags)
    # return a single-row DataFrame with exactly the model’s features
    return r.to_frame().T[features]


# grab the “last row” for your stream as baseline
baseline = _get_last_row_for_stream(df, stream_name)


# 4) Make predictions for each model
results = []
for idx, pipe in enumerate(pipes, start=1):
    X = make_feature_row(baseline, selected_game, selected_start_time, selected_tags, features)
    y = pipe.predict(X)[0]
    results.append({
        "model": f"pipe{idx}",
        "y_pred": round(y, 2)
    })

pd.DataFrame(results)
