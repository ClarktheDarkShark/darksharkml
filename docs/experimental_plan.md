# Stream Recommendation Experimental Plan

Goal: produce a daily recommender for `thelegendyagami` that ranks actionable choices such as category/game, start time, duration, and tags by expected subscriptions and follower gain.

## Validation Contract

The validation unit is future `thelegendyagami` rows only. Peer streamer rows can be used for training, but never from dates after the test cutoff. This keeps trend data useful without letting future platform behavior leak into evaluation.

Default backtest:

- Sort `thelegendyagami` history by date.
- Use the first 58% of target-stream dates as initial history.
- Evaluate the remaining dates in blocked rolling folds.
- For every fold, train on rows before the fold start and test only `thelegendyagami` rows inside the fold window.
- Compare learned models against target-only baselines: last value, trailing 7-row mean, and training mean.

Primary metrics:

- MAE and RMSE for calibration.
- R2 and Spearman correlation as secondary checks.
- Skill against `target_roll7`; a model that cannot beat this baseline is not deployment-worthy.

## Data Scope Tests

These scopes explicitly answer whether large streamer data helps or hurts:

- `target_only`: only prior `thelegendyagami` rows.
- `all_streams`: all prior streamers.
- `all_streams_yagami_x5`: all prior streamers, with target-stream rows oversampled 5x.
- `no_large_streamers`: prior streamers with median concurrent viewers below the configured threshold, plus `thelegendyagami`.
- `no_large_streamers_yagami_x5`: peer-filtered data with target-stream oversampling.

## Model Families

The first harness uses Heroku-friendly scikit-learn models:

- Baselines: zero, last value, rolling 7-row mean, train mean.
- Linear: Ridge over scaled numeric, one-hot categorical, and tag text features.
- Tree ensembles: HistGradientBoosting, Poisson HistGradientBoosting for non-negative targets, RandomForest, ExtraTrees.
- Zero-inflated counts: hurdle HGB, which models positive-event probability separately from positive amount.
- Neural baseline: MLPRegressor as the lightweight deep-learning proxy that does not add TensorFlow or PyTorch slug weight.

Optional heavier models such as XGBoost, LightGBM, CatBoost, PyTorch, or TensorFlow should only be added after the scikit-learn ceiling is known and the Heroku size impact is measured.

## Feature Sets

The default `planner` feature set reuses the existing engineered frame: day, hour, weekend, days since previous stream, game/category, duration, tags, and shifted historical rollups. Other supported sets:

- `planner_no_tags`: tests whether Twitch tags add signal or noise.
- `planner_with_streamer_id`: lets peer-trained models learn streamer-specific offsets.
- `controllable_only`: isolates directly actionable inputs from historical momentum.

## Promotion Gate

A model is a candidate for deployment only if it:

- Beats `target_roll7` on future `thelegendyagami` MAE for subscriptions and followers.
- Does not rely on future-dated peer rows.
- Produces stable recommendations across adjacent folds.
- Fits within the Heroku slug/runtime budget with the saved artifact.
- Keeps inference fast enough for a single web request or can be cached at boot.

## Next Implementation Steps

1. Run the backtest harness and identify the best scope/model/feature set for subscriptions and follower change.
2. Export the best per-target models with the exact feature list and validation summary.
3. Replace the current dashboard artifact with the promoted artifact only after the above gate passes.
4. Add a daily recommendation page that shows combined score, expected subs, expected followers, confidence, and factor deltas for game/category, start hour, duration, and tags.
5. Measure artifact size and Heroku boot time before pushing.
