# Initial Experiment Findings

Date: 2026-05-17

Data source: live `daily_stats` table via `DATABASE_URL`.

Prepared data:

- 3,079 total rows.
- 235 `thelegendyagami` rows from 2025-07-12 through 2026-05-16.
- Backtest evaluated 94 future `thelegendyagami` rows across 4 blocked rolling folds.

## Target Distribution

`thelegendyagami` is strongly zero-inflated:

- Subscriptions: 131 zero rows, 104 positive rows.
- Net followers: 11 negative rows, 174 zero rows, 50 positive rows.

This means every model must beat a zero baseline before it should be called predictive.

## Best Current Results

| Target | Best model | Scope | Feature set | MAE | Baseline MAE | Result |
| --- | --- | --- | --- | ---: | ---: | --- |
| `total_subscriptions` | HGB | `no_large_streamers` | `planner` | 1.226 | 1.415 zero / 2.460 rolling-7 | Promote candidate |
| `net_follower_change` | HGB | `no_large_streamers_yagami_x5` | `planner_with_streamer_id` | 0.403 | 0.404 zero | Not enough lift |
| `positive_follower_change` | HGB/zero tie | mixed | mixed | 0.351 | 0.351 zero | Not predictive |

## Interpretation

The subscription model has useful signal on unseen `thelegendyagami` rows. Peer streamer data helps only after excluding very large streamers; the full all-streamer pool is weaker. Tags add a small amount of signal for subscriptions because `planner` beats `planner_no_tags` in the HGB sweep.

Follower gain is not ready for recommendation. The best net-follower model only improves MAE by roughly 0.002 over zero, and positive follower gain ties the zero baseline. That is not strong predictive power; it is mostly learning that follower gains are rare.

## Current Candidate

Use this as the next promotion candidate for subscriptions:

- Model: HistGradientBoostingRegressor wrapped in the existing planner feature set.
- Training scope: streamers whose historical median concurrent viewers are below the peer threshold, always including `thelegendyagami`.
- Feature set: day, start hour, weekend flag, days since previous stream, game/category, duration, tags, and shifted historical rollups.
- Artifact size in the latest exported run: about 2.1 MB for three target candidates, which is acceptable for Heroku.

## Next Experiments

- Add classification metrics for follower-positive event prediction: average precision, ROC AUC, calibration, and top-k lift.
- Try a recommendation-specific objective for followers, such as ranking observed positive-follower rows above zero rows, instead of optimizing MAE.
- Test peer thresholds below 1,000 CCV because the current best subscription result suggests very large creators distort small-stream recommendations.
- Add fold-level stability checks for the top game/hour/duration recommendations before replacing the production artifact.
