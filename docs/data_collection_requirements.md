# Data Collection Requirements

The current recommender can rank Yagami-compatible categories with a transfer score from the broader streamer sample, but many categories still have thin direct evidence. No-data or peer-only categories are shown as lower-confidence tests, not evidence-backed predictions.

Current audit from the training table:

- 234 `thelegendyagami` streams.
- Median Yagami stream length is 2.95 hours; the 75th percentile is 3.48 hours. The planner should focus on 3-4 hour streams, not 7-hour streams.
- 43 Saturday streams; 23 of those are fighting-game streams, so Saturday should be treated as a separate Fight Night mode.
- 84 current candidate categories, but 26 Yagami-played candidate categories have fewer than 5 target rows.
- 711 comparable peer rows across 28 peer streamers for the current candidate pool. This is useful but still thin for confident category transfer.

To make category recommendations meaningfully stronger, collect:

- 30-50 streams per candidate game/category across comparable small and mid-size streamers.
- At least 5-10 streamers per category so the model does not confuse one creator's audience with the game itself.
- Continued `thelegendyagami` observations for repeated games, especially 2XKO, Street Fighter 6, Zelda/Souls-like games, roguelikes, and any new game being tested.
- For every stream: category segments, exact start/end time, duration, average/peak viewers, unique viewers, follows, subs, gifted subs, chatters, chat count, raids, tags, title, and stream date.
- Context features: day-of-week, holiday/event flags, game launch/patch/tournament windows, and whether larger adjacent creators were live in the same category.

Minimum next target:

- 300-500 more comparable rows across Yagami-like categories.
- 100+ fresh Saturday fighting-game rows across comparable streamers.
- 100+ rows across games `thelegendyagami` has not tried but could plausibly play, such as Tekken 8, Mortal Kombat, Granblue Fantasy Versus, Dragon Ball FighterZ, Under Night, Marvel vs. Capcom, Souls-likes, Zelda-likes, roguelikes, and retro-adjacent challenge games.

Without that, no-data categories can be ranked only as exploration candidates, not validated recommendations.
