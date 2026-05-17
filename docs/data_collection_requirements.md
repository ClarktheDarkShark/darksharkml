# Data Collection Requirements

The current recommender can rank fighting-game options with a transfer score from the broader streamer sample, but several categories still have no direct observations. Those no-data categories are shown as low-confidence exploration ideas, not evidence-backed predictions.

To make category recommendations meaningfully stronger, collect:

- 30-50 streams per candidate fighting game across comparable small and mid-size streamers.
- At least 5-10 streamers per category so the model does not confuse one creator's audience with the game itself.
- Continued `thelegendyagami` observations for repeated games, especially 2XKO, Street Fighter 6, Guilty Gear, Smash, and any new game being tested.
- For every stream: category segments, exact start/end time, duration, average/peak viewers, unique viewers, follows, subs, gifted subs, chatters, chat count, raids, tags, title, and stream date.
- Context features: day-of-week, holiday/event flags, game launch/patch/tournament windows, and whether larger adjacent creators were live in the same category.

Minimum next target:

- 300-500 more fighting-game stream rows across comparable streamers.
- 100+ rows across games `thelegendyagami` has not tried but could plausibly play, such as Tekken 8, Mortal Kombat, Granblue Fantasy Versus, Dragon Ball FighterZ, Under Night, and Marvel vs. Capcom.

Without that, no-data categories can be ranked only as exploration candidates, not validated recommendations.
