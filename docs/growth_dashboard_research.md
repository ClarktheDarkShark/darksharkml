# Growth Dashboard Research Notes

The added dashboard cues are based on three growth themes that are useful for a small or growing Twitch channel:

- Live engagement timing: Twitch Creator Camp says viewers who chat on their first visit are materially more likely to return, and the first and last 15 minutes are strong call-to-action windows. The dashboard now creates a simple live CTA clock for those windows.
- Category/content review: Twitch Creator Camp recommends using analytics to compare followers, viewership, revenue, categories, and longer time periods before deciding what to keep or pivot away from. The dashboard now shows a category pulse so the streamer can tell whether the recommendation is fresh, stale, or low evidence.
- Community bridges: Twitch Creator Camp highlights collaboration, raids, and suggested channels as growth paths. The dashboard now surfaces a small raid/collab watchlist from adjacent fighting-game streamers in the dataset, with stale data called out through recent-row counts.

Primary sources:

- https://www.twitch.tv/creatorcamp/en/level1/establish-your-brand/engaging-viewers/
- https://www.twitch.tv/creatorcamp/en/level1/establish-your-brand/what-analytics-matter/
- https://www.twitch.tv/creatorcamp/en/level1/growing-your-community/collaborating/
- https://help.twitch.tv/s/article/about-twitch-categories

Data caveat: the current fighting-game sample is still thin. The features are useful because they expose that uncertainty instead of hiding it, but the community bridge feature needs current peer-stream collection to become operational rather than historical.
