# Lineup / injury overrides for the live script

One file per game date (ET): `data/overrides/YYYY-MM-DD.csv`, columns `team,player,status[,note]`.

- `team`: full NBA team name (e.g. `Detroit Pistons`) or abbreviation (`DET`).
- `player`: as written in the box scores (`Jalen Duren`).
- `status`: one of `OUT`, `DOUBTFUL`, `QUESTIONABLE`, `PROBABLE`, `AVAILABLE` (case-insensitive).
- `note`: free text, source of the news (ignored by the script, useful for the review).

The live script (`src/live_odds.py --overrides auto`) downloads the latest official NBA injury report for the date and
then applies this file on top of it, so you only need to list players whose status differs from the official report
(late scratches, "will play" news, a starter listed questionable who has been confirmed in/out), not everyone on it.
Statuses OUT/DOUBTFUL remove the player from the team's available minutes in the model's injury features;
QUESTIONABLE/PROBABLE/AVAILABLE keep him available (the historical model treats questionable players as 54% likely to play).
