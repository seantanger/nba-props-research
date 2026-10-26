---
name: nba-props
description: Daily NBA player-prop paper-trading run (points, rebounds, assists, threes, points+rebounds+assists). Settles last night's picks, feeds tonight's injury report and lineup news to the model, pulls FanDuel / DraftKings / BetMGM / BetRivers prices from The Odds API, prints candidate bets, then has Claude review each one against current news and keep or veto it. Use when the user says /nba-props, asks for tonight's prop picks, or wants to settle/report the NBA prop ledger.
---

# /nba-props — tonight's NBA prop picks with a Claude review layer

The statistical model finds candidate bets; the ledger records every candidate with Claude's verdict so that, over
time, we can measure three things separately: the model alone, the kept picks, and the vetoed picks. Be honest and
consistent; never veto to protect a record. Everything is **paper trading, flat 1 unit per pick**; the user bets from
Ontario, so only FanDuel, DraftKings, BetMGM and BetRivers prices count (no offshore books).

All commands run from the workspace root (`/Users/seantang/git/nba-ai/workspace`) with `.venv/bin/python`. Each bash
call starts a fresh shell, so chain with `&&`. Read `JOURNAL.md` first if you are resuming.

## 0. What the pipeline knows (read before reviewing)

- Projection model: LightGBM mean + dispersion per market, trained at run time on every logged game through yesterday,
  with **injury-aware features** built from the official NBA injury report (teammates' vacated minutes/points/shots,
  starters out, the player's own status, the opponent's absences). On dev they improved every accuracy and ROI
  measure (REPORT.md §10.2) but the model still misses by more than FanDuel's line (3.48 vs 3.43 points on average)
  and under-reacts for the direct replacement of an absent starter, so review those rows with extra care.
- The decision-time status is the latest official report (`--injury auto`, PDF from `ak-static.cms.nba.com`, published
  on the half hour from 11 AM ET) **plus** your overrides file. Players listed Out/Doubtful are removed from the team's
  available minutes; Questionable/Probable stay in (historically 54% of Questionables play).
- Two bet sources are printed: `rule` = phase-1 FanDuel-anchored rule (bet DraftKings/BetMGM/BetRivers when their
  line/price disagrees with FanDuel's by ≥3% after the model's dispersion; dev +6.1% on n=6,424, holdout n=35 −1.8%,
  see REPORT.md), and `model_fd` / `model_book` = the projection model's own probability against the book's price at
  edge ≥ 0.08 (dev, injury-aware model, honest tier: FanDuel −0.4% on n=57k; DraftKings/BetMGM/BetRivers +1.2% on
  n=14.6k with an interval of [−0.7%, +3.0%] — break-even, not proven; logged to keep measuring it).
- Books are slow on secondary markets (rebounds, assists, threes, PRA) and when news breaks between a market's last
  update and tip. The rule's known failure is the opposite case: another book has the news and FanDuel does not
  (e.g. Jarrett Allen under 16.5 at an offshore book vs FanDuel 12.5 when a teammate was ruled out). The `news_flag`
  column marks this mechanically (see §4).

## 1. Settle last night first (if the ledger exists)

```
.venv/bin/python src/ledger.py settle && .venv/bin/python src/ledger.py close && .venv/bin/python src/ledger.py report
```
Box scores come from stats.nba.com via `nba_api` (cached under `data/raw/nba/daily/`); run after ~1 AM ET so every
game has final stats. `close` fills closing lines from the latest cached pre-tip odds snapshot, so it only has data
when the live script was run again near tip the night before. Show the report table (model-only vs kept vs vetoed)
to the user.

## 2. Feed tonight's injury and lineup news to the model

Search for tonight's NBA injury report and lineup news for each game on the slate (official report, team beat
writers, "starting lineup" posts, late scratches). Write `data/overrides/<YYYY-MM-DD>.csv` (ET game date):

```
team,player,status,note
Detroit Pistons,Jalen Duren,OUT,ruled out (hamstring) per beat reporter 5:40 PM
Orlando Magic,Anthony Black,AVAILABLE,confirmed starting per team PR
```

- `status` ∈ `OUT`, `DOUBTFUL`, `QUESTIONABLE`, `PROBABLE`, `AVAILABLE`. Team = full name or abbreviation; player as in
  the box score. Only list players whose status differs from (or is newer than) the official report; the script
  downloads the report itself.
- Leave genuine Questionables as `QUESTIONABLE` and handle them in the review step; mark `OUT` only when ruled out.

## 3. Get prices and candidates

```
.venv/bin/python src/live_odds.py --books fanduel,draftkings,betmgm,betrivers --mode both
```
Options: `--books fanduel` for a FanDuel-only slate (then only `model_fd` rows appear; the `rule` needs a second
book), `--mode rule` or `--mode model`, `--hours 8` to restrict to tonight, `--max-events N`, `--dry` to replay the
latest cached run folder with no API calls, `--injury none|<pdf>` and `--overrides none|<csv>` to override the
defaults (`auto`).

**Credit cost**: 1 credit for the events list + 5 credits per game (5 markets × 1 region). A 10-game slate costs
about 51 credits per run; the script stops when fewer than 150 credits would remain. Check the quota line it prints.
Every response is cached under `data/raw/odds_api/<UTC date>/<HHMM>/` and never re-fetched.

**When to run**: (1) a first run 2–3 hours before the first tip, after the 5:30 PM ET report for 7 PM games, to
review with time to search news; (2) a second run 20–40 minutes before tip, after starting lineups are posted
(~30 min before tip), because most injury news breaks between the afternoon report and tip and the rule's edge comes
from stale lines in exactly that window. Only pre-game markets are used (games already tipped are skipped). The
second run also provides the closing snapshot for `ledger.py close`.

Output: `results/live_bets_<stamp>.csv` with columns player, team, market, side, line, price, book, source, p_model,
edge, line_fd, p_fd, mu, sigma, age_book_min, age_fd_min, mins_to_tip, live, news_flag, news, report_ts,
claude_verdict, claude_reason. The `live` flag is False when either market is older than 120 minutes (The Odds API
drops suspended markets after ~15 minutes; a stale timestamp usually means the book pulled the market).

## 4. Review every candidate (the skill check)

For each row check, using only what is knowable before tip:

1. **The player's own status** — Questionable/Doubtful, minutes restriction, first game back, resting on a
   back-to-back, starting vs bench (the model knows the official status but not a restriction).
2. **Teammates' status** — the `news` column lists rotation players (≥15 minutes a game) listed
   Questionable/Doubtful/Out on the latest report or in your overrides (`*` = override). Confirm with the latest
   lineup news; a late scratch not on the report is the most common thing the model misses.
3. **`news_flag`** (rule rows only; mechanical, from the line gap and the status list):
   - `explained`: the other book's line has moved in the direction the news implies and FanDuel's has not → the book
     has the news and FanDuel is stale. On 2022-25 data such picks lost (−5.1% on n=186, versus +6.4% for the rest).
     **Veto** unless you verify FanDuel's line is current.
   - `against`: FanDuel's line already reflects the news and the other book's does not → keep; historically the best
     bucket (+25% on n=141, wide interval).
   - `same_line` / `none`: no news-direction story; review normally.
   - model rows show `news` when a listed absence exists: confirm it is real and still current.
4. **Stale line** — `age_book_min` or `age_fd_min` large, or `live = False`: the price may be gone; veto if you cannot
   confirm the market is open.
5. **Suspicious edges** (edge ≥ 0.15 or a book line 3+ points from FanDuel's): almost always news the model lacks;
   find it before keeping.
6. **Tip time** — `mins_to_tip` < 0 means the game started; drop the row.

Then fill both columns for **every** row:
- `claude_verdict`: `keep` or `veto`
- `claude_reason`: one short sentence citing the checkable fact (and its source); if nothing is found write
  `no relevant news found` and keep.

Rules: veto only for a concrete, checkable reason (status, lineup, line already moved), not a view on the player.
If the news supports the model's side, keep and say so. Model-only rows (`model_fd`, `model_book`) are logged for
measurement; apply the same review but note that the dev evidence for them is negative.

Write the verdicts back into the same CSV, then log it (the command refuses rows without a verdict/reason):

```
.venv/bin/python src/ledger.py log --picks results/live_bets_<stamp>.csv
```
Re-logging the same file after edits is safe (duplicates on game date/player/market/side/book/source are skipped).

## 5. Report to the user

A compact table: player, market, side, line, price, book, source, model probability, book's breakeven, edge,
news_flag, verdict, reason. Then say plainly:
- This is **paper trading**, flat 1 unit per pick. The rule was +6.1% on development data (n=6,424 bets, 2022-25) but
  −1.8% on the n=35 pre-registered holdout and the holdout has since been reused; the model-only path was not
  profitable on development data. Nothing is proven until this forward ledger has a few hundred settled picks.
- Give the ledger totals from §1 (picks, units, ROI, interval, closing-line value) whenever they exist.
