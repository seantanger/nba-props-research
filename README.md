# nba-ai

Research into whether NBA player-prop betting markets (points, rebounds, assists, threes, points+rebounds+assists) can be beaten
with a statistical model, tested honestly: leak-free features, bets graded at prices that were actually on offer, and a holdout
set aside before any result was seen.

**Status: no proven edge.** The one rule frozen on development data (+6.1% ROI on 6,424 bets, 2020-21 → 2024-25) returned −1.8% on
its 35-bet pre-registered holdout. The forward paper-trading ledger (started 2026-10-25) has 1,027 settled picks over 7 game
dates through 2026-11-02: +2.2% ROI overall, 95% CI [−4.1%, +7.7%]. Picks kept after review are at −0.6% and vetoed picks at
+13.9%, so far the review is removing winners. Since 2026-10-29 the assists model also uses player-tracking data. It looked
better in backtests, but its unreviewed assists picks are at −28.8% on 82 bets (old model −11.3% on 101). On log-loss it leads
the old model by +29e-4 with a 95% CI of [−81, +117], so that is unresolved. Seven dates is far too few to call any of this.
Nothing here is betting advice.

## What's here

| Path | What it is |
|---|---|
| `workspace/REPORT.md` | The write-up: data, approaches tried, the chosen rule, holdout results, caveats |
| `workspace/JOURNAL.md` | Running record of every experiment, decision and dead end, with numbers |
| `workspace/README.md` | Pipeline: commands in run order, key files, honesty rules enforced in code |
| `workspace/src/` | Data fetching, features, LightGBM mean + dispersion models, evaluation, live odds, ledger |
| `workspace/data/ledger/ledger.csv` | Forward paper-trading ledger (flat 1 unit per pick, model-only vs reviewed picks) |
| `workspace/data/ledger/ast_compare.csv` | Every FanDuel assists line with both the old and the tracking assists projection |
| `workspace/data/overrides/` | Daily injury/lineup overrides used by the live run |
| `workspace/results/` | Bet lists, headline JSONs and small result tables |
| `.claude/skills/nba-props/SKILL.md` | Daily routine as a Claude Code skill: settle, data refresh, injury news, live odds, review, log |

## How it works

1. **Projection model.** LightGBM predicts the mean and dispersion of each stat from box-score history (only games strictly before
   the one being predicted) plus injury-aware features built from the NBA's official injury reports. The assists model also
   uses stats.nba.com player tracking (potential assists, passes, touches, time of possession, shot zones, opponent potential
   assists allowed), each count rescaled by the league's season-to-date rate because stats.nba.com changed its definitions in
   2024-25 and 2025-26. The old assists model still runs alongside for comparison.
2. **Bet selection.**
   - `rule`: use FanDuel's line as the anchor, apply the model's dispersion, and bet DraftKings, BetMGM or BetRivers when their
     price disagrees by at least 3 percentage points.
   - `model_fd` / `model_book`: the model's own probability against the book's price at an edge of at least 0.08. Logged for
     measurement only; it was break-even on development data.
3. **Review.** Every candidate is checked against current injury and lineup news and marked `keep` or `veto` with a reason. Vetoed
   picks stay in the ledger so kept and vetoed picks can be compared.

## Setup

```
cd workspace
uv venv --python 3.12
uv pip install -r requirements.txt
export ODDS_API_KEY=...        # https://the-odds-api.com, for live prices
```

Every command below runs from `workspace/`; `uv run` picks up the `.venv` there.

Raw data is not committed (about 2.2 GB). Game logs, injury reports and player tracking can be re-downloaded with the scripts in
`workspace/src/` (full tracking history: `bash scripts/run_tracking_backfill.sh`, several hours because of stats.nba.com throttling).
The historical prop prices used for the backtests came from BettingPros and are not included, and neither is a downloader for them.

## Daily run

```
uv run python src/ledger.py settle && uv run python src/ledger.py close && uv run python src/ledger.py report
bash scripts/daily_refresh.sh
uv run python src/live_odds.py --books fanduel,draftkings,betmgm,betrivers --mode both
uv run python src/ledger.py log --picks results/live_bets_<stamp>.csv
```

`daily_refresh.sh` uses only stats.nba.com and the NBA injury PDFs (no Odds API credits). It re-downloads the current season's box
scores, parses the last 3 days of injury reports, fetches tracking for new game dates and rebuilds features, and it ends with
`REFRESH OK: box scores through <date> ...`. It uses macOS `date -v`, so on Linux change the `SINCE=` line. `live_odds.py --no-trk`
falls back to the old assists model if the tracking data is broken.
