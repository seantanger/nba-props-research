# nba-ai

Research into whether NBA player-prop betting markets (points, rebounds, assists, threes, points+rebounds+assists) can be beaten
with a statistical model, tested honestly: leak-free features, bets graded at prices that were actually on offer, and a holdout
set aside before any result was seen.

**Status: no proven edge.** The one rule frozen on development data (+6.1% ROI on 6,424 bets, 2020-21 → 2024-25) returned −1.8% on
its 35-bet pre-registered holdout. A forward paper-trading ledger started on 2026-10-25 and has too few settled picks to say
anything yet. Nothing here is betting advice.

## What's here

| Path | What it is |
|---|---|
| `workspace/REPORT.md` | The write-up: data, approaches tried, the chosen rule, holdout results, caveats |
| `workspace/JOURNAL.md` | Running record of every experiment, decision and dead end, with numbers |
| `workspace/README.md` | Pipeline: commands in run order, key files, honesty rules enforced in code |
| `workspace/src/` | Data fetching, features, LightGBM mean + dispersion models, evaluation, live odds, ledger |
| `workspace/data/ledger/ledger.csv` | Forward paper-trading ledger (flat 1 unit per pick, model-only vs reviewed picks) |
| `workspace/data/overrides/` | Daily injury/lineup overrides used by the live run |
| `workspace/results/` | Bet lists, headline JSONs and small result tables |
| `.claude/skills/nba-props/SKILL.md` | Daily routine as a Claude Code skill: settle, injury news, live odds, review, log |

## How it works

1. **Projection model.** LightGBM predicts the mean and dispersion of each stat from box-score history (only games strictly before
   the one being predicted) plus injury-aware features built from the NBA's official injury reports.
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

Raw data is not committed (about 870 MB). Game logs and injury reports can be re-downloaded with the scripts in `workspace/src/`.
The historical prop prices used for the backtests came from BettingPros and are not included, and neither is a downloader for them.

## Daily run

```
uv run python src/ledger.py settle && uv run python src/ledger.py report
uv run python src/live_odds.py --books fanduel,draftkings,betmgm,betrivers --mode both
uv run python src/ledger.py log --picks results/live_bets_<stamp>.csv
```

Before the live run, refresh the current season's game logs: `fetch_nba.py` caches each season file and never re-downloads it, so
delete the current-season files under `data/raw/nba/` first, then rebuild features:

```
rm data/raw/nba/*2026-27*.parquet
uv run python src/fetch_nba.py 2026-27
uv run python src/features.py
```

See `workspace/JOURNAL.md`, P2-8, for the injury-report step in between.
