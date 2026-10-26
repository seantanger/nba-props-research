# NBA player-prop edge research

Everything runs from the workspace root with the project virtualenv (`uv venv --python 3.12 && uv pip install -r requirements.txt`; commands use `uv run`, which picks up `.venv`).
Results, decisions and dead ends are in `JOURNAL.md`; the write-up is `REPORT.md`.

## Pipeline (run in this order)

| Step | Command | Output |
|---|---|---|
| 1. Game logs (nba_api, 2016-17 → today) | `uv run python src/fetch_nba.py` | `data/raw/nba/*.parquet` |
| 2. Historical prop prices (BettingPros, 11 US books, Jan 2021 → today) | not included in this repo: supply your own historical odds as raw JSON in the BettingPros offers format | `data/raw/bettingpros/` |
| 3. Leak-free pre-game features | `uv run python src/features.py` | `data/processed/player_games.parquet` |
| 4. Parse + grade prop lines | `uv run python src/build_bp.py` then `src/build_pairs.py` | `data/processed/bp_lines.parquet`, `results/bp_pairs_cache.parquet` |
| 5. Walk-forward projection distributions | `uv run python src/run_bp.py dev` (and `holdout` once, at the end) | `results/bp_dist_{dev,holdout}.parquet` |
| 6. Price every book line | `uv run python src/eval_bp.py dev|holdout|kambi` | `results/bp_eval_*.parquet` |
| 7. Rule search on dev only | `uv run python src/grid_bp.py`, `src/select_rule.py` | stdout (logged in JOURNAL) |
| 8. Final one-shot evaluation of the chosen rule | `uv run python src/eval_holdout.py holdout` (and `kambi`) | `results/bets_*.parquet`, `results/headline_*.json` |
| 8b. Post-hoc diagnostics (timestamp regime, "seen at tip" tier; labelled post-hoc in REPORT) | `uv run python scripts/diag_timestamps.py`, `scripts/posthoc_holdout.py` | `logs/posthoc_holdout.log`, `results/headline_holdout_posthoc_close5.json` |
| 9. Tonight's bets from live odds | `uv run python src/live_odds.py --books fanduel,draftkings,betmgm,betrivers --mode both` | `results/live_bets_<stamp>.csv` |
| 10. Official injury reports (phase 2) | `uv run python src/fetch_injury.py` then `src/parse_injury.py` then `src/injury_status.py`, then re-run step 3 | `data/raw/injury/pdf/`, `data/processed/injury_{reports,long,status}.parquet` |
| 11. Injury-aware model on dev / reused holdout | `uv run python src/run_bp.py dev inj`, `src/eval_bp.py dev inj`, `scripts/eval_inj.py` | `results/bp_{dist,eval}_dev_inj.parquet`, `logs/eval_inj.log` |
| 12. Review-step proxy (news-explained gaps) | `uv run python scripts/review_proxy.py` | `logs/review_proxy.log` |
| 13. Forward paper-trading ledger | `uv run python src/ledger.py log|settle|close|report` | `data/ledger/ledger.csv` |

Daily routine (settle, overrides, live run, review, log) is the skill in `.claude/skills/nba-props/SKILL.md`; lineup overrides go in `data/overrides/<date>.csv` (see `data/overrides/README.md`).

The original Kambi (Polish book) pipeline from the first two days is still runnable: `src/build_offers.py`, `src/run_dev.py`,
`src/analyze_dev.py`, `src/run_tabpfn.py` (the last one needs `.venv-tabpfn`, see JOURNAL).

## Key files
- `src/common.py` — splits (holdout definitions), pair construction, de-vig, bet grading at actual odds, cluster bootstrap.
- `src/models.py` — LightGBM mean + dispersion models (`proj_dist`), negative-binomial tail probabilities.
- `src/eval_bp.py` — model-alone, consensus-anchored and FanDuel-anchored probabilities, logistic stack.
- `src/select_rule.py` — availability tiers (`all`, `clean120`, `clean30`, `both15`) applied before bet selection.
- `results/chosen_rule.json` — the single rule frozen on dev data before the holdout was evaluated.
- `src/injury_features.py` — report-based (honest) and box-score-absence (leaky, ceiling only) injury features; `src/live_odds.py` — live run with `--books`, `--mode`, `--injury auto`, `--overrides auto`; `src/ledger.py` — forward ledger.

## Honesty rules enforced in code
- Features use only box scores dated strictly before the game (`shift(1)` everywhere) plus static attributes.
- Models are trained on logs strictly before each bi-monthly cutoff and predict only games after it.
- Bets are graded at the stored American/decimal price of the specific book, pushes return the stake, voids are excluded.
- Holdout = seasons 2025-26 and 2026-27 (plus the March–June 2023 window), defined in `JOURNAL.md` before any result was seen.
- The Odds API key is only read from the environment; every raw response is cached under `data/raw/odds_api/`.
