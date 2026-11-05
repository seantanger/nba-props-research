# JOURNAL — NBA player prop edge research

Started 2026-10-21. Workspace: ~/git/nba-ai/workspace

## Status
- [x] Env setup
- [x] Odds API quota check
- [x] Historical prop price source found
- [x] Game logs source
- [x] Features / models / backtest
- [x] Holdout eval
- [x] Live script + report

## Constraints
- Odds API free tier ~448 credits; reserve >=150 for live script. Do NOT pull history from it.
- Never print the key.

## Experiments
(append below)

## 2026-10-21 Data source hunt (done)
- Odds API: 448 credits remaining at start (1 credit spent on /sports). Not used for history.
- Network: a Fortinet web filter blocks gambling-category hosts (bettingpros, actionnetwork, covers, prizepicks, underdog, draftkings, kambi, superbet, sportsgameodds, boxscorelab). Reachable: api.the-odds-api.com, github, huggingface, kaggle (no creds), basketball-reference, site.api.espn.com, rotowire, stats.nba.com (via nba_api).
- HF datasets: propline/graded-player-props-sample = MLB/NFL only (Sep 2026). minmaxxxer = game lines only. Karmane polymarket = 2026 sample. Nothing NBA props.
- GitHub: dozens of model repos, almost none archive prices. FOUND: github.com/wojtrze/nba-stats `offers.csv`:
  129,835 rows / 80,744 unique outcome ids (Kambi/Unibet PL), 2022-11-10 .. 2023-05-11, markets PTS/REB/AST/3PM/ARP(=PRA), decimal odds, both sides,
  `closed_date` = tip time (not scrape time). Same bet_id appears 1-5x = repeated scrapes (price moves) in file order, scrape timestamps NOT recorded.
  Decision: primary backtest uses LAST snapshot per bet_id (closest to close, strictest); secondary uses FIRST snapshot; first->last move used as CLV proxy.
  Players who did not play -> treat as void (Kambi rule), stake returned.
- Also in repo: ESPN game logs (2022-23 heavily duplicated; dedup needed) and 2024-25/2025-26/2026-27 logs. Prefer nba_api logs instead.
- nba_api PlayerGameLogs works (25,894 rows for 2022-23 reg in 1s).

## Exp 1 — Leak found and fixed (2026-10-21)
- First run: proj_lgbm log-loss 0.497, +48% ROI → impossible. Cause: raw same-game columns `FP_HIGH_SCORE`, `TEAM_COUNT` slipped through the feature deny-list, and `pts_share10` divided by the sum over players in THIS game's box score (reveals who played).
- Fix: `feature_cols` is now an explicit regex allow-list of engineered lagged/static columns; `pts_share10` uses the previous game's roster. Lesson: never use a deny-list for features.

## Exp 2 — Standalone models vs closing market, dev walk-forward (last snapshot, 2022-12-01..2023-03-12, n=21,569 pairs with model preds)
- Log-loss: market 0.6887 | proj_lgbm (Poisson mean + dispersion → NegBin) 0.6998 | EB per-minute rate 0.7050 | clf_lgbm (binary on offers incl. implied prob) 0.7004.
- Flat-stake grids: proj thr=0.04 n=8076 ROI −0.16% [−2.4, +2.1]; unders +1.58% (n=5895), overs −4.86% (n=2181). EB all negative (−1 to −3%). clf −3.2% at thr≤0.04.
- Blind baselines: all overs −8.32% (n=24,872), all unders −2.87%. Vig ≈ 6%, so unders lose less than vig → over-side bias in the market (or in the bettor pool).
- Conclusion: none of the three standalone models beats the closing line. Market is well calibrated. Next: stacking (market + model), under-side bias, first-snapshot/CLV, TabPFN.

## Exp 3 — More training seasons for the projection model (2016-17..2022-23 logs, 188k player-games)
- proj_lgbm dev log-loss 0.6998 → 0.6964 (market 0.6887). EB/clf unchanged (~0.705/0.701).
- Flat grids, dev, last snapshot: proj thr=0.04 n=7239 ROI +1.44% [−1.3,+4.0]; thr=0.06 n=4982 +3.22% [+0.1,+6.2]; unders carry it (+1.97%, n=5674), overs ≈ 0.
- Still weak and the grid has ~16 cells per model, so the best cell is selection-biased. Not a decision yet.
- TabPFN: `tabpfn` 6.4.1 installed in separate venv `.venv-tabpfn` (Intel Mac → torch 2.2.2, numpy<2). Weights download without a token; CPU fit OK. Not yet run on NBA data.

## 2026-10-21 (later) — Web filter lifted; revisiting gambling-site historical price sources
(findings below)
- BettingPros: historical NBA prop prices per player/market: the opening line {line, cost, book, created ts} and each book's final
  line {line, cost, updated ts}. (Access details omitted from the public copy of this repo.) Books: 0=consensus, 12=DK, 10=FD,
  13=Caesars, 19=BetMGM, 18=BetRivers, 15=SugarHouse, 27=PartyCasino, 14=Fanatics, 24=bet365, 33=theScore, 49=HardRock, 60=Novig,
  2=Pinnacle, 37=PrizePicks, 36=Underdog, 63=Sleeper.
  - History probe: nothing for 2019/2020 events; 2021-01-10 event has 83 offers (6 markets); 2023-11 onward ~145-170 offers (10 markets). So ~5.5 seasons: 2020-21 (partial books), 2021-22, 2022-23, 2023-24, 2024-25, 2025-26, plus 2026-27 to date.
  - Only open and final state per book are stored (no full tick history). Final `updated` ts is often minutes before tip (FD/DK) → treat the stored line as the closing line; the `updated` timestamp tells us how stale it is.
  - limit max 50 → ~3-4 calls per event. ~1,300 events/season → ~25k calls for everything. Fetch politely (≤2 req/s, backoff), cache each page to data/raw/bettingpros/.

### HOLDOUT DECISION FOR THE NEW SOURCE (written before fetching any results)
- Primary holdout becomes the **2025-26 season (regular season + play-in + playoffs) and any 2026-27 games** in the BettingPros data. Reason: it is the most recent stretch, the most relevant to live betting today, contains the current book set, and is a full season (large n). Never inspected until final evaluation.
- Development data: BettingPros 2020-21 .. 2024-25.
- The original Kambi holdout (2023-03-13 .. 2023-05-11) stays untouched: BettingPros offers for games dated 2023-03-13 .. 2023-06-12 are **excluded from development** (they would reveal the same outcomes), and the Kambi holdout is still evaluated once at the end as a secondary holdout.
- Decision time for the new data: the per-book closing line (final stored state). Secondary: the opening line (first book to post, with its timestamp), which is the most conservative "price you could have had" since it is earliest; difference open→close per book gives real CLV.

## 2026-10-24 BettingPros download complete; data facts
- 7,809 events, 22,470 cached pages, 12.9M book-side rows; 2 events missing (Feb 12 2026, refetched), 107 events in Dec 2020–Jan 2021 have no offers (pre-coverage). nba_api logs 2016-17..2026-27 fetched (play-in files for 2021-22/2022-23 were missing; fixed). Grading: 3 games unmatched of 7,697; voids = players without a box-score row (fixed PLAYER_ID assignment so voids stay in pairs).
- Dev (2020-21..2024-25 minus Kambi window) bettable pairs: 3.0M across 11 sportsbooks (BetMGM, PartyCasino, DK, Caesars, FD, theScore, bet365, BetRivers, SugarHouse, Fanatics, HardRock). No Pinnacle. Median vig 5.9% (FD) .. 7.8% (Caesars). Book "closing" update median 31 min before tip; 25% > 2 h.
- Blind: all overs −9.57%, all unders −3.24% (n=3.0M each). Under bias everywhere but < vig.

## Exp 4 — Line-gap vs reference (model-free), dev
- Rule: bet toward FanDuel when a book's closing line is ≥2 units from FD's closing line. All rows: n=45,458, ROI +15.6% [+13.4,+17.7], hit 63%. Holds every season, every book, both sides, grows with gap size, also with consensus or DK as reference (+7..+10%).
- Availability problem: the edge is concentrated where FD's final update is ≤15 min before tip (median 2 min) and the soft book's stored line is older (median 139 min older). Those stale lines were plausibly suspended on late news; BettingPros stores only the last seen state, so this cannot be verified.
- Strict tier (book's line updated AFTER FD's final update): 2023-24 −1.5% (n=616), 2024-25 +1.75% [−3.7,+8.5] (n=3,074). Earlier seasons strongly positive (2021-22 +42%, n=1,135) but book coverage/scrape cadence differed.
- Tier "both updated within 15 min of tip": 2023-24 +7.0% (n=1,264, CI crosses 0), 2024-25 +15.5% [+10.5,+21.6] (n=2,882). Adding "book after FD": 2024-25 +9.6% (n=496, CI [−0.5,+21.3]).
- When FD is the stale one (FD final >60 min before tip, book updated later): −8.7% (n=1,303) → information is in the freshest line, which is exactly the one you cannot bet against.
- Conclusion: most of the gap edge is likely phantom (suspended markets). A verifiable residual may exist in near-tip rows but the sample is small. Decision deferred until the model-based pricing is evaluated with the same tiers.

## Exp 5 — Model-based pricing on BettingPros, dev walk-forward (2026-10-24)
- Setup: `src/run_bp.py dev` → LightGBM NegBin-style projection (mu, var) per player/game/market, trained on logs strictly before bi-monthly cutoffs (2021-01-01 … 2025-04-01; 2024-25 fixed via `dev2` after a stale-cutoff bug). `src/eval_bp.py dev` adds p_proj (model alone), p_anc_c (consensus-anchored Normal with model σ), p_anc_fd (FanDuel-anchored), and a walk-forward logistic stack (`grid_bp.py`, fit on prior seasons; 2024-25 coefs ≈ [0.07 market, 0.135 proj, 0.035 cons, 0.79 FD]).
- Log-loss (2.7M settled non-FD rows): book implied 0.6854, p_proj 0.6988, p_anc_c 0.6847, p_anc_fd 0.6854. Model alone is clearly worse than the market; it only adds a sliver on top of a market anchor.
- p_proj as standalone bettor (one bet/selection, flat): thr 0.04 all n=291k +1.5% [+1.1,+2.1]; in the verifiable tiers ≈ +1.7% after_fd, −0.4% both15. Not a usable edge after sizing realism.
- p_anc_fd / p_stack "all rows": +13% at thr 0.04 — but this is the Exp 4 gap rule generalized, with the same availability problem.
- **Timestamp anomaly in 2021-22:** in the "book updated after FD" tier the 2021-22 ROI is +25% (n=2,280), but FD's final-update timestamp there is a median 44 h before tip and the book's update a median 30 h after FD's. A line that is really two days stale cannot predict outcomes at +25%, so FD's `updated` field in that season is not the time of the last line change. Treat 2021-22 timestamp-based tiers as unreliable.
- **Clean tier** (FD final update 0–120 min before tip, book's final update later than FD's and before tip; filter rows before choosing the best book per selection — `src/select_rule.py`): p_anc_fd thr 0.03 → n=5,034 ROI +4.8% [+1.6,+8.2], positive in all five seasons (2020-21 +0.3%, 2021-22 +4.5%, 2022-23 +11.2%, 2023-24 +5.6%, 2024-25 +5.2%). With FD ≤30 min: n=2,946 +5.1% [+1.7,+8.3]. p_anc_c in the same tier: +0.7%..+2.3% (CI crosses 0). p_stack: +5.7% but 2024-25-driven (and 2024-25 was then predicted by a stale model; re-checked after dev2 below).
- Why this tier is credible: FanDuel's line is at its final state and was touched within 2 h of tip (a live market), and the other book updated its line *after* that, so the book was still active with FanDuel's final line known. Residual risk: the book's last "update" could itself be the moment of suspension (BettingPros stores no status), and scrape cadence is unknown.

### Exp 5 (cont.) — dev grid after the 2024-25 model fix (`logs/select_rule.log`, tiers applied before best-book selection)
- p_proj alone, clean120/clean30: −0.2% / −0.6% (n≈50k/39k) → the projection model has no standalone edge against these books. Dead.
- p_anc_c (consensus anchor): +2.6% clean120 thr 0.03 (n=4,044) but CI crosses 0 and 2023-24 is −4.0% → unreliable.
- p_anc_fd (FanDuel anchor, model σ), clean120: thr 0.02 +4.7% (n=10,157), thr 0.03 **+6.10% [+2.8,+8.9] (n=6,424)**, thr 0.04 +6.2% (n=4,254), thr 0.05 +8.1% (n=2,994), thr 0.06 +10.3% (n=2,169). thr 0.03 by season: 2020-21 +0.2% (n=780), 2021-22 +6.2% (310), 2022-23 +4.9% (247), 2023-24 +6.4% (1,257), 2024-25 +7.3% (3,830). Edge rises monotonically with threshold, which is what a real mispricing looks like.
- p_stack (logistic on 4 logits, walk-forward): clean120 thr 0.03 +6.1% (n=9,154), by season +6.4/+1.4/+4.0/+7.7. Same information as p_anc_fd (FD coefficient ≈0.8) with fitted weights; not better per bet.
- Variants examined on dev in total: Exp 2–3 (3 models × 3 thresholds × 2 snapshots), Exp 4 (gap sizes × 3 references × 6 tiers × sides), Exp 5 (4 probability columns × 5 thresholds × 4 tiers, twice). Roughly 150 rule variants. One is chosen below.

### DECISION (frozen before holdout, 2026-10-24 12:00 EDT) — see results/chosen_rule.json
Rule: p_anc_fd (FanDuel final line+price → centre; LightGBM dispersion → σ; Normal tail at the other book's line), bet the side whose model probability exceeds the breakeven at the book's price by ≥ 0.03, decimal odds 1.3–3.0, one bet per player/game/market (largest edge), flat 1 unit, **clean120 tier only** (FanDuel's final update 0–120 min before tip, the book's final update later than FanDuel's and before tip). Why: no fitted coefficients, positive in all five dev seasons, the middle threshold rather than the best-looking one, and the only tier whose lines I can argue were actually on the board when the signal was known. Secondary holdout: the 2023-03-13..2023-06-12 window (excluded from dev from the start). The 'all rows' tier will be reported alongside, labelled unverifiable.

## Secondary holdout (Kambi window 2023-03-13..2023-06-12), chosen rule, evaluated once (`logs/eval_holdout_kambi.log`, `results/headline_kambi.json`)
- clean120: n=38, ROI +10.9%, CI [−17.4%, +35.2%], hit 60.5% vs breakeven 54.1%, CLV proxy +1.38 pts. Uninformative (n too small). Other tiers for context: all n=882 +15.2% [+9.8,+20.3] (unverifiable availability), clean30 n=31 +16.8%, both15 n=117 −3.3%.

## PRIMARY HOLDOUT (2025-26 + 2026-27 through 2026-10-23), chosen rule, evaluated once 2026-10-24 13:03 EDT (`logs/eval_holdout.log`, `results/headline_holdout.json`, `results/bets_holdout.parquet`)
- Walk-forward dists: `src/run_bp.py holdout` (5 folds, cutoffs 2025-10-01/12-01/2026-02-01/04-01/10-01; 204,572 player/game/market rows). Log-loss on 1.06M settled rows: book 0.6872, p_proj 0.7043, p_anc_c 0.6867, p_anc_fd 0.6886 (n=810k).
- **HEADLINE (pre-registered rule, clean120 tier): n=35, ROI −1.78%, −0.6 units, CI95 [−33.1%, +28.0%], hit 51.4% vs breakeven 52.1%, avg odds 1.96, CLV proxy +2.80 pts (positive on 60%), max drawdown −6.4 units. P(ROI≤0)=0.54.** The pre-registered test did not confirm the edge. It is also nearly uninformative: 35 bets in a full season vs 3,830 in 2024-25 on dev.
- Same rule, other tiers (context, not headline): all n=24,192 +5.37% [+3.7,+7.2], CLV +4.39 pts positive on 82%, max DD −82 units, top-5 days 20% of profit; clean30 n=14 +5.6%; both15 n=14 +15.6%.
- **Why only 35 bets — the archive's timestamp semantics changed in 2025-26** (`scripts/diag_timestamps.py`, counts only, no outcomes): in dev the book's final-state timestamp was spread out (2024-25 book mins-before-tip q10/50/90 = 2/29/422, FanDuel median 5 min before tip, 90.7% of FD lines updated ≤120 min before tip). In 2025-26, ~75% of ALL rows, every book incl. FanDuel, are stamped ≈1.5 min AFTER scheduled tip (FD quantiles −3.3/−1.6/+1.9). The at-tip stamps vary line by line within an event and book (median 38 distinct timestamps per event+book, per-event spread median 5.5 min), so they are per-line last-seen times from scraping up to tip-off, not a bulk close stamp. The ~25% of rows with a pre-tip stamp (median 3.7 h before tip) are therefore lines that disappeared from the board before tip — the opposite of what the clean120 tier assumes (it requires both stamps ≥0 min before tip and the book's after FanDuel's). Result: clean120 covers 1.4% of FD-line rows in 2025-26 vs 19.5% in 2024-25, and 0% of 2026-27.
- Book set also changed in 2025-26: BettingPros now carries DFS/pick'em (PrizePicks, Underdog, Sleeper, Betr, Fliff — excluded by `bettable`), exchanges/prediction markets (Novig, ProphetX, Polymarket — included as bettable), and FanDuel's row count fell (113k rows vs ~100k in 2024-25 — similar) while HardRock became the largest sportsbook.

### POST-HOC look (NOT pre-registered; defined after seeing the headline and the 'all' tier above) — `scripts/posthoc_holdout.py`, `logs/posthoc_holdout.log`
- Tier that corresponds to availability in the new regime: both FanDuel's and the book's line last seen within 5 min of tip (either side). Same frozen rule otherwise (p_anc_fd, thr 0.03, odds 1.3–3.0, one bet/selection, flat).
- n=19,491, ROI +4.51% [+2.7%, +6.4%], +879 units, hit 54.1% vs be 52.0%, CLV proxy +4.23 pts positive on 82%, max drawdown −99.6 units over 205 days, top-5 days 24% of profit. Edge-bucket monotone: 0.03–0.04 +2.3%, 0.04–0.05 +2.8%, 0.05–0.07 +4.3%, ≥0.07 +8.7%. By month: Oct −0.8%, Nov +3.9%, Dec −1.2%, Jan +4.9%, Feb +9.7%, Mar +9.7%, Apr +9.4%, May +9.7% (n=84), 2026-10 +24% (n=150). Under +6.0% (n=10.6k), over +2.7% (n=8.9k).
- By book: Polymarket +14.5% (n=2,917, avg odds 2.17), Novig +8.2% (n=1,145), ProphetX −0.2%; sportsbooks: BetMGM +4.4%, HardRock +3.1% (n=5,660), DK +3.2%, Caesars +2.0%, PartyCasino +0.3%, BetRivers/SugarHouse/theScore ≈ −1..−2%. **Sportsbooks only (no Polymarket/Novig/ProphetX): n=13,854, +2.64% [+0.5%, +5.1%]**; exchanges only +9.1% (n=5,637) — exchange prices are thin and may not fill, so treat the exchange part as unverified.
- Caveats: defined after the fact (I knew the 'all' tier was +5.4% when I defined it); 'last seen at tip' still cannot distinguish a listed-but-suspended line from a live one; scrape cadence unknown. This number needs its own untouched holdout (the rest of 2026-27) before it can be believed.

## Exp 6 — TabPFN v6.4 (tabular foundation model) as the projection model, Kambi dev walk-forward (`logs/tabpfn2.log`, `results/dev_preds_tabpfn.parquet`)
- Setup: TabPFNRegressor (CPU, 4 estimators), per market, trained on the last 6,000 player-games before each of the 3 dev cutoffs, top-30 LightGBM-gain features; predictive distribution → P(over) at the Kambi line. 3 folds × 5 markets took 88 min on 4 threads while sharing the CPU.
- Log-loss 0.7076 vs market 0.6891 vs LightGBM NegBin 0.6998 (same n=21,569). As a bettor: thr 0.02/0.04/0.06 → −3.3%/−3.7%/−3.3%, all CIs below zero.
- Conclusion: dead. Worse than LightGBM on the same slice, far too slow for the 6.4M-row BettingPros backtest, and no sign it would help the anchored rule (the anchored rule only uses σ, where LightGBM's dispersion model already suffices).

## Live run (final) 2026-10-24 13:36 EDT — `logs/live_final.log`, `results/live_bets_20261024_1336.csv`
- 8 games, 7 fetched (35 credits; MIA@CHA served from the 12:07 cache), 408 credits remaining. 4,432 outcome rows, 6 books, 1,807 selections priced vs FanDuel, 17 bets at the frozen rule, 4.5–8.5 h before tip. Largest edges are large line gaps (Allen PTS 17.5 vs FD 12.5; Miller PTS 26.5 vs 22.5) — exactly the stale-line pattern; flagged in REPORT. Nothing staked.

## Status (end of session 2026-10-24)
- Pre-registered holdout: chosen rule −1.78% on 35 bets → NO_EDGE_FOUND. Cause: archive timestamp regime changed in 2025-26, so the pre-registered availability tier is nearly empty. Post-hoc 'both seen at tip' tier: +4.51% (n=19,491), sportsbooks only +2.64% (n=13,854) — not pre-registered, needs the rest of 2026-27 as a fresh holdout.
- Deliverables: README.md, REPORT.md, src/ pipeline, scripts/ (post_dev2.sh, diag_timestamps.py, posthoc_holdout.py), results/chosen_rule.json, headline_*.json, bets_*.parquet, live_bets_*.csv.

---
# PHASE 2 (started 2026-10-25): injury-aware workflow

## P2-1 Injury data source (`src/fetch_injury.py`, `src/parse_injury.py`, `src/injury_status.py`)
- Source: official NBA injury-report PDFs at `https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH><AM|PM>.pdf`
  (hourly files through 2025-12-19; `_<HH>_<MM><AM|PM>.pdf` from 2025-12-22). Found via the `nbainjuries` package (its parser needs Java;
  I wrote my own with pdfplumber). The file label is the hour; the report's own header says HH:30 (e.g. `_05PM` = "05:30 PM" report), so
  `report_ts` is parsed from the header, not the filename.
- Fetched hours 11AM..09PM for every game date 2020-12-22..2026-10-24: 12,218 PDFs cached in `data/raw/injury/pdf/`, 1,539 missing
  (2020-21 only published 01PM/05PM/08PM; a handful of other gaps). 2016-2020 reports are not available at this URL (nothing before 2020-12).
- Parser: word positions vs header x0s; a row is anchored by its status word, other words attach to the nearest anchor (reasons wrap
  across lines in the 2023+ layout). 961,467 rows, 12,197 reports, 27 files with notes (5 empty reports, rest minor). Columns: report_ts,
  game_date, game_time, matchup, team, player ("Last, First"), status, reason.
- Mapping: matchup+date -> nba_api GAME_ID (94.5% of rows; the rest are postponed games and next-day games not yet in logs); "Last, First"
  -> PLAYER_ID via normalized name within team/season (97.0%; unmatched are two-way/long-term players with no logged game).
- Game time in the report is a 12-hour clock without AM/PM ("07:00 (ET)" = 7 PM); hours 1-10 are PM. (First version got this wrong and
  showed 40% of listings "after tip"; fixed.)
- Decision-time status = latest report at or before tip-off minus 60 min (`status_dec`); also `status_final` (<= tip) and `status_first`.
  87,532 (game, team, player) listings over 7,742 games; 98.6% have a decision-time status; median first listing ~24 h before tip.

## P2-2 Coverage / leak quantification (`logs/injury_coverage.log`, 2020-21..2025-26, team-games with any listing = 96.8%)
- Rotation absentees = players with a 10-game minutes norm >= 20 who played the team's previous game but have no box-score row tonight: n=7,526.
  Status at tip-60: Out 64.2%, Doubtful 3.4%, Questionable 15.6%, Probable/Available 1.0%, **not listed 15.9%**.
  Of the ~32% not Out/Doubtful at tip-60: 46.5% became Out by tip (news in the final hour), 41.9% never appeared on the report
  (coach's decision / late scratch), 9.6% stayed Questionable.
- Reliability of the decision-time label: Out -> played 0.1% (n=21,624); Doubtful -> played 1.9% (n=731); Questionable -> played 53.9%
  (n=6,136); Probable -> played 92.7% (n=2,367); not listed & played last game -> played 90.5% (n=144,781).
- Conclusion: a box-score-absence feature would leak about one third of rotation absences (plus all DNP-CDs). The report-based feature at
  tip-60 is honest and covers two thirds of them. Both are built (`vac_*` honest, `bsvac_*` leaky) so the gap can be measured.
- Early-season artefact fixed: the "recent roster" (last 5 team games) resets each season, otherwise October absences are roster turnover.

## P2-3 Review-step proxy (idea 3) on DEV (`scripts/review_proxy.py`, `logs/review_proxy.log`, `logs/review_proxy_slack60.log`)
Setup: frozen rule candidates (p_anc_fd thr 0.03, odds 1.3-3.0, one per selection), phase-1 dev distributions. "News" = an injury-report
status change on the player's team published after FanDuel's last update and no later than the book's last update (capped at tip):
teammate (>=15-min norm) to Out/Doubtful, teammate back from Out, the player himself to Questionable/Doubtful/Out, or upgraded.
Explained gap = news direction matches the book-minus-FanDuel line gap (gap up & teammate out / self upgraded; gap down & teammate back /
self downgraded). VETO = explained.
- clean120 tier (honest availability): all n=6,424 +6.10%; **kept n=6,238 +6.44% [+3.9,+9.0]; vetoed n=186 −5.10% [−21.6,+18.2]**
  (hit 52.2% vs breakeven 54.7%). Vetoed by teammate minutes ruled out: 0-15 +20% (n=17), 15-25 −3.7% (57), 25-35 −19.6% (47), 35+ −9.4% (60).
  By |gap|: 0.5-1.5 +8.7% (94), 1.5-2.5 −15.2% (68), >=2.5 −30.5% (24). 161 of the 186 are 2024-25 (clean120 is mostly 2024-25).
- News AGAINST the gap (FanDuel has the news, the other book is stale): n=141 **+25.3% [+6.3,+45.1]**, hit 68.8%. Best subset by far.
- No news at all: n=6,007 +6.21%. Gap==0 (price-only edge): n=1,601 +5.95%.
- Bet WITH the book instead (take FanDuel's stale line on the news side at FanDuel's final price): n=186 +6.44% [−17.4,+26.1];
  restricted to starter out (>=25 min) or |gap|>=1.5: n=148 +14.1% [−12.1,+36.9]. Positive but not significant, and FanDuel's line may
  have been suspended at that point (BettingPros stores no status) -> treat as unverified; worth logging in the forward ledger.
- Sensitivity: counting news up to 60 min BEFORE FanDuel's last touch as well ("slack 60"): vetoed n=571 +4.0% [−6.4,+14.5] — news
  FanDuel had time to price is not a problem; the strict definition is the right one.
- 'all' tier (unverifiable availability, context only): vetoed n=961 +17.2%, against n=673 +25.8%; the vetoed set is profitable there
  because the book's "final" line in that tier is often a line that was taken down long before tip.
- Conclusion: veto rule adopted for the skill = explained gap with the strict window (dev evidence: −5% vs +6%, n=186, CI wide). Also
  flag "FanDuel has the news" bets as the highest-priority keeps. Counting: 2 window variants x 1 rule; subsets reported, none chosen on.

## P2-4 Diagnosis of the live model-only under lean (172 unders vs 40 overs on 2026-10-25, `results/proj_only_20261025.csv`)
- Live stub features vs October rows of past seasons (gp_season<=4): no NaN-pattern anomalies, no feature off by more than ~1 sd
  (stubs are simply better players than the all-player reference). Not a live-feature bug.
- Dev baseline: the phase-1 model's P(over) at FanDuel's line is below 0.5 on 64% of selections in every month (67% in October);
  mean mu - FanDuel line for PTS is -0.05..-0.34 while actual - line is +0.1..+0.3 (right-skewed outcomes). So a strong under lean is
  this model's normal behaviour, not an early-season effect alone.
- The largest live gaps are injury-driven lines the model cannot see: Paul Reed PRA 27.5 (mu 18.1) and Cade Cunningham PRA 41.5 (35.0)
  with Jalen Duren out; Franz Wagner PRA 33.5 (27.6) and Desmond Bane PRA 27.5 (20.3) with Paolo Banchero out. Books raise the
  replacements' lines; the injury-blind model says under. That is the case the injury features are built for (P2-5).

## P2-5 Injury-aware projection model on DEV (`src/run_bp.py dev inj`, `scripts/eval_inj.py`, `logs/eval_inj.log`)
Setup: same walk-forward folds and LightGBM settings as phase 1; feature set = phase-1 288 cols + report-based injury features
(vac_*, ret_*, share_*, inj_self_*, avail_*, opp_vac_*; box-score-absence `bs*` features excluded = honest). Rows before 2020-12-22 have
the injury columns NaN. Training 16:41-18:20 EDT 2026-10-25 (`logs/run_bp_dev_inj.log`, 99 min). Same rows for both models (4,105,678 lines).
- Importance (`logs/feat_importance.log`, PTS model, cutoff 2024-10-01): share_pts = 3rd feature (10.0% of gain), injury block 13.9% of gain
  for PTS, 5.1% for REB, (AST similar).
- (a) FanDuel rows n=292,919: avg miss 3.483 (inj) vs 3.503 (p1) vs 3.432 (FD line); right side 52.88% vs 52.74%; log-loss 0.7000 vs 0.7035
  vs 0.6867 (FD price); bias −0.101 vs −0.138 (PTS −0.200 vs −0.256; PRA −0.216 vs −0.303). On teams with vac_min>=20 (n=140,091) bias
  −0.089 vs −0.349 (the injury-driven under-projection is mostly gone) but miss 3.508 vs 3.536 vs FD 3.448. Every market's log-loss improves.
- (b) model-only (p_proj vs book price), 30 comparisons (5 book groups x 2 tiers x thr .03/.05/.08): injury model better in 30/30.
  FanDuel: −0.87/−0.53/−0.40% (n=132k/96k/57k) vs p1 −1.29/−0.85/−0.96%. Ontario (DK+MGM+BR) clean120: +0.07/+0.85/+1.17%
  (n=33.7k/24.5k/14.6k; CI at .08 [−0.7,+3.0]) vs p1 −0.02/+0.05/+0.40%; 'all' tier (inflated) +0.91/+1.59/+2.10%. Per book clean120 thr .08:
  DK −1.25% (5,937), MGM +1.31% (5,859), BR +1.31% (6,402). under_frac 0.75-0.85 for both models.
- (c) chosen rule p_anc_fd clean120 thr .03 sportsbooks: +6.54% n=6,743 [+3.6,+9.4] (inj) vs +6.07% n=6,651 (p1), same rows.
- Review proxy with the inj model (`logs/review_proxy_inj.out`): kept n=6,326 +6.92%; vetoed n=188 −5.28% [−20.3,+16.6]; against n=143
  +24.96% [+2.6,+47.5]. Same picture as with the phase-1 model.
- DECISION: injury features earn their place (consistent improvement everywhere; it is a feature-set change, not a tuned rule) -> the live
  script uses them. Model-only path: break-even, not profitable: still negative at FanDuel; +0.9..+1.2% at DK/MGM/BR in the honest tier with
  intervals including zero. Model-only threshold for the ledger set to 0.08 (chosen on dev; counts as one chosen variant among 3 thresholds).

## P2-6 Workflow built (2026-10-25)
- `src/ledger.py` log/settle/close/report (flat 1u; settle via nba_api PlayerGameLogs by date, cached `data/raw/nba/daily/`; DNP = void;
  closing line = newest cached odds snapshot after the pick's run and before tip, same book/market/player/side; report by source x verdict
  with a game-date bootstrap). Tested: logged the 17 legacy 2026-10-24 picks into a temp ledger and settled them from nba_api (10-7, +0.12u;
  Jarrett Allen under 17.5 lost, scored 21) — temp ledger deleted, real ledger starts 2026-10-25.
- `src/live_odds.py`: `--books` (default fanduel,draftkings,betmgm,betrivers), `--mode rule|model|both` (rows carry `source`
  rule / model_fd / model_book), `--injury auto` (downloads today's latest official report PDF; tries :30/:00 slots backwards; cached),
  `--overrides auto` (`data/overrides/<date>.csv` team,player,status; wins over the report), `--model-thr`. Output adds player_id, team,
  event_id, commence, news (rotation players Q/D/Out on the player's team, `*` = override, SELF = own status), news_flag
  (rule rows: explained / against / same_line / none from the gap direction vs the news; model rows: news / none), report_ts, and empty
  claude_verdict / claude_reason columns for the review.
- Dry run on the cached 2026-10-25 17:59 UTC snapshot (`logs/live_dry_p2.log`): 7 games, 4:30 PM report parsed (50 rows matched),
  overrides applied (Duren OUT, Banchero OUT, Thompson QUESTIONABLE); 1,261 priced selections; rule=3 (2 explained -> vetoed, 1 against
  -> kept), model_fd=145, model_book=165 at thr .05 (the model-only path flags a quarter of all selections; see P2-5 before trusting it).
  Reviewed with web news (Duren ruled out Sat evening; Murray GTD; five Clippers out), logged 313 rows (209 keep / 104 veto) to
  `data/ledger/ledger.csv`. These are the first forward-ledger picks (prices as of 13:59 EDT, all pre-tip).
- Skill: `.claude/skills/nba-props/SKILL.md`.
- Live test with fresh odds (`logs/live_close_20261025.log`, `--hours 3 --max-events 2`): ORL@NYK + LAC@OKC at 22:43 UTC, **11 credits
  spent, 286 remaining**; 6:30 PM report auto-downloaded; 73 bets printed, NOT logged (purpose was a closing-line snapshot). `ledger.py close`
  then filled closing lines for 85 of the 313 logged picks (mean clv_line −0.035; keep −0.03 n=66, veto −0.05 n=19). Not evidence yet.
- Day-1 ledger model rows were logged at thr .05 (default raised to .08 afterwards); the `edge` column allows filtering ≥.08 in `report`.
- Ledger status at end of session: 313 open picks for 2026-10-25, 0 settled. Settle after ~1 AM ET on 2026-10-26.

## P2-7 REUSED HOLDOUT (2025-10-21..2026-10-23), injury-aware model, evaluated once after all dev choices (`scripts/eval_inj_holdout.py`, `logs/eval_inj_holdout.log`, `results/bp_dist_holdout_inj.parquet`, `results/bp_eval_holdout_inj.parquet`)
- Status of this data: NOT clean. Opened in phase 1 (headline + post-hoc close5 tier defined after seeing it). This look: 2 thresholds x 2 tiers
  x 5 book groups for model-only, 3 tiers for the rule. All choices (features, thr .08, strict review window) were made on dev first.
- 1,837,691 aligned rows (identical between models). clean120 is nearly empty (n=34) because of the BettingPros timestamp change.
- (a) FD rows n=113,250: miss 3.792 (inj) vs 3.821 (p1) vs 3.726 (FD line); right side 52.59% vs 52.45%; log-loss 0.7005 vs 0.7056 vs 0.6880 (FD
  price); bias −0.001 vs −0.087 (FD −0.122). PTS miss 4.942 vs 4.985 vs 4.853. Under-bias gone; FD still more accurate. Same as dev.
- (b) model-only thr .08: FD all −0.16% n=23,299 [−2.0,+1.9] (p1 −0.72%); Ontario all +1.94% n=39,345 [+0.6,+3.4] (p1 +0.81%); Ontario close5
  +1.63% n=24,855 [+0.0,+3.4] (p1 +1.13%). Per book close5: DK +0.34% n=19,811; MGM −1.22% n=16,710; BR −1.05% n=6,158.
  thr .05: FD −0.41%/−0.49%; Ontario all +0.80% n=63,318 [−0.3,+1.9], close5 +0.73% [−0.7,+2.2]; DK −0.43%, MGM −1.72%, BR −2.52% [−4.4,−0.6].
  Injury model > phase-1 model in all 20 cells (as on dev, 30/30).
- (c) chosen rule with inj dispersion: clean120 n=34 −4.19% (p1 n=35 −1.78%); close5 +4.33% n=19,728 [+2.4,+6.3] (p1 +4.55%); all +5.22% (p1 +5.39%).
- INTERPRETATION: the pooled Ontario model-only number (+1.6–1.9%, interval touching zero) is driven by DraftKings alone, whereas on dev DK was the
  negative book and MGM/BR the positive ones (+1.3% each). Book-level sign flips between periods = noise, not a robust edge. FanDuel-only
  model-only is ≤0 on both dev and holdout. The rule's post-hoc close5 number is unchanged by the new dispersion. Nothing here qualifies as
  "positive on data it was not chosen on": the holdout is reused and the forward ledger has 0 settled picks.
- Review proxy NOT run on holdout (clean tier n=34 → vetoed subset empty). Dev numbers with inj model are in P2-5 / REPORT §10.4.

## Status (end of session 2026-10-25)
- FINAL_STATUS: NO_EDGE_FOUND. Dev-chosen rule +6.5% (clean, n=6.7k) has only a reused holdout; model-only path is break-even (FD ≤0; Ontario
  +0–2% with unstable per-book signs). Forward ledger started: 313 picks logged 2026-10-25, none settled. Credits: 11 spent this phase, 286 left.
- Next: run `.claude/skills/nba-props/SKILL.md` daily (settle → overrides → live run → review → log → report). Do not re-tune on the ledger
  before ~300 settled rule picks.

## P2-8 Daily run 2026-10-26 + STALE-DATA FIX
- Settled 2026-10-25: 300 graded / 13 void, +5.74u (+1.9%); kept −4.15u (−2.1%, n=196), vetoed +9.88u (+9.5%, n=104); rule 1–2.
  At the current model threshold (edge ≥ .08) kept model rows −16.3u on n=117. One date: noise.
- BUG FOUND: `fetch_nba.cached()` never re-fetches an existing file, so the 2026-27 logs were frozen at 2026-10-24 10:03 (games
  through 10-23). The 10-25 ledger picks and the first 10-26 run were scored on a model that had not seen 10-24/10-25 games (15 of 43
  games this season). Fixed for today: moved the stale 2026-27 raw files to `data/raw/nba/_stale_20261024/`, re-ran
  `fetch_nba.py 2026-27`, appended the new injury PDFs (10-23..10-26) to injury_reports.parquet (incremental parse, full re-parse takes
  ~40 min), then features.py -> injury_status.py -> features.py. Backups: `data/processed/*_pre20261026.parquet`.
  TODO: make the season refresh part of the daily skill (delete current-season raw files before fetch_nba, or add a --refresh flag).
- Live run 16:38 (45 credits, 241 left); first output on stale data kept as `results/live_bets_20261026_1638_staledata.csv` (308 rows, not
  logged). Replayed with --dry on refreshed data -> `results/live_bets_20261026_1654.csv`, 227 rows (rule 4, model_fd 97, model_book 126),
  reviewed and logged: 165 keep / 62 veto.

## P2-9 Tracking-data probe (2026-10-27; `scripts/check_tracking_fields.py`, `logs/tracking_probe.log`, raw JSON in `data/raw/nba/tracking_probe/`)
Fields-only check, 13 stats.nba.com calls via nba_api 1.11.4, 2.5 s apart. Nothing built, no retrain, ledger untouched, no Odds API credits.
- `BoxScorePlayerTrackV3` (per game, one call per GAME_ID). Player statistics: minutes, speed, distance, reboundChancesOffensive/
  Defensive/Total, touches, passes, assists, secondaryAssists, freeThrowAssists, contested/uncontested FGM/FGA/%, fieldGoalPercentage,
  defendedAtRimFGM/FGA/%. **Has** rebound chances (O/D/T), touches, passes, secondary + free-throw assists, contested/uncontested shots.
  **Does NOT have potential assists.** Identical field list and populated values for 2016-17 (0021600001, LeBron: 95 touches, 71 passes,
  22 rebound chances) and 2026-27 (0022600116). Per-game coverage therefore spans the whole training window (tracking exists from 2013-14).
- Potential assists live in `LeagueDashPtStats(pt_measure_type='Passing', player_or_team='Player')`: PASSES_MADE, PASSES_RECEIVED, AST,
  FT_AST, SECONDARY_AST, POTENTIAL_AST, AST_PTS_CREATED, AST_ADJ, AST_TO_PASS_PCT(_ADJ). Accepts DateFrom/DateTo; with a one-day window it
  returns one row per player who played that day (72 rows for 2016-10-25, GP=1, Totals), i.e. per-game values since nobody plays twice
  a day. Same endpoint with 'Rebounding' (OREB/DREB/REB chances, contested/uncontested, chance %, deferred, avg distance) and
  'Possessions' (touches, front-court/elbow/post/paint touches, time of possession, pts per touch) also date-filters. `PlayerDashPtPass`
  is per player-season (pass-to/pass-from by teammate), no POTENTIAL_AST, not needed.
- `ShotChartDetail`: one row per FGA with GAME_ID, GAME_DATE, period/clock, ACTION_TYPE, SHOT_TYPE, SHOT_ZONE_BASIC/AREA/RANGE,
  SHOT_DISTANCE, LOC_X/Y, made flag. Works for 2016-17 and for a single 2026-27 game (game_id_nullable). `player_id=0` + DateFrom/DateTo
  returns every player's shots for that day in one call (521 FGA, 3 games, 67 players on 2016-10-25) -> league-wide per-game zone counts
  are cheap. Per-shot rows keyed by game -> leak-free if aggregated over prior games only.
- `SynergyPlayTypes`: season aggregates only (no date/game filter in the endpoint; returns 0 rows unless play_type is given). Fields:
  PLAY_TYPE, POSS_PCT, PPP, FG_PCT, FT/TOV/SF/plus-one/score poss %, EFG, POSS, PTS, FGM/FGA/FGMX, PERCENTILE. 2016-17 Isolation returned
  288 players; 2026-27 only 18 (early-season minimum). **Leaks in a backtest**: a season row includes the games being predicted. Usable
  only as previous-season context (one row per player per play type per season), which is weak for props.
- Leakage summary: box-score tracking, Passing/Rebounding/Possessions by date, and shot charts are all game-keyed and safe when shifted to
  prior games (same discipline as the existing rolling features). Synergy is not. None of the tracking numbers are available for the
  current game before tip, so they can only enter as rolling averages / shares, like everything else.
- Backfill cost 2016-17..2026-27 (12,892 distinct games, 2,075 distinct game dates, all season types):
  BoxScorePlayerTrackV3 = 12,892 calls (~9 h at 2.5 s/call, ~4 h at 1 s; stats.nba.com throttles, expect retries);
  LeagueDashPtStats Passing by date = 2,075 calls (~1.5 h) for potential assists (Rebounding/Possessions another 2,075 each but the box
  score already has chances/touches); ShotChartDetail player_id=0 by date = 2,075 calls (~1.5 h, ~500 rows each). Synergy = 10 play types
  x 2 groupings x 11 seasons = 220 calls, season-only. Total for the useful, leak-safe set ≈ 17k calls, roughly 12-14 h sequential.
- Not decided here whether to add any of this; the prior that matters is that rebound chances / potential assists are the one tracking
  signal with a direct link to REB/AST props (they are usage proxies with less noise than the stat itself).

## P2-10 tracking features (2023-24..2026-27) — `src/fetch_tracking.py`, `src/tracking_features.py`, `scripts/eval_trk.py`, `scripts/paired_trk.py`
- Fetch (stats.nba.com only, no Odds API): 637 game dates (2023-10-24..2026-10-25, all season types) x 4 endpoints = 2,548 calls, one
  JSON per date under `data/raw/nba/tracking/{passing,rebounding,possessions,shots}/`. Season type (Regular/PlayIn/Playoffs) passed
  per date from the GAME_ID prefix (a Regular-Season call on a playoff date returns 0 rows). Resumable; dates >= today (ET) are always
  re-fetched; `--refresh-from DATE` forces older ones (so the current season cannot freeze like `fetch_nba.cached()` did).
  Throttling: stats.nba.com answers bursts with silent 60-s read-timeouts for 10-20 min; first run died at 600 calls with 6 retries,
  re-run with 40 retries / 5-min cap and 3.0-s sleep. Wall clock 12:10-17:06 for 1,945 calls incl. 6 throttle episodes (~3 h of
  backoff) -> ~400 calls/h effective, ~1,000/h when not throttled. Logs `logs/fetch_tracking.log`, `logs/fetch_tracking2.log`.
- Spot-check (zero extra calls, cached P2-9 probes): one-day LeagueDashPtStats window == BoxScorePlayerTrackV3 for the same game:
  touches / rebound chances (O,D,T) identical for all 21 OKC-LAC players (2026-10-25); passes / secondary / FT assists / AST
  identical for all 26 CLE-NYK players (2016-10-25). Built tables: `tracking_player_games.parquet` 85,818 player-dates
  (GP==1 everywhere, no duplicate player-dates), `tracking_shots.parquet` 710,906 FGA. Coverage vs logs 2023-24+: passing/possessions
  99.7% of player-games, shots: per-player FGA sums equal the box score 100%; Rebounding endpoint omits players with 0 REB (12% of
  rows, mean 8.7 min) -> chances imputed 0 for those. One date with no tracking at all on stats.nba.com: 2024-03-09 (shots exist).
- Features (90 cols, NaN before 2023-24; env PLAYER_GAMES + FEATURE_EXTRA='^trk_' so production player_games.parquet is untouched):
  trk_{pot_ast, ast_pts, sec_ast, ft_ast, passes, oreb_ch, dreb_ch, reb_ch, reb_cont, touches, poss_time, paint_t, elbow_t, post_t,
  fc_touches}_m{5,10,20}; per-zone FGA (rim / paint non-RA / mid / corner3 / above-break3, rim FGM, all) _m{5,10,20}; ratios _r{10,20}
  (contested share, reb per chance, ast per potential ast, pts per touch, zone shares of FGA, rim FG%); opp_trk_{pot_ast, reb_ch,
  rim_fga, paint_t}_15 allowed. All shift(1) WITHIN player/team.
  BUG NOTED (production, not changed): `features.lagged_roll` rolls the group-shifted series across group boundaries, so the first
  w-1 games of each player's/team's history borrow the previous PLAYER_ID's last games (e.g. Carmelo 2016 gets LeBron 2026 values).
  Not target leakage, but noise in ~w games per player; worth fixing in a future rebuild (changes the production model -> re-validate).
- DEV2 = 2024-25 season walk-forward (cutoffs 2024-10-01/12-01/2025-02-01/04-01/09-01), training from 2016-17 (tracking present for
  2023-24 onward). Baseline = injury-aware model re-run on today's data (`bp_dist_dev2_base`; mu corr 0.9999 with P2-5's run).
  Same 1,450,301 priced rows both models. Variants: ONE feature set (no tuning, no selection), 2 thresholds x 2 tiers x 5 book groups,
  10 markets. Logs `logs/eval_trk_dev2.log`, `logs/paired_trk_dev2.log`, `logs/feat_importance_trk.log`.
  (a) FanDuel rows n=99,584: miss 3.916 (trk) vs 3.919 (base) vs 3.842 (FD line); right side 52.27% vs 52.10% (FD price 48.9%);
      log-loss 0.7026 vs 0.7030 vs 0.6877 (FD). Paired bootstrap by date (tracking minus baseline, positive = better):
      ALL d_ll +3.8e-4 [+1.3,+6.1], d_miss +2.7e-3 [+1.4,+4.1]; AST n=8,669 d_ll +21e-4 [+5,+37] (0.6985->0.6964), miss 1.902->1.895,
      side 53.0->53.6%; REB n=15,057 d_ll +5.5e-4 [+0.6,+10.8], miss 2.120->2.117, side 52.95->53.2%; PTS n=17,302 d_ll -1.9e-4
      [-6.0,+2.4] (no gain from shot zones); PRA/PA/PR/FG3M intervals include 0. Injury features (P2-5) were worth ~35e-4 on ALL.
  (b) model-only thr .08 'all': FanDuel -3.22% n=15,924 [-5.5,-1.1] vs base -3.71% n=15,757; Ontario +1.03% n=34,519 [-0.4,+2.6] vs
      +0.53% n=34,333; clean120 Ontario +0.49% n=6,941 vs -0.85%. thr .05: FD -1.98% vs -2.73%; Ontario +0.76% vs +0.45%.
      By market thr .08: AST Ontario +4.11% n=3,614 [+0.9,+7.2] vs +2.45% n=3,495; AST FD +2.37% n=1,616 vs +0.23% n=1,506;
      REB Ontario +3.64% n=3,542 vs +3.08%; PTS Ontario +0.39% vs +0.28%, FD -4.97% vs -4.26%; PRA FD -7.2% vs -6.0%;
      PA/PR less negative with tracking (-1.6/-1.2% vs -3.1/-2.1% Ontario). (2024-25 alone is a worse FanDuel year than the 5-season dev.)
  (c) anchored rule: clean120 +7.06% n=3,954 vs +7.49% n=3,931; all +10.49% vs +10.40%. Unchanged, as expected.
  Importance (single fit, cutoff 2025-04-01, 240,803 rows): tracking gain share AST 0.5%, REB 0.2%, PTS 0.1% — tracking is NaN for
  ~80% of training rows, so the trees barely use it; AST_ewm alone is 71% of AST gain.
  DEV DECISION: small, statistically real accuracy gain in AST and REB; nothing in PTS/FG3M from shot zones; model-only ROI moves
  ~+0.5pp but stays inside noise and FanDuel stays clearly negative. Evaluate the reused holdout once with this single feature set.
- REUSED HOLDOUT (2025-10-21..2026-10-23; cutoffs 2025-10-01..2027-09-01; this window has already been looked at in P2-7/P2-8, so it
  is NOT a clean holdout). One look, same single feature set, same variant grid (2 thr x 2 tiers x 5 book groups x 10 markets);
  tiers here are 'all' and 'close5' (mins_before_tip <= 5) because the 2025-26 timestamp regime makes clean120 nearly empty.
  1,837,691 aligned rows both models. Logs `logs/eval_trk_holdout.log`, `logs/paired_trk_holdout.log`.
  (a) FanDuel rows n=113,250: miss 3.788 (trk) vs 3.792 (base) vs 3.726 (FD); right side 52.64% both; log-loss 0.7006 vs 0.7005
      vs 0.6880 (FD); bias -0.068 vs -0.001 (tracking re-introduces some under-projection; PTS -0.117 vs -0.050).
      Paired: ALL d_ll -1.0e-4 [-3.7,+1.6] (flat), d_miss +4.4e-3 [+2.8,+5.9]; AST n=10,914 d_ll +17.8e-4 [+5.3,+29.7]
      (0.6963->0.6946), miss 1.762->1.755, side 53.2->53.8%  -> the AST gain REPLICATES; REB n=16,397 d_ll 0.0 [-6.6,+6.7],
      miss 2.054->2.047 (+6.2e-3 [+4.3,+8.1]) -> miss better, log-loss flat; PTS d_ll -7.2e-4 [-11.6,-2.4] WORSE; PR d_ll -8.9e-4
      [-13.2,-4.2] WORSE; PRA -2.0e-4 [-7.3,+3.3]; FG3M +0.6e-4 (side 55.2->55.5%).
  (b) model-only thr .08: FD all +0.53% n=24,086 [-1.4,+2.6] vs base -0.12% n=23,340; FD close5 +0.42% n=22,684 vs -0.11%;
      Ontario all +1.97% n=40,924 [+0.6,+3.4] vs +1.87% n=39,235 [+0.4,+3.4]; Ontario close5 +1.57% n=25,977 [-0.1,+3.2] vs +1.51%;
      DK all +0.70% vs +0.64%; MGM all -0.18% vs -0.42%; BR all -0.60% vs -0.35%. Every pooled difference (+0.1 to +0.6pp) is inside
      the intervals. By market thr .08 'all': AST FD +5.93% n=2,364 [+2.1,+9.9] vs +2.26% n=2,324; AST Ontario +4.89% n=4,657
      [+2.3,+7.5] vs +5.37% n=4,480; REB FD -1.25% n=3,120 vs -1.34%; REB Ontario +2.71% n=4,901 [+0.2,+5.1] vs +2.26%;
      PTS FD +0.81% n=4,387 vs +0.04%; PTS Ontario +1.28% vs +1.49%; PRA FD +0.68% vs -0.85%; FG3M Ontario +8.00% n=2,853
      [+4.0,+12.0] vs +7.06%; PR FD -2.15% vs -1.19%.
  (c) anchored rule: close5 +4.45% n=19,871 vs +4.40%; all +5.28% n=24,624 vs +5.19%; clean120 n=34 -4.19% both. Unchanged.
  (Before the tier fix, `eval_trk.py`'s holdout numbers were wrong: 'all'/'clean120' used mins_before_tip >= 0, nearly empty in
  2025-26. Fixed; numbers above are after the fix.)
  INTERPRETATION: the only effect that survives both windows is assists — potential assists / passes / time of possession make the
  AST projection measurably sharper (log-loss -0.002, miss -0.007, right side +0.6pp in both seasons). Rebound chances cut REB miss
  slightly in both windows but do not move REB log-loss on holdout. Shot-zone mix does nothing for PTS or FG3M and PTS/PR log-loss is
  slightly WORSE on holdout (tracking NaN for ~80% of training rows adds noise to the big markets). Pooled model-only ROI differences
  are +0.1 to +0.6pp, inside every interval; FanDuel model-only is still -3.2% on dev. The selection rule is unaffected.
  BACKFILL ESTIMATE (2016-17..2022-23): ~1,440 game dates x 4 endpoints ~ 5,760 calls; ~6 h unthrottled, ~14 h with stats.nba.com
  throttling at the observed ~400 calls/h; 0 Odds API credits. Tracking would then be present for ~95% of training rows instead of
  ~20%, which is the main reason to expect the AST (and maybe REB) gain to grow and the PTS noise to shrink. Worth doing as an
  accuracy improvement for AST/REB, not as an edge.
  DAILY REFRESH (described, NOT wired in): after the day's games are final (~1-3 AM ET) and after the current-season Base logs are
  refreshed (see P2-8 cached() fix), run `python src/fetch_tracking.py --refresh-from <yesterday>` = 4 calls per game date
  (Passing, Rebounding, Possessions, ShotChartDetail), ~10 s, then `--build` and `python src/tracking_features.py`. The production
  model (`src/features.py`, `src/run_bp.py`, `live_odds.py`, daily skill, ledger) is unchanged.
  VERDICT: NO_EDGE_FOUND. Tracking features are a small, real accuracy gain for AST (replicated) and REB (miss only), nothing for
  PTS/FG3M, and no rule became profitable on data it was not chosen on. Backfill to 2016-17 is cheap (~6-14 h, no credits) and
  justified only as a model-quality step before any future re-validation.

## P2-11 tracking variants, full history (2026-10-29; `src/tracking_features2.py`, `scripts/run_trk2.py`, `scripts/eval_trk2.py`)
- Backfill audit (`scripts/check_tracking_complete.py`): 2,075 game dates 2016-10-25..2026-10-25, all 6 per-date kinds have a file for
  every date (0 missing, 0 FAILED/MISSING lines in `logs/fetch_tracking_backfill.log`). Dates with games but 0 tracking rows:
  2022-01-06 (4 games) and 2024-03-09 (7 games) for passing/rebounding/possessions/drives; re-fetched once (8 calls) -> still 0,
  so stats.nba.com has no tracking for those 11 games. catchshoot/pullup complete. Shots: per-player FGA from shot charts equals
  box-score FGA in 100% of player-games for every season 2016-17..2025-26 (2026-27 99.8%). `build_tables()` extended to join
  drives/catchshoot/pullup -> `tracking_player_games.parquet` 274,635 player-dates x 56; `tracking_shots.parquet` 2,268,002 shots.
- Feature file `data/processed/player_games_trk2.parquet` (274,634 x 572, 182 new cols, built on the ROLL-FIXED production file;
  `player_games_trk.parquet` is stale and unused). Groups: core (P2-10 passing/rebounding/possessions), zone, opp4, scor
  (drives / catch-and-shoot / pull-up), cv (40-game conversion rates, expected = volume x long-run rate, luck = actual - expected),
  pm (per-minute), vac (teammate tracking volume listed Out/Doubtful; NaN before 2020-21), mx (own volume x opponent allowed /
  league). Coverage of trk_touches_m10: 98.3% (2016-17), 99.5%+ after. Sanity: trk_cv_exp_ast_m10 corr with AST 0.718 vs
  AST_m10 0.714; luck columns mean ~0.
- PROTOCOL (written before any result): screening folds = one mid-season fold per early-dev season: 2021-01-01..03-01,
  2022-02-01..04-01, 2022-12-01..2023-02-01, 2023-12-01..2024-02-01 (~73k FanDuel rows). 12 variants on the screening folds:
  A (fixed baseline), B (A+core+zone+opp4), C (B+scor), S (per-market subsets of C: AST<-passing/possessions/drives, REB<-rebounding,
  PTS<-zones/possessions/scoring types, FG3M<-3pt zones/cs/pu, combos = unions, STL/BLK none), CV (C+cv), VAC (C+vac), MX (C+mx),
  PM (C+pm), DO (tracking in dispersion model only), MO (mean only), ALL (C+cv+vac+mx+pm), F (C + union of the add-on groups whose
  screening ALL paired log-loss vs C is >0 with 95% CI excluding 0; if none, F = best of C/S/DO/MO and costs no extra run).
  Primary metric = paired log-loss vs actual outcome on FanDuel rows (book 10, played, deduplicated per player-game-market, pushes
  dropped), per market and pooled, bootstrap by game date; secondary = |miss| and right side of the FanDuel line. ROI reported only.
  Selection: final = variant with the largest pooled log-loss gain vs A on the screening folds among those whose PTS and PRA
  intervals do not exclude 0 on the negative side. Then full 15-fold early dev for A, B and the final (plus C if final != C);
  promote only if the full-early-dev pooled gain vs A has CI > 0 and PTS/PRA are not significantly worse. Then dev2 once (A, B,
  final), then the reused holdout once (A, B, final). Model identical to production except the column lists. LGB 8 threads
  total (two 4-thread jobs in parallel).
- SCREENING RESULTS (4 folds, 73,156 paired FanDuel rows, 226 game dates; `logs/trk2/eval_screen_all_vsA.log` + pairwise logs).
  Pooled paired log-loss gain vs A (e-4, [95% CI]): B +8.7 [+4.1,+13.4]; C +8.9 [+4.6,+13.4]; S +7.4 [+3.2,+11.6]; CV +10.1
  [+5.5,+14.7]; VAC +10.3 [+5.9,+15.0]; MX +10.1 [+5.4,+14.8]; PM +9.6; DO -0.1 (dispersion-only = nothing); MO +8.8 (= C: the
  whole gain is in the mean model); ALL +11.3 [+6.3,+16.3]. Per market, ALL vs A: AST +40.9 [+19.8,+62.9] (side 53.3->54.3%),
  REB +11.1 [+3.3,+18.7], PA +25.0 [+8.7,+40.8], PRA +9.4 [-4.2,+23.0], PTS -0.1 [-5.4,+5.5], FG3M +4.0 [-3.9,+11.7]. B vs A:
  AST +35.8, REB +12.3, PTS -2.1 (full history roughly doubles the P2-10 AST gain). Add-on groups vs C: CV +1.2 [-1.2,+3.5],
  VAC +1.5 [-0.5,+3.4], MX +1.2 [-0.8,+3.3], PM +0.7 [-1.0,+2.5] -> none individually significant, so by protocol F = C (no extra
  run). ALL vs C: +2.4 [-0.2,+5.1] pooled (miss +2.2e-3 [+0.7,+3.7]; AST +8.1 [+0.4,+16.2]). Scoring types (C vs B) add nothing to
  PTS (-1.0 vs -2.1) or FG3M (+3.0 vs +1.1). Per-market subsets (S) are not better than giving every market everything.
  SELECTION (pre-registered rule): final = ALL (largest pooled gain vs A; PTS/PRA not worse). Variant count: 11 distinct model
  fits on the screening folds (A B C S CV VAC MX PM DO MO ALL); F coincided with C. Finalists for the full 15-fold early dev:
  A, B, C, ALL.
- FULL EARLY DEV (15 folds 2021-01-01..2024-10-01, 204,681 paired FanDuel rows, 744 game dates; `logs/trk2/eval_early_{B,C,ALL}.log`,
  `eval_early_ALL_vs_C.log`). Pooled paired log-loss gain vs A (e-4): B +7.8 [+5.3,+10.4]; C +7.8 [+5.4,+10.5]; ALL +11.3 [+8.8,+14.1].
  ALL vs A by market: AST +42.3 [+30.3,+54.3] (ll 0.7017->0.6974, miss 1.820->1.807, side 53.7->54.5%), REB +8.7 [+3.8,+13.4]
  (side 53.3->53.8%), PTS +3.2 [-0.2,+6.6] (miss 5.229->5.226), FG3M +5.0 [+0.3,+10.0], PRA +6.3 [-1.5,+13.4] (miss 7.016->7.006),
  PA +19.3 [+10.3,+28.2], PR +3.2 [-3.0,+9.6], RA +8.6 [-3.1,+20.3], STL +0.3, BLK +9.3 [-6.2,+23.2].
  ALL vs C: pooled +3.5 [+1.9,+5.1]; PTS +4.6 [+1.5,+7.6]; AST +5.5 [+0.7,+10.0]; FG3M +3.9 [+0.2,+7.8]; PA +10.2 [+3.7,+16.6];
  REB/PRA/PR/RA flat. So the add-on groups (conversion/luck, vacated tracking, matchups, per-minute) together are worth a real
  +3.5e-4 even though none was significant alone, and they are what turns PTS from flat to slightly positive.
  B vs A: AST +37.6, REB +8.7, PTS -0.1, PRA -1.5. C vs A: PTS -1.4, PRA +2.6 (scoring types alone do nothing for points).
  EARLY-DEV DECISION: final = ALL; passes the gate (pooled CI > 0, PTS/PRA not worse). Next: dev2 once (A, B, ALL), holdout once.
- EARLY-DEV ROI, report only (`logs/trk2/roi_early_ALL_vs_A.log`; "tracking" = ALL, "baseline" = A; 2,655,377 aligned priced rows,
  2021-01-06..2024-06-17; these seasons were already known to be softer than 2024-25). Model-only thr .08 'all': FanDuel ALL +2.24%
  n=43,030 [+1.1,+3.4] vs A +0.85% n=41,813 [-0.3,+2.1]; Ontario ALL +3.98% n=85,713 [+3.1,+4.9] vs A +2.94% n=82,511 [+2.0,+3.8].
  By market (ALL vs A): AST FD +5.18% n=7,143 [+2.9,+7.4] vs -0.08% n=6,686; AST Ontario +6.67% n=14,543 vs +3.35% n=12,600;
  REB FD +5.39% vs +4.71%; REB Ontario +7.47% vs +6.18%; PTS FD -0.12% vs +0.03%; PTS Ontario +2.28% vs +1.96%; PRA FD -2.56%
  n=2,208 vs -3.51%; PRA Ontario +0.55% vs +0.47%; FG3M FD +7.16% vs +6.01%; PA Ontario +1.22% vs -0.48%; RA Ontario +2.71% vs +0.59%.
- DEV2 (2024-25, evaluated ONCE for A, B, ALL; 99,553 paired FanDuel rows, 213 game dates; `logs/trk2/eval_dev2.log`,
  `eval_dev2_ALL_vs_B.log`, `roi_dev2_{ALL,B}_vs_A.log`). Paired log-loss gain vs A (e-4): B pooled +5.9 [+2.3,+9.6]; PTS +4.4
  [-0.4,+8.8]; REB +7.8 [-0.1,+15.4]; AST +6.5 [-17.2,+30.3]; FG3M +6.2 [+0.1,+12.2]; PRA +1.9 [-5.2,+9.1]; RA +12.9 [+1.1,+24.6].
  ALL pooled +5.2 [+1.1,+9.4]; PTS +5.5 [+0.5,+10.3]; REB +1.6 [-7.4,+10.7]; AST +6.9 [-18.5,+32.6] (miss 1.901->1.893, side
  53.3->53.0%); FG3M +4.7 [-3.3,+12.1]; PRA +3.9 [-3.8,+11.4]; PA +5.5; PR +5.1; RA +10.8 [-1.2,+22.6]. ALL vs B: pooled -0.7
  [-3.1,+1.8]; REB -6.1 [-10.8,-1.3] (ALL worse); PTS +1.1; AST +0.4 [-10.0,+10.0]; PRA +2.1. So on 2024-25 the add-on groups
  bought nothing over B and cost rebounds. The AST gain is +6..7e-4 with a +-25e-4 interval (n=8,669): underpowered, see diagnosis.
  ROI report only, model-only thr .08 'all': FanDuel A -3.68% n=15,708, B -2.91% n=15,980, ALL -2.77% n=15,841 [-4.9,-0.6];
  Ontario A +0.66% n=34,224, B +1.06% n=34,574, ALL +0.89% n=34,479 [-0.5,+2.3]. AST FD A +0.58% n=1,530 / B +2.03% / ALL +2.79%
  n=1,694; AST Ontario +1.91% / +3.13% / +3.73% n=4,049 [+0.9,+6.6]; REB FD +0.10/+1.74/+1.69; REB Ont +2.37/+3.05/+2.18;
  PTS FD -4.09/-4.53/-3.87; PTS Ont +0.07/+0.95/+0.07; PRA FD -5.92/-5.80/-7.63; PRA Ont -1.17/-2.49/-1.98; FG3M FD +0.67/+0.69/+1.37.
  Anchored chosen rule (p_anc_fd .03 clean120): A +7.19% n=3,922 vs ALL +7.51% n=3,950 (the rule is market-anchored, so the
  projection change barely moves it). Mean bias on FD rows: A -0.069, ALL -0.065; REB A -0.039, ALL -0.052.
- REUSED HOLDOUT (2025-10-21..2026-10-23; this window was first opened 2026-10-24 and reused in P2-7, P2-10 and here, so it is NOT
  a clean holdout; evaluated ONCE for A, B, ALL after every dev decision; 113,250 paired FanDuel rows, 215 game dates; 1,837,691
  aligned priced rows; `logs/trk2/eval_holdout.log`, `eval_holdout_ALL_vs_B.log`, `roi_holdout_{ALL,B}_vs_A.log`).
  Paired log-loss gain vs A (e-4): B pooled -0.7 [-4.2,+2.8] (miss +4.3e-3 [+2.4,+6.1] better, side +2.0e-3); PTS -3.6 [-8.4,+1.0];
  REB -12.4 [-23.4,-1.2] (WORSE, although REB |miss| is 6.0e-3 BETTER); AST +28.2 [+9.4,+46.7] (side 53.2->54.2%); FG3M +0.4;
  PRA +0.3 [-6.0,+6.9]; PR -7.1 [-12.1,-1.7].
  ALL pooled +0.1 [-3.9,+4.2] (miss +4.5e-3 [+2.3,+6.8], side +2.3e-3 [+0.4,+4.2]); PTS -3.5 [-8.7,+1.8]; REB -16.6 [-28.5,-4.6]
  (miss +5.7e-3 better); AST +33.2 [+14.0,+51.4] (ll 0.6965->0.6932, miss 1.762->1.751, side 53.2->54.5%); FG3M -1.4; PRA +2.2
  [-4.9,+9.8]; PA +3.7; PR -5.9 [-11.3,-0.5]; RA +4.9 [-7.6,+16.8]; STL +2.5; BLK -17.5 [-41.9,+8.1].
  ALL vs B: pooled +0.8 [-1.3,+3.0]; REB -4.2 [-8.9,+0.6]; AST +5.0 [-3.0,+12.5]; PTS +0.2; PRA +1.9.
  ROI report only, model-only thr .08 'all': FanDuel A -0.33% n=23,289 [-2.2,+1.7], B +0.46% n=24,522, ALL +0.61% n=24,609
  [-1.3,+2.6]; Ontario A +1.62% n=39,332 [+0.2,+3.1], B +2.02% n=41,405, ALL +2.04% n=41,583 [+0.7,+3.4]. AST FD A +1.79% n=2,320
  [-1.9,+5.5], B +7.19% n=2,486 [+3.4,+11.1], ALL +8.30% n=2,491 [+4.5,+12.0] (hit 56.1% vs break-even 51.9%); AST Ontario
  +4.70% n=4,477 / +5.56% / +5.51% n=4,999 [+2.9,+8.1]. REB FD -1.88/-1.28/-0.05; REB Ont +2.55/+2.88/+2.91; PRA FD -0.68/-0.49/0.00
  n=3,142; PRA Ont +0.02/+0.36/0.00; PTS FD +0.35/-0.17/-0.32; PTS Ont +1.05/+1.92/+0.83; FG3M FD +4.35/+5.91/+4.22; FG3M Ont
  +7.57/+6.89/+8.03. Anchored chosen rule: close5 A +4.50% n=19,751 vs ALL +4.52% n=19,956; 'all' A +5.32% n=24,475 vs ALL +5.38%.
  Mean bias on holdout FD rows: A -0.005 vs ALL -0.065; REB A +0.012 vs ALL -0.121; PTS -0.053 vs -0.082; AST +0.014 vs -0.021.
- DIAGNOSIS of the holdout REB loss (`scripts/trk_drift.py` -> `logs/trk2/drift.log`): stats.nba.com tracking definitions are not
  stationary. League REB per REB_CHANCE by season: 0.505 (2016-17), 0.555 (2018-19), 0.570-0.581 (2019-20..2023-24), 0.592 (2024-25),
  0.680 (2025-26), 0.524 (2026-27 first 967 player-dates); chances per minute 0.356 -> 0.315 (2023-24) -> 0.308 -> 0.266 (2025-26)
  -> 0.354 (2026-27). Also drives/min 0.201 (2023-24) -> 0.290 (2024-25) -> 0.274 -> 0.266; touches/min 1.66 -> 1.56 (2024-25 on);
  time-of-possession/min 0.086 -> 0.146 (2024-25) -> 0.100 -> 0.108; AST per potential AST 0.58 (2023-24) -> 0.54/0.56/0.52;
  potential AST/min stable 0.18-0.21; catch-and-shoot and pull-up FGA/min stable. The tracking vendor/definition changed for 2024-25
  (Hawk-Eye era) and the rebound-chance definition changed again for 2025-26 and again for 2026-27. Consequence: a model trained on
  pre-2025-26 chance rates sees fewer chances per minute in 2025-26 and under-projects rebounds (holdout REB mean mu: A 4.791 vs
  ALL 4.684, var/mu equal 1.39, so it is a mean shift not a dispersion problem). By fold cutoff the under-projection is in the
  2025-12-01 (4.774 vs 4.563) and 2026-02-01 (4.786 vs 4.623) folds and gone by 2026-04-01 (4.839 vs 4.921) once a season of
  new-definition rows is in the training set. That explains the paradox of better REB |miss| with worse REB log-loss: the sharper
  point estimate is paid for with a biased P(over). The same mechanism is the most likely reason the AST gain shrank to +7e-4 on
  dev2 (2024-25 is the first season under the new touch/drive/time-of-possession definitions; models trained mostly on the old
  scale were mis-scaled in exactly that season) and recovered to +33e-4 on the holdout once 2024-25 was in training. Any production
  use must normalise the drifting stats per season (e.g. z-score or divide by the trailing-season league rate, as a ratio to the
  league value over the same window) and must be re-validated.
- P2-11 DECISION (2026-10-29): NO PROMOTION. Reasons: (1) on data the variants were not selected on, the pooled primary metric is
  flat for both B (-0.7e-4 holdout) and ALL (+0.1e-4 holdout; +5.2e-4 dev2 but not better than B there); (2) the gain is a real,
  replicated accuracy improvement in assists only (AST paired log-loss vs A: early +42e-4, dev2 +7e-4 underpowered, holdout +33e-4
  [+14,+51]; AST FD model-only ROI +8.3% n=2,491 holdout vs +1.8% baseline) with nothing for points (PTS +3.2 / +5.5 / -3.5e-4,
  only dev2 excludes 0) and nothing for PRA (+6.3 / +3.9 / +2.2e-4, every interval includes 0; PRA FD model-only ROI still <= 0 in
  every window); (3) rebounds are significantly WORSE on the holdout for both B and ALL because of the un-normalised chance
  definition shift, and ALL is worse than B on REB in both confirmation windows (dev2 -6.1e-4 significant, holdout -4.2e-4);
  (4) the chosen anchored rule does not move (holdout +4.50% vs +4.52%). Between B and ALL: ALL vs B is +3.5e-4 on early dev
  (the data it was picked on), -0.7e-4 on dev2 and +0.8e-4 on the holdout, i.e. the add-on groups do not replicate; B is the
  simpler set and would be the candidate if anything is promoted later. The holdout was reused, so even the AST result is a
  confirmation on a window that had already been looked at three times. Variant count for the whole phase: 11 distinct model
  fits on the screening folds (A B C S CV VAC MX PM DO MO ALL; F coincided with C), 4 on full early dev (A B C ALL), 3 on dev2
  (A B ALL), 3 on the reused holdout (A B ALL); nothing changed in production.
- PROMOTION STEPS (recorded as required; NOT wired in; only to be executed after the re-validation below passes).
  Candidate = B (A + core passing/rebounding/possessions + zone + opp4 groups) with per-season normalisation, or an AST-only
  variant (tracking columns fed to the AST, PA and RA mean models only).
  1. Feature build: `.venv/bin/python -u src/fetch_tracking.py --build` (joins the cached per-date JSON into
     `data/processed/tracking_player_games.parquet` and `tracking_shots.parquet`), then `.venv/bin/python -u src/tracking_features2.py`
     (writes `player_games_trk2.parquet`; to go live, fold its column builders into `src/features.py` after `lagged_roll` so the
     production file gains the `trk_*`/`opp_trk_*`/`vac_trk_*` columns, and add `^trk_` to `FEATURE_PAT` in `src/common.py`).
     Before that, add the normalisation step: divide REB_CHANCES, DRIVES, TOUCHES, TIME_OF_POSS (and anything derived from them)
     by the league per-minute rate over the trailing 60 game dates, computed from rows strictly before the game date.
  2. Daily refresh, per game date, after games are final (about 1-3 AM ET) and after the current-season Base box-score refresh,
     before `features.py` runs: `.venv/bin/python -u src/fetch_tracking.py --refresh-from <yesterday>` = 7 stats.nba.com calls per
     game date (passing, rebounding, possessions, drives, catchshoot, pullup, shots), about 20 s, cached under `data/raw/nba/tracking/`;
     then `--build`. If a date returns 0 rows with games (as 2022-01-06 and 2024-03-09 did) retry once the next day and otherwise
     leave it; features tolerate NaN.
  3. Re-validation before promotion: rerun `scripts/run_trk2.py B early` and `dev2` with the normalised columns, require pooled
     paired log-loss vs A > 0 with CI excluding 0 on early dev, REB not significantly worse on dev2, and AST gain >= +15e-4 on
     dev2; then one look at a NEW holdout (2026-27 games after 2026-10-29) rather than the reused window.
  4. Only then: retrain the production model with the new feature list, re-run `src/run_bp.py dev|dev2|holdout trk` for the ledger
     comparison, and record the swap date in the ledger so pre/post live performance can be separated.

## Status (end of session 2026-10-29, P2-11 complete)
- Production unchanged: `data/processed/player_games.parquet` (2026-10-27 roll-fixed build), `src/live_odds.py`, `src/ledger.py`,
  ledger, daily skill, production feature set all untouched this session (verified by mtime). Odds API credits: 241, unused.
- Tracking cache complete 2016-17..2026-27 (6 kinds per date + shots); `tracking_player_games.parquet` / `tracking_shots.parquet`
  built; experimental feature file `player_games_trk2.parquet`; all P2-11 artefacts in `results/bp_dist_*_t2*.parquet`, `logs/trk2/`.
- P2-11 verdict: NO_EDGE_FOUND; no promotion. If revisited: normalise drifting tracking stats per season, test B (or AST-only) on
  early dev + dev2, then a NEW holdout (games after 2026-10-29). Promotion steps recorded in the P2-11 section above.

## P2-12 Assists model switched to tracking features + daily refresh (2026-10-29; `src/tracking_live.py`, `scripts/run_astn.py`, `scripts/daily_refresh.sh`)
- User decision after P2-11: upgrade the existing AST model in place (the model already fits one mean+dispersion model per market)
  rather than log a second shadow source; keep the old AST model running alongside for a paired forward comparison.
- Features (`src/tracking_live.py`): Fable's group B (90 columns: passing/rebounding/possessions m5/m10/m20 + ratios, shot-zone
  volumes, opp-allowed pot_ast/reb_ch/fga_rim/paint_t), with every stats.nba.com tracking count (the 15 CORE stats) rescaled by
  the league's SEASON-TO-DATE per-minute rate (dates <= the game's date, same season) x the all-history rate. Shot-chart zones
  not rescaled. Check: league per-minute rate by season 2022-23..2026-27, raw vs rescaled: REB_CHANCES .312/.315/.308/.266/.354 ->
  .317/.314/.323/.306/.318; TOUCHES 1.66/1.66/1.56/1.55/1.57 -> 1.65/1.66/1.66/1.65/1.67; TIME_OF_POSS .087/.086/.146/.100/.108 ->
  .094/.095/.094/.093/.093. Corr with Fable's raw columns: pot_ast_m10 0.999, opp_pot_ast_15 0.939.
- GATE (written before any result): switch if (1) early-dev AST paired log-loss gain vs A has a 95% CI > 0 and (2) dev2 AST gain
  vs A > 0 and >= raw B's +6.45e-4. Holdout not looked at again (already reused 4 times).
- RESULTS (`scripts/run_astn.py`, AST only, same folds/model as run_trk2; `logs/p212/eval_{early,dev2}.log`; 2 new fits):
  early dev (n=29,527 FanDuel AST rows, 744 dates): BN +34.2e-4 [+22.1,+45.5] (raw B +37.6, ALL +42.3); miss 1.820->1.808;
  side +0.86pp [+0.44,+1.28]. dev2 (n=8,669, 213 dates): BN +22.8e-4 [-1.1,+47.2] (raw B +6.5 [-17.2,+30.3], ALL +6.9);
  miss +18.3e-3 [+11.2,+25.1] (1.900->1.882); side +0.39pp [-0.5,+1.2]. Gate passed; per-season rescaling recovers most of the
  2024-25 shrink (consistent with the P2-11 drift diagnosis) and clears Fable's +15e-4 re-validation bar on dev2.
- PRODUCTION CHANGE (from game date 2026-10-29): `live_odds.py` adds the BN columns to the AST mean+dispersion models only
  (`models.proj_dist(extra={"AST": ...})`; other markets exclude trk_* explicitly); `--no-trk` falls back. The old AST model is
  fitted alongside and every FanDuel AST line is appended with both mu/sigma to `data/ledger/ast_compare.csv`; `ledger.py settle`
  grades it, `ledger.py report` prints paired log-loss (normal CDF, as used live) with a by-date bootstrap, side, |miss|, paper
  ROI at edge >= .08, plus AST model-only ledger picks before/after `TRK_SWITCH`. Expect ~+30e-4 in log-loss, which needs
  several hundred game-dates' worth of lines to resolve; treat early numbers as noise.
- FROZEN-LOG FIX (P2-8 TODO): `fetch_nba.py --refresh <season>` re-downloads (atomic replace, refuses a shorter file);
  `parse_injury.py --since D` incremental (seconds instead of ~40 min); `fetch_injury.py` and `injury_status.py` read game dates /
  games from the raw logs so they run before `features.py`. `scripts/daily_refresh.sh` = box scores -> injury PDFs (3 days) ->
  parse -> statuses -> tracking (7 calls/new date) -> build -> features; ends `REFRESH OK`. First run 2026-10-29 18:24-18:31:
  box scores / tracking / injury statuses through 2026-10-28 (the model had been missing 10-26..10-28). Daily skill updated.
