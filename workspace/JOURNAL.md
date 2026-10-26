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
