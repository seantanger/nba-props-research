# NBA player-prop edge research — report

Written 2026-10-24; §10 (phase 2, injury-aware workflow) added 2026-10-25. All numbers below come from files in `results/` and logs in `logs/`; the running record of every experiment, including the dead ends, is `JOURNAL.md`.

## 1. Headline

**No edge survived the pre-registered holdout test.** The one rule frozen on development data (FanDuel-anchored pricing with model dispersion, bet other books at ≥3 points of edge, only when both lines were verifiably live) produced **35 bets on the 2025-26 and early 2026-27 holdout, ROI −1.78%, −0.6 units, 95% bootstrap interval [−33.1%, +28.0%], hit rate 51.4% against a breakeven of 52.1%** (`results/headline_holdout.json`, `logs/eval_holdout.log`). On development data the same rule had made +6.10% on 6,424 bets with the interval [+2.8%, +8.9%].

The holdout test is also close to uninformative, and the reason matters more than the number: the price archive changed how it timestamps lines in 2025-26, so the availability filter that defines the rule almost never fires (details in §7). A post-hoc variant that uses the availability filter appropriate to the new timestamp regime — the frozen rule otherwise unchanged — made **+4.51% on 19,491 bets, interval [+2.7%, +6.4%]**, and +2.64% [+0.5%, +5.1%] on 13,854 bets at regulated sportsbooks only (excluding the Polymarket, Novig and ProphetX exchanges). That variant was defined after I had seen the holdout, so it is reported as "what came closest" and not as the headline. It needs its own untouched holdout (the rest of the 2026-27 season) before anyone should stake money on it.

| | Dev (2020-21 → 2024-25) | Secondary holdout (Mar–Jun 2023, Kambi prices) | **Primary holdout (2025-26 → 2026-10-23)** | Post-hoc tier on primary holdout (not pre-registered) |
|---|---|---|---|---|
| Bets | 6,424 | 38 | **35** | 19,491 (13,854 at sportsbooks) |
| ROI | +6.10% | +10.9% | **−1.78%** | +4.51% (+2.64% sportsbooks) |
| 95% CI (cluster bootstrap by date) | [+2.8%, +8.9%] | [−17.4%, +35.2%] | **[−33.1%, +28.0%]** | [+2.7%, +6.4%] ([+0.5%, +5.1%]) |
| Units (flat 1) | +392 | +4.2 | **−0.6** | +879 (+365) |
| Hit rate vs breakeven | — | 60.5% vs 54.1% | **51.4% vs 52.1%** | 54.1% vs 52.0% |
| CLV proxy vs consensus close | — | +1.38 pts | **+2.80 pts, positive on 60%** | +4.23 pts, positive on 82% |
| Max drawdown | — | — | **−6.4 units** | −99.6 units over 205 betting days |
| Share of profit from top-5 days | — | — | n/a | 24% |

Variants examined on development data before the rule was frozen: roughly 150 (three model families × thresholds × snapshots in the Kambi phase; gap sizes × references × tiers × sides in the gap-rule phase; four probability columns × five thresholds × four availability tiers, twice, in the final phase). One post-hoc variant was examined on the holdout.

## 2. What "an honest test" meant here

- **Prices.** Every graded bet uses the stored American price of one specific sportsbook at that book's last recorded state before tip-off, with the vig included. Pushes return the stake; voids (player did not play) are excluded from both the numerator and the denominator.
- **Decision time.** For the chosen rule, the decision time is the moment FanDuel's line reached its final pre-game state. A bet is only allowed at another book if that book's own line was updated *after* FanDuel's final update and before tip-off, and FanDuel's final update was within 120 minutes of tip-off. This is the "clean120" availability tier. It is the only tier in which I can argue that both the signal (FanDuel's final line) and the bettable price (the other book's line) were on the board at the same time.
- **Features.** Every model feature is built from box scores dated strictly before the game (`shift(1)` on every rolling statistic) plus static attributes. Models are retrained walk-forward on logs strictly before each bi-monthly cutoff and only predict games after that cutoff.
- **Holdout.** Before any BettingPros row was evaluated, I wrote in `JOURNAL.md` that seasons 2025-26 and 2026-27 (through 2026-10-23) would be the primary holdout, and that the Kambi window 2023-03-13 to 2023-06-12 (set aside on day one) would be a secondary holdout. The rule was frozen in `results/chosen_rule.json` at 12:00 EDT on 2026-10-24 and run once on each holdout.
- **Staking.** Flat one unit per bet, one bet per player/game/market (largest edge). No Kelly, so no single bet can carry the result.

## 3. Data

| Source | Coverage | Used for |
|---|---|---|
| nba_api box scores | 2016-17 → 2026-10-23, 274k player-games | Features, grading |
| BettingPros prop archive (`api.bettingpros.com`) | 2021-01 → 2026-10-23, 7,809 events, 12.9M book-side rows, 11 US sportsbooks (FanDuel, DraftKings, BetMGM, Caesars, bet365, BetRivers, PartyCasino, theScore, SugarHouse, Fanatics, HardRock) | Historical prices. Stores only the opening and the final state of each book's line, with an `updated` timestamp for the final state. No Pinnacle. |
| Kambi (Polish book archive) | 2022-11 → 2023-05 | First two days of work; a second, independent price source for the secondary holdout |
| The Odds API (free tier) | Live only | Tonight's bets. 443 credits remained after phase-1 testing, 286 after the phase-2 live test (§10.5); nothing historical was pulled from it. |

Markets: points, rebounds, assists, threes made, points+rebounds+assists, steals, blocks, and combined markets where offered (`BP_MARKETS` in `src/common.py`).

## 4. Approaches tried (dev data only, 2020-21 → 2024-25 minus the Kambi window)

Roughly 150 rule variants were examined across the experiments below. One was chosen.

| # | Approach | Result on dev (out-of-sample, walk-forward) | Verdict |
|---|---|---|---|
| 1 | LightGBM negative-binomial projection (mean + dispersion models), bet when the model's tail probability beats the book's breakeven | Log-loss 0.6988 vs book-implied 0.6854. As a standalone bettor: +1.5% on all rows at threshold 0.04 (n=291k), but −0.2% / −0.6% in the verifiable availability tiers (n≈50k/39k). | Dead. The model alone does not beat these books. |
| 2 | Empirical-Bayes shrinkage of recent rates, and a direct over/under classifier (Kambi pipeline) | Both worse than the closing market on log-loss. | Dead. |
| 3 | TabPFN (tabular foundation model, v6.4) as the projection model | See §6. | Not better than LightGBM on log-loss; too slow for the full backtest. |
| 4 | Model-free "line gap" rule: bet toward FanDuel when another book's final line is ≥2 units away | +15.6% on all rows (n=45k) — but concentrated where the other book's line is hours older than FanDuel's. Those lines were plausibly suspended on injury news; BettingPros does not store status. Strict tier (book updated after FanDuel): 2024-25 +1.75%, CI [−3.7, +8.5]. | Mostly phantom. The information is in the freshest line, which is the one you cannot bet against. |
| 5 | Consensus-anchored pricing: centre a Normal at the de-vigged consensus line, σ from the LightGBM dispersion model | Log-loss 0.6847 (better than any single book). As a bettor, clean120 tier: +2.6% (n=4,044), CI crosses zero, 2023-24 −4.0%. | Unreliable. |
| 6 | **FanDuel-anchored pricing** (`p_anc_fd`): centre at FanDuel's final line and price, σ from the model, evaluate the tail at the other book's line | Clean120 tier, threshold 0.03: **+6.10% ROI, CI [+2.8, +8.9], n=6,424**, positive in all five dev seasons (+0.2, +6.2, +4.9, +6.4, +7.3%). Edge rises monotonically with threshold (0.02 → +4.7%, 0.06 → +10.3%). | **Chosen.** |
| 7 | Walk-forward logistic stack of the four probability sources (market, model, consensus, FanDuel) | Clean120 thr 0.03: +6.1% (n=9,154). Fitted FanDuel weight ≈0.8, so it is approach 6 with fitted coefficients. | Not better per bet; rejected in favour of the rule without fitted weights. |

Why approach 6 was chosen over the others: no fitted coefficients, positive in every dev season, the middle threshold rather than the best-looking one, and the only availability tier whose prices I can argue were really there. The dev number is still selection-biased (it is the survivor of ~150 variants), which is exactly why the holdout exists.

## 5. The chosen rule (frozen in `results/chosen_rule.json`)

1. Take FanDuel's final pre-game line and price for a player/market; de-vig to get FanDuel's probability of the over.
2. Use the LightGBM dispersion model's σ for that player/market/game. Solve for the Normal centre μ that reproduces FanDuel's over-probability at FanDuel's line.
3. For every other sportsbook whose final line was updated after FanDuel's final update (and FanDuel's within 120 minutes of tip), compute the Normal tail probability at *that* book's line for each side.
4. Bet the side whose probability exceeds the breakeven implied by the book's price by at least 0.03, decimal odds between 1.30 and 3.00. One bet per player/game/market (largest edge). Flat one unit.

This is a "soft book lags sharp book" strategy with a model supplying only the shape of the distribution, which is the part the books price in a templated way and the part that matters when the lines differ.

## 6. TabPFN and the two named models (Jev, Clairic)

**TabPFN (v6.4, tabular foundation model).** Run as a drop-in replacement for the LightGBM projection on the Kambi development slice (3 walk-forward folds × 5 markets, 6,000 most recent player-games per fit, top-30 features). Log-loss 0.7076 versus the market's 0.6891 and LightGBM's 0.6998 on the same 21,569 pairs; as a standalone bettor −3.3% to −3.7% at every threshold with intervals below zero (`logs/tabpfn2.log`). It took 88 minutes on CPU for 3% of the data the final backtest uses. Dead end: worse than the gradient-boosting model it was meant to replace.

**Jev (TypeSafe AI "System One").** No `TYPESAFE_API_KEY` or `OPENROUTER_API_KEY` was present in the environment, so it was not run. Access needed: an OpenRouter key (`POST https://openrouter.ai/api/v1/systemone`, model `typesafe/jev-1.14`) or a TypeSafe key for `https://api.typesafe.ai/v1/systemone`; Python SDK `typesafe-sdk`. How I would use it: for each candidate bet in the clean120 tier, submit a typed decision request ("will player X exceed line L in market M tonight?") with the pre-game feature row and the FanDuel/other-book lines as structured context, and take its calibrated probability as a fifth input to the walk-forward stack (approach 7). It would be evaluated the same way as every other probability column: log-loss against the book, then ROI in the clean tier on dev, with the holdout touched once.

**Clairic (Cloudflare `@cf/cloudflare/clairic`).** No `CLOUDFLARE_ACCOUNT_ID` / API token was present, so it was not run. Access needed: a Cloudflare account with Workers AI enabled; REST call `POST https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/cloudflare/clairic`. How I would use it: the `estimate` answer type returns a distribution over a bounded step grid, which maps directly onto a player's points/rebounds/assists for a game. I would give it the player's last-N game log and the pre-game context and compare its distribution's σ (and its tail probability at the book's line) against the LightGBM dispersion model, first on log-loss, then as a replacement for σ in step 2 of the rule. The `forecast` answer type (up to 64 steps) could be used on the player's game-log series directly.

## 7. Risks and caveats on the result

**Why the holdout produced only 35 bets.** The chosen rule's availability tier ("clean120") requires FanDuel's final line to be stamped 0–120 minutes before tip-off and the other book's final line to be stamped after FanDuel's and before tip-off. In the development seasons the archive's `updated` field behaved like the time of the last line change (2024-25: book stamps spread from 2 to 422 minutes before tip at the 10th–90th percentiles; 90.7% of FanDuel lines stamped within 120 minutes of tip). In 2025-26, about 75% of all rows from every book, FanDuel included, are stamped roughly 1.5 minutes *after* the scheduled tip. Those stamps vary line by line inside each event and book (median 38 distinct timestamps per event-book pair), so they are per-line "last seen" times from scraping right up to tip-off, not a single bulk closing stamp. The remaining 25% of rows, stamped a median 3.7 hours before tip, are lines that *disappeared* before tip — the opposite of what the tier assumes. The tier therefore covered 1.4% of FanDuel-priced rows in 2025-26 against 19.5% in 2024-25, and none of the 2026-27 rows. (`scripts/diag_timestamps.py`; counts only, no outcomes were used.)

The post-hoc tier ("both lines last seen within 5 minutes of tip") is the natural translation of the same idea into the new regime, but it was defined after I saw the headline and the unfiltered "all rows" result (+5.37% on 24,192 bets), so it carries selection bias of unknown size.

Other caveats:

1. **The archive stores only two states per line** (open and final). A "last seen at tip" line could have been listed but suspended; the archive has no status field. Scrape cadence is unknown.
2. **Exchanges inflate the post-hoc number.** Polymarket (+14.5% on 2,917 bets at average odds 2.17) and Novig (+8.2%) are thin markets whose displayed prices may not fill in size. The sportsbook-only figure (+2.64%) is the one to take seriously, and its interval nearly touches zero.
3. **2021-22 timestamps are unreliable** in the dev data (FanDuel stamps a median 44 hours before tip in one tier), so one dev season contributes little.
4. **No injury or lineup data.** The signal is FanDuel's information propagating to slower books. A live implementation needs a fast FanDuel feed; the live script uses The Odds API, which adds latency.
5. **Limits.** Books limit winning prop bettors quickly; 13,854 bets a season at one account is not realistic. Expect the achievable fraction to be a small part of the backtest.
6. **FanDuel dependency.** DraftKings and the consensus were weaker anchors on dev.
7. **Selection bias on dev.** The dev ROI is the survivor of ~150 variants; the holdout numbers are the ones to believe, and the pre-registered one is negative.
8. **Unknown book ids 68 and 75** appear in 2025-26 (101 bets, not in the archive's book list I parsed); they are included in the "sportsbooks" subtotal and are immaterial.
## 8. Tonight's bets (live script output)

Run once at 13:36 EDT on 2026-10-24 (`logs/live_final.log`, `results/live_bets_20261024_1336.csv`), 4.5–8.5 hours before the eight tip-offs, using The Odds API (`regions=us`, five markets: points, rebounds, assists, threes, and points+rebounds+assists; six books returned: FanDuel, DraftKings, BetMGM, BetRivers, Bovada, BetOnline). 4,432 outcome rows, 1,807 selections priced against FanDuel, 17 bets at the frozen rule (≥3 points of edge, odds 1.30–3.00, one per player/game/market). Credits: 35 spent, 408 remaining; every raw response is cached under `data/raw/odds_api/2026-10-24/`.

| player | market | side | line | price | book | p_model | edge | line_fd | mins_to_tip |
|---|---|---|---|---|---|---|---|---|---|
| Jarrett Allen | PTS | under | 17.5 | -300 | bovada | 0.81 | 0.06 | 12.5 | 364 |
| Naz Reid | FG3M | under | 3.5 | -210 | bovada | 0.737 | 0.059 | 2.5 | 274 |
| Brandon Miller | FG3M | under | 4.5 | -280 | bovada | 0.793 | 0.056 | 3.5 | 274 |
| Kyshawn George | PTS | over | 13.5 | 100 | betonlineag | 0.555 | 0.055 | 14.5 | 394 |
| Jamal Murray | FG3M | over | 2.5 | -165 | betmgm | 0.676 | 0.053 | 3.5 | 454 |
| Tyrese Maxey | REB | under | 3.5 | 120 | bovada | 0.506 | 0.052 | 3.5 | 364 |
| Nikola Jokic | AST | over | 9.5 | -140 | betmgm | 0.622 | 0.039 | 10.5 | 454 |
| Dylan Harper | AST | under | 4.5 | -150 | betrivers | 0.636 | 0.036 | 3.5 | 394 |
| Donovan Mitchell | AST | under | 5.5 | -200 | bovada | 0.702 | 0.036 | 4.5 | 364 |
| James Harden | AST | under | 7.5 | -155 | betmgm | 0.642 | 0.035 | 6.5 | 364 |
| Matas Buzelis | FG3M | under | 2.5 | -200 | betmgm | 0.701 | 0.034 | 1.5 | 394 |
| Kevin Durant | FG3M | under | 2.5 | -210 | betmgm | 0.711 | 0.034 | 1.5 | 424 |
| Yaxel Lendeborg | PTS | under | 17.5 | -107 | draftkings | 0.551 | 0.034 | 16.5 | 514 |
| Michael Porter Jr | FG3M | over | 2.5 | -210 | bovada | 0.709 | 0.031 | 3.5 | 364 |
| VJ Edgecombe | AST | under | 2.5 | 165 | bovada | 0.409 | 0.031 | 3.5 | 364 |
| James Harden | REB | under | 3.5 | 165 | bovada | 0.408 | 0.031 | 4.5 | 364 |
| Brandon Miller | PTS | under | 26.5 | -250 | bovada | 0.745 | 0.031 | 22.5 | 274 |

How to read it: `p_model` is the FanDuel-anchored probability of the bet side at the book's line; `edge` is `p_model` minus the breakeven at `price`. **Two caveats.** First, these were computed hours before tip; the rule is meant to run in the last two hours when both FanDuel's and the book's lines are fresh, so the script should be re-run then (the `live` flag in the CSV is only meaningful near tip). Second, the biggest edges here come from large line gaps (Jarrett Allen points 17.5 at Bovada against 12.5 at FanDuel; Brandon Miller points 26.5 against 22.5). On the historical data gaps that large were usually a stale or suspended line, not a bettable one; anyone placing these should confirm the market is open at that price before treating the edge as real. Nothing was staked.

## 9. What to do next

1. **Do not stake money on the post-hoc number yet.** Treat the remainder of the 2026-27 season as a fresh holdout for the "both lines seen at tip" tier. The rule, threshold, odds range and tier are now fixed (`results/chosen_rule.json` plus the tier definition in `scripts/posthoc_holdout.py`); run it once at season's end on the sportsbook-only subset. A sportsbook-only ROI above +2% with the interval clear of zero would be a pass.
2. **Replace the archive with timestamped snapshots.** The whole question of "was the line really there" disappears with a feed that stores every line at every poll: The Odds API historical tier (paid; roughly 10 credits per market-region snapshot) or a self-run poller of the free live endpoint every few minutes through the pre-game window, which the live script already caches. One season of such snapshots lets you grade the strategy at the exact price available at a fixed decision time (e.g., 10 minutes before tip) instead of at a reconstructed final state.
3. **Paper-trade the live script** for a few weeks: run it 15 and 5 minutes before each tip, log the bets it would have placed, and grade them. That measures the latency gap between FanDuel moving and the slow book moving, which the archive cannot show.
4. **Add injury and lineup data** (official NBA injury reports, starting-lineup confirmations). The current features are box-score only; a minutes projection that reacts to the report would make the model's σ and μ meaningful on its own instead of leaning entirely on FanDuel.
5. **If Jev or Clairic credentials are provided**, wire them in as described in §6 and evaluate on log-loss first; neither can be judged until it is shown to beat the market anchor on a held-out season.

---

## 10. Phase 2 (2026-10-25): injuries, the model-only path, and the review step

Three questions were set: (1) can pre-game injury information be obtained honestly; (2) does a projection model that knows the injuries get anywhere near profitable on its own; (3) can a mechanical review step catch the rule's known failure, where another book has the news and FanDuel does not. All choices below were made on development data (2020-21 → 2024-25); the 2025-26 holdout was reused once at the end and is labelled as such. The real test from here is the forward ledger (§10.6).

### 10.1 Injury data (`src/fetch_injury.py`, `src/parse_injury.py`, `src/injury_status.py`)

The NBA publishes an official injury report as a PDF every hour (every 15–30 minutes since December 2025) at a predictable URL. I downloaded 12,218 reports covering every game date from 2020-12-22 to 2026-10-24 (1,539 hourly slots missing, mostly 2020-21 when only three reports a day were published; nothing exists before December 2020), parsed them to 961,467 (report, game, team, player, status, reason) rows, and keyed 94.5% of rows to a game id and 97.0% of players to an nba_api id. The report carries its own timestamp, which is what the features use. The decision-time status for a game is the latest report published at least 60 minutes before tip-off.

How much does that miss? Among 7,526 rotation players (≥20 minutes a game) who played the team's previous game and did not appear in the box score, 64% were listed Out and 3% Doubtful an hour before tip; 16% were Questionable, 1% Probable/Available and **16% were not on the report at all**. Of the third not yet Out/Doubtful, 46% became Out in the final hour and 42% never appeared (late scratches and coach's decisions). Reliability of the decision-time label: Out → played 0.1% (n=21,624), Doubtful → played 1.9% (n=731), Questionable → played 54% (n=6,136), Probable → played 93% (n=2,367). So a box-score-absence feature would leak roughly one third of rotation absences; the report-based feature is honest and sees the other two thirds.

### 10.2 Injury-aware features and the retrained model (`src/injury_features.py`, `logs/eval_inj.log`)

For each player-game the features are built from the team's recent roster (players who appeared in the last five team games that season) and each roster member's 10-game averages as of his previous game: minutes, points, shots, rebounds, assists, threes and usage vacated by players listed Out/Doubtful (`vac_*`), the same for newly-listed players and for Questionables, minutes returning from an absence, the available total (`avail_*`), vacated minutes by position and same position, the number of starters out, the player's own status, his expected share of the available minutes/points (`share_*`), and the opponent's vacated minutes. Rows before December 2020 have these columns missing (not zero). The box-score-absence versions (`bs*`) exist for a leak ceiling but are excluded from every model reported here.

The projection model (LightGBM mean + dispersion, same walk-forward folds as phase 1) was retrained with the 38 new features. Gain-based importance on a PTS model: the player's share of available points is the third most important feature (10% of gain); the injury block carries 14% of gain for points, 5% for rebounds.

**(a) Accuracy against FanDuel's line** (292,919 settled FanDuel selections on dev, one per player/game/market, same rows for both models):

| | Phase-1 model | Injury-aware model | FanDuel's line |
|---|---|---|---|
| Mean absolute miss (all markets) | 3.503 | **3.483** | 3.432 |
| Right side of FanDuel's line (ties excluded) | 52.74% | **52.88%** | — |
| Log-loss of P(over) | 0.7035 | **0.7000** | 0.6867 (de-vigged price) |
| Mean bias (projection − actual), all / points / PRA | −0.138 / −0.256 / −0.303 | **−0.101 / −0.200 / −0.216** | −0.086 / −0.154 / +0.100 |
| Bias when the team has ≥20 vacated minutes (n=140,091) | −0.349 | **−0.089** | −0.087 |

The injury features fix most of the under-projection on games with absences (the Paul Reed / Franz Wagner case from the live run), and every market's log-loss improves, but the model still misses by more than FanDuel's line and its log-loss stays 0.013 worse than the price. The under lean of the live run is structural: on dev the model's P(over) at FanDuel's line is below 0.5 for 64% of selections and 75–85% of its model-only bets are unders, for both models.

**(b) Model-only betting** (the model's own probability against each book's price, edge ≥ threshold, odds 1.3–3.0, flat 1 unit, one bet per player/game/market; "clean" = both the line and FanDuel's line verifiably updated within 120 minutes of tip, the honest tier from phase 1; "all" = every archived line, which includes lines that may have been pulled early):

| Book group | Tier | thr | Phase-1 model | Injury-aware model |
|---|---|---|---|---|
| FanDuel | all | 0.03 | −1.29% (n=135,901) | −0.87% (n=132,098) [−1.6%, −0.2%] |
| FanDuel | all | 0.05 | −0.85% (n=100,842) | −0.53% (n=96,100) [−1.3%, +0.3%] |
| FanDuel | all | 0.08 | −0.96% (n=63,180) | −0.40% (n=57,454) [−1.5%, +0.7%] |
| DK + BetMGM + BetRivers | clean | 0.03 | −0.02% (n=34,566) | +0.07% (n=33,679) [−1.2%, +1.3%] |
| DK + BetMGM + BetRivers | clean | 0.05 | +0.05% (n=25,754) | +0.85% (n=24,512) [−0.6%, +2.2%] |
| DK + BetMGM + BetRivers | clean | 0.08 | +0.40% (n=16,067) | **+1.17% (n=14,608) [−0.7%, +3.0%]** |
| DK + BetMGM + BetRivers | all | 0.08 | +1.29% (n=125,695) | +2.10% (n=116,867) [+1.3%, +2.8%] |
| DraftKings / BetMGM / BetRivers separately | clean | 0.08 | −2.33% / +0.58% / +0.78% | −1.25% (n=5,937) / +1.31% (n=5,859) / +1.31% (n=6,402) |

Thirty book-group × tier × threshold comparisons were run and the injury-aware model is better in all thirty, so the features earned their place in the live script. But the answer to "does the model-only path get near profitable once it knows the injuries" is: **it gets to break-even, not to profit.** At FanDuel it is still negative at every threshold (−0.4% to −0.9%). At the three other Ontario books it reaches +0.9% to +1.2% in the honest tier at thresholds 0.05–0.08, with intervals of ±2 points that include zero, and DraftKings alone stays negative. The "all" tier's +2.1% with a clear interval is the same inflated tier that phase 1 showed cannot be trusted. For the forward ledger the model-only threshold is set to 0.08 (chosen on dev; the only threshold at which every book group other than FanDuel is positive in the honest tier).

**(c) The chosen FanDuel-anchored rule with the new dispersion model**: clean tier, sportsbooks, thr 0.03: +6.54% on n=6,743 [+3.6%, +9.4%] versus +6.07% on n=6,651 with the phase-1 dispersion (same rows, same selection). The rule's profit comes from FanDuel's line, not the model's mean; a slightly better σ helps a little.

### 10.3 Reused holdout (2025-26 → 2026-10-23), injury-aware model

**This holdout is reused.** It was opened once in phase 1 (§7), and the post-hoc `close5` tier was defined after seeing it. Everything in this phase (injury features, the 0.08 model-only threshold, the strict review window) was chosen on dev; the holdout was then evaluated once more, with the script written before it ran (`scripts/eval_inj_holdout.py`, `logs/eval_inj_holdout.log`). Two thresholds × two tiers × five book groups were looked at, so treat every number below as a reused, multiply-looked-at estimate, not a clean test. 1,837,691 aligned selections, 2025-10-21 → 2026-10-23. The BettingPros timestamp change (§7) means the pre-registered `clean120` tier is almost empty here (n=34).

**(a) Accuracy against FanDuel's line** (FanDuel rows, n=113,250): the injury-aware model's average miss is 3.792 versus 3.821 for the phase-1 model and 3.726 for FanDuel's own line; it picks the right side of FanDuel's line 52.6% of the time (phase 1: 52.4%); log-loss 0.7005 versus 0.7056 versus 0.6880 for FanDuel's price. Bias against the line drops from −0.087 to −0.001 (points: −0.133 → −0.049; points+rebounds+assists: −0.143 → +0.050). The same picture as dev: the under-bias is largely gone, FanDuel is still more accurate.

**(b) Model-only ROI** (flat stakes, actual prices):

| Book group | Tier | thr 0.08, injury model | thr 0.08, phase 1 | thr 0.05, injury model |
|---|---|---|---|---|
| FanDuel | all | −0.16% n=23,299 [−2.0, +1.9] | −0.72% n=26,206 | −0.41% n=38,402 |
| FanDuel | close5 | −0.14% n=21,993 | −0.67% n=24,758 | −0.49% n=36,200 |
| Ontario (DK+MGM+BR) | all | **+1.94% n=39,345 [+0.6, +3.4]** | +0.81% n=43,278 | +0.80% n=63,318 [−0.3, +1.9] |
| Ontario (DK+MGM+BR) | close5 | **+1.63% n=24,855 [+0.0, +3.4]** | +1.13% n=27,709 | +0.73% n=40,872 [−0.7, +2.2] |
| DraftKings | close5 | +0.34% n=19,811 | −0.44% | −0.43% |
| BetMGM | close5 | −1.22% n=16,710 | −1.28% | −1.72% |
| BetRivers | close5 | −1.05% n=6,158 | −1.97% | −2.52% [−4.4, −0.6] |

The injury model beats the phase-1 model in every cell, as on dev. At FanDuel it is still at or below zero. At the Ontario books the pooled number at the dev-chosen threshold is +1.6% to +1.9% with the interval touching zero, but the per-book picture does not agree with dev: on dev BetMGM and BetRivers were the positive books (+1.3% each) and DraftKings negative; here DraftKings is the only book above zero and BetMGM and BetRivers lose. The pooled positive is driven by one book and flips which book between the two periods, which is what noise looks like. Conclusion unchanged: the model-only path is break-even, not profitable, and "near profitable" only at non-FanDuel books.

**(c) Chosen FanDuel-anchored rule with the injury model's dispersion:** pre-registered `clean120` n=34 −4.19% (phase 1: n=35 −1.78%); post-hoc `close5` +4.33% n=19,728 [+2.4%, +6.3%] (phase 1: +4.55% n=19,523); `all` +5.22% n=24,443 (phase 1: +5.39%). The new dispersion neither helps nor hurts the rule here; its edge, on this reused holdout, remains the FanDuel-anchored line gap.

### 10.4 The review step as a mechanical rule (`scripts/review_proxy.py`, `logs/review_proxy.log`)

Definition, using only pre-game information: "news" is an official-report status change on the player's team published after FanDuel's last update of the market and no later than the other book's last update (capped at tip): a rotation teammate (≥15 minutes a game) to Out/Doubtful or back from Out, or the player himself downgraded/upgraded. The gap is "explained" when its direction matches the news (the book's line is higher than FanDuel's and a teammate is out; lower and the player himself is Questionable, and so on). Veto = explained.

On dev, chosen rule, clean tier (n=6,424 bets, +6.10%): **kept n=6,238 +6.44% [+3.9%, +9.0%]; vetoed n=186 −5.10% [−21.6%, +18.2%]** (hit 52.2% against a 54.7% breakeven). The vetoed bets lose most when a starter is out (25–35 minutes vacated: −19.6%, n=47) and when the gap is large (≥2.5 points: −30.5%, n=24; 1.5–2.5: −15.2%, n=68); small-gap vetoes (0.5–1.5) were actually +8.7% (n=94). Bets where the news runs *against* the gap — FanDuel has already moved and the other book has not — made **+25.3% on n=141 [+6.3%, +45.1%]**, the best bucket in the whole study. A looser window (news up to 60 minutes before FanDuel's last touch) vetoes 571 bets at +4.0%, i.e. news FanDuel had time to price is not a problem; the strict window is the right definition.

Should explained gaps be bet *with* the book instead? Taking FanDuel's stale line on the news side at FanDuel's last price: n=186 +6.44% [−17.4%, +26.1%]; restricted to a starter out or a gap ≥1.5 points: n=148 +14.1% [−12.1%, +36.9%]. Positive but far from significant, and the archive cannot say whether FanDuel had already suspended that line, so this is unverified. The live script now labels these rows `explained` (veto) and `against` (keep), and the ledger records both so the forward test can settle the question. Variants counted: two windows × one rule; the subsets above are reported, not chosen on.

Re-run with the injury-aware model's probabilities (`logs/review_proxy_inj.out`, dev, clean tier, n=6,514): kept n=6,326 +6.92% [+4.4%, +9.2%]; vetoed n=188 −5.28% [−20.3%, +16.6%]; against n=143 +24.96% [+2.6%, +47.5%]; with-the-book at FanDuel n=188 +7.13%, starter-out or gap ≥1.5 subset n=152 +13.39% [−9.1%, +33.5%]. Same conclusions as with the phase-1 model. The review proxy was not evaluated on the holdout: the clean tier has 34 bets there and the vetoed subset would be empty.

### 10.5 What was built

- `src/live_odds.py`: `--books fanduel,draftkings,betmgm,betrivers` (Ontario set, default; `--books fanduel` for a FanDuel-only slate), `--mode rule|model|both`, `--injury auto` (downloads and parses today's latest official report), `--overrides auto` (reads `data/overrides/<date>.csv`, `team,player,status`), `--model-thr`. Output rows carry `source` (rule / model_fd / model_book), the listed rotation absences (`news`), a `news_flag` (explained / against / same_line / none for rule rows), and empty `claude_verdict` / `claude_reason` columns.
- `src/ledger.py`: `log` (refuses rows without a verdict and reason; de-duplicates), `settle` (nba_api box scores by date, DNP = void), `close` (closing line from a later cached pre-tip snapshot, same book/market/player/side), `report` (model-only vs kept vs vetoed by source, units, ROI, bootstrap interval, average closing-line move). Tested by settling the 17 legacy 2026-10-24 picks from nba_api in a scratch ledger (10–7, +0.12 units; Jarrett Allen under lost, 21 points).
- `.claude/skills/nba-props/SKILL.md`: the daily routine (settle → overrides from news → live run → review with keep/veto and a one-sentence reason → log → report), when to run (2–3 hours before the first tip and again 20–40 minutes before tip, after lineups post), and the credit cost (1 for the events list + 5 per game; ~51 credits for a 10-game slate).
- End-to-end dry run on the cached 2026-10-25 17:59 UTC odds (7 games): the 4:30 PM report parsed (50 rows on tonight's teams), overrides applied, 1,261 priced selections, 3 rule bets (2 `explained` → vetoed: Paul Reed PRA under at DraftKings 27.5 vs FanDuel 26.5 with Duren out; Sabonis rebounds under at DraftKings with Keegan Murray doubtful; 1 `against` → kept: Hachimura threes over at BetMGM with five Clippers out), plus model-only rows, reviewed against web news and logged to `data/ledger/ledger.csv` as the first forward picks (prices as of 13:59 ET, all pre-tip). The model-only rows were logged at threshold 0.05 (the day-1 default before it was raised to 0.08); the `edge` column lets the report filter to ≥0.08. One live test with fresh odds (`logs/live_close_20261025.log`, `--hours 3 --max-events 2`, Orlando@New York and LA Clippers@Oklahoma City, 22:43 UTC) spent **11 credits (286 remaining)**; it auto-downloaded the 6:30 PM report, priced 73 bets (printed, not logged), and its cached snapshot gave `ledger.py close` closing lines for 85 of the logged picks (mean line move −0.035 points against us; kept −0.03 n=66, vetoed −0.05 n=19 — a four-hour move on two games, not evidence of anything yet).

### 10.6 What to do now

1. Run the skill daily. The ledger is the only clean test left: the rule's dev number (+6.1%) has a holdout that is both tiny and reused, and the model-only path is break-even on dev. A few hundred settled kept picks with closing-line data will say more than any further backtest.
2. Keep the veto rule and the `against` keeps as they are; do not re-tune them on the ledger until it has at least 300 settled rule picks.
3. The FanDuel-only option (`--books fanduel`) only produces model-only picks; on dev those lose 0.4–0.9%. If the user wants FanDuel-only action, the honest advice is to paper-trade it, not stake it.
