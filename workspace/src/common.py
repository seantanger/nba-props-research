"""Shared loading, splitting, grading and statistics for the backtests."""
import pathlib
import os, re
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

HOLDOUT_START = pd.Timestamp("2023-03-13")   # fixed before any modelling; evaluated once at the end
DEV_FOLDS = [  # (train_end_exclusive, test_start, test_end_exclusive) walk-forward inside dev
    ("2022-12-01", "2022-12-01", "2023-01-01"),
    ("2023-01-01", "2023-01-01", "2023-02-01"),
    ("2023-02-01", "2023-02-01", "2023-03-13"),
]
MARKETS = ["PTS", "REB", "AST", "FG3M", "PRA"]

NON_FEATURES = {"SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "NICKNAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID",
                "GAME_DATE", "MATCHUP", "WL", "opp", "season", "POSITION", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
                "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS",
                "NBA_FANTASY_PTS", "DD2", "TD3", "WNBA_FANTASY_PTS", "PRA", "USG_PCT", "PACE", "TS_PCT", "AST_PCT", "REB_PCT",
                "E_USG_PCT", "POSS", "MIN_SEC", "AVAILABLE_FLAG", "PA", "PR", "RA"}


FEATURE_PAT = re.compile(r"(_m\d+$|_s\d+$|_ewm$|_last$|_max10$|_min10$|_std_season$|_per_min20$|^ps_|^tm_|^opp_|^pos_|^exp_margin$|"
                         r"^prev_|^pts_share10$|_trend$|^rest_days$|^b2b$|^games_prior$|^gp_season$|^home$|^playoffs$|^dow$|^month$|"
                         r"^height_in$|^DRAFT_YEAR$|^vac_|^ret_|^share_|^inj_|^avail_)")
# Switches for experiments (env vars): FEATURE_EXTRA adds a regex of otherwise-excluded columns (e.g. "^bs" for the LEAKY
# box-score-absence features, ceiling test only); FEATURE_EXCL drops a regex (e.g. "^(vac_|ret_|share_|inj_|avail_|opp_vac_)" = baseline).
FEATURE_EXTRA = re.compile(os.environ["FEATURE_EXTRA"]) if os.environ.get("FEATURE_EXTRA") else None
FEATURE_EXCL = re.compile(os.environ["FEATURE_EXCL"]) if os.environ.get("FEATURE_EXCL") else None


def feature_cols(df):
    """Explicit allow-list: only engineered, lagged (shift(1)) or static columns. Raw same-game box-score columns never qualify."""
    cols = [c for c in df.columns if (FEATURE_PAT.search(c) or (FEATURE_EXTRA and FEATURE_EXTRA.search(c))) and c not in NON_FEATURES
            and not c.endswith("_RANK") and df[c].dtype.kind in "fiub"]
    return [c for c in cols if not (FEATURE_EXCL and FEATURE_EXCL.search(c))]


def load_player_games():
    return pd.read_parquet(PROC / "player_games.parquet")


def load_pairs(snapshot="last"):
    """One row per (bet_id pair) = (player, game, market, line) at the chosen snapshot, with both prices.
    snapshot: 'last' (closest to tip) or 'first' (earliest scrape)."""
    o = pd.read_parquet(PROC / "offers.parquet")
    o = o[o.is_last] if snapshot == "last" else o[o.is_first]
    key = ["player_id", "game_id", "market", "line"]
    ov = o[o.side == "over"].drop_duplicates(key)
    un = o[o.side == "under"].drop_duplicates(key)
    p = ov.merge(un[key + ["dec_odds", "result", "pnl", "row"]], on=key, suffixes=("_over", "_under"))
    p = p.rename(columns={"dec_odds_over": "odds_over", "dec_odds_under": "odds_under", "result_over": "res_over",
                          "result_under": "res_under", "pnl_over": "pnl_over", "pnl_under": "pnl_under"})
    io, iu = 1 / p.odds_over, 1 / p.odds_under
    p["vig"] = io + iu - 1
    p["p_over_imp"] = io / (io + iu)           # multiplicative de-vig
    p["p_under_imp"] = 1 - p.p_over_imp
    p["be_over"] = io                            # breakeven win prob for the over at its price
    p["be_under"] = iu
    p = p.drop(columns=["side", "bet_id", "snap", "n_snaps", "is_first", "is_last"])
    p["is_holdout"] = p.game_date >= HOLDOUT_START
    return p.reset_index(drop=True)


def attach_first_snapshot(pairs):
    """Add the first-snapshot price/line for the same (player, game, market) to compute movement features/CLV."""
    f = load_pairs("first")[["player_id", "game_id", "market", "line", "odds_over", "odds_under", "p_over_imp"]]
    f = f.rename(columns={"line": "line_first", "odds_over": "odds_over_first", "odds_under": "odds_under_first", "p_over_imp": "p_over_imp_first"})
    # same line -> price move; different line -> line move. Match on player/game/market, pick the first line
    f = f.drop_duplicates(["player_id", "game_id", "market"])
    return pairs.merge(f, on=["player_id", "game_id", "market"], how="left")


def grade(bets):
    """bets: DataFrame with columns side ('over'/'under'), odds, result, stake. Returns pnl per bet."""
    win = bets.result == "win"
    loss = bets.result == "loss"
    return np.where(win, bets.stake * (bets.odds - 1), np.where(loss, -bets.stake, 0.0))


def make_bets(pairs, p_over, side_rule="edge", thr=0.03, markets=None, sides=("over", "under"), stake=1.0, max_odds=3.0, min_odds=1.3):
    """Turn model probabilities into bets. Edge = model prob - breakeven prob at the offered price."""
    p = pairs.copy()
    p["p_over_model"] = np.asarray(p_over)
    p["edge_over"] = p.p_over_model - p.be_over
    p["edge_under"] = (1 - p.p_over_model) - p.be_under
    rows = []
    for side in sides:
        e = p[f"edge_{side}"]
        sel = (e > thr) & (p[f"odds_{side}"] <= max_odds) & (p[f"odds_{side}"] >= min_odds)
        if markets:
            sel &= p.market.isin(markets)
        b = p[sel].copy()
        b["side"] = side
        b["odds"] = b[f"odds_{side}"]
        b["result"] = b[f"res_{side}"]
        b["edge"] = e[sel]
        b["p_model"] = np.where(side == "over", b.p_over_model, 1 - b.p_over_model)
        b["stake"] = stake
        rows.append(b)
    bets = pd.concat(rows, ignore_index=True) if rows else p.iloc[0:0]
    # one bet per (player, game, market): keep the larger edge if both sides somehow qualify (cannot with vig>0)
    bets = bets.sort_values("edge", ascending=False).drop_duplicates(["player_id", "game_id", "market"])
    bets["pnl"] = grade(bets)
    return bets


def summarize(bets, n_boot=2000, seed=0, label=""):
    b = bets[bets.result.isin(["win", "loss", "push"])]
    n = len(b)
    if n == 0:
        return {"label": label, "n": 0}
    settled = b[b.result != "push"]
    stake = b.stake.sum()
    units = b.pnl.sum()
    roi = units / stake
    hit = (settled.result == "win").mean() if len(settled) else np.nan
    be = (1 / settled.odds).mean() if len(settled) else np.nan   # avg breakeven rate at the taken prices
    # cluster bootstrap by game date (bets on one slate are correlated)
    rng = np.random.default_rng(seed)
    dates = b.game_date.values
    ud = np.unique(dates)
    idx_by_date = {d: np.where(dates == d)[0] for d in ud}
    pnl = b.pnl.values; st = b.stake.values
    rois = []
    for _ in range(n_boot):
        samp = rng.choice(ud, size=len(ud), replace=True)
        ix = np.concatenate([idx_by_date[d] for d in samp])
        rois.append(pnl[ix].sum() / st[ix].sum())
    lo, hi = np.percentile(rois, [2.5, 97.5])
    out = {"label": label, "n": n, "roi": roi, "units": units, "ci_lo": lo, "ci_hi": hi, "hit": hit, "breakeven": be,
           "avg_odds": settled.odds.mean(), "n_void": int((bets.result == "void").sum()), "p_roi_le0": float(np.mean(np.array(rois) <= 0))}
    return out


def fmt(s):
    if s.get("n", 0) == 0:
        return f"{s.get('label','')}: no bets"
    return (f"{s['label']}: n={s['n']} ROI={s['roi']*100:+.2f}% units={s['units']:+.1f} CI95=[{s['ci_lo']*100:+.1f}%, {s['ci_hi']*100:+.1f}%] "
            f"hit={s['hit']*100:.1f}% be={s['breakeven']*100:.1f}% avg_odds={s['avg_odds']:.3f} P(ROI<=0)={s['p_roi_le0']:.2f}")


def dev_pairs(pairs):
    return pairs[~pairs.is_holdout]


def holdout_pairs(pairs):
    return pairs[pairs.is_holdout]


# ---------------------------------------------------------------------------------------------
# BettingPros multi-book data (2020-21 .. 2026-27). Holdout policy (fixed before looking at results, see JOURNAL):
#   holdout = season 2025-26 and later; the Kambi holdout window 2023-03-13..2023-06-12 is excluded from development.
BP_HOLDOUT_SEASON = "2025-26"
KAMBI_HOLDOUT_WINDOW = (pd.Timestamp("2023-03-13"), pd.Timestamp("2023-06-13"))
BP_MARKETS = ["PTS", "REB", "AST", "FG3M", "PRA", "PA", "PR", "RA", "STL", "BLK"]
BOOK_NAMES = {0: "Consensus", 12: "DraftKings", 10: "FanDuel", 19: "BetMGM", 13: "Caesars", 49: "HardRock", 2: "Pinnacle", 33: "theScore",
              14: "Fanatics", 60: "Novig", 37: "PrizePicks", 38: "ProphetX", 36: "Underdog", 45: "Betr", 39: "Fliff", 63: "Sleeper",
              18: "BetRivers", 15: "SugarHouse", 27: "PartyCasino", 24: "bet365", 26: "Tipico", 32: "Borgata", 73: "Polymarket"}
DFS_BOOKS = {37, 36, 63, 70, 69, 74, 39, 53, 54, 57, 45}   # pick'em / exchange-style; not a sportsbook price -> excluded from betting


def season_of(d):
    d = pd.DatetimeIndex(d)
    y = np.where(d.month >= 9, d.year, d.year - 1)
    return pd.Series([f"{a}-{str(a + 1)[2:]}" for a in y], index=d if False else None).values


def load_bp_pairs(min_odds=1.3, max_odds=3.0):
    """One row per (event, player, market, book) with both sides at the book's final (closing) line.
    Adds consensus (book 0) line/price and the opening line for the same offer."""
    d = pd.read_parquet(PROC / "bp_lines.parquet")
    d = d[d.main & d.GAME_ID.notna() & d.PLAYER_ID.notna()].copy()
    d["PLAYER_ID"] = d.PLAYER_ID.astype(int)
    key = ["event_id", "PLAYER_ID", "GAME_ID", "market", "book_id"]
    ov = d[d.side == "over"].drop_duplicates(key)
    un = d[d.side == "under"].drop_duplicates(key)[key + ["line", "dec_odds", "result", "updated", "open_line", "open_dec", "open_created", "open_book"]]
    p = ov.merge(un, on=key, suffixes=("", "_under"))
    p = p[p.line == p.line_under].drop(columns=["line_under"])
    p = p.rename(columns={"dec_odds": "odds_over", "dec_odds_under": "odds_under", "result": "res_over", "result_under": "res_under",
                          "open_dec": "open_odds_over", "open_dec_under": "open_odds_under", "open_line": "open_line_over", "open_line_under": "open_line_under"})
    p["updated"] = p[["updated", "updated_under"]].max(axis=1)
    p["mins_before_tip"] = (p.scheduled - p.updated).dt.total_seconds() / 60
    io, iu = 1 / p.odds_over, 1 / p.odds_under
    p["vig"] = io + iu - 1
    p["p_over_imp"] = io / (io + iu); p["be_over"] = io; p["be_under"] = iu
    p["player_id"] = p.PLAYER_ID; p["game_id"] = p.GAME_ID
    p["season"] = season_of(p.game_date)
    p["is_holdout"] = p.season >= BP_HOLDOUT_SEASON
    p["in_kambi_holdout"] = (p.game_date >= KAMBI_HOLDOUT_WINDOW[0]) & (p.game_date < KAMBI_HOLDOUT_WINDOW[1])
    # consensus reference (book 0) at its own line
    c = p[p.book_id == 0][["event_id", "PLAYER_ID", "market", "line", "p_over_imp", "odds_over", "odds_under"]].rename(
        columns={"line": "cons_line", "p_over_imp": "cons_p_over", "odds_over": "cons_odds_over", "odds_under": "cons_odds_under"})
    p = p.merge(c, on=["event_id", "PLAYER_ID", "market"], how="left")
    p["is_dfs"] = p.book_id.isin(DFS_BOOKS)
    p["bettable"] = (p.book_id != 0) & ~p.is_dfs & (p.odds_over >= min_odds) & (p.odds_over <= max_odds) & (p.odds_under >= min_odds) & (p.odds_under <= max_odds)
    keep = ["event_id", "player_id", "game_id", "market", "book_id", "player_name", "game_date", "scheduled", "season", "season_type", "line",
            "odds_over", "odds_under", "res_over", "res_under", "actual", "played", "updated", "mins_before_tip", "vig", "p_over_imp", "be_over",
            "be_under", "open_line_over", "open_odds_over", "open_odds_under", "open_created", "open_book", "cons_line", "cons_p_over",
            "cons_odds_over", "cons_odds_under", "is_holdout", "in_kambi_holdout", "is_dfs", "bettable"]
    return p[keep].reset_index(drop=True)


def bp_dev(p):
    return p[~p.is_holdout & ~p.in_kambi_holdout]


def bp_holdout(p):
    return p[p.is_holdout]
