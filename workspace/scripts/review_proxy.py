"""Mechanical proxy for the human review step (phase 2, idea 3), evaluated on dev only.

Candidate bets: the frozen rule (p_anc_fd, thr 0.03, odds 1.3-3.0, one per selection) in the clean120 tier (headline) and the
'all' tier (context). For each candidate, look for injury-report "news" on the player's team published after FanDuel's last
update and no later than the book's last update (capped at tip-off):
  teammate_out : a teammate with a >= MIN_T minute norm moved to Out/Doubtful (or was first listed as such)
  teammate_ret : a teammate moved from Out/Doubtful to playing status
  self_worse   : the player himself moved to Questionable/Doubtful/Out (or first listed so)
  self_better  : the player moved from Questionable/Doubtful to Probable/Available or was removed
The line gap (book line minus FanDuel line) is "explained" when the news would move the player's number in that direction:
  gap up   (book higher -> rule bets UNDER at the book): teammate_out or self_better
  gap down (book lower  -> rule bets OVER  at the book): teammate_ret or self_worse
VETO = explained. Also reported: news against the gap (book is the stale one), and the "bet WITH the book" alternative = take
FanDuel's stale line on the news side (unverifiable: FanDuel may have suspended the market; BettingPros stores no status).
Usage: .venv/bin/python scripts/review_proxy.py [--tag _inj] (uses results/bp_eval_dev{tag}.parquet)
"""
import argparse, sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import make_bets, summarize, fmt, PROC, RESULTS, BOOK_NAMES
from src.select_rule import tier_rows

MIN_T = 15.0


def events_table():
    """Status-change events per (GAME_ID, team, player) with UTC timestamps."""
    L = pd.read_parquet(PROC / "injury_long.parquet")
    L = L.dropna(subset=["PLAYER_ID"]).copy()
    L["PLAYER_ID"] = L.PLAYER_ID.astype("int64")
    L["ts"] = L.report_ts.dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC").dt.tz_localize(None)
    L["tip"] = L.tip_ts.dt.tz_localize("US/Eastern", ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC").dt.tz_localize(None)
    L = L.dropna(subset=["ts"]).sort_values(["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "ts"])
    g = L.groupby(["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"], sort=False)
    L["prev"] = g.code.shift(1)
    first = L.prev.isna()
    L["ev"] = ""
    L.loc[(L.code >= 3) & (first | (L.prev < 3)), "ev"] = "out"
    L.loc[(L.code < 3) & (L.prev >= 3), "ev"] = "ret"
    L.loc[(L.code == 2) & (first | (L.prev < 2)), "ev"] = "q"
    L.loc[(L.code < 2) & (L.prev == 2), "ev"] = "up"
    # a player who disappears from the report after being Questionable is effectively upgraded: ignore (rare, needs full-roster diff)
    E = L[L.ev != ""][["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "ts", "tip", "ev", "player", "status"]].copy()
    # minute norms of the subject player as of his last game before this one
    pg = pd.read_parquet(PROC / "player_games.parquet", columns=["PLAYER_ID", "GAME_DATE", "MIN", "GAME_ID"])
    pg = pg[pg.MIN > 0].sort_values("GAME_DATE")
    pg["min_r10"] = pg.groupby("PLAYER_ID").MIN.transform(lambda s: s.rolling(10, min_periods=1).mean())
    gd = pd.read_parquet(PROC / "player_games.parquet", columns=["GAME_ID", "GAME_DATE"]).drop_duplicates("GAME_ID")
    E = E.merge(gd, on="GAME_ID").sort_values("GAME_DATE")
    E = pd.merge_asof(E, pg[["PLAYER_ID", "GAME_DATE", "min_r10"]].sort_values("GAME_DATE"), on="GAME_DATE", by="PLAYER_ID", allow_exact_matches=False)
    E["min_r10"] = E.min_r10.fillna(0)
    return E


def attach_news(bets, E, slack_min=0):
    """For each bet, flag news events on the team in (updated_fd - slack, min(updated, tip)]."""
    teams = pd.read_parquet(PROC / "player_games.parquet", columns=["PLAYER_ID", "GAME_ID", "TEAM_ABBREVIATION"])
    b = bets.merge(teams.rename(columns={"PLAYER_ID": "player_id", "GAME_ID": "game_id"}), on=["player_id", "game_id"], how="left")
    b["t_end"] = b[["updated", "scheduled"]].min(axis=1)
    b["t_start"] = b.updated_fd - pd.Timedelta(minutes=slack_min)
    e = E.rename(columns={"GAME_ID": "game_id", "PLAYER_ID": "subj"})
    m = b[["event_id", "player_id", "game_id", "market", "book_id", "TEAM_ABBREVIATION", "t_start", "t_end"]].merge(
        e[["game_id", "TEAM_ABBREVIATION", "subj", "ts", "ev", "min_r10", "player", "status"]], on=["game_id", "TEAM_ABBREVIATION"], how="inner")
    m = m[(m.ts > m.t_start) & (m.ts <= m.t_end)]
    m["self"] = m.subj == m.player_id
    flags = pd.DataFrame(index=b.index)
    key = ["event_id", "player_id", "game_id", "market", "book_id"]
    def has(mask, name):
        s = m[mask].groupby(key).size().rename(name)
        return s
    agg = pd.concat([
        has(~m.self & (m.ev == "out") & (m.min_r10 >= MIN_T), "teammate_out"),
        has(~m.self & (m.ev == "ret") & (m.min_r10 >= MIN_T), "teammate_ret"),
        has(m.self & m.ev.isin(["q", "out"]), "self_worse"),
        has(m.self & (m.ev == "up"), "self_better"),
        m[~m.self & (m.ev == "out")].groupby(key).min_r10.sum().rename("out_min"),
    ], axis=1).reset_index()
    b = b.merge(agg, on=key, how="left")
    for c in ["teammate_out", "teammate_ret", "self_worse", "self_better", "out_min"]:
        b[c] = b[c].fillna(0)
    b["gap"] = b.line - b.line_fd
    up = b.gap > 0
    down = b.gap < 0
    b["explained"] = (up & ((b.teammate_out > 0) | (b.self_better > 0))) | (down & ((b.teammate_ret > 0) | (b.self_worse > 0)))
    b["against"] = (up & ((b.teammate_ret > 0) | (b.self_worse > 0))) | (down & ((b.teammate_out > 0) | (b.self_better > 0)))
    b["any_news"] = (b.teammate_out + b.teammate_ret + b.self_worse + b.self_better) > 0
    return b


def report(b, label):
    print(f"\n=== {label}: n={len(b)}")
    print("   ", fmt(summarize(b, n_boot=500, label="all candidates")))
    for name, mask in [("kept (not explained)", ~b.explained), ("VETOED (news explains gap)", b.explained),
                       ("  of which gap up / bet under", b.explained & (b.gap > 0)), ("  of which gap down / bet over", b.explained & (b.gap < 0)),
                       ("news AGAINST the gap (book stale)", b.against & ~b.explained), ("no news at all", ~b.any_news),
                       ("gap == 0 (price-only edge)", b.gap == 0)]:
        q = b[mask]
        if len(q) >= 5:
            print("   ", fmt(summarize(q, n_boot=500, label=name)))
    # 'bet WITH the book': on explained rows, take FanDuel's line on the news side (opposite of the rule's side), at FanDuel's price
    q = b[b.explained].copy()
    if len(q) >= 5:
        print("    bet WITH the book at FanDuel's stale line (unverifiable availability):")
        for s, g in q.groupby("season"):
            print(f"      {s}: n={len(g)}")


def with_book_at_fd(b, p_all):
    """Return a bets frame for the alternative: for explained gaps, bet the news side at FanDuel's final line/price."""
    q = b[b.explained].copy()
    fd = p_all[p_all.book_id == 10][["event_id", "player_id", "game_id", "market", "line", "odds_over", "odds_under", "res_over", "res_under", "season", "game_date"]]
    fd = fd.rename(columns={"line": "line_fd2"})
    q = q[["event_id", "player_id", "game_id", "market", "gap", "side"]].merge(fd, on=["event_id", "player_id", "game_id", "market"], how="inner")
    q["side"] = np.where(q.gap > 0, "over", "under")  # news side at FanDuel: gap up -> FD too low -> over
    q["odds"] = np.where(q.side == "over", q.odds_over, q.odds_under)
    q["result"] = np.where(q.side == "over", q.res_over, q.res_under)
    q = q[q.result.isin(["win", "loss", "push"])]
    q["stake"] = 1.0
    q["pnl"] = np.select([q.result == "win", q.result == "loss"], [q.odds - 1, -1.0], 0.0)
    return q


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); ap.add_argument("--slack", type=float, default=0)
    a = ap.parse_args()
    p_full = pd.read_parquet(RESULTS / f"bp_eval_dev{a.tag}.parquet")
    p = p_full[p_full.bettable & (p_full.book_id != 10)]
    E = events_table()
    print("events:", E.ev.value_counts().to_dict())
    for tier in ["clean120", "all"]:
        q = tier_rows(p, tier)
        q = q[q.p_anc_fd.notna()]
        bets = make_bets(q, q.p_anc_fd, thr=0.03)
        bets = bets.sort_values("edge", ascending=False).drop_duplicates(["event_id", "player_id", "market"])
        b = attach_news(bets, E, slack_min=a.slack)
        report(b, f"tier {tier} (rule p_anc_fd thr 0.03)")
        alt = with_book_at_fd(b, p_full)
        if len(alt):
            print("    WITH-the-book alternative at FanDuel:", fmt(summarize(alt, n_boot=500, label="news side at FD")))
            big = alt.merge(b[["event_id", "player_id", "game_id", "market", "out_min"]], on=["event_id", "player_id", "game_id", "market"])
            big = big[(big.gap.abs() >= 1.5) | (big.out_min >= 25)]
            if len(big) >= 5:
                print("      ... restricted to starter out (>=25 min) or |gap|>=1.5:", fmt(summarize(big, n_boot=500, label="")))
        b.to_parquet(RESULTS / f"review_proxy_{tier}{a.tag}.parquet", index=False)
