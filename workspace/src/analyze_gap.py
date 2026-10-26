import numpy as np, pandas as pd, sys
from src.common import *
p = pd.read_parquet(RESULTS / "bp_pairs_cache.parquet")
d = bp_dev(p); d = d[d.bettable].copy()
ref = d[d.book_id.isin([10, 12])].pivot_table(index=["event_id", "player_id", "market"], columns="book_id", values=["line", "p_over_imp", "mins_before_tip"], aggfunc="first")
ref.columns = [f"{a}_{'fd' if b == 10 else 'dk'}" for a, b in ref.columns]
d = d.merge(ref.reset_index(), on=["event_id", "player_id", "market"], how="left")
d["gap_fd"] = d.line - d.line_fd
s = d[(d.gap_fd.abs() >= 2) & (d.book_id != 10)].copy()
s["side"] = np.where(s.gap_fd < 0, "over", "under")
s["odds"] = np.where(s.side == "over", s.odds_over, s.odds_under); s["result"] = np.where(s.side == "over", s.res_over, s.res_under); s["stake"] = 1.0; s["pnl"] = grade(s)
s["fresh"] = pd.cut(s.mins_before_tip, [-1e9, 15, 60, 180, 720, 1e9], labels=["<=15m", "15-60m", "1-3h", "3-12h", ">12h"])
s["fd_fresh"] = pd.cut(s.mins_before_tip_fd, [-1e9, 15, 60, 180, 720, 1e9], labels=["<=15m", "15-60m", "1-3h", "3-12h", ">12h"])
print("n", len(s), "void share", (s.result == "void").mean().round(3), "share of all bettable rows:", round(len(s) / len(d), 4))
def show(g, by):
    for k, q in g.groupby(by, observed=True):
        r = summarize(q, n_boot=200, label=str(k)); print("   ", fmt(r) if r["n"] else k, "| void", round((q.result == "void").mean(), 3))
print("\n-- by book freshness (time since the soft book's line last changed)"); show(s, "fresh")
print("\n-- by FanDuel freshness"); show(s, "fd_fresh")
print("\n-- book updated AFTER FanDuel's last update (book saw the later state)"); show(s, s.updated > s.updated_fd if "updated_fd" in s else (s.mins_before_tip < s.mins_before_tip_fd))
print("\n-- by side"); show(s, "side")
print("\n-- by season"); show(s, "season")
print("\n-- by market"); show(s, "market")
print("\n-- by book"); show(s, s.book_id.map(BOOK_NAMES))
print("\n-- by gap size"); show(s, pd.cut(s.gap_fd.abs(), [1.9, 2.1, 3.1, 5.1, 99]))
print("\n-- fresh (<=60m) only, by side"); show(s[s.mins_before_tip <= 60], "side")
print("\n-- fresh (<=60m) only, by season"); show(s[s.mins_before_tip <= 60], "season")
