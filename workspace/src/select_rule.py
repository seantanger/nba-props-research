"""Rule selection on BettingPros DEV rows with availability tiers applied BEFORE picking the best book per selection.
Usage: .venv/bin/python src/select_rule.py [eval_parquet]   (default results/bp_eval_dev_stack.parquet)
Tiers (row filters on the per-book closing lines):
  all      : every bettable row (availability of the stored line at its timestamp is NOT verifiable)
  clean120 : FanDuel's final update 0-120 min before tip AND this book's final update is later than FanDuel's AND before tip
  clean30  : same with FanDuel's final update 0-30 min before tip
  both15   : both FanDuel's and this book's final updates within 15 min of tip (either order)
"""
import sys
import numpy as np, pandas as pd
from src.common import *
from src.eval_bp import one_per_selection



def tier_rows(df, name):
    if name == "all":
        return df
    has = df.line_fd.notna() & (df.mins_before_tip >= 0)
    if name == "clean120":
        return df[has & (df.mbt_fd >= 0) & (df.mbt_fd <= 120) & df.book_after_fd]
    if name == "clean30":
        return df[has & (df.mbt_fd >= 0) & (df.mbt_fd <= 30) & df.book_after_fd]
    if name == "both15":
        return df[has & (df.mbt_fd >= 0) & (df.mbt_fd <= 15) & (df.mins_before_tip <= 15)]
    raise ValueError(name)


TIERS = ["all", "clean120", "clean30", "both15"]


def run(df, col, thr, tier, by_season=False, n_boot=300):
    q = tier_rows(df, tier)
    q = q[q[col].notna()]
    if len(q) == 0:
        return
    b = make_bets(q, q[col], thr=thr)
    print("   ", fmt(summarize(b, n_boot=n_boot, label=f"{col} thr={thr} {tier:9s}")))
    if by_season:
        for s, g in b.groupby("season"):
            print("        ", fmt(summarize(g, n_boot=100, label=s)))
    return b


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "bp_eval_dev_stack.parquet"
    d = pd.read_parquet(RESULTS / src)
    d = d[d.bettable & (d.book_id != 10)].copy()
    cols = [c for c in ["p_anc_fd", "p_stack", "p_anc_c", "p_proj"] if c in d.columns]
    for col in cols:
        print(f"\n==== {col} ====")
        for thr in [0.02, 0.03, 0.04, 0.05, 0.06]:
            for tier in TIERS:
                run(d, col, thr, tier, by_season=(tier in ("clean120", "clean30") and thr in (0.03, 0.04)))
