"""Model-free analyses on BettingPros DEV pairs: blind sides, line disagreement vs consensus / sharp books, stale lines."""
import numpy as np, pandas as pd
from scipy import stats
from src.common import *

p = pd.read_parquet(RESULTS / "bp_pairs_cache.parquet")
d = bp_dev(p); d = d[d.bettable].copy()

def blind(df, side, label, n_boot=200):
    b = df.copy(); b["side"] = side; b["odds"] = b[f"odds_{side}"]; b["result"] = b[f"res_{side}"]; b["stake"] = 1.0; b["pnl"] = grade(b)
    return fmt(summarize(b, n_boot=n_boot, label=label))

print("=== Blind sides, all books (dev, n pairs=%d) ===" % len(d))
print(blind(d, "over", "all over")); print(blind(d, "under", "all under"))
for mk in BP_MARKETS:
    s = d[d.market == mk]; print("  ", mk, "| over:", blind(s, "over", "").split("ROI=")[1][:7], "| under:", blind(s, "under", "").split("ROI=")[1][:7], "| n=", len(s))
print("\n=== Blind under by book ===")
for bk, s in d.groupby("book_id"):
    print("  ", BOOK_NAMES.get(bk, bk), blind(s, "under", "").split(": ")[1][:60])

# reference lines: consensus (cons_line) and FanDuel / DraftKings closing line for the same selection
ref = d[d.book_id.isin([10, 12])].pivot_table(index=["event_id", "player_id", "market"], columns="book_id", values=["line", "p_over_imp"], aggfunc="first")
ref.columns = [f"{a}_{'fd' if b == 10 else 'dk'}" for a, b in ref.columns]
d = d.merge(ref.reset_index(), on=["event_id", "player_id", "market"], how="left")

print("\n=== Line disagreement: book line vs reference line (bet toward the reference), every qualifying book row ===")
for refname, rl in [("consensus", "cons_line"), ("FanDuel", "line_fd"), ("DraftKings", "line_dk")]:
    s = d[d[rl].notna() & (d.book_id != 10) & (d.book_id != 12)] if refname != "consensus" else d[d[rl].notna()]
    diff = s.line - s[rl]
    for lo, hi, lab in [(-99, -1.5, "book <= ref-2"), (-1.5, -0.5, "book = ref-1"), (0.5, 1.5, "book = ref+1"), (1.5, 99, "book >= ref+2")]:
        q = s[(diff > lo) & (diff <= hi)]
        side = "over" if hi <= 0 else "under"
        print(f"  {refname:10s} {lab:14s} {blind(q, side, side)}")
    # stale filter: book updated > 3h before tip while reference exists
    q = s[(diff.abs() >= 1) & (s.mins_before_tip > 180)]
    print(f"  {refname:10s} |diff|>=1 & stale(>3h): over-when-low:", blind(q[q.line < q[rl]], "over", "").split(": ")[1][:70])
    print(f"  {refname:10s} |diff|>=1 & stale(>3h): under-when-high:", blind(q[q.line > q[rl]], "under", "").split(": ")[1][:70])
    q = s[(diff.abs() >= 1) & (s.mins_before_tip <= 60)]
    print(f"  {refname:10s} |diff|>=1 & fresh(<=1h): over-when-low:", blind(q[q.line < q[rl]], "over", "").split(": ")[1][:70])
    print(f"  {refname:10s} |diff|>=1 & fresh(<=1h): under-when-high:", blind(q[q.line > q[rl]], "under", "").split(": ")[1][:70])

print("\n=== Same line as consensus, but price disagreement (book implied over prob vs consensus) ===")
s = d[(d.line == d.cons_line)]
dp = s.p_over_imp - s.cons_p_over
for lo, hi in [(-1, -0.04), (-0.04, -0.02), (0.02, 0.04), (0.04, 1)]:
    q = s[(dp > lo) & (dp <= hi)]
    side = "over" if hi <= 0 else "under"
    print(f"  dp in ({lo},{hi}] bet {side}: {blind(q, side, '')}")
