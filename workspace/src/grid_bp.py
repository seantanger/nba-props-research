"""Strategy grids on the evaluated BettingPros DEV rows (results/bp_eval_dev.parquet)."""
import sys
import numpy as np, pandas as pd
from src.common import *
from src.eval_bp import stack_fit, stack_predict, STACK_COLS, tiers, one_per_selection, lg

p = pd.read_parquet(RESULTS / "bp_eval_dev.parquet")
d = p[p.bettable & (p.book_id != 10)].copy()
# walk-forward stack: fit on all prior seasons, predict the season
d["p_stack"] = np.nan
for s in sorted(d.season.unique()):
    tr = d[d.season < s]
    if len(tr) < 50000:
        continue
    m = stack_fit(tr, STACK_COLS)
    te = d.season == s
    d.loc[te, "p_stack"] = stack_predict(m, d[te], STACK_COLS)
    print(f"stack {s}: coefs={np.round(m.coef_[0], 3)} n_train={len(tr)}")
d.to_parquet(RESULTS / "bp_eval_dev_stack.parquet", index=False)

def bets_for(df, col, thr):
    q = df[df[col].notna()]
    return make_bets(q, q[col], thr=thr)

for col in ["p_anc_c", "p_anc_fd", "p_stack", "p_proj"]:
    print(f"\n==== {col} ====")
    for thr in [0.02, 0.04, 0.06, 0.08]:
        b = bets_for(d, col, thr)
        if len(b) == 0:
            continue
        print(f" thr={thr}")
        for tname, tb in tiers(b).items():
            print("   ", fmt(summarize(tb, n_boot=200, label=f"{tname:9s} all-books")))
            print("   ", fmt(summarize(one_per_selection(tb), n_boot=200, label=f"{tname:9s} one/sel ")))
        if thr == 0.04:
            print("    by season (after_fd tier, one/sel):")
            for s_, q in one_per_selection(tiers(b)["after_fd"]).groupby("season"):
                print("      ", fmt(summarize(q, n_boot=100, label=s_)))
            print("    by season (all rows, one/sel):")
            for s_, q in one_per_selection(b).groupby("season"):
                print("      ", fmt(summarize(q, n_boot=100, label=s_)))
