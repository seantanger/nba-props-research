"""POST-HOC (not pre-registered) look at the holdout: the frozen rule unchanged, but on the availability tier that exists in the
2025-26 timestamp regime: both FanDuel's and the book's line were last seen within 5 minutes of tip (either side of it).
Run once. This is NOT the headline; it is reported as 'what came closest' with that label."""
import json, sys
import numpy as np, pandas as pd
from src.common import *
sys.argv = ["x", "holdout"]
import src.eval_holdout as eh  # runs the headline again (cheap) and gives us d, rule, report

d, rule, col = eh.d, eh.rule, eh.col
W = 5
q = d[d.line_fd.notna() & (d.mbt_fd.abs() <= W) & (d.mins_before_tip.abs() <= W) & d[col].notna()]
print(f"\n=== POST-HOC tier 'both last seen within {W} min of tip' (rows={len(q):,}) ===")
b = make_bets(q, q[col], thr=rule["thr"], min_odds=rule["min_odds"], max_odds=rule["max_odds"])
pc = np.where(b.side == "over", b.p_anc_c, 1 - b.p_anc_c)
b["clv"] = pc - 1 / b.odds
b["book"] = b.book_id.map(BOOK_NAMES).fillna(b.book_id.astype(str))
b["month"] = b.game_date.dt.to_period("M").astype(str)
s = eh.report(b, f"POSTHOC_close{W}")
for by in ["month", "book", "market", "side"]:
    print(f"  by {by}:")
    for k, g in b.groupby(by):
        if len(g) >= 30:
            print("     ", fmt(summarize(g, n_boot=300, label=str(k))))
# edge-threshold monotonicity (information only; the rule's threshold is fixed at 0.03)
print("  by edge bucket (same bets split by the model edge, not a re-selection):")
for lo, hi in [(0.03, 0.04), (0.04, 0.05), (0.05, 0.07), (0.07, 1)]:
    g = b[(b.edge >= lo) & (b.edge < hi)]
    if len(g) >= 30:
        print("     ", fmt(summarize(g, n_boot=300, label=f"edge {lo:.2f}-{hi:.2f}")))
b.to_parquet(RESULTS / "bets_holdout_posthoc_close5.parquet", index=False)
json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in s.items()}, open(RESULTS / "headline_holdout_posthoc_close5.json", "w"), indent=1, default=str)
