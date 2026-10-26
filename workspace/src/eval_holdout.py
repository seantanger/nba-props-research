"""Evaluate the ONE chosen rule (results/chosen_rule.json) on an evaluated BettingPros slice.
Usage: .venv/bin/python src/eval_holdout.py holdout|kambi|dev
Reports n, ROI, units, cluster-bootstrap CI (by game date), hit rate vs breakeven, CLV proxy vs consensus close,
breakdowns by season / month / book / market / side, drawdown, and the unverifiable 'all rows' tier for comparison.
"""
import json, sys
import numpy as np, pandas as pd
from src.common import *
from src.select_rule import tier_rows

which = sys.argv[1]
rule = json.loads((RESULTS / "chosen_rule.json").read_text())
p = pd.read_parquet(RESULTS / f"bp_eval_{which}.parquet")
d = p[p.bettable & (p.book_id != 10)].copy()
col = rule["prob_col"]


def bets_in(tier):
    q = tier_rows(d, tier)
    q = q[q[col].notna()]
    b = make_bets(q, q[col], thr=rule["thr"], min_odds=rule["min_odds"], max_odds=rule["max_odds"])
    # CLV proxy: consensus-anchored probability of the bet side at the bet's line minus the breakeven at the price taken
    pc = np.where(b.side == "over", b.p_anc_c, 1 - b.p_anc_c)
    b["clv"] = pc - 1 / b.odds
    b["book"] = b.book_id.map(BOOK_NAMES)
    b["month"] = b.game_date.dt.to_period("M").astype(str)
    return b


def report(b, label):
    s = summarize(b, n_boot=2000, label=label)
    print(fmt(s))
    settled = b[b.result.isin(["win", "loss"])]
    if len(settled):
        print(f"      CLV proxy vs consensus close: mean {settled.clv.mean()*100:+.2f} pts, positive on {(settled.clv>0).mean()*100:.1f}% of bets (n with consensus={settled.clv.notna().sum()})")
        daily = b.groupby("game_date").pnl.sum()
        cum = daily.cumsum()
        dd = (cum - cum.cummax()).min()
        print(f"      units by day: best {daily.max():+.1f}, worst {daily.min():+.1f}, max drawdown {dd:+.1f} units over {len(daily)} betting days; "
              f"top-5 days share of profit: {daily.nlargest(5).sum()/max(daily.sum(),1e-9)*100:.0f}%")
    return s


print("rule:", rule, "\nslice:", which, "rows", len(d), "dates", d.game_date.min().date(), "..", d.game_date.max().date())
print("\n=== CHOSEN RULE (tier=%s) ===" % rule["tier"])
b = bets_in(rule["tier"])
s = report(b, "HEADLINE")
for by in ["season", "month", "book", "market", "side"]:
    print(f"  by {by}:")
    for k, g in b.groupby(by):
        if len(g) >= 20:
            print("     ", fmt(summarize(g, n_boot=300, label=str(k))))
print("\n=== same rule, other tiers (for context; 'all' includes lines whose availability cannot be verified) ===")
for t in ["all", "clean120", "clean30", "both15"]:
    if t != rule["tier"]:
        report(bets_in(t), t)
b.to_parquet(RESULTS / f"bets_{which}.parquet", index=False)
json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in s.items()}, open(RESULTS / f"headline_{which}.json", "w"), indent=1, default=str)
