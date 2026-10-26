"""Structural hypotheses on dev pairs (no model needed): PRA vs component lines, line moves, star overs."""
import numpy as np, pandas as pd
from src.common import *

pairs = attach_first_snapshot(load_pairs("last"))
d = dev_pairs(pairs).copy()

def blind(df, side, label):
    b = df.copy(); b["side"] = side; b["odds"] = b[f"odds_{side}"]; b["result"] = b[f"res_{side}"]; b["stake"] = 1.0; b["pnl"] = grade(b)
    b = b[(b.odds >= 1.3) & (b.odds <= 3.0)]
    return fmt(summarize(b, n_boot=300, label=label))

print("=== PRA line vs sum of component lines (same player/game, last snapshot) ===")
w = d.pivot_table(index=["player_id", "game_id"], columns="market", values="line", aggfunc="first")
w = w.dropna(subset=["PTS", "REB", "AST", "PRA"])
w["comp_sum"] = w.PTS + w.REB + w.AST
w["diff"] = w.PRA - w.comp_sum
print("  n=", len(w), " diff (PRA line - sum): mean", w["diff"].mean().round(3), " quantiles", w["diff"].quantile([.1, .25, .5, .75, .9]).round(2).tolist())
pra = d[d.market == "PRA"].merge(w[["diff", "comp_sum"]].reset_index(), on=["player_id", "game_id"])
for lo, hi in [(-99, -1.5), (-1.5, -0.5), (-0.5, 0.5), (0.5, 1.5), (1.5, 99)]:
    s = pra[(pra["diff"] >= lo) & (pra["diff"] < hi)]
    print("  diff in [%s,%s): " % (lo, hi), blind(s, "over", "PRA over"), "|", blind(s, "under", "PRA under").split(":")[1][:40])

print("\n=== Line moves first->last: follow vs fade at the LAST price ===")
mv = d[d.line_first.notna()].copy()
mv["dline"] = mv.line - mv.line_first
mv["dprob"] = mv.p_over_imp - mv.p_over_imp_first
print("  n pairs with first snap:", len(mv), " line moved:", (mv.dline != 0).sum(), " price moved same line:", ((mv.dline == 0) & (mv.dprob.abs() > 1e-6)).sum())
up = mv[mv.dline > 0]; dn = mv[mv.dline < 0]
print("  line UP  :", blind(up, "over", "follow(over)"), "\n            ", blind(up, "under", "fade(under)"))
print("  line DOWN:", blind(dn, "under", "follow(under)"), "\n            ", blind(dn, "over", "fade(over)"))
pu = mv[(mv.dline == 0) & (mv.dprob > 0.01)]; pdn = mv[(mv.dline == 0) & (mv.dprob < -0.01)]
print("  price->over:", blind(pu, "over", "follow(over)"), "\n            ", blind(pu, "under", "fade(under)"))
print("  price->under:", blind(pdn, "under", "follow(under)"), "\n            ", blind(pdn, "over", "fade(over)"))
print("  same-line price move at FIRST price (would need to bet early): ")
f = load_pairs("first"); fdev = dev_pairs(f)
ff = fdev.merge(mv[["player_id", "game_id", "market", "line", "dprob"]], on=["player_id", "game_id", "market", "line"])
print("    ", blind(ff[ff.dprob > 0.01], "over", "bet over at first price when later moves to over"))
print("    ", blind(ff[ff.dprob < -0.01], "under", "bet under at first price when later moves to under"))

print("\n=== Star overs (public bias?) by PTS line level, blind unders ===")
pts = d[d.market == "PTS"]
for lo, hi in [(0, 10), (10, 15), (15, 20), (20, 25), (25, 40)]:
    s = pts[(pts.line >= lo) & (pts.line < hi)]
    print("  PTS line [%d,%d):" % (lo, hi), blind(s, "under", "under"))
print("\n=== By market: blind under, odds buckets (juice on over side?) ===")
for mk in MARKETS:
    s = d[d.market == mk]
    print(f"  {mk}: over-implied mean={s.p_over_imp.mean():.3f}  P(over wins)={(s.res_over=='win').mean():.3f} (settled only: {(s.res_over=='win').sum()/(s.res_over.isin(['win','loss'])).sum():.3f})  void={(s.res_over=='void').mean():.3f}")
