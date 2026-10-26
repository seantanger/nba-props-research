"""Post-hoc analysis on saved dev predictions: stacking with market, under bias, CLV proxy."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from src.common import *

snap = sys.argv[1] if len(sys.argv) > 1 else "last"
d = pd.read_parquet(RESULTS / f"dev_preds_{snap}.parquet")
ok = d.y.notna() & d.p_proj.notna()
d = d[ok].copy()
lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
d["l_imp"] = lg(d.p_over_imp); d["l_proj"] = lg(d.p_proj); d["l_eb"] = lg(d.p_eb); d["l_clf"] = lg(d.p_clf)

print("=== Stacking: walk-forward logistic on logits (fit on earlier folds, test on later) ===")
d["p_stack"] = np.nan
for i, (tr_end, ts, te) in enumerate(DEV_FOLDS):
    if i == 0:
        continue
    trm = (d.game_date < ts); tem = (d.game_date >= ts) & (d.game_date < te)
    for cols in [["l_imp"], ["l_imp", "l_proj"], ["l_imp", "l_proj", "l_eb"], ["l_imp", "l_proj", "l_eb", "l_clf"]]:
        m = LogisticRegression(C=100).fit(d.loc[trm, cols], d.y[trm])
        p = m.predict_proba(d.loc[tem, cols])[:, 1]
        print(f"  fold {ts}: {cols} coef={np.round(m.coef_[0],3)} test logloss={log_loss(d.y[tem], p):.4f} (market {log_loss(d.y[tem], d.p_over_imp[tem]):.4f})")
        if cols == ["l_imp", "l_proj", "l_eb"]:
            d.loc[tem, "p_stack"] = p

print("\n=== Under bias by market x side-favourite (dev, blind) ===")
for mk in MARKETS:
    s = d[d.market == mk]
    for lab, sel in [("under all", s.index), ("under when under odds>=1.9", s.index[s.odds_under >= 1.9]), ("under when under odds<1.9", s.index[s.odds_under < 1.9])]:
        b = s.loc[sel].copy(); b["side"] = "under"; b["odds"] = b.odds_under; b["result"] = b.res_under; b["stake"] = 1.0; b["pnl"] = grade(b)
        print("  ", fmt(summarize(b, n_boot=300, label=f"{mk} {lab}")))

print("\n=== Stacked model bets (folds 2-3 only) ===")
s = d[d.p_stack.notna()]
for thr in [0.01, 0.02, 0.03, 0.04]:
    b = make_bets(s, s.p_stack, thr=thr)
    print("  ", fmt(summarize(b, n_boot=300, label=f"stack thr={thr}")))
    for side in ["over", "under"]:
        print("      ", fmt(summarize(b[b.side == side], n_boot=300, label=f"side={side}")))

print("\n=== CLV proxy: does model edge at this snapshot predict first->last movement? ===")
# only pairs where the first-snapshot line is the same as this one
m = d[d.line_first == d.line].copy()
m["move"] = m.p_over_imp - m.p_over_imp_first  # positive = market moved toward over
m["edge_over_model"] = m.p_proj - m.p_over_imp_first
print("  n same-line pairs:", len(m), " corr(edge_over_model_vs_first, move):", np.corrcoef(m.edge_over_model, m.move)[0, 1].round(4))
m["q"] = pd.qcut(m.edge_over_model, 5, labels=False)
print(m.groupby("q").agg(n=("move", "size"), edge=("edge_over_model", "mean"), move=("move", "mean"), y=("y", "mean")).round(4))
