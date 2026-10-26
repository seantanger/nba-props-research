"""Walk-forward evaluation on the DEV period only (holdout untouched).
Produces out-of-fold P(over) per model for every dev pair -> results/dev_preds.parquet, and prints
calibration + strategy grids. Usage: .venv/bin/python src/run_dev.py [snapshot]"""
import sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from src.common import *
from src.models import proj_lgbm, eb_rate, clf_lgbm

snapshot = sys.argv[1] if len(sys.argv) > 1 else "last"
pg = load_player_games()
pairs = attach_first_snapshot(load_pairs(snapshot))
dev = dev_pairs(pairs).copy()
dev["y"] = (dev.res_over == "win").astype(float)
dev.loc[~dev.res_over.isin(["win", "loss"]), "y"] = np.nan

preds = {}
t0 = time.time()
for name in ["proj", "eb", "clf"]:
    preds[name] = np.full(len(dev), np.nan)
mu_all = np.full(len(dev), np.nan); var_all = np.full(len(dev), np.nan)
for train_end, ts, te in DEV_FOLDS:
    te_mask = ((dev.game_date >= ts) & (dev.game_date < te)).values
    test = dev[te_mask]
    cutoff = pd.Timestamp(train_end)
    p, mu, var = proj_lgbm(pg, test, cutoff)
    preds["proj"][te_mask] = p; mu_all[te_mask] = mu; var_all[te_mask] = var
    preds["eb"][te_mask] = eb_rate(pg, test, cutoff)
    tr = dev[(dev.game_date < cutoff)]
    preds["clf"][te_mask] = clf_lgbm(pg, tr, test)[0]
    print(f"fold {ts}..{te}: n={te_mask.sum()} train_pairs={len(tr)} elapsed={time.time()-t0:.0f}s")

for k, v in preds.items():
    dev[f"p_{k}"] = v
dev["mu_proj"] = mu_all; dev["var_proj"] = var_all
dev.to_parquet(RESULTS / f"dev_preds_{snapshot}.parquet", index=False)

# ---- probabilistic quality vs market
ok = dev.y.notna()
print("\nLog-loss / Brier of P(over) (lower is better), dev out-of-fold:")
for k in ["p_over_imp", "p_proj", "p_eb", "p_clf"]:
    m = ok & dev[k].notna()
    pr = dev.loc[m, k].clip(0.01, 0.99)
    print(f"  {k:12s} n={m.sum()} logloss={log_loss(dev.y[m], pr):.4f} brier={brier_score_loss(dev.y[m], pr):.4f}")

# ---- strategy grids (flat 1u)
print("\nStrategy grid (dev, flat stakes):")
for k in ["p_proj", "p_eb", "p_clf"]:
    for w in [1.0, 0.5]:
        pm = w * dev[k] + (1 - w) * dev.p_over_imp
        for thr in [0.02, 0.04, 0.06, 0.08]:
            b = make_bets(dev, pm, thr=thr)
            print("  ", fmt(summarize(b, n_boot=300, label=f"{k} w={w} thr={thr}")))
            if thr == 0.04:
                for side in ["over", "under"]:
                    print("      ", fmt(summarize(b[b.side == side], n_boot=300, label=f"side={side}")))
                for mk in MARKETS:
                    print("      ", fmt(summarize(b[b.market == mk], n_boot=300, label=f"mk={mk}")))
