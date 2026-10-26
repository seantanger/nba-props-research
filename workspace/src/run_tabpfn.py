"""TabPFN (tabular foundation model) as a distributional projection model, evaluated walk-forward on the Kambi DEV pairs.
Runs in .venv-tabpfn (torch 2.2 / numpy<2). Train rows: last `n_train` player-games before cutoff per market; features: top-k by
LightGBM gain (saved by run_dev)."""
import sys, time, json
import numpy as np, pandas as pd, torch
from tabpfn import TabPFNRegressor
sys.path.insert(0, ".")
from src.common import *

n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
k_feat = int(sys.argv[2]) if len(sys.argv) > 2 else 40
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
pg = load_player_games()
pairs = load_pairs("last"); dev = dev_pairs(pairs).copy()
top = json.load(open(RESULTS / "feature_gain.json"))
out = np.full(len(dev), np.nan); mu_all = np.full(len(dev), np.nan)
t0 = time.time()
for train_end, ts, te in DEV_FOLDS:
    cutoff = pd.Timestamp(train_end)
    te_mask = ((dev.game_date >= ts) & (dev.game_date < te)).values
    test = dev[te_mask]
    for mk in MARKETS:
        cols = top[mk][:k_feat]
        tr = pg[(pg.GAME_DATE < cutoff) & (pg.MIN > 0)].sort_values("GAME_DATE").tail(n_train)
        X = test[test.market == mk].merge(pg[["PLAYER_ID", "GAME_ID"] + cols], left_on=["player_id", "game_id"], right_on=["PLAYER_ID", "GAME_ID"], how="left")
        ok = X.PLAYER_ID.notna().values
        if ok.sum() == 0:
            continue
        m = TabPFNRegressor(device="cpu", n_estimators=4, random_state=0, ignore_pretraining_limits=True)
        m.fit(tr[cols].values.astype(np.float32), tr[mk].values.astype(np.float32))
        full = m.predict(X.loc[ok, cols].values.astype(np.float32), output_type="full")
        crit, logits = full["criterion"], full["logits"]
        ys = torch.tensor(X.loc[ok, "line"].values, dtype=torch.float32)
        cdf = np.diag(crit.cdf(logits, ys).detach().numpy())
        p_over = 1 - cdf
        idx = np.where(te_mask)[0][(test.market == mk).values][ok]
        out[idx] = p_over; mu_all[idx] = full["mean"]
        print(f"{ts} {mk}: train={len(tr)} test={ok.sum()} elapsed={time.time()-t0:.0f}s", flush=True)
dev["p_tabpfn"] = out; dev["mu_tabpfn"] = mu_all
dev[["player_id", "game_id", "market", "line", "p_tabpfn", "mu_tabpfn"]].to_parquet(RESULTS / "dev_preds_tabpfn.parquet", index=False)
from sklearn.metrics import log_loss
y = (dev.res_over == "win").astype(float); okk = dev.res_over.isin(["win", "loss"]) & dev.p_tabpfn.notna()
print("tabpfn logloss", log_loss(y[okk], dev.p_tabpfn[okk].clip(0.01, 0.99)), "market", log_loss(y[okk], dev.p_over_imp[okk]), "n", okk.sum())
for thr in [0.02, 0.04, 0.06]:
    print(fmt(summarize(make_bets(dev[okk], dev.p_tabpfn[okk], thr=thr), n_boot=300, label=f"tabpfn thr={thr}")))
