"""P2-12 check: walk-forward AST mu/var with the production baseline columns + tracking group B normalised per season
(src/tracking_live.py). Same folds, model and params as scripts/run_trk2.py; only the AST market is fitted.
Usage: .venv/bin/python scripts/run_astn.py <early|dev2> [threads]
Writes results/bp_dist_<foldset>_t2BN.parquet (AST rows only) for scripts/eval_trk2.py <foldset> A BN."""
import os, sys, time, pathlib
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common import feature_cols, load_bp_pairs, RESULTS, PROC
from src.models import LGB_PARAMS
from src.tracking_live import add_trk, trk_cols

foldset = sys.argv[1]
LGB_PARAMS["num_threads"] = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get("LGB_THREADS", 8))
CUTS = {
    "early": [("2021-01-01", "2021-03-01"), ("2021-03-01", "2021-05-01"), ("2021-05-01", "2021-10-01"), ("2021-10-01", "2021-12-01"),
              ("2021-12-01", "2022-02-01"), ("2022-02-01", "2022-04-01"), ("2022-04-01", "2022-10-01"), ("2022-10-01", "2022-12-01"),
              ("2022-12-01", "2023-02-01"), ("2023-02-01", "2023-04-01"), ("2023-04-01", "2023-10-01"), ("2023-10-01", "2023-12-01"),
              ("2023-12-01", "2024-02-01"), ("2024-02-01", "2024-04-01"), ("2024-04-01", "2024-10-01")],
    "dev2": [("2024-10-01", "2024-12-01"), ("2024-12-01", "2025-02-01"), ("2025-02-01", "2025-04-01"), ("2025-04-01", "2025-09-01")],
}[foldset]

t0 = time.time()
pg = pd.read_parquet(PROC / "player_games.parquet")
base = feature_cols(pg)
pg = add_trk(pg)
tc = trk_cols(pg)
print(f"features built in {time.time() - t0:.0f}s: base={len(base)} trk={len(tc)}", flush=True)
cols = base + tc
pairs = load_bp_pairs()
pairs = pairs[~pairs.is_holdout & (pairs.market == "AST")]
keys = pairs[["player_id", "game_id", "game_date"]].drop_duplicates(["player_id", "game_id"]).rename(columns={"player_id": "PLAYER_ID", "game_id": "GAME_ID"})
out = []
for c0, c1 in CUTS:
    k = keys[(keys.game_date >= c0) & (keys.game_date < c1)][["PLAYER_ID", "GAME_ID"]]
    if len(k) == 0:
        continue
    train = pg[(pg.GAME_DATE < pd.Timestamp(c0)) & (pg.MIN > 0)]
    X = k.merge(pg[["PLAYER_ID", "GAME_ID"] + cols], on=["PLAYER_ID", "GAME_ID"], how="inner")
    m = lgb.train(LGB_PARAMS, lgb.Dataset(train[cols], train["AST"]), num_boost_round=600)
    r2 = (train["AST"].values - m.predict(train[cols])) ** 2
    d = lgb.train(dict(LGB_PARAMS, objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200),
                  lgb.Dataset(train[cols], np.log(r2 + 0.25)), num_boost_round=300)
    sc = r2.mean() / np.exp(d.predict(train[cols])).mean()
    out.append(pd.DataFrame({"PLAYER_ID": X.PLAYER_ID.values, "GAME_ID": X.GAME_ID.values, "market": "AST", "mu": m.predict(X[cols]),
                             "var": np.exp(d.predict(X[cols])) * sc, "cutoff": c0}))
    print(f"fold {c0}..{c1}: keys={len(k)} train={len(train):,} elapsed={time.time() - t0:.0f}s", flush=True)
res = pd.concat(out, ignore_index=True)
res.to_parquet(RESULTS / f"bp_dist_{foldset}_t2BN.parquet", index=False)
print("saved", len(res), f"results/bp_dist_{foldset}_t2BN.parquet", flush=True)
