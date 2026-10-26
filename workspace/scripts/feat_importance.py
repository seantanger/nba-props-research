"""Gain-based importance of the injury features in one PTS/REB/AST mean model (cutoff 2024-10-01, i.e. trained on 2016-17..2023-24)."""
import sys, pathlib
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common import feature_cols
from src import models as M
M.LGB_PARAMS["num_threads"] = 2
pg = pd.read_parquet(ROOT / "data/processed/player_games.parquet")
cols = feature_cols(pg)
inj = [c for c in cols if c.startswith(("vac_", "ret_", "share_", "inj_", "avail_", "opp_vac_"))]
print(f"{len(cols)} features, {len(inj)} injury features")
train = pg[(pg.GAME_DATE < "2024-10-01") & (pg.GAME_DATE >= "2020-12-22") & (pg.MIN > 0)]
out = open(ROOT / "logs/feat_importance.log", "w")
for mk in ["PTS", "REB", "AST"]:
    m = lgb.train(M.LGB_PARAMS, lgb.Dataset(train[cols], train[mk]), num_boost_round=600)
    imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
    tot = imp.sum()
    s = f"{mk}: injury features share of gain = {imp[inj].sum() / tot:.3%}; top-25 overall:\n" + "\n".join(
        f"  {i + 1:2d}. {c:28s} {v / tot:7.3%}{'  <- injury' if c in inj else ''}" for i, (c, v) in enumerate(imp.head(25).items()))
    s += "\n  top injury features: " + ", ".join(f"{c} {v / tot:.3%}" for c, v in imp[inj].head(8).items())
    print(s); print(s, file=out); out.flush()
