"""Walk-forward projection distributions for the BettingPros pairs.
Usage: .venv/bin/python src/run_bp.py dev [tag]  -> results/bp_dist_dev[_tag].parquet (env FEATURE_EXTRA/FEATURE_EXCL select features)   (seasons 2020-21..2024-25)
       .venv/bin/python src/run_bp.py holdout  -> results/bp_dist_holdout.parquet (2025-26 and later; run ONCE at the end)
Each fold trains on game logs strictly before the fold's cutoff date and predicts games in [cutoff, next cutoff)."""
import os, sys, time
import numpy as np, pandas as pd
from src.common import *
from src.models import proj_dist, LGB_PARAMS

which = sys.argv[1]
tag = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""  # e.g. "inj" -> results/bp_dist_dev_inj.parquet
LGB_PARAMS["num_threads"] = int(os.environ.get("LGB_THREADS", 12))
cutoffs = {
    "dev": ["2021-01-01", "2021-03-01", "2021-05-01", "2021-10-01", "2021-12-01", "2022-02-01", "2022-04-01", "2022-10-01", "2022-12-01",
            "2023-02-01", "2023-04-01", "2023-10-01", "2023-12-01", "2024-02-01", "2024-04-01", "2024-10-01", "2024-12-01", "2025-02-01", "2025-04-01", "2025-09-01"],
    "dev2": ["2024-10-01", "2024-12-01", "2025-02-01", "2025-04-01", "2025-09-01"],
    "holdout": ["2025-10-01", "2025-12-01", "2026-02-01", "2026-04-01", "2026-10-01", "2027-09-01"],
}[which]
pg = load_player_games()
pairs = load_bp_pairs()
pairs = bp_holdout(pairs) if which == "holdout" else pairs[~pairs.is_holdout]
if which == "dev2":
    pairs = pairs[pairs.game_date >= "2024-10-01"]
keys_all = pairs[["player_id", "game_id", "market", "game_date"]].drop_duplicates(["player_id", "game_id", "market"]).rename(columns={"player_id": "PLAYER_ID", "game_id": "GAME_ID"})
out = []
t0 = time.time()
for c0, c1 in zip(cutoffs[:-1], cutoffs[1:]):
    k = keys_all[(keys_all.game_date >= c0) & (keys_all.game_date < c1)]
    if len(k) == 0:
        continue
    d = proj_dist(pg, k, pd.Timestamp(c0), BP_MARKETS)
    d["cutoff"] = c0
    out.append(d)
    print(f"fold {c0}..{c1}: keys={len(k)} got={len(d)} elapsed={time.time()-t0:.0f}s", flush=True)
res = pd.concat(out, ignore_index=True)
res.to_parquet(RESULTS / f"bp_dist_{which}{tag}.parquet", index=False)
print("saved", len(res))
