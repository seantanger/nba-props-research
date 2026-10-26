#!/bin/bash
# After run_bp.py dev2: merge corrected 2024-25 distributions into bp_dist_dev, re-price dev rows, re-fit stack, run rule grid.
cd "$(dirname "$0")/.." || exit 1
.venv/bin/python - <<'PY'
import pandas as pd
from src.common import RESULTS
a = pd.read_parquet(RESULTS / "bp_dist_dev.parquet"); b = pd.read_parquet(RESULTS / "bp_dist_dev2.parquet")
k = ["PLAYER_ID", "GAME_ID", "market"]
a = a.merge(b[k].drop_duplicates().assign(_x=1), on=k, how="left"); a = a[a._x.isna()].drop(columns="_x")
m = pd.concat([a, b], ignore_index=True)
assert not m.duplicated(k).any()
m.to_parquet(RESULTS / "bp_dist_dev.parquet", index=False)
print("merged", len(a), "+", len(b), "=", len(m), m.cutoff.value_counts().sort_index().tail(6).to_dict())
PY
.venv/bin/python src/eval_bp.py dev && .venv/bin/python src/grid_bp.py > logs/grid_bp2.log 2>&1 && .venv/bin/python src/select_rule.py > logs/select_rule.log 2>&1
echo POST_DEV2_DONE
