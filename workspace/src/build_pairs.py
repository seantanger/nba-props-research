"""Build results/bp_pairs_cache.parquet (one row per event/player/market/book with both sides) from data/processed/bp_lines.parquet."""
from src.common import *
p = load_bp_pairs()
p.to_parquet(RESULTS / "bp_pairs_cache.parquet", index=False)
print("pairs", len(p), "voids", (p.res_over == "void").sum(), "holdout rows", p.is_holdout.sum())
