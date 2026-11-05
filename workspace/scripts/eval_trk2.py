"""P2-11 paired evaluation of variant distributions on FanDuel rows.
Usage: .venv/bin/python scripts/eval_trk2.py <foldset> <ref> <variant> [<variant> ...]
Rows: FanDuel (book 10), played, actual known, one per player-game-market (last snapshot), pushes dropped; holdout limited to
game_date <= 2026-10-23. Per market + pooled: log-loss vs actual (NegBin P(over) from mu/var), |mu - actual|, right side of the
FanDuel line; paired differences vs <ref> with 2,000 bootstrap resamples by game date. Positive d = variant better."""
import sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common import RESULTS, load_bp_pairs, bp_holdout
from src.models import nb_sf
foldset, ref, variants = sys.argv[1], sys.argv[2], sys.argv[3:]
pairs = load_bp_pairs()
pairs = bp_holdout(pairs) if foldset == "holdout" else pairs[~pairs.is_holdout]
if foldset == "holdout":
    pairs = pairs[pairs.game_date <= "2026-10-23"]
fd = pairs[(pairs.book_id == 10) & (pairs.played == 1) & pairs.actual.notna()].sort_values("updated").drop_duplicates(["player_id", "game_id", "market"], keep="last")
fd = fd[fd.actual != fd.line][["player_id", "game_id", "market", "game_date", "line", "actual", "p_over_imp"]]


def load(v):
    d = pd.read_parquet(RESULTS / f"bp_dist_{foldset}_t2{v}.parquet").rename(columns={"PLAYER_ID": "player_id", "GAME_ID": "game_id"})
    d = d[d.mu.notna()].drop_duplicates(["player_id", "game_id", "market"])
    m = fd.merge(d, on=["player_id", "game_id", "market"], how="inner")
    y = (m.actual > m.line).astype(float)
    p = np.clip(nb_sf(m.line.values, m.mu.values, m["var"].values), 1e-4, 1 - 1e-4)
    m["ll"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    m["miss"] = (m.mu - m.actual).abs()
    m["side"] = ((m.mu > m.line) == (m.actual > m.line)).astype(float)
    m["ll_fd"] = -(y * np.log(np.clip(m.p_over_imp, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(m.p_over_imp, 1e-4, 1 - 1e-4)))
    return m[["player_id", "game_id", "market", "game_date", "line", "actual", "ll", "miss", "side", "ll_fd"]]


def boot(g, cols, n=2000, seed=0):
    byd = g.groupby("game_date")[cols].sum(); cnt = g.groupby("game_date").size().values
    rng = np.random.default_rng(seed); b = []
    for _ in range(n):
        i = rng.integers(0, len(byd), len(byd)); b.append(byd.values[i].sum(0) / cnt[i].sum())
    b = np.array(b)
    return np.percentile(b, 2.5, axis=0), np.percentile(b, 97.5, axis=0)


R = load(ref)
print(f"{foldset}: ref={ref} FanDuel rows n={len(R):,}  games dates={R.game_date.nunique()}  "
      f"logloss={R.ll.mean():.4f} (FD price {R.ll_fd.mean():.4f}) miss={R.miss.mean():.3f} side={R.side.mean():.4f}")
print("  per market (ref):", "  ".join(f"{mk}:n={len(g):,} ll={g.ll.mean():.4f} miss={g.miss.mean():.3f} side={g.side.mean():.3f}" for mk, g in R.groupby("market")))
MK = ["ALL", "PTS", "REB", "AST", "FG3M", "PRA", "PA", "PR", "RA", "STL", "BLK"]
for v in variants:
    V = load(v)
    m = R.merge(V, on=["player_id", "game_id", "market", "game_date", "line", "actual"], suffixes=("_r", "_v"))
    m["d_ll"] = m.ll_r - m.ll_v; m["d_miss"] = m.miss_r - m.miss_v; m["d_side"] = m.side_v - m.side_r
    print(f"\n== {v} vs {ref}  paired n={len(m):,}  (positive = {v} better)")
    for mk in MK:
        g = m if mk == "ALL" else m[m.market == mk]
        if len(g) == 0:
            continue
        lo, hi = boot(g, ["d_ll", "d_miss", "d_side"])
        flag = "+" if lo[0] > 0 else ("-" if hi[0] < 0 else " ")
        print(f"  {mk:4s} n={len(g):7,d} d_ll={g.d_ll.mean()*1e4:+6.2f}e-4 [{lo[0]*1e4:+6.2f},{hi[0]*1e4:+6.2f}]{flag} "
              f"d_miss={g.d_miss.mean()*1e3:+6.2f}e-3 [{lo[1]*1e3:+6.2f},{hi[1]*1e3:+6.2f}] "
              f"d_side={g.d_side.mean()*1e3:+6.2f}e-3 [{lo[2]*1e3:+6.2f},{hi[2]*1e3:+6.2f}]  ll={g.ll_v.mean():.4f} miss={g.miss_v.mean():.3f} side={g.side_v.mean():.4f}")
