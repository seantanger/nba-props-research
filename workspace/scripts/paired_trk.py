"""P2-10: paired comparison tracking vs baseline on the same FanDuel rows: log-loss and |miss| differences, bootstrap by game date."""
import sys, pathlib, numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common import RESULTS
which = sys.argv[1] if len(sys.argv) > 1 else "dev2"
KEY = ["event_id", "player_id", "game_id", "market", "book_id", "line"]
def load(tag):
    e = pd.read_parquet(RESULTS / f"bp_eval_{which}_{tag}.parquet"); e = e[e.mu.notna()]
    if which == "holdout": e = e[e.game_date <= "2026-10-23"]
    f = e[(e.book_id == 10) & (e.played == 1) & e.actual.notna() & e.line_fd.notna()].drop_duplicates(["player_id", "game_id", "market"], keep="last")
    return f[KEY + ["game_date", "mu", "actual", "p_proj"]]
a, b = load("base"), load("trk")
m = a.merge(b, on=KEY + ["game_date", "actual"], suffixes=("_b", "_t"))
m = m[m.actual != m.line]
y = (m.actual > m.line).astype(float)
ll = lambda p: -(y * np.log(np.clip(p, 1e-4, 1 - 1e-4)) + (1 - y) * np.log(1 - np.clip(p, 1e-4, 1 - 1e-4)))
m["d_ll"] = ll(m.p_proj_b) - ll(m.p_proj_t)           # >0 = tracking better
m["d_miss"] = (m.mu_b - m.actual).abs() - (m.mu_t - m.actual).abs()
rng = np.random.default_rng(0)
print(f"{which}: paired FanDuel rows n={len(m):,}")
for mk, g in [("ALL", m)] + list(m.groupby("market")):
    byd = g.groupby("game_date")[["d_ll", "d_miss"]].sum(); n = g.groupby("game_date").size().values
    boots = []
    for _ in range(2000):
        idx = rng.integers(0, len(byd), len(byd)); boots.append(byd.values[idx].sum(0) / n[idx].sum())
    boots = np.array(boots)
    print(f"  {mk:4s} n={len(g):6,d}  d_logloss={g.d_ll.mean()*1e4:+.2f}e-4 CI[{np.percentile(boots[:,0],2.5)*1e4:+.2f},{np.percentile(boots[:,0],97.5)*1e4:+.2f}]"
          f"  d_miss={g.d_miss.mean()*1e3:+.2f}e-3 CI[{np.percentile(boots[:,1],2.5)*1e3:+.2f},{np.percentile(boots[:,1],97.5)*1e3:+.2f}]  (positive = tracking better)")
