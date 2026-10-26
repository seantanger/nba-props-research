"""Price every BettingPros book line with the walk-forward projection distributions and evaluate betting rules.

Probabilities per row:
  p_proj   : P(actual > line) from the projection model alone (NegBin with model mu/var)
  p_anc_c  : market-anchored: model sigma, centre solved from the consensus line+price (Normal approx)
  p_anc_fd : same, centred on FanDuel's closing line+price
  p_stack  : walk-forward logistic regression (fit on seasons before the test season) on logits of the above + book implied
Usage: .venv/bin/python src/eval_bp.py dev|holdout|kambi
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from src.common import *
from src.models import nb_sf

lg = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))


def anchored(line, ref_line, ref_p_over, sigma):
    m = ref_line + sigma * stats.norm.ppf(np.clip(ref_p_over, 1e-3, 1 - 1e-3))
    return 1 - stats.norm.cdf((line - m) / sigma)


def build(which, tag=""):
    p = pd.read_parquet(RESULTS / "bp_pairs_cache.parquet")
    # 'kambi' = the 2023-03-13..2023-06-12 window excluded from dev from the start (secondary holdout); its distributions
    # come from the dev walk-forward run (cutoffs 2023-02-01 / 2023-04-01, trained only on earlier games).
    p = bp_holdout(p) if which == "holdout" else (p[p.in_kambi_holdout] if which == "kambi" else bp_dev(p))
    dist = pd.read_parquet(RESULTS / f"bp_dist_{'dev' if which == 'kambi' else which}{tag}.parquet").rename(columns={"PLAYER_ID": "player_id", "GAME_ID": "game_id"})
    p = p.merge(dist, on=["player_id", "game_id", "market"], how="inner")
    p["sigma"] = np.sqrt(p["var"])
    p["p_proj"] = nb_sf(p.line.values, p.mu.values, p["var"].values)
    p["p_anc_c"] = anchored(p.line, p.cons_line, p.cons_p_over, p.sigma)
    fd = p[p.book_id == 10][["event_id", "player_id", "market", "line", "p_over_imp", "updated", "mins_before_tip"]].rename(
        columns={"line": "line_fd", "p_over_imp": "p_fd", "updated": "updated_fd", "mins_before_tip": "mbt_fd"})
    p = p.merge(fd, on=["event_id", "player_id", "market"], how="left")
    p["p_anc_fd"] = anchored(p.line, p.line_fd, p.p_fd, p.sigma)
    p["gap_fd"] = p.line - p.line_fd
    p["book_after_fd"] = p.updated > p.updated_fd
    p["y"] = np.where(p.res_over == "win", 1.0, np.where(p.res_over == "loss", 0.0, np.nan))
    return p


def stack_fit(train, cols):
    ok = train.y.notna()
    for c in cols:
        ok &= train[c].notna()
    X = np.column_stack([lg(train.loc[ok, c]) for c in cols])
    return LogisticRegression(C=10).fit(X, train.y[ok])


def stack_predict(m, df, cols):
    ok = np.ones(len(df), bool)
    for c in cols:
        ok &= df[c].notna().values
    out = np.full(len(df), np.nan)
    out[ok] = m.predict_proba(np.column_stack([lg(df.loc[ok, c]) for c in cols]))[:, 1]
    return out


STACK_COLS = ["p_over_imp", "p_proj", "p_anc_c", "p_anc_fd"]


def tiers(b):
    return {"all": b, "fresh60": b[b.mins_before_tip <= 60], "after_fd": b[b.book_after_fd], "both15": b[(b.mins_before_tip <= 15) & (b.mbt_fd <= 15)]}


def one_per_selection(bets):
    return bets.sort_values("edge", ascending=False).drop_duplicates(["event_id", "player_id", "market"])


if __name__ == "__main__":
    which = sys.argv[1]
    tag = ("_" + sys.argv[2]) if len(sys.argv) > 2 else ""
    p = build(which, tag)
    p.to_parquet(RESULTS / f"bp_eval_{which}{tag}.parquet", index=False)
    d = p[p.bettable & (p.book_id != 10)]
    ok = d.y.notna()
    print("rows", len(d), "settled", ok.sum())
    print("log-loss vs book implied:")
    for c in ["p_over_imp", "p_proj", "p_anc_c", "p_anc_fd"]:
        m = ok & d[c].notna()
        print(f"  {c:10s} n={m.sum()} ll={log_loss(d.y[m], d[c][m].clip(0.01, 0.99)):.4f}")
