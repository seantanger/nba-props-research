"""REUSED holdout (2025-26 .. 2026-10-23): injury-aware model vs phase-1 model. Same measurements as scripts/eval_inj.py.
The holdout archive timestamps lines ~1.5 min after tip, so the honest 'clean120' tier is empty here; tiers used:
'all' (every archived line) and 'close5' (both the line and FanDuel's line last seen within 5 min of tip; post-hoc tier from phase 1).
Usage: .venv/bin/python scripts/eval_inj_holdout.py -> logs/eval_inj_holdout.log"""
import sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common import RESULTS, make_bets, summarize, fmt
sys.path.insert(0, str(ROOT / "scripts"))
from eval_inj import accuracy, logloss, KEY, ONT

def close5(d):
    return d[d.line_fd.notna() & (d.mbt_fd.abs() <= 5) & (d.mins_before_tip.abs() <= 5)]

def main():
    log = open(ROOT / "logs" / "eval_inj_holdout.log", "w")
    def P(*a):
        print(*a); print(*a, file=log); log.flush()
    new = pd.read_parquet(RESULTS / "bp_eval_holdout_inj.parquet"); old = pd.read_parquet(RESULTS / "bp_eval_holdout.parquet")
    new, old = new[new.mu.notna()], old[old.mu.notna()]
    k = new[KEY].merge(old[KEY]).drop_duplicates()
    new = new.merge(k, on=KEY); old = old.merge(k, on=KEY)
    P(f"REUSED HOLDOUT aligned rows: new {len(new):,} old {len(old):,}; dates {new.game_date.min().date()}..{new.game_date.max().date()}")
    P("\n(a) accuracy vs FanDuel's line (FanDuel rows, settled)")
    pd.set_option("display.width", 250)
    P(pd.concat([accuracy(old, "phase1"), accuracy(new, "injury")]).sort_values(["market", "model"]).to_string(index=False))
    P("\n(b) model-only ROI, thr 0.08 (the dev-chosen threshold) and 0.05")
    for thr in [0.08, 0.05]:
        for lab, e in [("phase1", old), ("injury", new)]:
            s = e[e.played.notna()]
            for name, sub in [("FanDuel", s[s.book_id == 10]), ("Ontario(DK/MGM/BR)", s[s.book_id.isin(ONT)]), ("DraftKings", s[s.book_id == 12]),
                              ("BetMGM", s[s.book_id == 19]), ("BetRivers", s[s.book_id == 18])]:
                for tier, t in [("all", sub), ("close5", close5(sub) if name != "FanDuel" else sub[sub.mins_before_tip.abs() <= 5])]:
                    b = make_bets(t, t.p_proj, thr=thr)
                    r = summarize(b, label=f"{lab} model-only @{name} {tier} thr={thr}")
                    P(fmt(r) + (f"  under_frac={(b.side == 'under').mean():.2f}" if len(b) else ""))
    P("\n(c) chosen FanDuel-anchored rule (p_anc_fd thr .03) with each model's dispersion")
    for lab, e in [("phase1", old), ("injury", new)]:
        sub = e[(e.book_id != 10) & e.played.notna() & ~e.is_dfs]
        for tier, t in [("clean120 (pre-registered; nearly empty here)", sub[sub.line_fd.notna() & (sub.mins_before_tip >= 0) & (sub.mbt_fd >= 0) & (sub.mbt_fd <= 120) & sub.book_after_fd]),
                        ("close5 (post-hoc)", close5(sub)), ("all", sub)]:
            b = make_bets(t, t.p_anc_fd, thr=0.03)
            P(fmt(summarize(b, label=f"{lab} anchored {tier} sportsbooks")))
    log.close()

if __name__ == "__main__":
    main()
