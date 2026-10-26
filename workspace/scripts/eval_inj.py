"""P2-5: does the injury-aware projection model (tag `inj`) beat the phase-1 model (no tag) on DEV?
(a) accuracy vs FanDuel's line on FanDuel rows (avg miss, correct side, log-loss of the over probability)
(b) model-only ROI (p_proj vs each book's price) at FanDuel and at the Ontario books (DK 12, BetMGM 19, BetRivers 18)
(c) the chosen FanDuel-anchored rule (p_anc_fd, clean120, thr .03) with each model's dispersion
Usage: .venv/bin/python scripts/eval_inj.py [tag_new=inj] [tag_old=]  -> logs/eval_inj.log
"""
import sys, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common import RESULTS, make_bets, summarize, fmt, bp_dev
from src.select_rule import tier_rows

tag_new = sys.argv[1] if len(sys.argv) > 1 else "inj"
tag_old = sys.argv[2] if len(sys.argv) > 2 else ""
ONT = {12: "DraftKings", 19: "BetMGM", 18: "BetRivers"}
KEY = ["event_id", "player_id", "game_id", "market", "book_id", "line"]


def load(tag):
    f = RESULTS / f"bp_eval_dev{('_' + tag) if tag else ''}.parquet"
    e = pd.read_parquet(f)
    e = e[e.mu.notna()]
    return e


def logloss(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def accuracy(e, label):
    """FanDuel rows (book 10), settled, played: model mu vs FD line."""
    f = e[(e.book_id == 10) & (e.played == 1) & e.actual.notna() & e.line_fd.notna()].drop_duplicates(["player_id", "game_id", "market"], keep="last")
    out = []
    for mk, g in [("ALL", f)] + list(f.groupby("market")):
        miss_m, miss_l = (g.mu - g.actual).abs().mean(), (g.line - g.actual).abs().mean()
        nt = g[g.actual != g.line]
        side_m = (np.sign(nt.mu - nt.line) == np.sign(nt.actual - nt.line)).mean()
        side_fd = (np.sign(nt.p_over_imp - 0.5) == np.sign(nt.actual - nt.line)).mean()
        yy = (nt.actual > nt.line).astype(float)
        out.append(dict(model=label, market=mk, n=len(g), miss_model=round(miss_m, 3), miss_fd_line=round(miss_l, 3),
                        side_model=round(side_m, 4), side_fd=round(side_fd, 4), ll_proj=round(logloss(nt.p_proj, yy), 4),
                        ll_fd=round(logloss(nt.p_over_imp, yy), 4), ll_anc=round(logloss(nt.p_anc_fd, yy), 4),
                        bias_model=round((g.mu - g.actual).mean(), 3), bias_fd=round((g.line - g.actual).mean(), 3)))
    return pd.DataFrame(out)


def model_only(e, label, thr):
    rows = []
    settled = e[e.played.notna()]
    for name, sub in [("FanDuel", settled[settled.book_id == 10]), ("Ontario(DK/MGM/BR)", settled[settled.book_id.isin(ONT)]),
                      ("DraftKings", settled[settled.book_id == 12]), ("BetMGM", settled[settled.book_id == 19]), ("BetRivers", settled[settled.book_id == 18])]:
        for tier in ["all", "clean120"]:
            s = tier_rows(sub, tier) if tier != "all" else sub[sub.mins_before_tip >= 0]
            if tier == "clean120" and name == "FanDuel":
                s = sub[(sub.mins_before_tip >= 0) & (sub.mins_before_tip <= 120)]
            if len(s) == 0:
                continue
            b = make_bets(s, s.p_proj, thr=thr)
            r = summarize(b, label=f"{label} model-only @{name} {tier} thr={thr}")
            r["n_offers"] = len(s); r["bet_frac"] = len(b) / max(len(s), 1)
            r["under_frac"] = (b.side == "under").mean() if len(b) else np.nan
            rows.append(r)
    return rows


def anchored(e, label):
    rows = []
    sub = e[(e.book_id != 10) & e.played.notna() & ~e.is_dfs]
    for tier in ["clean120", "all"]:
        s = tier_rows(sub, tier)
        b = make_bets(s, s.p_anc_fd, thr=0.03)
        rows.append(summarize(b, label=f"{label} anchored p_anc_fd {tier} thr=.03 (sportsbooks)"))
    return rows


def main():
    log = open(ROOT / "logs" / "eval_inj.log", "w")
    def P(*a):
        print(*a); print(*a, file=log); log.flush()
    new, old = load(tag_new), load(tag_old)
    # align on the same rows so the comparison is like for like
    k = new[KEY].merge(old[KEY]).drop_duplicates()
    new = new.merge(k, on=KEY); old = old.merge(k, on=KEY)
    P(f"aligned rows: new {len(new):,} old {len(old):,}; dev dates {new.game_date.min().date()}..{new.game_date.max().date()}")
    P("\n(a) accuracy vs FanDuel's line (FanDuel rows, settled, one per player/game/market)")
    acc = pd.concat([accuracy(old, "phase1"), accuracy(new, "injury")]).sort_values(["market", "model"])
    pd.set_option("display.width", 250)
    P(acc.to_string(index=False))
    P("\n(b) model-only ROI (p_proj vs the book's price), flat 1u, odds 1.3-3.0")
    for thr in [0.03, 0.05, 0.08]:
        for lab, e in [("phase1", old), ("injury", new)]:
            for r in model_only(e, lab, thr):
                P(fmt(r) + f"  offers={r['n_offers']:,} bet_frac={r['bet_frac']:.3f} under_frac={r['under_frac']:.2f}")
    P("\n(c) chosen FanDuel-anchored rule with each model's dispersion")
    for lab, e in [("phase1", old), ("injury", new)]:
        for r in anchored(e, lab):
            P(fmt(r))
    # (d) where do the models disagree most? injury-driven rows
    P("\n(d) rows where injury context exists (vac_min >= 20 on the player's team) — model-only at FanDuel, thr .05")
    pg = pd.read_parquet(ROOT / "data" / "processed" / "player_games.parquet", columns=["PLAYER_ID", "GAME_ID", "vac_min", "vac_starters", "inj_self_code"])
    pg = pg.rename(columns={"PLAYER_ID": "player_id", "GAME_ID": "game_id"})
    pg["player_id"] = pg.player_id.astype("int64")
    for lab, e in [("phase1", old), ("injury", new)]:
        e2 = e.merge(pg, on=["player_id", "game_id"], how="left")
        for cond, cname in [(e2.vac_min >= 20, "vac_min>=20"), (e2.vac_min < 20, "vac_min<20"), (e2.inj_self_code >= 2, "self Q/D")]:
            s = e2[cond & (e2.book_id == 10) & e2.played.notna() & (e2.mins_before_tip >= 0)]
            b = make_bets(s, s.p_proj, thr=0.05)
            P(fmt(summarize(b, label=f"{lab} FD model-only {cname}")))
            f = s[(s.played == 1) & s.actual.notna()].drop_duplicates(["player_id", "game_id", "market"], keep="last")
            nt = f[f.actual != f.line]
            P(f"    acc {cname}: n={len(f):,} miss_model={(f.mu - f.actual).abs().mean():.3f} miss_fd={(f.line - f.actual).abs().mean():.3f} "
              f"side_model={(np.sign(nt.mu - nt.line) == np.sign(nt.actual - nt.line)).mean():.4f} side_fd={(np.sign(nt.p_over_imp - .5) == np.sign(nt.actual - nt.line)).mean():.4f} "
              f"bias_model={(f.mu - f.actual).mean():+.3f} bias_fd={(f.line - f.actual).mean():+.3f}")
    log.close()


if __name__ == "__main__":
    main()
