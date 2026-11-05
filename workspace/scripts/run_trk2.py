"""P2-11 variant runner: walk-forward mu/var for BettingPros keys with a per-market feature set.
Usage: .venv/bin/python scripts/run_trk2.py <variant> <foldset> [threads]
  variant : A B C S CV VAC MX PM DO MO ALL F (see VARIANTS)      foldset : screen | early | dev2 | holdout
Writes results/bp_dist_<foldset>_t2<variant>.parquet (same schema as src/run_bp.py so src/eval_bp.py / scripts/eval_trk.py work on dev2/holdout).
Model = production (src/models._fit_mean_disp: LightGBM poisson mean 600 rounds + log-residual dispersion 300 rounds, same params);
the only difference is which columns each market's models see. Baseline columns = feature_cols() of the PRODUCTION file's columns,
so opp_trk_* never leaks into the baseline through the ^opp_ pattern."""
import os, re, sys, time, pathlib
import numpy as np, pandas as pd, lightgbm as lgb
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from src.common import feature_cols, load_bp_pairs, bp_holdout, BP_MARKETS, RESULTS, PROC
from src.models import LGB_PARAMS

variant, foldset = sys.argv[1], sys.argv[2]
LGB_PARAMS["num_threads"] = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("LGB_THREADS", 8))
CUTS = {
    "screen": [("2021-01-01", "2021-03-01"), ("2022-02-01", "2022-04-01"), ("2022-12-01", "2023-02-01"), ("2023-12-01", "2024-02-01")],
    "early": [("2021-01-01", "2021-03-01"), ("2021-03-01", "2021-05-01"), ("2021-05-01", "2021-10-01"), ("2021-10-01", "2021-12-01"),
              ("2021-12-01", "2022-02-01"), ("2022-02-01", "2022-04-01"), ("2022-04-01", "2022-10-01"), ("2022-10-01", "2022-12-01"),
              ("2022-12-01", "2023-02-01"), ("2023-02-01", "2023-04-01"), ("2023-04-01", "2023-10-01"), ("2023-10-01", "2023-12-01"),
              ("2023-12-01", "2024-02-01"), ("2024-02-01", "2024-04-01"), ("2024-04-01", "2024-10-01")],
    "dev2": [("2024-10-01", "2024-12-01"), ("2024-12-01", "2025-02-01"), ("2025-02-01", "2025-04-01"), ("2025-04-01", "2025-09-01")],
    "holdout": [("2025-10-01", "2025-12-01"), ("2025-12-01", "2026-02-01"), ("2026-02-01", "2026-04-01"), ("2026-04-01", "2026-10-01"), ("2026-10-01", "2027-09-01")],
}[foldset]
G = {  # tracking column groups (regex over the new columns only)
    "core": r"^trk_(pot_ast|ast_pts|sec_ast|ft_ast|passes|oreb_ch|dreb_ch|reb_ch|reb_cont|touches|poss_time|paint_t|elbow_t|post_t|fc_touches)_m\d+$|^trk_(reb_cont_share|reb_per_ch|ast_per_pot|pts_per_touch)_r\d+$",
    "zone": r"^trk_(fga_(rim|paint|mid|c3|ab3|all)|fgm_rim)_m\d+$|^trk_(z_\w+|rim_fg)_r\d+$",
    "opp4": r"^opp_trk_(pot_ast|reb_ch|fga_rim|paint_t)_15$",
    "scor": r"^trk_(dr|cs|pu)_\w+_(m\d+|r\d+)$",
    "cv": r"^trk_cv_", "pm": r"^trk_pm_", "vac": r"^trk_vac_", "mx": r"^trk_mx_|^opp_trk_(drives|cs_fg3a|pu_fga)_15$",
}
SUB = {  # per-market subsets of C
    "AST": r"^trk_(pot_ast|ast_pts|sec_ast|ft_ast|passes|touches|poss_time|fc_touches|elbow_t|post_t|paint_t)_m\d+$|^trk_(ast_per_pot|pts_per_touch)_r\d+$|^trk_dr_(drives|pass|ast)_m\d+$|^opp_trk_pot_ast_15$",
    "REB": r"^trk_(oreb_ch|dreb_ch|reb_ch|reb_cont)_m\d+$|^trk_(reb_cont_share|reb_per_ch)_r\d+$|^opp_trk_reb_ch_15$",
    "PTS": r"^trk_(touches|poss_time|paint_t|elbow_t|post_t|fc_touches)_m\d+$|^trk_pts_per_touch_r\d+$|" + G["zone"] + r"|" + G["scor"] + r"|^opp_trk_(fga_rim|paint_t)_15$",
    "FG3M": r"^trk_(fga_(c3|ab3|all))_m\d+$|^trk_(z_\w+)_r\d+$|^trk_(cs|pu)_(fga|fg3a|fgm|share|fg3pct|fgpct)_(m\d+|r\d+)$",
}
SUB["PRA"] = "|".join([SUB["PTS"], SUB["REB"], SUB["AST"]]); SUB["PA"] = "|".join([SUB["PTS"], SUB["AST"]])
SUB["PR"] = "|".join([SUB["PTS"], SUB["REB"]]); SUB["RA"] = "|".join([SUB["REB"], SUB["AST"]]); SUB["STL"] = SUB["BLK"] = r"$^"
B = "|".join([G["core"], G["zone"], G["opp4"]]); C = "|".join([B, G["scor"]])
VARIANTS = {  # name -> (mean regex per market or str, disp regex per market or str)
    "A": (r"$^", r"$^"), "B": (B, B), "C": (C, C), "S": (SUB, SUB),
    "CV": ("|".join([C, G["cv"]]),) * 2, "VAC": ("|".join([C, G["vac"]]),) * 2, "MX": ("|".join([C, G["mx"]]),) * 2,
    "PM": ("|".join([C, G["pm"]]),) * 2, "DO": (r"$^", C), "MO": (C, r"$^"),
    "ALL": ("|".join([C, G["cv"], G["vac"], G["mx"], G["pm"]]),) * 2,
    "F": None,  # filled in after screening (see JOURNAL P2-11)
}
F_EXTRA = os.environ.get("F_GROUPS", "")  # e.g. "cv,vac" -> F = C + those groups
if variant == "F":
    VARIANTS["F"] = ("|".join([C] + [G[k] for k in F_EXTRA.split(",") if k]),) * 2


def rx(spec, mk):
    return re.compile(spec[mk] if isinstance(spec, dict) else spec)


pg = pd.read_parquet(PROC / "player_games_trk2.parquet")
base_cols = feature_cols(pd.read_parquet(PROC / "player_games.parquet").head(50))
trk_cols = [c for c in pg.columns if c.startswith("trk_") or c.startswith("opp_trk_")]
assert not set(base_cols) & set(trk_cols)
pairs = load_bp_pairs()
pairs = bp_holdout(pairs) if foldset == "holdout" else pairs[~pairs.is_holdout]
keys_all = pairs[["player_id", "game_id", "market", "game_date"]].drop_duplicates(["player_id", "game_id", "market"]).rename(columns={"player_id": "PLAYER_ID", "game_id": "GAME_ID"})
mspec, dspec = VARIANTS[variant]
out, t0 = [], time.time()
for c0, c1 in CUTS:
    k = keys_all[(keys_all.game_date >= c0) & (keys_all.game_date < c1)]
    if len(k) == 0:
        continue
    train = pg[(pg.GAME_DATE < pd.Timestamp(c0)) & (pg.MIN > 0)]
    for mk in BP_MARKETS:
        kk = k[k.market == mk][["PLAYER_ID", "GAME_ID"]].drop_duplicates()
        if len(kk) == 0:
            continue
        cm = base_cols + [c for c in trk_cols if rx(mspec, mk).search(c)]
        cd = base_cols + [c for c in trk_cols if rx(dspec, mk).search(c)]
        X = kk.merge(pg[["PLAYER_ID", "GAME_ID"] + sorted(set(cm) | set(cd))], on=["PLAYER_ID", "GAME_ID"], how="inner")
        m = lgb.train(LGB_PARAMS, lgb.Dataset(train[cm], train[mk]), num_boost_round=600)
        r2 = (train[mk].values - m.predict(train[cm])) ** 2
        d = lgb.train(dict(LGB_PARAMS, objective="regression", learning_rate=0.03, num_leaves=15, min_data_in_leaf=200),
                      lgb.Dataset(train[cd], np.log(r2 + 0.25)), num_boost_round=300)
        sc = r2.mean() / np.exp(d.predict(train[cd])).mean()
        out.append(pd.DataFrame({"PLAYER_ID": X.PLAYER_ID.values, "GAME_ID": X.GAME_ID.values, "market": mk, "mu": m.predict(X[cm]),
                                 "var": np.exp(d.predict(X[cd])) * sc, "cutoff": c0}))
        if mk == "PTS":
            print(f"  {c0} {mk}: train={len(train):,} cols mean={len(cm)} disp={len(cd)}", flush=True)
    print(f"fold {c0}..{c1}: keys={len(k)} elapsed={time.time()-t0:.0f}s", flush=True)
res = pd.concat(out, ignore_index=True)
res.to_parquet(RESULTS / f"bp_dist_{foldset}_t2{variant}.parquet", index=False)
print("saved", len(res), f"results/bp_dist_{foldset}_t2{variant}.parquet", flush=True)
