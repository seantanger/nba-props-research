"""P2-11: tracking feature groups on FULL history (2016-17 onward) built on the roll-fixed production file.

Reads data/processed/player_games.parquet (production, untouched) + tracking_player_games.parquet + tracking_shots.parquet
(src/fetch_tracking.py --build, now incl. drives / catch-and-shoot / pull-up) and writes player_games_trk2.parquet with ALL
candidate columns. Variants pick subsets by regex in scripts/run_trk2.py (nothing here is used by production code).
Every column starts with trk_ or opp_trk_ so the production FEATURE_PAT never picks it up by accident except opp_trk_ (handled by
the runner, which derives the baseline column list from the production file). All series are shift(1) WITHIN player / team.

Groups (prefix -> content):
  core  trk_{pot_ast,ast_pts,sec_ast,ft_ast,passes,oreb_ch,dreb_ch,reb_ch,reb_cont,touches,poss_time,paint_t,elbow_t,post_t,fc_touches}_m{5,10,20}
        + ratios trk_{reb_cont_share,reb_per_ch,ast_per_pot,pts_per_touch}_r{10,20}            (P2-10 passing/rebounding/possessions)
  zone  trk_fga_{rim,paint,mid,c3,ab3}_m*, trk_fgm_rim_m*, trk_fga_all_m*, trk_z_*_r*, trk_rim_fg_r*   (P2-10 shot zones)
  opp4  opp_trk_{pot_ast,reb_ch,fga_rim,paint_t}_15                                            (P2-10 opponent-allowed)
  dr    trk_dr_{drives,fga,pts,pass,ast,fta}_m*, trk_dr_share_r*            cs/pu  trk_cs_{fga,fg3a,pts,fgm}_m*, trk_pu_*, trk_{cs,pu}_{share,fg3pct|fgpct}_r*
  cv    conversion & luck: long-run rates trk_cv_*_r40, expected trk_cv_exp_{ast,reb,fg3m,fgpts}_m10, luck trk_cv_luck_*_m10
  pm    per-minute: trk_pm_{touches,pot_ast,reb_ch,drives,cs_fga,pu_fga,fga_rim,poss_time}_r10
  vac   trk_vac_{pot_ast,touches,reb_ch,drives,cs_fga,fga_rim,passes} vacated by teammates listed Out/Doubtful (report-based, 2020-21+)
  mx    opp_trk_{drives,cs_fg3a,pu_fga}_15 (more allowed) + trk_mx_{rim,drive,cs3,potast,rebch}_m10 = own volume x opp allowed / league
Usage: .venv/bin/python src/tracking_features2.py
"""
import pathlib, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
from injury_features import _roster  # noqa: E402

ZONES = {"Restricted Area": "rim", "In The Paint (Non-RA)": "paint", "Mid-Range": "mid", "Left Corner 3": "c3", "Right Corner 3": "c3",
         "Above the Break 3": "ab3"}
ZL = ["rim", "paint", "mid", "c3", "ab3"]
ZPTS = {"rim": 2, "paint": 2, "mid": 2, "c3": 3, "ab3": 3}
CORE = {"POTENTIAL_AST": "pot_ast", "AST_PTS_CREATED": "ast_pts", "SECONDARY_AST": "sec_ast", "FT_AST": "ft_ast", "PASSES_MADE": "passes",
        "OREB_CHANCES": "oreb_ch", "DREB_CHANCES": "dreb_ch", "REB_CHANCES": "reb_ch", "REB_CONTEST": "reb_cont", "TOUCHES": "touches",
        "TIME_OF_POSS": "poss_time", "PAINT_TOUCHES": "paint_t", "ELBOW_TOUCHES": "elbow_t", "POST_TOUCHES": "post_t",
        "FRONT_CT_TOUCHES": "fc_touches"}
SCOR = {"DRIVES": "dr_drives", "DRIVE_FGA": "dr_fga", "DRIVE_PTS": "dr_pts", "DRIVE_PASSES": "dr_pass", "DRIVE_AST": "dr_ast", "DRIVE_FTA": "dr_fta",
        "CATCH_SHOOT_FGA": "cs_fga", "CATCH_SHOOT_FG3A": "cs_fg3a", "CATCH_SHOOT_PTS": "cs_pts", "CATCH_SHOOT_FGM": "cs_fgm",
        "PULL_UP_FGA": "pu_fga", "PULL_UP_FG3A": "pu_fg3a", "PULL_UP_PTS": "pu_pts", "PULL_UP_FGM": "pu_fgm"}
VOL = [f"fga_{z}" for z in ZL] + ["fgm_rim", "fga_all"]
RATIOS = {"reb_cont_share": ("REB_CONTEST", "REB_CHANCES"), "reb_per_ch": ("REB", "REB_CHANCES"), "ast_per_pot": ("AST", "POTENTIAL_AST"),
          "pts_per_touch": ("POINTS", "TOUCHES"), "rim_fg": ("fgm_rim", "fga_rim"), **{f"z_{z}": (f"fga_{z}", "fga_all") for z in ZL},
          "dr_share": ("DRIVES", "FRONT_CT_TOUCHES"), "cs_share": ("CATCH_SHOOT_FGA", "fga_all"), "pu_share": ("PULL_UP_FGA", "fga_all"),
          "cs_fg3pct": ("CATCH_SHOOT_FG3M", "CATCH_SHOOT_FG3A"), "pu_fgpct": ("PULL_UP_FGM", "PULL_UP_FGA")}
CV_RATES = {"ast_rate": ("AST", "POTENTIAL_AST"), "reb_rate": ("REB", "REB_CHANCES"), "cs3": ("CATCH_SHOOT_FG3M", "CATCH_SHOOT_FG3A"),
            "pu": ("PULL_UP_FGM", "PULL_UP_FGA"), **{z: (f"fgm_{z}", f"fga_{z}") for z in ZL}}
PM = {"touches": "TOUCHES", "pot_ast": "POTENTIAL_AST", "reb_ch": "REB_CHANCES", "drives": "DRIVES", "cs_fga": "CATCH_SHOOT_FGA",
      "pu_fga": "PULL_UP_FGA", "fga_rim": "fga_rim", "poss_time": "TIME_OF_POSS"}
VAC = {"pot_ast": "POTENTIAL_AST", "touches": "TOUCHES", "reb_ch": "REB_CHANCES", "drives": "DRIVES", "cs_fga": "CATCH_SHOOT_FGA",
       "fga_rim": "fga_rim", "passes": "PASSES_MADE"}
ALLOWED = {"pot_ast": "POTENTIAL_AST", "reb_ch": "REB_CHANCES", "fga_rim": "fga_rim", "paint_t": "PAINT_TOUCHES", "drives": "DRIVES",
           "cs_fg3a": "CATCH_SHOOT_FG3A", "pu_fga": "PULL_UP_FGA"}
MX = {"rim": ("fga_rim", "fga_rim"), "drive": ("DRIVES", "drives"), "cs3": ("CATCH_SHOOT_FG3A", "cs_fg3a"), "potast": ("POTENTIAL_AST", "pot_ast"),
      "rebch": ("REB_CHANCES", "reb_ch")}


def shot_table(shots):
    s = shots[shots.SHOT_ZONE_BASIC.isin(ZONES)].copy()
    s["zone"] = s.SHOT_ZONE_BASIC.map(ZONES)
    g = s.groupby(["PLAYER_ID", "GAME_DATE", "zone"]).agg(fga=("SHOT_ATTEMPTED_FLAG", "sum"), fgm=("SHOT_MADE_FLAG", "sum")).unstack("zone")
    out = pd.DataFrame({f"fga_{z}": g["fga"][z] for z in ZL})
    for z in ZL:
        out[f"fgm_{z}"] = g["fgm"][z]
    out = out.fillna(0)
    out["fga_all"] = out[[f"fga_{z}" for z in ZL]].sum(axis=1)
    return out.reset_index()


def lag_roll(g, col, w, fn="mean"):
    f = (lambda s: s.shift(1).rolling(w, min_periods=1).mean()) if fn == "mean" else (lambda s: s.shift(1).rolling(w, min_periods=1).sum())
    return g[col].transform(f)


def add_tracking_features(df, trk, shots, status):
    trk = trk.merge(shot_table(shots), on=["PLAYER_ID", "GAME_DATE"], how="outer")
    trk["PLAYER_ID"] = trk.PLAYER_ID.astype("int64")
    raw = sorted(set(list(CORE) + list(SCOR) + ["REB", "AST", "POINTS", "MIN", "CATCH_SHOOT_FG3M"] + VOL + [f"fgm_{z}" for z in ZL]))
    T = lambda c: f"_t_{c}"
    df = df.merge(trk[["PLAYER_ID", "GAME_DATE"] + raw].rename(columns={c: T(c) for c in raw}), on=["PLAYER_ID", "GAME_DATE"], how="left")
    gap = df[T("TOUCHES")].notna() & df[T("REB_CHANCES")].isna()  # Rebounding endpoint omits 0-REB players
    for c in ["OREB_CHANCES", "DREB_CHANCES", "REB_CHANCES", "REB_CONTEST"]:
        df.loc[gap, T(c)] = 0.0
    df.loc[gap, T("REB")] = df.loc[gap, "REB"]
    df[T("fgpts")] = sum(df[T(f"fgm_{z}")] * ZPTS[z] for z in ZL)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    g = df.groupby("PLAYER_ID", sort=False)
    F = {}
    for rawc, name in {**CORE, **SCOR}.items():
        for w in [5, 10, 20]:
            F[f"trk_{name}_m{w}"] = lag_roll(g, T(rawc), w)
    for v in VOL:
        for w in [5, 10, 20]:
            F[f"trk_{v}_m{w}"] = lag_roll(g, T(v), w)
    for name, (num, den) in RATIOS.items():
        for w in [10, 20]:
            F[f"trk_{name}_r{w}"] = lag_roll(g, T(num), w, "sum") / lag_roll(g, T(den), w, "sum").replace(0, np.nan)
    # conversion (long-run 40-game rates), expected output from recent volume x long-run rate, and luck = actual - expected
    for name, (num, den) in CV_RATES.items():
        F[f"trk_cv_{name}_r40"] = lag_roll(g, T(num), 40, "sum") / lag_roll(g, T(den), 40, "sum").replace(0, np.nan)
    F["trk_cv_exp_ast_m10"] = F["trk_pot_ast_m10"] * F["trk_cv_ast_rate_r40"]
    F["trk_cv_exp_reb_m10"] = F["trk_reb_ch_m10"] * F["trk_cv_reb_rate_r40"]
    F["trk_cv_exp_fg3m_m10"] = F["trk_fga_c3_m10"] * F["trk_cv_c3_r40"] + F["trk_fga_ab3_m10"] * F["trk_cv_ab3_r40"]
    F["trk_cv_exp_fgpts_m10"] = sum(F[f"trk_fga_{z}_m10"] * F[f"trk_cv_{z}_r40"] * ZPTS[z] for z in ZL)
    F["trk_cv_luck_ast_m10"] = lag_roll(g, T("AST"), 10) - F["trk_cv_exp_ast_m10"]
    F["trk_cv_luck_reb_m10"] = lag_roll(g, T("REB"), 10) - F["trk_cv_exp_reb_m10"]
    F["trk_cv_luck_fg3m_m10"] = lag_roll(g, "FG3M", 10) - F["trk_cv_exp_fg3m_m10"]
    F["trk_cv_luck_fgpts_m10"] = lag_roll(g, T("fgpts"), 10) - F["trk_cv_exp_fgpts_m10"]
    # per-minute volumes (10-game sums / minutes)
    min10 = lag_roll(g, T("MIN"), 10, "sum").replace(0, np.nan)
    for name, rawc in PM.items():
        F[f"trk_pm_{name}_r10"] = lag_roll(g, T(rawc), 10, "sum") / min10
    F = pd.DataFrame(F, index=df.index)
    df = pd.concat([df, F], axis=1)

    # ---- opponent-allowed (lagged 15-game team mean of what the defending team gave up) + matchup interactions
    tg = df.groupby(["GAME_ID", "TEAM_ABBREVIATION", "opp", "GAME_DATE"], as_index=False).agg(
        **{k: (T(v), "sum") for k, v in ALLOWED.items()}, n_trk=(T("TOUCHES"), "count"))
    tg.loc[tg.n_trk == 0, list(ALLOWED)] = np.nan
    allowed = tg.rename(columns={"TEAM_ABBREVIATION": "_scorer", "opp": "TEAM_ABBREVIATION"}).drop(columns="_scorer")
    allowed = allowed.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    ag = allowed.groupby("TEAM_ABBREVIATION", sort=False)
    oc = [f"opp_trk_{k}_15" for k in ALLOWED]
    for k in ALLOWED:
        allowed[f"opp_trk_{k}_15"] = lag_roll(ag, k, 15)
    allowed = allowed.rename(columns={"TEAM_ABBREVIATION": "opp"})[["GAME_ID", "opp"] + oc]
    df = df.merge(allowed, on=["GAME_ID", "opp"], how="left")
    # league scale for the interaction: mean of the 30 teams' lagged allowed value on that date (all lagged information)
    lg = df.drop_duplicates(["GAME_DATE", "opp"]).groupby("GAME_DATE")[oc].mean().rename(columns=lambda c: c + "_lg")
    df = df.merge(lg, left_on="GAME_DATE", right_index=True, how="left")
    for name, (rawc, k) in MX.items():
        df[f"trk_mx_{name}_m10"] = df[f"trk_{ {'fga_rim':'fga_rim','DRIVES':'dr_drives','CATCH_SHOOT_FG3A':'cs_fg3a','POTENTIAL_AST':'pot_ast','REB_CHANCES':'reb_ch'}[rawc] }_m10"] \
            * df[f"opp_trk_{k}_15"] / df[f"opp_trk_{k}_15_lg"].replace(0, np.nan)
    df = df.drop(columns=[c + "_lg" for c in oc])

    # ---- vacated teammate tracking volume (report-based Out/Doubtful, same roster logic as injury_features)
    st = df.loc[df.MIN > 0, ["PLAYER_ID", "GAME_DATE"] + [T(v) for v in VAC.values()]].sort_values(["PLAYER_ID", "GAME_DATE"])
    sg = st.groupby("PLAYER_ID", sort=False)
    for k, v in VAC.items():
        st[f"v_{k}"] = sg[T(v)].transform(lambda s: s.rolling(10, min_periods=1).mean())  # incl. own last game (state as of last game)
    ros = _roster(df).sort_values("GAME_DATE")
    st = st[["PLAYER_ID", "GAME_DATE"] + [f"v_{k}" for k in VAC]].sort_values("GAME_DATE")
    ros = pd.merge_asof(ros, st, on="GAME_DATE", by="PLAYER_ID", allow_exact_matches=False, direction="backward")
    s = status[["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "code_dec"]].dropna(subset=["PLAYER_ID"]).drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"])
    s["PLAYER_ID"] = s.PLAYER_ID.astype("int64")
    ros = ros.merge(s, on=["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"], how="left")
    ros["out"] = (ros.code_dec.fillna(0) >= 3).astype(int)
    key = ["TEAM_ABBREVIATION", "GAME_ID"]
    V = ros[ros.out == 1].groupby(key).agg(**{f"trk_vac_{k}": (f"v_{k}", "sum") for k in VAC}).reset_index()
    df = df.merge(V, on=key, how="left")
    own = ros.loc[ros.out == 1, key + ["PLAYER_ID"] + [f"v_{k}" for k in VAC]]
    df = df.merge(own, on=key + ["PLAYER_ID"], how="left")
    for k in VAC:  # a listed-out player who plays is not vacated
        df[f"trk_vac_{k}"] = df[f"trk_vac_{k}"].fillna(0) - df[f"v_{k}"].fillna(0)
    df = df.drop(columns=[f"v_{k}" for k in VAC])
    has_rep = df.GAME_DATE >= pd.Timestamp("2020-10-01")  # no injury reports before 2020-21 -> NaN, like vac_* (which are 0 there)
    df.loc[~has_rep, [f"trk_vac_{k}" for k in VAC]] = np.nan
    df = df.drop(columns=[c for c in df.columns if c.startswith("_t_")])
    return df


if __name__ == "__main__":
    pg = pd.read_parquet(PROC / "player_games.parquet")
    trk = pd.read_parquet(PROC / "tracking_player_games.parquet")
    shots = pd.read_parquet(PROC / "tracking_shots.parquet")
    status = pd.read_parquet(PROC / "injury_status.parquet", columns=["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "code_dec"])
    out = add_tracking_features(pg, trk, shots, status)
    assert len(out) == len(pg)
    new = [c for c in out.columns if c.startswith("trk_") or c.startswith("opp_trk_")]
    print(f"{out.shape}; {len(new)} tracking columns; coverage by season (share of rows with trk_touches_m10 / trk_dr_drives_m10):")
    print(out.groupby("season")[["trk_touches_m10", "trk_dr_drives_m10", "trk_vac_touches", "trk_mx_rim_m10"]].apply(lambda d: d.notna().mean().round(3)).to_string())
    out.to_parquet(PROC / "player_games_trk2.parquet", index=False)
