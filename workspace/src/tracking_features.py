"""P2-10: rolling, prior-games-only tracking and shot-location features (2023-24 onward; NaN before).

Reads data/processed/player_games.parquet (production features, untouched) + tracking_player_games.parquet + tracking_shots.parquet
(src/fetch_tracking.py --build) and writes data/processed/player_games_trk.parquet with extra columns:
  trk_<stat>_m{5,10,20}            lagged rolling means of potential assists, assist points created, secondary+FT assists, passes,
                                   rebound chances (O/D/T), contested rebounds, touches, time of possession, paint/elbow/post touches
  trk_<ratio>_r{10,20}             lagged rolling ratios (sum/sum): contested share of chances, reb per chance, ast per potential ast,
                                   pts per touch, zone shares of FGA (rim / paint non-RA / midrange / corner 3 / above-break 3)
  trk_fga_<zone>_m{5,10,20}        lagged rolling per-zone FGA volume (+ rim FGM, 3PA by corner/above-break)
  opp_trk_<stat>_15                opponent-allowed (lagged 15-game team mean) potential assists, rebound chances, rim FGA, paint touches
Column names end in _m\\d+ / _r\\d+ / start with opp_ so src/common.feature_cols picks them up unchanged. Every series is shift(1)
within player (or team) after sorting by date, so a game only sees earlier games. Used via env PLAYER_GAMES=... by run_bp.py.
Usage: .venv/bin/python src/tracking_features.py
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ZONES = {"Restricted Area": "rim", "In The Paint (Non-RA)": "paint", "Mid-Range": "mid", "Left Corner 3": "c3", "Right Corner 3": "c3",
         "Above the Break 3": "ab3"}
STATS = {"POTENTIAL_AST": "pot_ast", "AST_PTS_CREATED": "ast_pts", "SECONDARY_AST": "sec_ast", "FT_AST": "ft_ast", "PASSES_MADE": "passes",
         "OREB_CHANCES": "oreb_ch", "DREB_CHANCES": "dreb_ch", "REB_CHANCES": "reb_ch", "REB_CONTEST": "reb_cont", "TOUCHES": "touches",
         "TIME_OF_POSS": "poss_time", "PAINT_TOUCHES": "paint_t", "ELBOW_TOUCHES": "elbow_t", "POST_TOUCHES": "post_t",
         "FRONT_CT_TOUCHES": "fc_touches"}
RATIOS = {"reb_cont_share": ("REB_CONTEST", "REB_CHANCES"), "reb_per_ch": ("REB", "REB_CHANCES"), "ast_per_pot": ("AST", "POTENTIAL_AST"),
          "pts_per_touch": ("POINTS", "TOUCHES"), "z_rim": ("fga_rim", "fga_all"), "z_paint": ("fga_paint", "fga_all"),
          "z_mid": ("fga_mid", "fga_all"), "z_c3": ("fga_c3", "fga_all"), "z_ab3": ("fga_ab3", "fga_all"), "rim_fg": ("fgm_rim", "fga_rim")}
VOL = ["fga_rim", "fga_paint", "fga_mid", "fga_c3", "fga_ab3", "fgm_rim", "fga_all"]


def shot_table(shots):
    s = shots[shots.SHOT_ZONE_BASIC.isin(ZONES)].copy()
    s["zone"] = s.SHOT_ZONE_BASIC.map(ZONES)
    g = s.groupby(["PLAYER_ID", "GAME_DATE", "zone"]).agg(fga=("SHOT_ATTEMPTED_FLAG", "sum"), fgm=("SHOT_MADE_FLAG", "sum")).unstack("zone")
    out = pd.DataFrame({f"fga_{z}": g["fga"][z] for z in ["rim", "paint", "mid", "c3", "ab3"]})
    out["fgm_rim"] = g["fgm"]["rim"]
    out = out.fillna(0)
    out["fga_all"] = out[[f"fga_{z}" for z in ["rim", "paint", "mid", "c3", "ab3"]]].sum(axis=1)
    return out.reset_index()


def lag_roll(g, col, w, fn="mean"):
    """shift(1) then roll WITHIN the group. (features.lagged_roll rolls the shifted series across group boundaries, so the first
    w-1 games of each player/team see the previous group's last games; noted in JOURNAL P2-10, production code left unchanged.)"""
    if fn == "mean":
        return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum())


def add_tracking_features(df, trk, shots):
    """df: player_games (sorted by PLAYER_ID, GAME_DATE). Returns df with trk_/opp_trk_ columns added."""
    trk = trk.merge(shot_table(shots), on=["PLAYER_ID", "GAME_DATE"], how="outer")
    trk["PLAYER_ID"] = trk.PLAYER_ID.astype("int64")
    raw_cols = list(STATS) + ["REB", "AST", "POINTS"] + VOL
    df = df.merge(trk[["PLAYER_ID", "GAME_DATE"] + raw_cols].rename(columns={c: f"_t_{c}" for c in raw_cols}),
                  on=["PLAYER_ID", "GAME_DATE"], how="left")
    # the Rebounding endpoint omits players with 0 rebounds (98.5% of its gaps): when a passing row exists but no rebounding row,
    # treat chances/contested as 0 for that game rather than NaN (NaN would bias the rolling means of low-minute players upward)
    gap = df["_t_TOUCHES"].notna() & df["_t_REB_CHANCES"].isna()
    for c in ["OREB_CHANCES", "DREB_CHANCES", "REB_CHANCES", "REB_CONTEST"]:
        df.loc[gap, f"_t_{c}"] = 0.0
    df.loc[gap, "_t_REB"] = df.loc[gap, "REB"]
    df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    g = df.groupby("PLAYER_ID", sort=False)
    feats = {}
    for raw, name in STATS.items():
        for w in [5, 10, 20]:
            feats[f"trk_{name}_m{w}"] = lag_roll(g, f"_t_{raw}", w)
    for v in VOL:
        for w in [5, 10, 20]:
            feats[f"trk_{v}_m{w}"] = lag_roll(g, f"_t_{v}", w)
    for name, (num, den) in RATIOS.items():
        for w in [10, 20]:
            feats[f"trk_{name}_r{w}"] = lag_roll(g, f"_t_{num}", w, "sum") / lag_roll(g, f"_t_{den}", w, "sum").replace(0, np.nan)
    F = pd.DataFrame(feats, index=df.index)
    # rows with no tracking history at all (pre-2023-24) stay NaN: rolling over all-NaN gives NaN already
    df = pd.concat([df, F], axis=1)

    # ---- opponent-allowed: team-game totals of the OPPONENT's players, rolled 15 games (lagged) per defending team
    tg = df.groupby(["GAME_ID", "TEAM_ABBREVIATION", "opp", "GAME_DATE"], as_index=False).agg(
        pot_ast=("_t_POTENTIAL_AST", "sum"), reb_ch=("_t_REB_CHANCES", "sum"), rim_fga=("_t_fga_rim", "sum"), paint_t=("_t_PAINT_TOUCHES", "sum"),
        n_trk=("_t_TOUCHES", "count"))
    tg.loc[tg.n_trk == 0, ["pot_ast", "reb_ch", "rim_fga", "paint_t"]] = np.nan
    # what team X allowed in game G = totals of its opponent in G; key by (GAME_ID, defending team = opp)
    allowed = tg.rename(columns={"TEAM_ABBREVIATION": "_scorer", "opp": "TEAM_ABBREVIATION"}).drop(columns="_scorer")
    allowed = allowed.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    ag = allowed.groupby("TEAM_ABBREVIATION", sort=False)
    for c in ["pot_ast", "reb_ch", "rim_fga", "paint_t"]:
        allowed[f"opp_trk_{c}_15"] = lag_roll(ag, c, 15)
    allowed = allowed.rename(columns={"TEAM_ABBREVIATION": "opp"})[["GAME_ID", "opp"] + [f"opp_trk_{c}_15" for c in ["pot_ast", "reb_ch", "rim_fga", "paint_t"]]]
    df = df.merge(allowed, on=["GAME_ID", "opp"], how="left")
    df = df.drop(columns=[c for c in df.columns if c.startswith("_t_")])
    return df


if __name__ == "__main__":
    pg = pd.read_parquet(PROC / "player_games.parquet")
    trk = pd.read_parquet(PROC / "tracking_player_games.parquet")
    shots = pd.read_parquet(PROC / "tracking_shots.parquet")
    out = add_tracking_features(pg, trk, shots)
    new = [c for c in out.columns if c.startswith("trk_") or c.startswith("opp_trk_")]
    print(f"{out.shape}; {len(new)} tracking columns; coverage by season (share of rows with trk_touches_m10):")
    print(out.groupby("season").trk_touches_m10.apply(lambda s: round(s.notna().mean(), 3)).to_string())
    out.to_parquet(PROC / "player_games_trk.parquet", index=False)
