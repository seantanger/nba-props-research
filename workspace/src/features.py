"""Build leak-free pre-game features for every player-game in the nba_api logs.

Rule: every feature for a game dated D uses only box scores from games dated < D (shift(1) on
player/team series after sorting by date), plus static player attributes (position/height) and
last-season aggregates.

Output: data/processed/player_games.parquet (one row per player-game with targets + features)
"""
import pathlib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nba"
OUT = ROOT / "data" / "processed"
STATS = ["MIN", "PTS", "REB", "AST", "FG3M", "FG3A", "FGA", "FTA", "PRA", "PA", "PR", "RA", "TOV", "STL", "BLK", "OREB", "DREB"]
TARGETS = ["PTS", "REB", "AST", "FG3M", "PRA", "PA", "PR", "RA", "STL", "BLK"]


def load_player_logs():
    base = pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob("player_*_Base.parquet"))], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob("player_*_Advanced.parquet"))], ignore_index=True)
    adv = adv[["GAME_ID", "PLAYER_ID", "USG_PCT", "PACE", "TS_PCT", "AST_PCT", "REB_PCT", "E_USG_PCT", "POSS"]]
    df = base.merge(adv, on=["GAME_ID", "PLAYER_ID"], how="left")
    df["GAME_DATE"] = pd.to_datetime(df.GAME_DATE).dt.tz_localize(None).dt.normalize()
    df["PRA"] = df.PTS + df.REB + df.AST
    df["PA"] = df.PTS + df.AST; df["PR"] = df.PTS + df.REB; df["RA"] = df.REB + df.AST
    df["home"] = df.MATCHUP.str.contains("vs.").astype(int)
    df["opp"] = df.MATCHUP.str.split(" ").str[-1]
    df["season"] = df.SEASON_YEAR
    df["playoffs"] = df.GAME_ID.astype(str).str[2] .isin(["4", "5"]).astype(int)  # 004=playoffs, 005=play-in
    df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    return df


def load_team_logs():
    base = pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob("team_*_Base.parquet"))], ignore_index=True)
    adv = pd.concat([pd.read_parquet(p) for p in sorted(RAW.glob("team_*_Advanced.parquet"))], ignore_index=True)
    adv = adv[["GAME_ID", "TEAM_ID", "OFF_RATING", "DEF_RATING", "PACE", "POSS"]]
    t = base.merge(adv, on=["GAME_ID", "TEAM_ID"], how="left")
    t["GAME_DATE"] = pd.to_datetime(t.GAME_DATE).dt.tz_localize(None).dt.normalize()
    t["opp"] = t.MATCHUP.str.split(" ").str[-1]
    t["home"] = t.MATCHUP.str.contains("vs.").astype(int)
    t["margin"] = t.PLUS_MINUS
    # stats allowed = opponent's stats in the same game
    opp = t[["GAME_ID", "TEAM_ABBREVIATION", "PTS", "REB", "AST", "FG3M", "FG3A", "FGA", "FTA", "PACE"]].rename(
        columns={"TEAM_ABBREVIATION": "opp", "PTS": "allow_PTS", "REB": "allow_REB", "AST": "allow_AST",
                 "FG3M": "allow_FG3M", "FG3A": "allow_FG3A", "FGA": "allow_FGA", "FTA": "allow_FTA", "PACE": "opp_pace_g"})
    t = t.merge(opp, on=["GAME_ID", "opp"], how="left")
    return t.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)


def lagged_roll(g, col, w, fn="mean", min_periods=1):
    s = g[col].shift(1)
    if fn == "mean":
        return s.rolling(w, min_periods=min_periods).mean()
    if fn == "std":
        return s.rolling(w, min_periods=2).std()
    if fn == "sum":
        return s.rolling(w, min_periods=min_periods).sum()
    raise ValueError(fn)


def build(player_stubs=None, team_stubs=None, out_path=None, extra_status=None):
    """player_stubs/team_stubs: optional DataFrames of UNPLAYED games (stats NaN) appended before feature computation,
    so that upcoming games get the same strictly-lagged features (used by src/live_odds.py).
    extra_status: optional injury statuses for the stub games (GAME_ID, TEAM_ABBREVIATION, PLAYER_ID, code_dec), appended to the
    historical decision-time statuses from data/processed/injury_status.parquet."""
    df = load_player_logs()
    tl = load_team_logs()
    if player_stubs is not None:
        df = pd.concat([df, player_stubs], ignore_index=True).sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        # stubs skip load_player_logs, so derive its matchup/season columns for them too
        df["home"] = df.MATCHUP.str.contains("vs.").astype(int)
        df["opp"] = df.MATCHUP.str.split(" ").str[-1]
        df["season"] = df.SEASON_YEAR
        df["playoffs"] = df.GAME_ID.astype(str).str[2].isin(["4", "5"]).astype(int)
    if team_stubs is not None:
        tl = pd.concat([tl, team_stubs], ignore_index=True).sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
        tl["opp"] = tl.MATCHUP.str.split(" ").str[-1]
        tl["home"] = tl.MATCHUP.str.contains("vs.").astype(int)
    pidx = pd.read_parquet(RAW / "playerindex_all.parquet")
    pidx["height_in"] = pidx.HEIGHT.str.split("-").apply(lambda x: int(x[0]) * 12 + int(x[1]) if isinstance(x, list) and len(x) == 2 else np.nan)
    pidx = pidx[["PERSON_ID", "POSITION", "height_in", "DRAFT_YEAR"]].rename(columns={"PERSON_ID": "PLAYER_ID"})
    df = df.merge(pidx, on="PLAYER_ID", how="left")
    df["pos_G"] = df.POSITION.fillna("").str.contains("G").astype(int)
    df["pos_F"] = df.POSITION.fillna("").str.contains("F").astype(int)
    df["pos_C"] = df.POSITION.fillna("").str.contains("C").astype(int)

    # ---- player rolling features (within player, across seasons; shift(1) guarantees strictly prior games)
    g = df.groupby("PLAYER_ID", sort=False)
    feats = {}
    for col in STATS + ["USG_PCT", "TS_PCT", "AST_PCT", "REB_PCT"]:
        for w in [3, 5, 10, 20, 40]:
            feats[f"{col}_m{w}"] = lagged_roll(g, col, w)
        if col in ["MIN", "PTS", "REB", "AST", "FG3M", "PRA", "PA", "PR", "RA", "STL", "BLK", "FGA"]:
            feats[f"{col}_s10"] = lagged_roll(g, col, 10, "std")
            feats[f"{col}_s20"] = lagged_roll(g, col, 20, "std")
    for col in ["MIN", "PTS", "REB", "AST", "FG3M", "PRA", "PA", "PR", "RA", "STL", "BLK"]:
        feats[f"{col}_ewm"] = g[col].transform(lambda s: s.shift(1).ewm(halflife=6, min_periods=1).mean())
        feats[f"{col}_last"] = g[col].shift(1)
        feats[f"{col}_max10"] = g[col].transform(lambda s: s.shift(1).rolling(10, min_periods=1).max())
        feats[f"{col}_min10"] = g[col].transform(lambda s: s.shift(1).rolling(10, min_periods=1).min())
    feats["games_prior"] = g.cumcount()
    feats["prev_date"] = g.GAME_DATE.shift(1)
    F = pd.DataFrame(feats, index=df.index)
    df = pd.concat([df, F], axis=1)
    df["rest_days"] = (df.GAME_DATE - df.prev_date).dt.days.clip(upper=15)
    df["b2b"] = (df.rest_days == 1).astype(int)
    # season-to-date (lagged) within season
    gs = df.groupby(["PLAYER_ID", "season"], sort=False)
    for col in ["MIN", "PTS", "REB", "AST", "FG3M", "PRA", "PA", "PR", "RA", "STL", "BLK"]:
        df[f"{col}_std_season"] = gs[col].transform(lambda s: s.shift(1).expanding().mean())
    df["gp_season"] = gs.cumcount()
    # per-minute rates (lagged, 20-game)
    for col in ["PTS", "REB", "AST", "FG3M", "PRA", "PA", "PR", "RA", "STL", "BLK", "FGA", "FTA", "FG3A"]:
        df[f"{col}_per_min20"] = df[f"{col}_m20"] / df["MIN_m20"].replace(0, np.nan)
    # last-season aggregates (static prior)
    prev = df.groupby(["PLAYER_ID", "season"]).agg(ps_MIN=("MIN", "mean"), ps_PTS=("PTS", "mean"), ps_REB=("REB", "mean"),
                                                    ps_AST=("AST", "mean"), ps_FG3M=("FG3M", "mean"), ps_GP=("MIN", "size")).reset_index()
    prev["season"] = prev.season.map(lambda s: f"{int(s[:4]) + 1}-{int(s[-2:]) + 1:02d}")
    df = df.merge(prev, on=["PLAYER_ID", "season"], how="left")

    # ---- team-level context (lagged): own team pace/off rating, opponent def rating, allowed stats, margin
    tl = tl.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    tg = tl.groupby("TEAM_ABBREVIATION", sort=False)
    team_feats = pd.DataFrame(index=tl.index)
    for col, w in [("PACE", 15), ("OFF_RATING", 15), ("DEF_RATING", 15), ("margin", 15), ("allow_PTS", 15), ("allow_REB", 15),
                   ("allow_AST", 15), ("allow_FG3M", 15), ("allow_FG3A", 15), ("allow_FTA", 15), ("PTS", 15), ("FGA", 15), ("FG3A", 15)]:
        team_feats[f"tm_{col}_{w}"] = lagged_roll(tg, col, w)
    team_feats["tm_prev_date"] = tg.GAME_DATE.shift(1)
    team_feats["tm_games"] = tg.cumcount()
    tl = pd.concat([tl, team_feats], axis=1)
    tl["tm_rest"] = (tl.GAME_DATE - tl.tm_prev_date).dt.days.clip(upper=15)
    # league average of allowed stats to normalize (lagged by date)
    tcols = [c for c in tl.columns if c.startswith("tm_") and c not in ("tm_prev_date", "tm_games")]
    own = tl[["GAME_ID", "TEAM_ABBREVIATION"] + tcols].copy()
    opp = tl[["GAME_ID", "TEAM_ABBREVIATION"] + tcols].rename(columns={c: c.replace("tm_", "opp_") for c in tcols})
    opp = opp.rename(columns={"TEAM_ABBREVIATION": "opp"})
    df = df.merge(own, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    df = df.merge(opp, on=["GAME_ID", "opp"], how="left")
    df["exp_margin"] = df.tm_margin_15 - df.opp_margin_15 + 2.5 * (2 * df.home - 1)

    # ---- teammate context: minutes/usage missing from the team's previous game relative to its 15-game norm
    # (who was out LAST game is known; who is out THIS game is not assumed known)
    pm = df[["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN", "USG_PCT", "PTS"]].copy()
    pm["MIN_norm"] = df["MIN_m10"]
    pm["PTS_norm"] = df["PTS_m10"]
    # team game -> set of players who appeared; for each player, sum of 10g-mean MIN of teammates that appeared
    team_game = pm.groupby(["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"]).agg(
        team_MIN_norm_present=("MIN_norm", "sum"), team_PTS_norm_present=("PTS_norm", "sum"), n_players=("PLAYER_ID", "size")).reset_index()
    team_game = team_game.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    tgg = team_game.groupby("TEAM_ABBREVIATION", sort=False)
    team_game["prev_team_PTS_norm_present"] = tgg.team_PTS_norm_present.shift(1)
    team_game["prev_n_players"] = tgg.n_players.shift(1)
    df = df.merge(team_game[["GAME_ID", "TEAM_ABBREVIATION", "prev_team_PTS_norm_present", "prev_n_players"]],
                  on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    # player's own share of team scoring norm, relative to the roster that appeared in the team's PREVIOUS game (leak-free)
    df["pts_share10"] = df.PTS_m10 / df.prev_team_PTS_norm_present.replace(0, np.nan)
    # minutes trend
    df["MIN_trend"] = df.MIN_m3 - df.MIN_m20
    df["PTS_trend"] = df.PTS_m5 - df.PTS_m20
    # ---- injury-aware features: official injury report as of tip-off minus 60 min (src/injury_status.py, src/injury_features.py)
    try:
        from src.injury_features import add_injury_features
    except ImportError:
        from injury_features import add_injury_features
    st_path = OUT / "injury_status.parquet"
    status = pd.read_parquet(st_path, columns=["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "code_dec"]) if st_path.exists() else None
    if extra_status is not None:
        status = pd.concat([status, extra_status], ignore_index=True) if status is not None else extra_status
    report_start = pd.read_parquet(st_path, columns=["GAME_DATE"]).GAME_DATE.min() if st_path.exists() else None
    df = add_injury_features(df, status, with_bs=True, report_start=report_start)
    df["dow"] = df.GAME_DATE.dt.dayofweek
    df["month"] = df.GAME_DATE.dt.month
    df = df.drop(columns=["prev_date"])
    OUT.mkdir(exist_ok=True, parents=True)
    df.to_parquet(out_path or OUT / "player_games.parquet", index=False)
    print(df.shape, df.GAME_DATE.min(), df.GAME_DATE.max())
    return df


if __name__ == "__main__":
    build()
