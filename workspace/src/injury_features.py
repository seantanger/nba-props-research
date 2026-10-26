"""Injury-aware features for player_games rows.

For each team-game the "recent roster" is every player who appeared (MIN>0) in any of the team's previous 5 games.
Each roster member carries his state as of his own last game before tonight (10-game means incl. that game), so absent
players have a state too. Tonight's absences come from one of two sources:
  * report-based (prefix vac_/ret_/share_/inj_): the official injury report as of tip-off minus 60 min (status code >=3 =
    Out/Doubtful; 2 = Questionable). Decision-time information only -> legitimate features.
  * box-score-based (prefix bsvac_): roster members with no MIN>0 row tonight. Knows late scratches -> LEAKS; kept only to
    measure the ceiling / the size of the leak, never for a bettable model.
Opponent aggregates get the opp_vac_ prefix (report-based only).
"""
import numpy as np
import pandas as pd

STATE = ["MIN", "PTS", "FGA", "REB", "AST", "FG3A", "USGMIN"]
STARTER_MIN = 28.0


def _player_state(df):
    d = df[["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN", "PTS", "FGA", "REB", "AST", "FG3A", "USG_PCT", "pos_G", "pos_F", "pos_C"]].copy()
    d = d[d.MIN > 0].sort_values(["PLAYER_ID", "GAME_DATE"])
    d["USGMIN"] = d.USG_PCT.fillna(0) * d.MIN
    g = d.groupby("PLAYER_ID", sort=False)
    for c in STATE:
        d[f"{c}_r10"] = g[c].transform(lambda s: s.rolling(10, min_periods=1).mean())
    d["pos"] = np.select([d.pos_C == 1, d.pos_G == 1], ["C", "G"], "F")
    return d


def _roster(df, n_prev=5):
    """(TEAM_ABBREVIATION, GAME_ID, GAME_DATE, PLAYER_ID, in_last1, n_last3): players who appeared in the team's last n_prev games."""
    tg = df[["TEAM_ABBREVIATION", "season", "GAME_ID", "GAME_DATE"]].drop_duplicates().sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"])
    tg["ti"] = tg.groupby(["TEAM_ABBREVIATION", "season"]).cumcount()  # resets each season: no roster carry-over across summers
    app = df.loc[df.MIN > 0, ["TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID"]].merge(tg[["TEAM_ABBREVIATION", "season", "GAME_ID", "ti"]], on=["TEAM_ABBREVIATION", "GAME_ID"])
    parts = []
    for k in range(1, n_prev + 1):
        a = app[["TEAM_ABBREVIATION", "season", "PLAYER_ID", "ti"]].copy()
        a["ti"] = a.ti + k
        a["k"] = k
        parts.append(a)
    r = pd.concat(parts, ignore_index=True)
    r = r.groupby(["TEAM_ABBREVIATION", "season", "ti", "PLAYER_ID"]).agg(kmin=("k", "min"), n_last3=("k", lambda s: (s <= 3).sum())).reset_index()
    r["in_last1"] = (r.kmin == 1).astype(int)
    r = r.merge(tg, on=["TEAM_ABBREVIATION", "season", "ti"], how="inner")  # drops horizons past the last game
    return r[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "PLAYER_ID", "in_last1", "n_last3"]]


def add_injury_features(df, status=None, with_bs=True, report_start=None):
    """df: player_games (needs MIN, raw stats, pos_*, MIN_m10, PTS_m10). status: DataFrame GAME_ID, TEAM_ABBREVIATION, PLAYER_ID, code_dec.
    Returns df with the new columns added (row order preserved)."""
    df = df.copy()
    df["PLAYER_ID"] = df.PLAYER_ID.astype("int64")
    st = _player_state(df)
    ros = _roster(df)
    # state of each roster member as of his last game strictly before tonight
    ros = ros.sort_values("GAME_DATE")
    stt = st[["PLAYER_ID", "GAME_DATE", "pos"] + [f"{c}_r10" for c in STATE]].sort_values("GAME_DATE")
    ros = pd.merge_asof(ros, stt, on="GAME_DATE", by="PLAYER_ID", allow_exact_matches=False, direction="backward")
    ros["starter"] = (ros.MIN_r10 >= STARTER_MIN).astype(int)
    # decision-time status (not listed -> available)
    if status is not None:
        s = status[["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "code_dec"]].dropna(subset=["PLAYER_ID"]).drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"])
        s["PLAYER_ID"] = s.PLAYER_ID.astype(df.PLAYER_ID.dtype)
        ros = ros.merge(s, on=["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"], how="left")
    else:
        ros["code_dec"] = np.nan
    ros["code_dec"] = ros.code_dec.fillna(0)
    # box-score absence (leaky): roster member with no MIN>0 row tonight
    played = df.loc[df.MIN > 0, ["GAME_ID", "PLAYER_ID"]].drop_duplicates().assign(played=1)
    ros = ros.merge(played, on=["GAME_ID", "PLAYER_ID"], how="left")
    ros["bs_out"] = ros.played.isna().astype(int)
    ros["out"] = (ros.code_dec >= 3).astype(int)
    ros["q"] = (ros.code_dec == 2).astype(int)
    ros["ret"] = ((ros.code_dec < 3) & (ros.in_last1 == 0)).astype(int)
    key = ["TEAM_ABBREVIATION", "GAME_ID"]

    def agg(mask, prefix, cols=STATE, extra=True):
        sub = ros[mask]
        a = sub.groupby(key).agg(**{f"{prefix}_{c.lower()}": (f"{c}_r10", "sum") for c in cols}, **{f"{prefix}_n": ("PLAYER_ID", "size")})
        if extra:
            a[f"{prefix}_starters"] = sub.groupby(key).starter.sum()
        return a

    T = pd.DataFrame(index=ros.groupby(key).size().index)
    T = T.join(agg(ros.out == 1, "vac"))
    T = T.join(agg((ros.out == 1) & (ros.in_last1 == 1), "vac_new", cols=["MIN", "PTS", "FGA"], extra=False))
    T = T.join(agg(ros.q == 1, "vac_q", cols=["MIN", "PTS"], extra=False))
    T = T.join(agg(ros.ret == 1, "ret", cols=["MIN", "PTS", "FGA"], extra=False))
    T = T.join(ros[ros.out == 0].groupby(key).agg(avail_min=("MIN_r10", "sum"), avail_pts=("PTS_r10", "sum"), avail_fga=("FGA_r10", "sum")))
    # position-specific vacated minutes
    for p in ["G", "F", "C"]:
        T[f"vac_min_pos{p}"] = ros[(ros.out == 1) & (ros.pos == p)].groupby(key).MIN_r10.sum()
    if with_bs:
        T = T.join(agg(ros.bs_out == 1, "bsvac"))
        T = T.join(agg((ros.bs_out == 1) & (ros.in_last1 == 1), "bsvac_new", cols=["MIN", "PTS", "FGA"], extra=False))
        T = T.join(ros[ros.bs_out == 0].groupby(key).agg(bsavail_min=("MIN_r10", "sum"), bsavail_pts=("PTS_r10", "sum")))
    T = T.reset_index()
    # every team-game present in df gets a row (zeros where nothing is vacated)
    allg = df[key].drop_duplicates()
    T = allg.merge(T, on=key, how="left")
    fill0 = [c for c in T.columns if c not in key and not c.startswith("avail") and not c.startswith("bsavail")]
    T[fill0] = T[fill0].fillna(0)
    out = df.merge(T, on=key, how="left")
    # own status + own roster membership
    own = ros[key + ["PLAYER_ID", "code_dec", "out", "MIN_r10", "pos"]].rename(columns={"code_dec": "inj_self_code", "out": "_own_out", "MIN_r10": "_own_min", "pos": "_own_pos"})
    out = out.merge(own, on=key + ["PLAYER_ID"], how="left")
    out["inj_self_code"] = out.inj_self_code.fillna(0)
    if status is not None:  # a player not on the recent roster can still be listed (e.g. first game back from a long absence)
        s2 = status[["GAME_ID", "PLAYER_ID", "code_dec"]].dropna(subset=["PLAYER_ID"]).drop_duplicates(["GAME_ID", "PLAYER_ID"])
        s2["PLAYER_ID"] = s2.PLAYER_ID.astype(df.PLAYER_ID.dtype)
        out = out.merge(s2.rename(columns={"code_dec": "_c2"}), on=["GAME_ID", "PLAYER_ID"], how="left")
        out["inj_self_code"] = out.inj_self_code.where(out.inj_self_code > 0, out._c2.fillna(0))
        out = out.drop(columns="_c2")
    out["inj_self_q"] = (out.inj_self_code == 2).astype(int)
    # exclude own contribution from vacated sums if the player himself was listed out (he is playing, so he is not vacated)
    ownout = (out._own_out == 1)
    for c in ["vac_min", "vac_n"]:
        out.loc[ownout, c] = out.loc[ownout, c] - (out.loc[ownout, "_own_min"] if c == "vac_min" else 1)
    # expected shares of the available roster (self included); new signings not yet in the roster add their own norm
    notros = out._own_min.isna()
    denom_min = out.avail_min + np.where(ownout, out._own_min.fillna(0), 0) + np.where(notros, out.MIN_m10.fillna(0), 0)
    denom_pts = out.avail_pts + np.where(notros, out.PTS_m10.fillna(0), 0)
    out["share_min"] = out.MIN_m10 / denom_min.replace(0, np.nan)
    out["share_pts"] = out.PTS_m10 / denom_pts.replace(0, np.nan)
    out["share_min_x240"] = out.share_min * 240
    out["vac_min_samepos"] = np.select([out._own_pos == "G", out._own_pos == "C"], [out.vac_min_posG, out.vac_min_posC], out.vac_min_posF)
    out["vac_frac"] = out.vac_min / (out.vac_min + out.avail_min).replace(0, np.nan)
    if with_bs:
        out["bsshare_min"] = out.MIN_m10 / (out.bsavail_min + np.where(notros, out.MIN_m10.fillna(0), 0)).replace(0, np.nan)
    # opponent vacated (report-based)
    oppT = T[key + ["vac_min", "vac_starters", "vac_new_min", "vac_pts", "vac_q_min"]].rename(
        columns={"TEAM_ABBREVIATION": "opp", "vac_min": "opp_vac_min", "vac_starters": "opp_vac_starters", "vac_new_min": "opp_vac_new_min",
                 "vac_pts": "opp_vac_pts", "vac_q_min": "opp_vac_q_min"})
    out = out.merge(oppT, on=["GAME_ID", "opp"], how="left")
    out = out.drop(columns=["_own_out", "_own_min", "_own_pos"])
    # games before the first available report have no injury information: missing, not "nobody out"
    if report_start is not None:
        rep_cols = [c for c in out.columns if c.startswith(("vac_", "ret_", "avail_", "share_", "inj_", "opp_vac_"))]
        out.loc[out.GAME_DATE < report_start, rep_cols] = np.nan
    return out
