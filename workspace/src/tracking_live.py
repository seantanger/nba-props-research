"""Tracking features for the production assists model (P2-12).

Fable's group B from P2-11 (passing / rebounding / possessions rolling means and ratios, shot-zone volumes, opponent-allowed
potential assists / rebound chances / rim attempts / paint touches), with one change: every stats.nba.com tracking count is
rescaled by the league's SEASON-TO-DATE per-minute rate of that count, so the definition changes P2-11 found (touches / time of
possession / drives in 2024-25, rebound chances in 2025-26 and 2026-27) do not shift the features. The rate for a game dated D
uses every tracked player-game of the same season on dates <= D; a game's features only use games before D (shift(1)), so
nothing after D is ever seen. Shot-chart zone counts are not rescaled (they equal box-score FGA in every season).

Only the markets in TRK_MARKETS get these columns (src/models.proj_dist `extra`); the other markets keep the production set.
Usage: add_trk(df) on a player_games-like frame (production file or the live file with tonight's stub rows).
"""
import pathlib, re, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
from tracking_features2 import CORE, ZL, VOL, shot_table, lag_roll  # noqa: E402

TRK_MARKETS = ["AST"]
TRK_RE = re.compile(r"^(opp_)?trk_")
RATIOS = {"reb_cont_share": ("REB_CONTEST", "REB_CHANCES"), "reb_per_ch": ("REB", "REB_CHANCES"), "ast_per_pot": ("AST", "POTENTIAL_AST"),
          "pts_per_touch": ("POINTS", "TOUCHES"), "rim_fg": ("fgm_rim", "fga_rim"), **{f"z_{z}": (f"fga_{z}", "fga_all") for z in ZL}}
ALLOWED = {"pot_ast": "POTENTIAL_AST", "reb_ch": "REB_CHANCES", "fga_rim": "fga_rim", "paint_t": "PAINT_TOUCHES"}


def season_of(d):
    y = np.where(d.dt.month >= 9, d.dt.year, d.dt.year - 1)
    return pd.Series(y, index=d.index)


def normalise(trk):
    """Rescale each tracking count by the league's season-to-date per-minute rate (dates <= the game's date), times the
    all-history rate so the scale stays readable. Returns a copy."""
    t = trk.copy()
    t["_season"] = season_of(t.GAME_DATE)
    ok = t.MIN.fillna(0) > 0
    day = t[ok].groupby(["_season", "GAME_DATE"])[["MIN"] + list(CORE)].sum().sort_index()
    cum = day.groupby(level="_season").cumsum()
    rate = cum[list(CORE)].div(cum.MIN, axis=0)
    ref = day[list(CORE)].sum() / day.MIN.sum()
    scale = (ref / rate).replace([np.inf, -np.inf], np.nan).reset_index()
    t = t.merge(scale.rename(columns={c: f"_k_{c}" for c in CORE}), on=["_season", "GAME_DATE"], how="left")
    for c in CORE:
        t[c] = t[c] * t[f"_k_{c}"]
    return t.drop(columns=["_season"] + [f"_k_{c}" for c in CORE])


def add_trk(df, trk=None, shots=None):
    """Append the trk_* / opp_trk_* columns (group B, normalised) to a player_games-like frame. Row order is preserved."""
    trk = pd.read_parquet(PROC / "tracking_player_games.parquet") if trk is None else trk
    shots = pd.read_parquet(PROC / "tracking_shots.parquet") if shots is None else shots
    trk = normalise(trk).merge(shot_table(shots), on=["PLAYER_ID", "GAME_DATE"], how="outer")
    trk["PLAYER_ID"] = trk.PLAYER_ID.astype("int64")
    raw = sorted(set(CORE) | {"REB", "AST", "POINTS"} | set(VOL))
    T = lambda c: f"_t_{c}"
    df = df.assign(_row=np.arange(len(df)))
    df = df.merge(trk[["PLAYER_ID", "GAME_DATE"] + raw].rename(columns={c: T(c) for c in raw}), on=["PLAYER_ID", "GAME_DATE"], how="left")
    gap = df[T("TOUCHES")].notna() & df[T("REB_CHANCES")].isna()  # Rebounding endpoint omits 0-REB players
    for c in ["OREB_CHANCES", "DREB_CHANCES", "REB_CHANCES", "REB_CONTEST"]:
        df.loc[gap, T(c)] = 0.0
    df.loc[gap, T("REB")] = df.loc[gap, "REB"]
    df = df.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    g = df.groupby("PLAYER_ID", sort=False)
    F = {}
    for rawc, name in CORE.items():
        for w in [5, 10, 20]:
            F[f"trk_{name}_m{w}"] = lag_roll(g, T(rawc), w)
    for v in VOL:
        for w in [5, 10, 20]:
            F[f"trk_{v}_m{w}"] = lag_roll(g, T(v), w)
    for name, (num, den) in RATIOS.items():
        for w in [10, 20]:
            F[f"trk_{name}_r{w}"] = lag_roll(g, T(num), w, "sum") / lag_roll(g, T(den), w, "sum").replace(0, np.nan)
    df = pd.concat([df, pd.DataFrame(F, index=df.index)], axis=1)
    # opponent-allowed: lagged 15-game mean of what the defending team gave up
    tg = df.groupby(["GAME_ID", "TEAM_ABBREVIATION", "opp", "GAME_DATE"], as_index=False).agg(
        **{k: (T(v), "sum") for k, v in ALLOWED.items()}, n_trk=(T("TOUCHES"), "count"))
    tg.loc[tg.n_trk == 0, list(ALLOWED)] = np.nan
    allowed = tg.drop(columns="TEAM_ABBREVIATION").rename(columns={"opp": "TEAM_ABBREVIATION"})
    allowed = allowed.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    ag = allowed.groupby("TEAM_ABBREVIATION", sort=False)
    for k in ALLOWED:
        allowed[f"opp_trk_{k}_15"] = lag_roll(ag, k, 15)
    allowed = allowed.rename(columns={"TEAM_ABBREVIATION": "opp"})[["GAME_ID", "opp"] + [f"opp_trk_{k}_15" for k in ALLOWED]]
    df = df.merge(allowed, on=["GAME_ID", "opp"], how="left")
    df = df.drop(columns=[c for c in df.columns if c.startswith("_t_")]).sort_values("_row").drop(columns="_row").reset_index(drop=True)
    return df


def trk_cols(df):
    return [c for c in df.columns if TRK_RE.match(c)]
