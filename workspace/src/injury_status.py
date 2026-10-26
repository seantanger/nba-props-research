"""Turn parsed official injury reports into per-game, decision-time player statuses keyed to nba_api ids.

Input : data/processed/injury_reports.parquet (src/parse_injury.py)
Output: data/processed/injury_long.parquet   one row per (report_ts, GAME_ID, TEAM_ABBREVIATION, PLAYER_ID/player, status)
        data/processed/injury_status.parquet one row per (GAME_ID, TEAM_ABBREVIATION, player) with
            status_dec/ts_dec  = latest report at or before tip-off minus DEC_MIN minutes (the decision-time view)
            status_final/ts_final = latest report at or before tip-off (what was known at tip)
            status_first/ts_first = first listing for this game
Decision time is tip-off minus 60 minutes by default: the model/bettor is assumed to act on the last report published
at least an hour before the game. Nothing after that cutoff is used by the features.
"""
import pathlib, re, sys
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
sys.path.insert(0, str(ROOT / "src"))
from build_bp import norm_name  # noqa: E402

DEC_MIN = 60
STATUS_CODE = {"Available": 0, "Probable": 1, "Questionable": 2, "Doubtful": 3, "Out": 4}
TEAM_FIX = {"LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "LA Lakers": "LAL"}


def last_first_to_first_last(s):
    s = str(s)
    if "," in s:
        last, first = s.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return s


def build(dec_min=DEC_MIN):
    rep = pd.read_parquet(PROC / "injury_reports.parquet")
    pg = pd.read_parquet(PROC / "player_games.parquet", columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP", "season"])
    # --- team name -> abbreviation (from the logs themselves, plus fixes)
    tmap = pg.drop_duplicates(["TEAM_NAME", "TEAM_ABBREVIATION"]).set_index("TEAM_NAME").TEAM_ABBREVIATION.to_dict()
    tmap.update(TEAM_FIX)
    rep["team_abbr"] = rep.team.map(tmap)
    unmapped = rep.loc[rep.team_abbr.isna(), "team"].value_counts()
    if len(unmapped):
        print("unmapped team names:", unmapped.head(10).to_dict())
    # --- game id via date + away@home
    rep["away"] = rep.matchup.str.split("@").str[0].str.strip()
    rep["home"] = rep.matchup.str.split("@").str[-1].str.strip()
    games = pg[pg.MATCHUP.str.contains("vs.")][["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "MATCHUP"]].drop_duplicates("GAME_ID")
    games["home"] = games.TEAM_ABBREVIATION
    games["away"] = games.MATCHUP.str.split(" ").str[-1]
    rep = rep.merge(games[["GAME_ID", "GAME_DATE", "home", "away"]], left_on=["game_date", "home", "away"], right_on=["GAME_DATE", "home", "away"], how="left")
    print(f"report rows {len(rep):,}; with GAME_ID {rep.GAME_ID.notna().mean():.3f}")
    rep = rep[rep.GAME_ID.notna()].copy()
    # --- tip-off time (ET) from the report's own game-time column
    hm = rep.game_time.str.extract(r"(\d{1,2}):(\d{2})")
    hour = hm[0].astype(float).fillna(19)
    hour = hour.where(hour >= 11, hour + 12)  # 12-hour clock without AM/PM: 01:00..10:30 are evening tips, 11:00/12:00 are day games
    rep["tip_ts"] = rep.game_date + pd.to_timedelta(hour * 60 + hm[1].astype(float).fillna(0), unit="m")
    # --- player id: match normalized "First Last" to the team's players in that season (by appearance count)
    rep["nm"] = rep.player.map(last_first_to_first_last).map(norm_name)
    pg["nm"] = pg.PLAYER_NAME.map(norm_name)
    pg["season_start"] = pg.season.str[:4].astype(int)
    rep["season_start"] = np.where(rep.game_date.dt.month >= 9, rep.game_date.dt.year, rep.game_date.dt.year - 1)
    roster = (pg.groupby(["season_start", "TEAM_ABBREVIATION", "nm", "PLAYER_ID"]).size().reset_index(name="n")
              .sort_values("n", ascending=False).drop_duplicates(["season_start", "TEAM_ABBREVIATION", "nm"]))
    rep = rep.merge(roster[["season_start", "TEAM_ABBREVIATION", "nm", "PLAYER_ID"]], left_on=["season_start", "team_abbr", "nm"],
                    right_on=["season_start", "TEAM_ABBREVIATION", "nm"], how="left").drop(columns=["TEAM_ABBREVIATION"])
    # fallback: any season for that team (players who never logged a game this season but did before)
    miss = rep.PLAYER_ID.isna()
    r2 = roster.sort_values("n", ascending=False).drop_duplicates(["TEAM_ABBREVIATION", "nm"])[["TEAM_ABBREVIATION", "nm", "PLAYER_ID"]]
    fb = rep.loc[miss, ["team_abbr", "nm"]].merge(r2, left_on=["team_abbr", "nm"], right_on=["TEAM_ABBREVIATION", "nm"], how="left")
    rep.loc[miss, "PLAYER_ID"] = fb.PLAYER_ID.values
    print(f"player id matched {rep.PLAYER_ID.notna().mean():.3f} (unmatched are mostly two-way/long-term players with no logged games)")
    rep = rep.rename(columns={"team_abbr": "TEAM_ABBREVIATION"})
    rep["code"] = rep.status.map(STATUS_CODE)
    long = rep[["report_ts", "GAME_ID", "GAME_DATE", "tip_ts", "TEAM_ABBREVIATION", "PLAYER_ID", "player", "nm", "status", "code", "reason"]].sort_values(["GAME_ID", "TEAM_ABBREVIATION", "nm", "report_ts"])
    long.to_parquet(PROC / "injury_long.parquet", index=False)
    # --- decision-time, final and first views
    key = ["GAME_ID", "TEAM_ABBREVIATION", "nm"]
    dec = long[long.report_ts <= long.tip_ts - pd.Timedelta(minutes=dec_min)].groupby(key).last()[["status", "code", "report_ts"]]
    dec.columns = ["status_dec", "code_dec", "ts_dec"]
    fin = long[long.report_ts <= long.tip_ts].groupby(key).last()[["status", "code", "report_ts"]]
    fin.columns = ["status_final", "code_final", "ts_final"]
    first = long.groupby(key).first()[["status", "report_ts", "GAME_DATE", "tip_ts", "PLAYER_ID", "player", "reason"]]
    first.columns = ["status_first", "ts_first", "GAME_DATE", "tip_ts", "PLAYER_ID", "player", "reason"]
    st = first.join(dec).join(fin).reset_index()
    st.to_parquet(PROC / "injury_status.parquet", index=False)
    print(f"status rows {len(st):,} games {st.GAME_ID.nunique():,} dec coverage {st.status_dec.notna().mean():.3f}")
    print(st.status_dec.value_counts(dropna=False).to_dict())
    return st


if __name__ == "__main__":
    build()
