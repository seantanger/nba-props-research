"""Completeness audit of data/raw/nba/tracking: per kind, dates with games but 0 rows; shots vs box-score FGA per season."""
import json, sys
sys.path.insert(0, "src")
import pandas as pd
from fetch_tracking import TRK, game_dates, n_rows, DEFAULT_SEASONS, _frame
from common import load_player_games
ALL = [f"{y}-{str(y+1)[2:]}" for y in range(2016, 2027)]
gd = game_dates(ALL)
print("game dates", len(gd), gd.GAME_DATE.min().date(), gd.GAME_DATE.max().date())
dates = set(gd.GAME_DATE.dt.strftime("%Y-%m-%d"))
for kind in ["passing", "rebounding", "possessions", "drives", "catchshoot", "pullup", "shots"]:
    files = {f.stem: f for f in (TRK / kind).glob("*.json")}
    want = dates if kind != "shots" else {d for d in dates if d >= "2023-10-01"}
    missing = sorted(want - set(files))
    zero = sorted(d for d in want & set(files) if n_rows(json.loads(files[d].read_text())) == 0)
    print(f"{kind:12s} files={len(files)} missing={len(missing)} {missing[:5]} zero_rows={len(zero)} {zero[:8]}")
# shots coverage: FGA from shots vs box score per season
pg = load_player_games()
parts = []
for f in sorted((TRK / "shots").glob("*.json")) + sorted((TRK / "shots_month").glob("*.json")):
    df = _frame(json.loads(f.read_text()))
    if len(df): parts.append(df[["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "GAME_DATE"]])
s = pd.concat(parts).drop_duplicates(["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID"])
s["GAME_DATE"] = pd.to_datetime(s.GAME_DATE, format="%Y%m%d")
sc = s.groupby(["PLAYER_ID", "GAME_DATE"]).size().rename("shots")
pg2 = pg[["PLAYER_ID", "GAME_DATE", "FGA", "season"]].merge(sc, left_on=["PLAYER_ID", "GAME_DATE"], right_index=True, how="left")
pg2["shots"] = pg2.shots.fillna(0)
g = pg2.groupby("season").apply(lambda d: pd.Series({"fga_box": d.FGA.sum(), "fga_shots": d.shots.sum(), "pg_match": (d.FGA == d.shots).mean()}))
print(g.to_string())
# drives field names
d = json.loads(next((TRK / "drives").glob("2024-11-*.json")).read_text())
for k in ["drives", "catchshoot", "pullup"]:
    d = json.loads(sorted((TRK / k).glob("2024-11-*.json"))[0].read_text())
    print(k, d["resultSets"][0]["headers"])
