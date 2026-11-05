"""Pull player & team game logs from stats.nba.com via nba_api and cache to data/raw/nba/.
Everything here is post-game box-score data; features built from it are lagged by game date.
Usage: .venv/bin/python src/fetch_nba.py [--refresh] [seasons...]  (default 2021-22 2022-23)
  --refresh  re-download the given seasons even if cached (use for the current season every day; cached files never update
             otherwise). Each file is replaced only after its new download succeeds.
"""
import sys, time, pathlib
import pandas as pd
from nba_api.stats.endpoints import playergamelogs, teamgamelogs

RAW = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "nba"
RAW.mkdir(parents=True, exist_ok=True)


def cached(path, fn, refresh=False):
    if path.exists() and not refresh:
        return pd.read_parquet(path)
    for attempt in range(4):
        try:
            df = fn()
            if refresh and path.exists() and len(df) < len(pd.read_parquet(path)):
                raise RuntimeError(f"refresh returned fewer rows ({len(df)}) than the cached file")
            tmp = path.with_suffix(".tmp")
            df.to_parquet(tmp)
            tmp.replace(path)
            return df
        except Exception as e:  # noqa
            print("  retry", attempt, type(e).__name__, str(e)[:80])
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"failed {path}")


def pull(season, refresh=False):
    for stype in ["Regular Season", "PlayIn", "Playoffs"]:
        tag = stype.replace(" ", "")
        for measure in ["Base", "Advanced"]:
            p = RAW / f"player_{season}_{tag}_{measure}.parquet"
            df = cached(p, lambda: playergamelogs.PlayerGameLogs(
                season_nullable=season, season_type_nullable=stype,
                measure_type_player_game_logs_nullable=measure, timeout=60).get_data_frames()[0], refresh)
            print(season, stype, measure, df.shape)
            time.sleep(1.5)
        for measure in ["Base", "Advanced"]:
            p = RAW / f"team_{season}_{tag}_{measure}.parquet"
            df = cached(p, lambda: teamgamelogs.TeamGameLogs(
                season_nullable=season, season_type_nullable=stype,
                measure_type_player_game_logs_nullable=measure, timeout=60).get_data_frames()[0], refresh)
            print(season, stype, "team", measure, df.shape)
            time.sleep(1.5)


if __name__ == "__main__":
    refresh = "--refresh" in sys.argv
    seasons = [a for a in sys.argv[1:] if a != "--refresh"] or ["2021-22", "2022-23"]
    for s in seasons:
        pull(s, refresh)
