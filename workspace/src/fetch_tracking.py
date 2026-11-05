"""Fetch per-DATE player tracking aggregates and shot charts from stats.nba.com (nba_api) and cache one JSON per date.

Endpoints (one call per game date each; a one-day DateFrom=DateTo window equals that game's box-score tracking numbers,
verified against BoxScorePlayerTrackV3 in P2-10):
  LeagueDashPtStats Player/Totals  pt_measure_type = Passing | Rebounding | Possessions | Drives | CatchShoot | PullUpShot
  ShotChartDetail player_id=0 (all players), one row per FGA
Cache: data/raw/nba/tracking/<kind>/<YYYY-MM-DD>.json
Kind `shots_month` (backfill only): ShotChartDetail per (season, season type, calendar month) instead of per date, ~9 calls a
season instead of ~200 -> data/raw/nba/tracking/shots_month/<season>_<type>_<YYYY-MM>.json. A whole-season call is truncated
(2016-17 RS returned 106,496 shots, games only through Jan 30), so a month (<40k shots) is the largest safe window; each month
file is checked against the game count from the player logs.
Resumable: a date is skipped if its file exists AND the date is before today (ET) AND the file is not empty-with-games
(dates on/after today are always re-fetched so the current season never freezes; `--refresh-from YYYY-MM-DD` forces re-fetch).
Usage: .venv/bin/python src/fetch_tracking.py [--seasons 2023-24 2024-25 ...] [--refresh-from DATE] [--sleep 2.5]
Then: .venv/bin/python src/fetch_tracking.py --build   -> data/processed/tracking_player_games.parquet, tracking_shots.parquet
"""
import argparse, json, pathlib, sys, time, datetime as dt
import pandas as pd
from zoneinfo import ZoneInfo

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nba"
TRK = RAW / "tracking"
PROC = ROOT / "data" / "processed"
ENDPOINTS = {"passing": "Passing", "rebounding": "Rebounding", "possessions": "Possessions", "drives": "Drives",
             "catchshoot": "CatchShoot", "pullup": "PullUpShot", "shots": None, "shots_month": None}
DAILY_KINDS = [k for k in ENDPOINTS if k != "shots_month"]
SHOT_CAP_WARN = 90000  # a month window should be far below the ~106k truncation seen on a full-season call
DEFAULT_SEASONS = ["2023-24", "2024-25", "2025-26", "2026-27"]


def game_dates(seasons):
    """Distinct game dates (all season types) from the cached player logs, with the season each belongs to."""
    rows = []
    for s in seasons:
        for p in sorted(RAW.glob(f"player_{s}_*_Base.parquet")):
            d = pd.read_parquet(p, columns=["GAME_DATE", "GAME_ID"])
            d["GAME_DATE"] = pd.to_datetime(d.GAME_DATE).dt.tz_localize(None).dt.normalize()
            d["stype"] = d.GAME_ID.astype(str).str[2].map({"2": "Regular Season", "4": "Playoffs", "5": "PlayIn"})
            rows.append(d.groupby(["GAME_DATE", "stype"]).GAME_ID.nunique().rename("n_games").reset_index().assign(season=s))
    out = pd.concat(rows).groupby(["GAME_DATE", "season"], as_index=False).agg(n_games=("n_games", "sum"), stypes=("stype", lambda x: sorted(set(x))))
    return out.sort_values("GAME_DATE")


def _call(fn, retries=40):
    """stats.nba.com answers bursts with silent read-timeouts that last 10-20 min; back off up to 5 min and keep trying."""
    delay = 10
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # throttling / timeouts
            print(f"    retry {attempt + 1}/{retries} {type(e).__name__}: {str(e)[:100]}; sleeping {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)
    raise RuntimeError("gave up")


def fetch_date(kind, date, season, sleep, stypes=("Regular Season",)):
    """One call per season type present on that date (Regular Season / PlayIn / Playoffs); rowSets are concatenated."""
    from nba_api.stats.endpoints import leaguedashptstats, shotchartdetail
    ds = date.strftime("%m/%d/%Y")
    out = None
    for st in stypes:
        if kind == "shots":
            fn = lambda: shotchartdetail.ShotChartDetail(player_id=0, team_id=0, season_nullable=season, context_measure_simple="FGA",
                                                         date_from_nullable=ds, date_to_nullable=ds, season_type_all_star=st,
                                                         timeout=60).get_dict()
        else:
            fn = lambda: leaguedashptstats.LeagueDashPtStats(pt_measure_type=ENDPOINTS[kind], player_or_team="Player", per_mode_simple="Totals",
                                                             season=season, season_type_all_star=st,
                                                             date_from_nullable=ds, date_to_nullable=ds, timeout=60).get_dict()
        d = _call(fn)
        time.sleep(sleep)
        if out is None:
            out = d
        else:
            for a, b in zip(out["resultSets"], d["resultSets"]):
                if a["name"] != "LeagueAverages":
                    a["rowSet"].extend(b["rowSet"])
    return out


def n_rows(d):
    return sum(len(rs["rowSet"]) for rs in d.get("resultSets", []) if rs["name"] != "LeagueAverages")


def fetch_shots_months(seasons, sleep):
    """Backfill shot charts one (season, season type, month) window per call; skip windows already cached with all games."""
    from nba_api.stats.endpoints import shotchartdetail
    today_et = pd.Timestamp(dt.datetime.now(ZoneInfo("America/New_York")).date())
    rows = []
    for s in seasons:
        for p in sorted(RAW.glob(f"player_{s}_*_Base.parquet")):
            d = pd.read_parquet(p, columns=["GAME_DATE", "GAME_ID"])
            d["GAME_DATE"] = pd.to_datetime(d.GAME_DATE).dt.tz_localize(None).dt.normalize()
            d["stype"] = d.GAME_ID.astype(str).str[2].map({"2": "Regular Season", "4": "Playoffs", "5": "PlayIn"})
            rows.append(d.assign(season=s))
    g = pd.concat(rows).drop_duplicates("GAME_ID")
    g["month"] = g.GAME_DATE.dt.to_period("M")
    windows = g.groupby(["season", "stype", "month"]).agg(n_games=("GAME_ID", "nunique"), d0=("GAME_DATE", "min"), d1=("GAME_DATE", "max")).reset_index()
    print(f"shots_month: {len(windows)} windows for {seasons}", flush=True)
    t0 = time.time()
    for i, w in enumerate(windows.itertuples()):
        f = TRK / "shots_month" / f"{w.season}_{w.stype.replace(' ', '')}_{w.month}.json"
        if f.exists() and w.d1 < today_et:
            got = _frame(json.loads(f.read_text()))
            if len(got) and got.GAME_ID.nunique() >= w.n_games:
                continue
        f.parent.mkdir(parents=True, exist_ok=True)
        fn = lambda: shotchartdetail.ShotChartDetail(player_id=0, team_id=0, season_nullable=w.season, context_measure_simple="FGA",
                                                     date_from_nullable=w.d0.strftime("%m/%d/%Y"), date_to_nullable=w.d1.strftime("%m/%d/%Y"),
                                                     season_type_all_star=w.stype, timeout=180).get_dict()
        d = _call(fn)
        time.sleep(sleep)
        f.write_text(json.dumps(d))
        got = _frame(d)
        ng = got.GAME_ID.nunique() if len(got) else 0
        flag = "  <-- MISSING GAMES" if ng < w.n_games else ""
        flag += "  <-- NEAR TRUNCATION CAP" if len(got) >= SHOT_CAP_WARN else ""
        print(f"  [{i + 1}/{len(windows)}] shots {w.season} {w.stype} {w.month}: shots={len(got)} games={ng}/{w.n_games} "
              f"elapsed={time.time() - t0:.0f}s{flag}", flush=True)
    print("done", flush=True)


def run_fetch(seasons, refresh_from, sleep, kinds):
    if "shots_month" in kinds:
        fetch_shots_months(seasons, sleep)
        kinds = [k for k in kinds if k != "shots_month"]
        if not kinds:
            return
    today_et = pd.Timestamp(dt.datetime.now(ZoneInfo("America/New_York")).date())
    dates = game_dates(seasons)
    print(f"{len(dates)} game dates {dates.GAME_DATE.min().date()}..{dates.GAME_DATE.max().date()}; kinds={kinds}", flush=True)
    todo = []
    for _, r in dates.iterrows():
        for kind in kinds:
            f = TRK / kind / f"{r.GAME_DATE.date()}.json"
            fresh = f.exists() and r.GAME_DATE < today_et and not (refresh_from is not None and r.GAME_DATE >= refresh_from)
            if fresh:
                try:
                    if n_rows(json.loads(f.read_text())) > 0:
                        continue
                except Exception:
                    pass
            todo.append((kind, r.GAME_DATE, r.season, r.n_games, r.stypes, f))
    print(f"{len(todo)} calls to make", flush=True)
    t0 = time.time()
    for i, (kind, date, season, ng, stypes, f) in enumerate(todo):
        f.parent.mkdir(parents=True, exist_ok=True)
        d = fetch_date(kind, date, season, sleep, stypes)
        f.write_text(json.dumps(d))
        if i % 25 == 0 or n_rows(d) == 0:
            print(f"  [{i + 1}/{len(todo)}] {kind} {date.date()} ({season}, {ng} games) rows={n_rows(d)} elapsed={time.time() - t0:.0f}s", flush=True)
    print("done", flush=True)


# ------------------------------------------------------------------ build tidy tables
def _frame(d):
    rs = [r for r in d.get("resultSets", []) if r["name"] != "LeagueAverages"]
    if not rs:
        return pd.DataFrame()
    return pd.DataFrame(rs[0]["rowSet"], columns=rs[0]["headers"])


def build_tables():
    """Per player-date tracking table (passing + rebounding + possessions + drives + catchshoot + pullup joined) and the per-shot table."""
    frames = {}
    for kind in ["passing", "rebounding", "possessions", "drives", "catchshoot", "pullup"]:
        parts = []
        for f in sorted((TRK / kind).glob("*.json")):
            df = _frame(json.loads(f.read_text()))
            if len(df):
                df["GAME_DATE"] = pd.Timestamp(f.stem)
                parts.append(df)
        frames[kind] = pd.concat(parts, ignore_index=True)
        print(kind, frames[kind].shape, flush=True)
    keep = {
        "passing": ["PLAYER_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_DATE", "GP", "MIN", "PASSES_MADE", "PASSES_RECEIVED", "AST", "FT_AST",
                    "SECONDARY_AST", "POTENTIAL_AST", "AST_PTS_CREATED", "AST_ADJ"],
        "rebounding": ["PLAYER_ID", "GAME_DATE", "OREB", "OREB_CONTEST", "OREB_CHANCES", "OREB_CHANCE_DEFER", "DREB", "DREB_CONTEST",
                       "DREB_CHANCES", "DREB_CHANCE_DEFER", "REB", "REB_CONTEST", "REB_CHANCES", "REB_CHANCE_DEFER", "AVG_REB_DIST"],
        "possessions": ["PLAYER_ID", "GAME_DATE", "POINTS", "TOUCHES", "FRONT_CT_TOUCHES", "TIME_OF_POSS", "AVG_SEC_PER_TOUCH",
                        "AVG_DRIB_PER_TOUCH", "ELBOW_TOUCHES", "POST_TOUCHES", "PAINT_TOUCHES"],
        "drives": ["PLAYER_ID", "GAME_DATE", "DRIVES", "DRIVE_FGM", "DRIVE_FGA", "DRIVE_FTM", "DRIVE_FTA", "DRIVE_PTS", "DRIVE_PASSES",
                   "DRIVE_AST", "DRIVE_TOV", "DRIVE_PF"],
        "catchshoot": ["PLAYER_ID", "GAME_DATE", "CATCH_SHOOT_FGM", "CATCH_SHOOT_FGA", "CATCH_SHOOT_PTS", "CATCH_SHOOT_FG3M", "CATCH_SHOOT_FG3A"],
        "pullup": ["PLAYER_ID", "GAME_DATE", "PULL_UP_FGM", "PULL_UP_FGA", "PULL_UP_PTS", "PULL_UP_FG3M", "PULL_UP_FG3A"],
    }
    t = frames["passing"][keep["passing"]]
    for kind in ["rebounding", "possessions", "drives", "catchshoot", "pullup"]:
        t = t.merge(frames[kind][keep[kind]], on=["PLAYER_ID", "GAME_DATE"], how="outer")
    dup = t.duplicated(["PLAYER_ID", "GAME_DATE"]).sum()
    assert dup == 0, f"{dup} duplicate player-dates (a player cannot play twice in one day)"
    assert (t.GP.dropna() == 1).all(), "GP != 1 found: one-day window did not isolate a single game"
    t = t.rename(columns={"TEAM_ABBREVIATION": "trk_team"})
    t.to_parquet(PROC / "tracking_player_games.parquet", index=False)
    print("tracking_player_games", t.shape, t.GAME_DATE.min().date(), t.GAME_DATE.max().date(), flush=True)

    parts = []
    for f in sorted((TRK / "shots").glob("*.json")) + sorted((TRK / "shots_month").glob("*.json")):
        df = _frame(json.loads(f.read_text()))
        if len(df):
            parts.append(df)
    s = pd.concat(parts, ignore_index=True)
    s["GAME_DATE"] = pd.to_datetime(s.GAME_DATE, format="%Y%m%d")
    s = s.drop_duplicates(["GAME_ID", "GAME_EVENT_ID", "PLAYER_ID"])
    s.to_parquet(PROC / "tracking_shots.parquet", index=False)
    print("tracking_shots", s.shape, s.GAME_DATE.min().date(), s.GAME_DATE.max().date(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", default=DEFAULT_SEASONS)
    ap.add_argument("--refresh-from", default=None)
    ap.add_argument("--sleep", type=float, default=2.5)
    ap.add_argument("--kinds", nargs="*", default=DAILY_KINDS, choices=list(ENDPOINTS))
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.build:
        build_tables()
    else:
        run_fetch(a.seasons, pd.Timestamp(a.refresh_from) if a.refresh_from else None, a.sleep, a.kinds)
