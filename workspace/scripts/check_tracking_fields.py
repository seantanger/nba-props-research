"""Probe stats.nba.com tracking / shot / play-type endpoints: fields, coverage, per-game vs season.
Read-only; caches raw JSON under data/raw/nba/tracking_probe/. Does not touch features, models or the ledger."""
import json, sys, time, traceback
from pathlib import Path

from nba_api.stats import endpoints as E

OUT = Path("data/raw/nba/tracking_probe"); OUT.mkdir(parents=True, exist_ok=True)
SLEEP = 2.5
GAME_NEW, GAME_OLD = "0022600116", "0021600001"   # OKC vs LAC 2026-10-25; CLE vs NYK 2016-10-25
WANT = ["potential", "rebound_chance", "reboundchance", "touch", "pass", "secondary", "freethrowassist", "ft_ast",
        "contested", "uncontested", "drive", "speed", "distance"]


def call(name, cls, **kw):
    f = OUT / f"{name}.json"
    if f.exists():
        return json.loads(f.read_text())
    print(f"-- calling {cls.__name__} {kw}", flush=True)
    try:
        r = cls(timeout=60, **kw)
        d = r.get_dict()
    except Exception as e:
        d = {"error": repr(e), "trace": traceback.format_exc()[-600:]}
    f.write_text(json.dumps(d))
    time.sleep(SLEEP)
    return d


def frames(d):
    """-> {dataset name: (headers, nrows)} for either resultSets or boxScore json shapes."""
    out = {}
    if "error" in d:
        return {"ERROR": (d["error"], 0)}
    if "resultSets" in d:
        for rs in d["resultSets"]:
            out[rs["name"]] = (rs["headers"], len(rs["rowSet"]))
    elif "boxScorePlayerTrack" in d or "boxScore" in d:
        bs = d.get("boxScorePlayerTrack") or d.get("boxScore")
        for side in ("homeTeam", "awayTeam"):
            t = bs.get(side, {})
            pl = t.get("players", [])
            if pl:
                stat_keys = list(pl[0].get("statistics", {}).keys())
                out[f"{side}.players"] = (list(pl[0].keys()) + [f"statistics.{k}" for k in stat_keys], len(pl))
    else:
        out["?"] = (list(d.keys()), 0)
    return out


def show(title, d):
    print(f"\n### {title}")
    for k, (hdr, n) in frames(d).items():
        print(f"  [{k}] rows={n}\n    {hdr}")
        flat = " ".join(map(str, hdr)).lower().replace("_", "")
        hits = [w for w in WANT if w.replace("_", "") in flat]
        if hits:
            print(f"    keyword hits: {hits}")


if __name__ == "__main__":
    # 1 & 2: per-game tracking box score, new and old game
    show("BoxScorePlayerTrackV3 2026-27 game", call("bspt_v3_new", E.BoxScorePlayerTrackV3, game_id=GAME_NEW))
    show("BoxScorePlayerTrackV3 2016-17 game", call("bspt_v3_old", E.BoxScorePlayerTrackV3, game_id=GAME_OLD))

    # 3: where potential assists / rebound chances live, and whether they date-filter
    show("LeagueDashPtStats Passing, date range 2016-10-25..2016-10-25",
         call("ldpt_passing_old", E.LeagueDashPtStats, pt_measure_type="Passing", player_or_team="Player",
              season="2016-17", date_from_nullable="10/25/2016", date_to_nullable="10/25/2016"))
    show("LeagueDashPtStats Rebounding, date range 2026-10-25",
         call("ldpt_reb_new", E.LeagueDashPtStats, pt_measure_type="Rebounding", player_or_team="Player",
              season="2026-27", date_from_nullable="10/25/2026", date_to_nullable="10/25/2026"))
    show("LeagueDashPtStats Possessions (touches), date range 2026-10-25",
         call("ldpt_poss_new", E.LeagueDashPtStats, pt_measure_type="Possessions", player_or_team="Player",
              season="2026-27", date_from_nullable="10/25/2026", date_to_nullable="10/25/2026"))
    show("PlayerDashPtPass SGA 2026-27 season",
         call("pdpt_pass_sga", E.PlayerDashPtPass, player_id=1628983, team_id=1610612760, season="2026-27"))

    # 4: shot locations and play types
    show("ShotChartDetail LeBron 2016-17 season (all games)",
         call("shotchart_lbj_1617", E.ShotChartDetail, player_id=2544, team_id=0, season_nullable="2016-17",
              context_measure_simple="FGA"))
    show("ShotChartDetail SGA 2026-27 single game",
         call("shotchart_sga_game", E.ShotChartDetail, player_id=1628983, team_id=0, season_nullable="2026-27",
              game_id_nullable=GAME_NEW, context_measure_simple="FGA"))
    show("SynergyPlayTypes 2026-27 player offensive",
         call("synergy_2627", E.SynergyPlayTypes, season="2026-27", player_or_team_abbreviation="P",
              type_grouping_nullable="offensive"))
    show("SynergyPlayTypes 2016-17 player offensive",
         call("synergy_1617", E.SynergyPlayTypes, season="2016-17", player_or_team_abbreviation="P",
              type_grouping_nullable="offensive"))

    # follow-ups: Synergy needs a play type; can ShotChartDetail return all players for one date?
    show("SynergyPlayTypes 2016-17 Isolation PerGame",
         call("synergy_1617_iso", E.SynergyPlayTypes, season="2016-17", player_or_team_abbreviation="P",
              type_grouping_nullable="offensive", play_type_nullable="Isolation", per_mode_simple="PerGame"))
    show("SynergyPlayTypes 2026-27 Isolation PerGame",
         call("synergy_2627_iso", E.SynergyPlayTypes, season="2026-27", player_or_team_abbreviation="P",
              type_grouping_nullable="offensive", play_type_nullable="Isolation", per_mode_simple="PerGame"))
    show("ShotChartDetail all players (player_id=0), 2016-10-25 only",
         call("shotchart_all_20161025", E.ShotChartDetail, player_id=0, team_id=0, season_nullable="2016-17",
              date_from_nullable="10/25/2016", date_to_nullable="10/25/2016", context_measure_simple="FGA"))
