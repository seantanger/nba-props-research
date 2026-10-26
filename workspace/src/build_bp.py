"""Parse cached BettingPros offer pages into a tidy table and grade them against nba_api box scores.

Output: data/processed/bp_lines.parquet — one row per (event, player, market, book, side) with the book's final
(closing) line/price + its `updated` timestamp, and the market's opening line/price/book/created timestamp.
"""
import json, gzip, pathlib, re, unicodedata
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "bettingpros"
PROC = ROOT / "data" / "processed"
MARKET_MAP = {156: "PTS", 157: "REB", 151: "AST", 162: "FG3M", 335: "PA", 336: "PR", 337: "RA", 338: "PRA", 160: "STL", 152: "BLK"}
BP_TEAM = {"UTH": "UTA", "NOR": "NOP", "NO": "NOP", "SA": "SAS", "GS": "GSW", "NY": "NYK", "PHO": "PHX", "WSH": "WAS", "BKN": "BKN"}


def parse():
    ev = pd.read_parquet(RAW / "events.parquet")
    ev["scheduled"] = pd.to_datetime(ev.scheduled)
    rows = []
    files = sorted((RAW / "offers").glob("*.json.gz"))
    for fp in files:
        with gzip.open(fp, "rt") as f:
            d = json.load(f)
        for o in d.get("offers", []):
            mk = MARKET_MAP.get(o["market_id"])
            if mk is None or not o.get("participants"):
                continue
            pl = o["participants"][0]
            pinfo = pl.get("player") or {}
            for s in o["selections"]:
                side = s["selection"]
                op = s.get("opening_line") or {}
                for b in s["books"]:
                    for ln in b["lines"]:
                        if ln.get("line") is None or ln.get("cost") is None:
                            continue
                        rows.append((o["event_id"], o["market_id"], mk, int(o["player_id"]) if o.get("player_id") else -1, pl["name"],
                                     pinfo.get("team"), pinfo.get("position"), side, b["id"], ln["line"], ln["cost"], ln.get("updated"),
                                     bool(ln.get("main")), bool(ln.get("is_off")), op.get("line"), op.get("cost"), op.get("book_id"), op.get("created")))
    df = pd.DataFrame(rows, columns=["event_id", "market_id", "market", "bp_player_id", "player_name", "bp_team", "bp_pos", "side", "book_id",
                                     "line", "cost", "updated", "main", "is_off", "open_line", "open_cost", "open_book", "open_created"])
    df = df.merge(ev[["event_id", "scheduled", "season", "season_type", "home", "visitor"]], on="event_id", how="left")
    for c in ["updated", "open_created"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["dec_odds"] = np.where(df.cost > 0, 1 + df.cost / 100, 1 + 100 / (-df.cost))
    df["open_dec"] = np.where(df.open_cost > 0, 1 + df.open_cost / 100, 1 + 100 / (-df.open_cost.astype(float)))
    # game date in US Eastern (events scheduled in UTC)
    df["game_date"] = (df.scheduled - pd.Timedelta(hours=5)).dt.normalize()
    df["mins_before_tip"] = (df.scheduled - df.updated).dt.total_seconds() / 60
    df["open_mins_before_tip"] = (df.scheduled - df.open_created).dt.total_seconds() / 60
    return df


ALIASES = {"herb jones": "herbert jones", "moe wagner": "moritz wagner", "cam johnson": "cameron johnson", "cam thomas": "cameron thomas",
           "nic claxton": "nicolas claxton", "kj martin": "kenyon martin", "og anunoby": "og anunoby", "pj washington": "pj washington",
           "jeff dowtin": "jeff dowtin", "gg jackson": "gregory jackson", "trey jemison": "trey jemison", "ron holland": "ronald holland",
           "nah shon hyland": "bones hyland", "bones hyland": "bones hyland", "jabari smith": "jabari smith", "kevin knox": "kevin knox",
           "jaren jackson": "jaren jackson", "dennis smith": "dennis smith", "derrick jones": "derrick jones", "tim hardaway": "tim hardaway"}


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z ]", "", s.lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    s = ALIASES.get(s.strip(), s)
    return re.sub(r"\s+", " ", s).strip()


def grade(df):
    """Attach actual stats from nba_api logs by (game_date, team, normalized name)."""
    logs = pd.concat([pd.read_parquet(p) for p in sorted((ROOT / "data" / "raw" / "nba").glob("player_*_Base.parquet"))], ignore_index=True)
    logs["GAME_DATE"] = pd.to_datetime(logs.GAME_DATE).dt.tz_localize(None).dt.normalize()
    logs["PRA"] = logs.PTS + logs.REB + logs.AST
    logs["PA"] = logs.PTS + logs.AST; logs["PR"] = logs.PTS + logs.REB; logs["RA"] = logs.REB + logs.AST
    logs["nm"] = logs.PLAYER_NAME.map(norm_name)
    logs["home_abbr"] = np.where(logs.MATCHUP.str.contains("vs."), logs.TEAM_ABBREVIATION, logs.MATCHUP.str.split(" ").str[-1])
    logs["away_abbr"] = np.where(logs.MATCHUP.str.contains("vs."), logs.MATCHUP.str.split(" ").str[-1], logs.TEAM_ABBREVIATION)
    df["nm"] = df.player_name.map(norm_name)
    df["home_n"] = df.home.map(lambda t: BP_TEAM.get(t, t)); df["away_n"] = df.visitor.map(lambda t: BP_TEAM.get(t, t))
    # match on date + home/away pair -> game, then name within game
    g = logs[["GAME_ID", "GAME_DATE", "home_abbr", "away_abbr"]].drop_duplicates()
    df = df.merge(g, left_on=["game_date", "home_n", "away_n"], right_on=["GAME_DATE", "home_abbr", "away_abbr"], how="left")
    # fallback: same teams, date +/- 1 day (UTC offset edge cases)
    miss = df.GAME_ID.isna()
    for delta in [1, -1]:
        if miss.sum() == 0:
            break
        g2 = g.copy(); g2["GAME_DATE"] = g2.GAME_DATE - pd.Timedelta(days=delta)
        fix = df.loc[miss, ["game_date", "home_n", "away_n"]].merge(g2, left_on=["game_date", "home_n", "away_n"], right_on=["GAME_DATE", "home_abbr", "away_abbr"], how="left")
        df.loc[miss, "GAME_ID"] = fix.GAME_ID.values
        miss = df.GAME_ID.isna()
    stats = logs[["GAME_ID", "nm", "PLAYER_ID", "MIN", "PTS", "REB", "AST", "FG3M", "PRA", "PA", "PR", "RA", "STL", "BLK"]]
    df = df.merge(stats, on=["GAME_ID", "nm"], how="left")
    # players with no box-score row (did not play) still get a PLAYER_ID from the name -> most recent id map, so voids are kept
    nm2id = logs.sort_values("GAME_DATE").drop_duplicates("nm", keep="last").set_index("nm").PLAYER_ID
    df["PLAYER_ID"] = df.PLAYER_ID.fillna(df.nm.map(nm2id))
    # players with no box-score row in a matched game did not play -> void
    df["played"] = df.MIN.fillna(0) > 0
    actual = np.full(len(df), np.nan)
    for mk in MARKET_MAP.values():
        s = (df.market == mk).values
        actual[s] = df.loc[s, mk].values
    df["actual"] = actual
    df["result"] = np.where(df.GAME_ID.isna(), "unmatched", np.where(~df.played, "void", np.where(
        df.actual == df.line, "push", np.where((df.side == "over") == (df.actual > df.line), "win", "loss"))))
    return df.drop(columns=["home_abbr", "away_abbr", "GAME_DATE", "home_n", "away_n"])


if __name__ == "__main__":
    df = parse()
    print("parsed rows", len(df), "events", df.event_id.nunique(), df.game_date.min(), df.game_date.max())
    df = grade(df)
    print(df.result.value_counts())
    print("unmatched games:", df[df.result == "unmatched"].event_id.nunique(), "of", df.event_id.nunique())
    PROC.mkdir(exist_ok=True)
    df.to_parquet(PROC / "bp_lines.parquet", index=False)
