"""Turn the raw Kambi/Unibet offer archive (data/raw/wojtrze/offers.csv) into a clean table of
priced over/under pairs joined to nba_api player ids, game ids and actual outcomes.

Output: data/processed/offers.parquet, one row per (bet_id snapshot) with columns
  player_id, game_id, game_date, market, line, side, dec_odds, snap (0=first..), n_snaps, is_last, is_first,
  actual, result (win/loss/push/void)
"""
import pathlib, re, unicodedata
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

ESPN2NBA = {"gs": "GSW", "ny": "NYK", "sa": "SAS", "no": "NOP", "utah": "UTA", "wsh": "WAS", "bkn": "BKN",
            "phx": "PHX", "lal": "LAL", "lac": "LAC", "bos": "BOS", "phi": "PHI", "tor": "TOR", "chi": "CHI",
            "cle": "CLE", "det": "DET", "ind": "IND", "mil": "MIL", "den": "DEN", "min": "MIN", "okc": "OKC",
            "por": "POR", "sac": "SAC", "atl": "ATL", "cha": "CHA", "mia": "MIA", "orl": "ORL", "dal": "DAL",
            "hou": "HOU", "mem": "MEM"}
MARKET = {"PTS": "PTS", "REB": "REB", "AST": "AST", "3PM": "FG3M", "ARP": "PRA"}


def norm_name(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def load_logs():
    frames = []
    for p in sorted((RAW / "nba").glob("player_*_Base.parquet")):
        frames.append(pd.read_parquet(p))
    logs = pd.concat(frames, ignore_index=True)
    logs["GAME_DATE"] = pd.to_datetime(logs["GAME_DATE"]).dt.tz_localize(None).dt.normalize()
    logs["PRA"] = logs.PTS + logs.REB + logs.AST
    logs["home_abbr"] = np.where(logs.MATCHUP.str.contains("vs."), logs.TEAM_ABBREVIATION,
                                 logs.MATCHUP.str.split(" ").str[-1])
    logs["away_abbr"] = np.where(logs.MATCHUP.str.contains("@"), logs.TEAM_ABBREVIATION,
                                 logs.MATCHUP.str.split(" ").str[-1])
    return logs


def main():
    o = pd.read_csv(RAW / "wojtrze" / "offers.csv")
    o["tip_utc"] = pd.to_datetime(o["closed_date"], utc=True)
    # US game date: tip in UTC minus 5h (ET, standard time Nov-Mar; DST shift only matters for ~1am UTC tips)
    o["game_date"] = (o.tip_utc - pd.Timedelta(hours=6)).dt.tz_localize(None).dt.normalize()
    o["home_abbr"] = o.home.map(ESPN2NBA)
    o["away_abbr"] = o.away.map(ESPN2NBA)
    o["market"] = o.bet_type.map(MARKET)
    o["side"] = o.over_under.str.lower()
    o["row"] = np.arange(len(o))
    o["snap"] = o.groupby("bet_id").cumcount()
    o["n_snaps"] = o.groupby("bet_id").bet_id.transform("size")
    o["is_last"] = o.snap == o.n_snaps - 1
    o["is_first"] = o.snap == 0
    o["name_key"] = o.player_ESPN.map(norm_name)

    logs = load_logs()
    # game key -> game_id
    games = logs.drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE", "home_abbr", "away_abbr"]]
    o = o.merge(games.rename(columns={"GAME_DATE": "game_date"}), on=["game_date", "home_abbr", "away_abbr"], how="left")
    miss_game = o.GAME_ID.isna().mean()
    # player key within game
    plog = logs[["GAME_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "MIN", "PTS", "REB", "AST", "FG3M", "PRA"]].copy()
    plog["name_key"] = plog.PLAYER_NAME.map(norm_name)
    # roster of names per player id (for players who did not play in the game we still need an id)
    pid_by_name = logs.drop_duplicates("PLAYER_ID")[["PLAYER_ID", "PLAYER_NAME"]].copy()
    pid_by_name["name_key"] = pid_by_name.PLAYER_NAME.map(norm_name)
    pid_by_name = pid_by_name.drop_duplicates("name_key")
    o = o.merge(pid_by_name[["name_key", "PLAYER_ID"]], on="name_key", how="left")
    # a few manual aliases
    alias = {"nicolasclaxton": "nicclaxton", "kjmartin": "kenyonmartinjr", "cameronthomas": "camthomas",
             "herbertjones": "herbjones", "ogunleye": "ogunleye", "jeffdowtin": "jeffdowtinjr",
             "xaviertillmansr": "xaviertillman", "robertwilliams": "robertwilliamsiii", "ishsmith": "ishmaelsmith",
             "moritzwagner": "mowagner", "kevinknox": "kevinknoxii", "marcusmorris": "marcusmorrissr",
             "jaimejaquez": "jaimejaquezjr", "trey murphy": "treymurphyiii", "treymurphy": "treymurphyiii",
             "dereckliveley": "derecklivelyii", "garytrent": "garytrentjr", "larrynance": "larrynancejr",
             "kellyoubre": "kellyoubrejr", "wendellcarter": "wendellcarterjr", "michaelporter": "michaelporterjr",
             "timhardaway": "timhardawayjr", "jarenjackson": "jarenjacksonjr", "troybrown": "troybrownjr",
             "otto porter": "ottoporterjr", "ottoporter": "ottoporterjr", "dennissmith": "dennissmithjr",
             "kirawlewis": "kiralewisjr", "jabariwalker": "jabariwalker", "jaimejaquezjr": "jaimejaquezjr",
             "pjwashington": "pjwashington", "cjmccollum": "cjmccollum", "nickeilalexanderwalker": "nickeilalexanderwalker",
             "ronholland": "ronaldhollandii", "lonniewalker": "lonniewalkeriv", "jarencollins": "jarencollins",
             "dariusbazley": "dariusbazley", "ggjackson": "ggjackson", "jaylinwilliams": "jaylinwilliams",
             "craigporter": "craigporterjr", "vinceedwards": "vincentedwards", "kevinporter": "kevinporterjr",
             "jaimejaquez jr": "jaimejaquezjr", "patrickbaldwin": "patrickbaldwinjr", "kennylofton": "kennethloftonjr",
             "mosesbrown": "mosesbrown", "jabarismith": "jabarismithjr", "jalenwilliams": "jalenwilliams",
             "aj green": "ajgreen", "ajgriffin": "ajgriffin", "derrickjones": "derrickjonesjr", "jamesnnaji": "jamesnnaji",
             "scottiebarnes": "scottiebarnes", "jrsmith": "jrsmith", "jtthor": "jtthor", "bryantbrown": "brucebrown",
             "charliebrown": "charliebrownjr", "mattryan": "mattryan", "boban": "bobanmarjanovic"}
    m = o.PLAYER_ID.isna()
    o.loc[m, "PLAYER_ID"] = o.loc[m, "name_key"].map(alias).map(pid_by_name.set_index("name_key").PLAYER_ID)
    # fallback: suffix-insensitive / initial+lastname match restricted to players who appeared for either team
    # of the game in that season (team rosters derived from logs up to any date; roster leak is irrelevant to pricing)
    def strip_suffix(k):
        return re.sub(r"(jr|sr|ii|iii|iv)$", "", k)
    roster = logs[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "SEASON_YEAR"]].drop_duplicates()
    roster["key"] = roster.PLAYER_NAME.map(norm_name).map(strip_suffix)
    roster["last"] = roster.PLAYER_NAME.map(lambda n: norm_name(n.split(" ")[-1] if not re.search(r"(Jr\.|Sr\.|II|III|IV)$", n) else n.split(" ")[-2]))
    roster["init"] = roster.PLAYER_NAME.str[0].str.lower()
    o["season"] = np.where(o.game_date >= "2022-08-01", "2022-23", "2021-22")
    todo = o[o.PLAYER_ID.isna()].drop_duplicates(["player_ESPN", "home_abbr", "away_abbr", "season"])
    fixes = {}
    for _, r in todo.iterrows():
        k = strip_suffix(r.name_key)
        cand = roster[(roster.SEASON_YEAR == r.season) & roster.TEAM_ABBREVIATION.isin([r.home_abbr, r.away_abbr])]
        hit = cand[cand.key == k]
        if hit.empty and re.match(r"^[A-Z]\.", str(r.player_ESPN)):
            last = norm_name(r.player_ESPN.split(".", 1)[1]); init = r.player_ESPN[0].lower()
            hit = cand[(cand["last"] == last) & (cand.init == init)]
        if hit.PLAYER_ID.nunique() == 1:
            fixes[(r.player_ESPN, r.home_abbr, r.away_abbr, r.season)] = int(hit.PLAYER_ID.iloc[0])
    key = list(zip(o.player_ESPN, o.home_abbr, o.away_abbr, o.season))
    fixed = pd.Series([fixes.get(k, np.nan) for k in key], index=o.index)
    m = o.PLAYER_ID.isna()
    o.loc[m, "PLAYER_ID"] = fixed[m]
    miss_pid = o.PLAYER_ID.isna().mean()
    print(f"unmatched game {miss_game:.3%}  unmatched player {miss_pid:.3%}")
    print("top unmatched names:", o[o.PLAYER_ID.isna()].player_ESPN.value_counts().head(25).to_dict())

    o = o.dropna(subset=["GAME_ID", "PLAYER_ID"]).copy()
    o["PLAYER_ID"] = o.PLAYER_ID.astype(int)
    # actual stat (NaN if the player did not appear in the box score => void)
    o = o.merge(plog[["GAME_ID", "PLAYER_ID", "MIN", "PTS", "REB", "AST", "FG3M", "PRA"]], on=["GAME_ID", "PLAYER_ID"], how="left")
    stat_cols = {"PTS": "PTS", "REB": "REB", "AST": "AST", "FG3M": "FG3M", "PRA": "PRA"}
    o["actual"] = np.nan
    for mk, col in stat_cols.items():
        o.loc[o.market == mk, "actual"] = o.loc[o.market == mk, col]
    o["played"] = o.MIN.notna() & (o.MIN > 0)
    diff = o.actual - o.line
    o["result"] = np.select(
        [~o.played, diff == 0, (o.side == "over") & (diff > 0), (o.side == "under") & (diff < 0)],
        ["void", "push", "win", "win"], default="loss")
    o["dec_odds"] = o.odds.astype(float)
    o["pnl"] = np.select([o.result == "win", o.result == "loss"], [o.dec_odds - 1.0, -1.0], 0.0)
    out = o[["bet_id", "row", "snap", "n_snaps", "is_first", "is_last", "PLAYER_ID", "player_ESPN", "GAME_ID", "game_date",
             "tip_utc", "home_abbr", "away_abbr", "market", "line", "side", "dec_odds", "actual", "played", "result", "pnl"]]
    out = out.rename(columns={"PLAYER_ID": "player_id", "GAME_ID": "game_id"})
    out.to_parquet(OUT / "offers.parquet", index=False)
    print(out.shape, out.game_date.min(), out.game_date.max())
    print(out[out.is_last].result.value_counts())
    print(out[out.is_last].groupby(["market", "side"]).pnl.agg(["mean", "size"]))


if __name__ == "__main__":
    main()
