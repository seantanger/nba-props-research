"""Pull tonight's NBA player-prop odds from The Odds API and print the bets the chosen rule would place.

Usage: .venv/bin/python src/live_odds.py [--max-events N] [--hours H] [--dry] [--books b1,b2] [--mode rule|model|both]
                                        [--overrides auto|none|path.csv] [--injury auto|none|path.pdf] [--model-thr X]
  --max-events  cap on games fetched this run (each costs len(MARKETS)=5 credits; the events list costs 1; default 15)
  --hours       only games commencing within H hours from now (default 30)
  --dry         re-use the latest cached run folder for today only; make no API calls
  --books       bookmakers allowed as bet targets (default fanduel,draftkings,betmgm,betrivers = the Ontario set);
                FanDuel is always loaded as the anchor. Use --books fanduel for a FanDuel-only slate (model-only bets).
  --mode        rule  = FanDuel-anchored rule at the other books (phase-1 chosen rule);
                model = projection-model probability vs each book's price incl. FanDuel (source model_fd / model_book);
                both  = both (default). Rows carry a `source` column.
  --overrides   data/overrides/<ET date>.csv (auto) with columns team,player,status (OUT|DOUBTFUL|QUESTIONABLE|PROBABLE|AVAILABLE)
                written by the reviewer from lineup news; it overrides the official report for tonight's stub games.
  --injury      auto = download today's latest official NBA injury report PDF (cached under data/raw/injury/pdf/) and
                use its statuses for tonight's games; none = ignore; or a path to a PDF.
  --model-thr   edge threshold for model-only bets (default 0.08, chosen on dev: the only threshold at which the model-only path is
                positive at DraftKings/BetMGM/BetRivers in the honest-availability tier; still negative at FanDuel).

Rule (from results/chosen_rule.json, written by src/choose_rule.py): FanDuel-anchored probability with the projection
model's dispersion; bet another book's over/under when model prob - breakeven >= thr, decimal odds in [1.3, 3.0],
one bet per player/market (best edge), flat 1-unit stakes, and only when BOTH the FanDuel market and the book's market
were updated within the last `max_age_min` minutes (The Odds API drops suspended markets after ~15 min).
Raw responses are cached under data/raw/odds_api/<date>/ and never fetched twice. The API key is read from
ODDS_API_KEY and is never printed or written to disk.
"""
import argparse, json, os, pathlib, sys, time
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd, requests
from scipy import stats
from src.common import ROOT, PROC, RESULTS, feature_cols
from src.models import proj_dist
from src.build_bp import norm_name
from src import features as F
from src.parse_injury import parse_pdf

API = "https://api.the-odds-api.com/v4"
SPORT = "basketball_nba"
MARKETS = {"player_points": "PTS", "player_rebounds": "REB", "player_assists": "AST", "player_threes": "FG3M",
           "player_points_rebounds_assists": "PRA"}
RESERVE = 150
CACHE = ROOT / "data" / "raw" / "odds_api"
INJ_PDF = ROOT / "data" / "raw" / "injury" / "pdf"
INJ_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{d}_{hh:02d}_{mm:02d}{ap}.pdf"
STATUS_CODE = {"AVAILABLE": 0, "PROBABLE": 1, "QUESTIONABLE": 2, "DOUBTFUL": 3, "OUT": 4}
TEAM_FIX = {"LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "LA Lakers": "LAL"}
ROTATION_MIN = 15  # a listed teammate counts as news only if he averages this many minutes (10-game mean)


def american_to_dec(a):
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def get(url, params, cache_file):
    if cache_file.exists():
        return json.loads(cache_file.read_text()), None
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("ODDS_API_KEY not set")
    r = requests.get(url, params={**params, "apiKey": key}, timeout=30)
    hdr = {k: r.headers.get(k) for k in ["x-requests-remaining", "x-requests-used", "x-requests-last"]}
    if r.status_code != 200:
        print("HTTP", r.status_code, r.text[:200].replace(key, "***"))
        return None, hdr
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(r.text)
    return r.json(), hdr


def fetch(args):
    # One cache folder per run, so a later run the same day fetches fresh prices; --dry replays the latest run.
    day_dir = CACHE / datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if args.dry:
        runs = sorted(p for p in day_dir.glob("*") if p.is_dir()) if day_dir.exists() else []
        if not runs:
            sys.exit("no cached runs for today")
        cdir = runs[-1]
    else:
        cdir = day_dir / datetime.now(timezone.utc).strftime("%H%M")
    ev_file = cdir / "events.json"
    events, hdr = get(f"{API}/sports/{SPORT}/events", {}, ev_file)
    if hdr:
        print("credits remaining:", hdr["x-requests-remaining"], "used:", hdr["x-requests-used"])
    now = datetime.now(timezone.utc)
    # Pre-game only: after tip, FanDuel's lines are in-game while other books' are stale pre-game lines.
    ev = [e for e in events if now <= datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")) <= now + timedelta(hours=args.hours)]
    print(f"{len(events)} events listed, {len(ev)} within {args.hours}h")
    out = []
    mk = ",".join(MARKETS)
    for e in ev[: args.max_events]:
        f = cdir / f"{e['id']}.json"
        if args.dry and not f.exists():
            continue
        if not f.exists() and hdr and hdr.get("x-requests-remaining") is not None and int(hdr["x-requests-remaining"]) - len(MARKETS) < RESERVE:
            print("stopping: credit reserve reached"); break
        data, h = get(f"{API}/sports/{SPORT}/events/{e['id']}/odds", {"regions": "us", "markets": mk, "oddsFormat": "american"}, f)
        if h and h.get("x-requests-remaining") is not None:
            hdr = h
            print(f"  fetched {e['away_team']} @ {e['home_team']}: cost {h['x-requests-last']}, remaining {h['x-requests-remaining']}")
        if data:
            out.append(data)
    return out


def parse(events):
    rows = []
    for e in events:
        for b in e.get("bookmakers", []):
            for m in b.get("markets", []):
                if m["key"] not in MARKETS:
                    continue
                for o in m.get("outcomes", []):
                    rows.append(dict(event_id=e["id"], home=e["home_team"], away=e["away_team"], commence=e["commence_time"], book=b["key"],
                                     market=MARKETS[m["key"]], updated=m.get("last_update"), player=o.get("description"), side=o["name"].lower(),
                                     line=o.get("point"), price=o["price"]))
    d = pd.DataFrame(rows)
    if d.empty:
        return d
    d["dec"] = d.price.map(american_to_dec)
    d["updated"] = pd.to_datetime(d.updated, utc=True)
    d["commence"] = pd.to_datetime(d.commence, utc=True)
    return d


def team_abbr():
    """Odds API team name -> nba_api abbreviation, matched on the nickname (last word) as in build_live_features."""
    t = pd.read_parquet(PROC / "player_games.parquet", columns=["TEAM_NAME", "TEAM_ABBREVIATION"]).drop_duplicates("TEAM_NAME")
    nick = {n.split()[-1].lower(): a for n, a in t.values}
    return lambda name: nick.get(str(name).split()[-1].lower())


def pairs_from(d):
    key = ["event_id", "book", "market", "player", "line"]
    ov = d[d.side == "over"].drop_duplicates(key)
    un = d[d.side == "under"].drop_duplicates(key)[key + ["price", "dec"]]
    p = ov.merge(un, on=key, suffixes=("_over", "_under"))
    io, iu = 1 / p.dec_over, 1 / p.dec_under
    p["p_over_imp"] = io / (io + iu); p["be_over"] = io; p["be_under"] = iu
    return p


def fetch_injury_report(date_et, dry=False):
    """Latest official injury-report PDF for the ET date (tries :30/:00 slots backwards from the current hour; 11AM earliest).
    Returns (DataFrame from parse_pdf, path) or (None, None). PDFs are cached and never re-downloaded."""
    INJ_PDF.mkdir(parents=True, exist_ok=True)
    cached = sorted(INJ_PDF.glob(f"Injury-Report_{date_et}_*.pdf"))
    now = pd.Timestamp.now(tz="US/Eastern")
    slots = []
    if str(now.date()) == date_et:
        h = now.hour
        for hh in range(h, 10, -1):
            for mm in ([30, 0] if hh < h or now.minute >= 35 else [0]):
                if hh == h and mm == 30 and now.minute < 35:
                    continue
                slots.append((hh, mm))
    else:
        slots = [(hh, mm) for hh in range(21, 10, -1) for mm in (30, 0)]
    if not dry:
        for hh, mm in slots:
            ap_ = "PM" if hh >= 12 else "AM"
            f = INJ_PDF / f"Injury-Report_{date_et}_{hh % 12 or 12:02d}_{mm:02d}{ap_}.pdf"
            if f.exists():
                cached.append(f); break
            try:
                r = requests.get(INJ_URL.format(d=date_et, hh=hh % 12 or 12, mm=mm, ap=ap_), headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            except requests.RequestException as e:
                print("injury report fetch failed:", e); break
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                f.write_bytes(r.content); cached.append(f); break
    if not cached:
        return None, None
    # newest by the report's own timestamp
    best, best_ts = None, None
    for f in sorted(set(cached)):
        df, note = parse_pdf(f)
        if df is None or df.empty:
            continue
        ts = df.report_ts.iloc[0]
        if best_ts is None or ts > best_ts:
            best, best_ts = df, ts
    if best is None:
        return None, None
    return best, best_ts


def load_overrides(path, date_et):
    """data/overrides/<date>.csv with columns team,player,status[,note]. Returns DataFrame or None."""
    if path == "none":
        return None
    f = ROOT / "data" / "overrides" / f"{date_et}.csv" if path == "auto" else pathlib.Path(path)
    if not f.exists():
        return None
    o = pd.read_csv(f, dtype=str).fillna("")
    o.columns = [c.strip().lower() for c in o.columns]
    o["status"] = o.status.str.strip().str.upper()
    bad = ~o.status.isin(STATUS_CODE)
    if bad.any():
        print("overrides with unknown status ignored:", o[bad].to_dict("records"))
    return o[~bad].copy()


def stub_status(ps, ts, pg_last, report, overrides, tmap):
    """Decision-time statuses for tonight's stub games: official report rows (+ reviewer overrides, which win).
    Returns (extra_status DataFrame: GAME_ID, TEAM_ABBREVIATION, PLAYER_ID, code_dec, nm, status, src; n_report, n_override)."""
    # roster lookup: normalized name -> (PLAYER_ID, team) using each player's latest logged team
    look = pg_last.set_index("key")[["PLAYER_ID", "TEAM_ABBREVIATION"]]
    rows = []
    teams_tonight = ts[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE"]]
    if report is not None and len(report):
        rep = report.copy()
        rep["TEAM_ABBREVIATION"] = rep.team.map(lambda t: TEAM_FIX.get(t, tmap.get(t)))
        rep["nm"] = rep.player.map(lambda x: norm_name(f"{x.split(',', 1)[1].strip()} {x.split(',', 1)[0].strip()}" if "," in str(x) else str(x)))
        rep = rep.merge(teams_tonight, on="TEAM_ABBREVIATION", how="inner")
        rep = rep[rep.game_date.isna() | (rep.game_date == rep.GAME_DATE)]
        for r in rep.itertuples():
            pid = look.PLAYER_ID.get(r.nm)
            rows.append(dict(GAME_ID=r.GAME_ID, TEAM_ABBREVIATION=r.TEAM_ABBREVIATION, PLAYER_ID=pid, nm=r.nm, status=r.status.upper(), src="report"))
    n_rep = len(rows)
    if overrides is not None and len(overrides):
        for r in overrides.itertuples():
            ab = TEAM_FIX.get(r.team, tmap.get(r.team, r.team.upper()))
            g = teams_tonight[teams_tonight.TEAM_ABBREVIATION == ab]
            if len(g) == 0:
                print(f"override skipped (team not in this run's games or unknown): {r.team} / {r.player}"); continue
            nm = norm_name(r.player)
            pid = look.PLAYER_ID.get(nm)
            if pid is None:
                print(f"override player not found in logs: {r.player} ({r.team}); status applied by name only")
            rows = [x for x in rows if not (x["GAME_ID"] == g.GAME_ID.iloc[0] and x["nm"] == nm)]
            rows.append(dict(GAME_ID=g.GAME_ID.iloc[0], TEAM_ABBREVIATION=ab, PLAYER_ID=pid, nm=nm, status=r.status, src="override"))
    if not rows:
        return None, 0, 0
    st = pd.DataFrame(rows)
    st["code_dec"] = st.status.map(STATUS_CODE)
    st = st.dropna(subset=["PLAYER_ID"]).drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"], keep="last")
    st["PLAYER_ID"] = st.PLAYER_ID.astype("int64")
    return st, n_rep, len(rows) - n_rep


def build_live_features(d, injury="auto", overrides="auto", dry=False):
    """Append stub rows for tonight's games to the logs and compute strictly-lagged features for them (incl. injury features
    from tonight's official report / overrides). Returns live rows, key map, full logs, status table, report timestamp."""
    pg = pd.read_parquet(PROC / "player_games.parquet", columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_DATE", "SEASON_YEAR", "MIN"])
    pg = pg.sort_values("GAME_DATE")
    pg["MIN10"] = pg.groupby("PLAYER_ID").MIN.transform(lambda x: x.rolling(10, min_periods=1).mean())
    last = pg.groupby("PLAYER_ID").tail(1).copy()
    last["key"] = last.PLAYER_NAME.map(norm_name)
    nick = {n.split()[-1].lower(): a for n, a in pg.drop_duplicates("TEAM_NAME")[["TEAM_NAME", "TEAM_ABBREVIATION"]].values}
    tmap = {n: a for n, a in pg.drop_duplicates("TEAM_NAME")[["TEAM_NAME", "TEAM_ABBREVIATION"]].values}
    season = pg.SEASON_YEAR.max()
    tz = "US/Eastern"
    prow, trow = [], []
    for ev, g in d.groupby("event_id"):
        home_ab, away_ab = nick.get(g.home.iloc[0].split()[-1].lower()), nick.get(g.away.iloc[0].split()[-1].lower())
        if not home_ab or not away_ab:
            print("unmapped team", g.home.iloc[0], g.away.iloc[0]); continue
        gdate = g.commence.iloc[0].tz_convert(tz).normalize().tz_localize(None)
        gid = "LIVE_" + ev[:10]
        for ab, opp, home in [(home_ab, away_ab, 1), (away_ab, home_ab, 0)]:
            tid = pg[pg.TEAM_ABBREVIATION == ab].TEAM_ID.iloc[-1]
            trow.append(dict(SEASON_YEAR=season, TEAM_ID=tid, TEAM_ABBREVIATION=ab, GAME_ID=gid, GAME_DATE=gdate, MATCHUP=f"{ab} {'vs.' if home else '@'} {opp}"))
        for name in g.player.unique():
            m = last[last.key == norm_name(name)]
            if len(m) == 0 or m.TEAM_ABBREVIATION.iloc[0] not in (home_ab, away_ab):
                continue
            ab = m.TEAM_ABBREVIATION.iloc[0]; home = int(ab == home_ab); opp = away_ab if home else home_ab
            prow.append(dict(SEASON_YEAR=season, PLAYER_ID=m.PLAYER_ID.iloc[0], PLAYER_NAME=m.PLAYER_NAME.iloc[0], TEAM_ID=m.TEAM_ID.iloc[0],
                             TEAM_ABBREVIATION=ab, GAME_ID=gid, GAME_DATE=gdate, MATCHUP=f"{ab} {'vs.' if home else '@'} {opp}", live_key=norm_name(name)))
    ps, ts = pd.DataFrame(prow), pd.DataFrame(trow)
    date_et = str(ts.GAME_DATE.min().date())
    # --- tonight's injury statuses: official report (+ overrides)
    report, report_ts = (None, None)
    if injury == "auto":
        report, report_ts = fetch_injury_report(date_et, dry=dry)
    elif injury != "none":
        report, _ = parse_pdf(pathlib.Path(injury)); report_ts = report.report_ts.iloc[0] if report is not None and len(report) else None
    ov = load_overrides(overrides, date_et)
    extra, n_rep, n_ov = stub_status(ps, ts, last, report, ov, tmap)
    print(f"injury statuses for tonight: report {'none' if report_ts is None else report_ts} ({n_rep} rows matched to tonight's teams), overrides {n_ov}")
    if extra is not None:
        extra = extra.merge(last[["PLAYER_ID", "PLAYER_NAME", "MIN10"]], on="PLAYER_ID", how="left")
        listed = extra[(extra.code_dec >= 2) & (extra.MIN10 >= ROTATION_MIN)]
        print("  rotation players questionable/doubtful/out: " + "; ".join(f"{r.TEAM_ABBREVIATION} {r.PLAYER_NAME} {r.status}{'*' if r.src == 'override' else ''}" for r in listed.sort_values(['TEAM_ABBREVIATION', 'MIN10'], ascending=[True, False]).itertuples()))
    out = PROC / "player_games_live.parquet"
    F.build(player_stubs=ps.drop(columns=["live_key"]), team_stubs=ts, out_path=out,
            extra_status=None if extra is None else extra[["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "code_dec"]])
    live = pd.read_parquet(out)
    live = live[live.GAME_ID.astype(str).str.startswith("LIVE_")]
    return live, ps[["PLAYER_ID", "GAME_ID", "TEAM_ABBREVIATION", "live_key"]].drop_duplicates(), pd.read_parquet(out), extra, report_ts


def news_for(extra, live_keys):
    """Per (GAME_ID, TEAM): text of rotation players listed Q/D/Out, and flags teammate_out / per-player self code."""
    if extra is None:
        return {}, {}
    e = extra[extra.MIN10 >= ROTATION_MIN]
    team_news, team_out = {}, {}
    for (gid, ab), g in e[e.code_dec >= 2].groupby(["GAME_ID", "TEAM_ABBREVIATION"]):
        team_news[(gid, ab)] = "; ".join(f"{r.PLAYER_NAME} {r.status}{'*' if r.src == 'override' else ''}" for r in g.sort_values("MIN10", ascending=False).itertuples())
        team_out[(gid, ab)] = int((g.code_dec >= 3).sum())
    return team_news, team_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=15)
    ap.add_argument("--hours", type=float, default=30)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--books", default="fanduel,draftkings,betmgm,betrivers")
    ap.add_argument("--mode", default="both", choices=["rule", "model", "both"])
    ap.add_argument("--overrides", default="auto")
    ap.add_argument("--injury", default="auto")
    ap.add_argument("--model-thr", type=float, default=0.08)
    args = ap.parse_args()
    rule = json.loads((RESULTS / "chosen_rule.json").read_text())
    books = [b.strip() for b in args.books.split(",") if b.strip()]
    print("rule:", rule, "| books:", books, "| mode:", args.mode)

    events = fetch(args)
    d = parse(events)
    if d.empty:
        print("no prop markets returned"); return
    d["key"] = d.player.map(norm_name)
    print(f"{len(d)} outcome rows, {d.book.nunique()} books ({', '.join(sorted(d.book.unique()))}), {d.event_id.nunique()} games, FanDuel present: {'fanduel' in set(d.book)}")
    d = d[d.book.isin(set(books) | {"fanduel"})]

    live, keymap, pg_all, extra, report_ts = build_live_features(d, injury=args.injury, overrides=args.overrides, dry=args.dry)
    keys = keymap.merge(d[["key", "market"]].drop_duplicates(), left_on="live_key", right_on="key")[["PLAYER_ID", "GAME_ID", "market"]].drop_duplicates()
    cutoff = pd.Timestamp(datetime.now()).normalize() + pd.Timedelta(days=1)
    pg_all["MIN"] = pg_all["MIN"].fillna(0)  # stub rows are excluded from training by MIN > 0
    dist = proj_dist(pg_all, keys, cutoff, sorted(keys.market.unique()))
    dist = dist.merge(keymap, on=["PLAYER_ID", "GAME_ID"])
    dist["sigma"] = np.sqrt(dist["var"])
    team_news, team_out = news_for(extra, keymap)
    self_code = {} if extra is None else {(r.GAME_ID, int(r.PLAYER_ID)): (r.code_dec, r.status) for r in extra.itertuples()}

    p = pairs_from(d)
    fd = p[p.book == "fanduel"][["event_id", "market", "key", "line", "p_over_imp", "updated"]].rename(
        columns={"line": "line_fd", "p_over_imp": "p_fd", "updated": "updated_fd"})
    p = p.merge(fd, on=["event_id", "market", "key"], how="left")
    p = p.merge(dist[["live_key", "GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "market", "mu", "sigma"]], left_on=["key", "market"], right_on=["live_key", "market"], how="inner")
    m = p.line_fd + p.sigma * stats.norm.ppf(np.clip(p.p_fd, 1e-3, 1 - 1e-3))
    p["p_anc_over"] = 1 - stats.norm.cdf((p.line - m) / p.sigma)
    p["p_proj_over"] = 1 - stats.norm.cdf((p.line - p.mu) / p.sigma)
    now = pd.Timestamp.now(tz="UTC")
    p["age_book_min"] = (now - p.updated).dt.total_seconds() / 60
    p["age_fd_min"] = (now - p.updated_fd).dt.total_seconds() / 60
    p["mins_to_tip"] = (p.commence - now).dt.total_seconds() / 60
    p["news"] = [team_news.get((g, t), "") for g, t in zip(p.GAME_ID, p.TEAM_ABBREVIATION)]
    p["teammate_out"] = [team_out.get((g, t), 0) for g, t in zip(p.GAME_ID, p.TEAM_ABBREVIATION)]
    sc = [self_code.get((g, int(i)), (0, "")) for g, i in zip(p.GAME_ID, p.PLAYER_ID)]
    p["self_code"] = [c for c, _ in sc]
    p["news"] = np.where(p.self_code >= 1, ["SELF " + s_ + ("; " if n else "") + n for (_, s_), n in zip(sc, p.news)], p.news)

    def collect(q, pcol, thr, source):
        bets = []
        for side in ["over", "under"]:
            r = q.copy(); r["side"] = side
            r["p_model"] = r[pcol] if side == "over" else 1 - r[pcol]
            r["edge"] = r.p_model - r[f"be_{side}"]
            r["dec"] = r[f"dec_{side}"]; r["price"] = r[f"price_{side}"]
            bets.append(r[(r.edge >= thr) & (r.dec >= rule["min_odds"]) & (r.dec <= rule["max_odds"])])
        b = pd.concat(bets); b["source"] = source
        return b

    out = []
    if args.mode in ("rule", "both"):
        q = p[(p.book != "fanduel") & p.line_fd.notna()]
        b = collect(q, "p_anc_over", rule["thr"], "rule").sort_values("edge", ascending=False).drop_duplicates(["event_id", "key", "market"])
        out.append(b)
    if args.mode in ("model", "both"):
        b = collect(p, "p_proj_over", args.model_thr, "model")
        bf = b[b.book == "fanduel"].assign(source="model_fd")
        bo = b[b.book != "fanduel"].sort_values("edge", ascending=False).drop_duplicates(["event_id", "key", "market"]).assign(source="model_book")
        out += [bf, bo]
    b = pd.concat(out, ignore_index=True) if out else p.iloc[0:0]
    if len(b) == 0:
        print("no bets"); return
    b["live"] = (b.age_book_min <= rule["max_age_min"]) & (b.age_fd_min.fillna(0) <= rule["max_age_min"])
    b["kelly_quarter"] = (0.25 * (b.p_model * b.dec - 1) / (b.dec - 1)).clip(0, 0.05).round(4)
    # news_flag (mechanical, pre-game info only). For rule bets: gap = book line - FanDuel line. A teammate out raises the
    # player's true mean, so book>FD with a teammate out means the BOOK has the news and FanDuel is stale -> "explained"
    # (the historical review proxy says veto); book<FD with a teammate out means FanDuel has the news -> "against" (keep;
    # historically the strongest bets). The player's own Q/D status works the other way. Model bets: "news" = check that the
    # listed absences are real and the model saw them (the stub features use the report/overrides).
    gap = b.line - b.line_fd
    expl = ((b.teammate_out > 0) & (gap > 0)) | ((b.self_code >= 2) & (gap < 0))
    agst = ((b.teammate_out > 0) & (gap < 0)) | ((b.self_code >= 2) & (gap > 0))
    has_news = b.news.astype(str).str.len() > 0
    b["news_flag"] = np.select([b.source != "rule", expl & ~agst, agst & ~expl, has_news], ["news", "explained", "against", "same_line"], "none")
    b.loc[(b.source != "rule") & ~has_news, "news_flag"] = "none"
    b["report_ts"] = report_ts
    b["player_id"] = b.PLAYER_ID.astype("int64"); b["team"] = b.TEAM_ABBREVIATION
    ab = team_abbr()
    home_ab, away_ab = b.home.map(ab), b.away.map(ab)
    b["game"] = away_ab + " @ " + home_ab
    b["opponent"] = np.where(b.team == home_ab, away_ab, home_ab)
    b["claude_verdict"] = ""; b["claude_reason"] = ""
    cols = ["player", "player_id", "team", "opponent", "game", "market", "side", "line", "price", "dec", "book", "source", "p_model", "edge", "line_fd", "p_fd", "mu", "sigma",
            "age_book_min", "age_fd_min", "mins_to_tip", "live", "kelly_quarter", "news_flag", "news", "report_ts", "event_id", "commence",
            "claude_verdict", "claude_reason"]
    b = b[cols].round(3).sort_values(["source", "edge"], ascending=[True, False])
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    b.to_csv(RESULTS / f"live_bets_{stamp}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300); pd.set_option("display.max_colwidth", 60)
    print(f"\n{len(p)} priced selections; bets: " + ", ".join(f"{k}={v}" for k, v in b.source.value_counts().items())
          + f" ({int(b.live.sum())} with markets updated within {rule['max_age_min']} min); news_flag: " + ", ".join(f"{k}={v}" for k, v in b.news_flag.value_counts().items()))
    print(b.drop(columns=["event_id", "commence", "claude_verdict", "claude_reason", "player_id"]).to_string(index=False))
    print(f"saved results/live_bets_{stamp}.csv  -> fill claude_verdict/claude_reason, then: src/ledger.py log --picks results/live_bets_{stamp}.csv")


if __name__ == "__main__":
    main()
