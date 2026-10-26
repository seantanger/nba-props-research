"""Forward paper-trading ledger for NBA player props (flat 1-unit stakes, no real bets).

Usage: .venv/bin/python src/ledger.py <command> [options]
  log    --picks results/live_bets_<stamp>.csv [--run-stamp S]
         Append every row of a reviewed live-bets file to data/ledger/ledger.csv. Each row must carry
         claude_verdict (keep|veto) and a one-sentence claude_reason. Rows already logged (same game_date,
         player, market, side, book, source) are skipped, so the file can be re-logged after edits.
  settle [--date YYYY-MM-DD]
         Fill actual/result/profit_units for open picks from NBA box scores (nba_api PlayerGameLogs, cached in
         data/raw/nba/daily/). A pick whose player has no box-score row (DNP) is voided (0 units).
  close  Fill closing_line/closing_price/clv_line from the latest cached pre-tip odds snapshot
         (data/raw/odds_api/<date>/<HHMM>/) for the same book, market, player and side.
  report [--since YYYY-MM-DD]
         Summarise settled picks by source (rule vs model_fd vs model_book) x verdict (kept vs vetoed):
         picks, W/L/P/void, units, ROI with a bootstrap interval, and average closing-line value.
Columns: logged_at, run_stamp, game_date, commence, event_id, player, player_id, team, opponent, game, market, side, line, price,
dec, book, source, p_model, edge, line_fd, p_fd, mu, sigma, news_flag, news, claude_verdict, claude_reason,
actual, result (open|win|loss|push|void), profit_units, closing_line, closing_price, clv_line.
"""
import argparse, json, os, pathlib, sys, time
from datetime import datetime
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
LEDGER = pathlib.Path(os.environ.get("LEDGER_PATH", ROOT / "data" / "ledger" / "ledger.csv"))
DAILY = ROOT / "data" / "raw" / "nba" / "daily"
ODDS = ROOT / "data" / "raw" / "odds_api"
sys.path.insert(0, str(ROOT))
from src.build_bp import norm_name  # noqa: E402

COLS = ["logged_at", "run_stamp", "game_date", "commence", "event_id", "player", "player_id", "team", "opponent", "game", "market", "side",
        "line", "price", "dec", "book", "source", "p_model", "edge", "line_fd", "p_fd", "mu", "sigma", "news_flag", "news",
        "claude_verdict", "claude_reason", "actual", "result", "profit_units", "closing_line", "closing_price", "clv_line"]
KEY = ["game_date", "player", "market", "side", "book", "source"]
STAT = {"PTS": ["PTS"], "REB": ["REB"], "AST": ["AST"], "FG3M": ["FG3M"], "PRA": ["PTS", "REB", "AST"]}
MARKET_KEYS = {"PTS": "player_points", "REB": "player_rebounds", "AST": "player_assists", "FG3M": "player_threes",
               "PRA": "player_points_rebounds_assists"}


def american_profit(price, stake=1.0):
    """Profit in units on a winning bet of `stake` units at American odds `price`."""
    price = float(price)
    return stake * (price / 100 if price > 0 else 100 / abs(price))


def american_to_dec(a):
    a = float(a)
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def load():
    if LEDGER.exists():
        df = pd.read_csv(LEDGER, dtype={"event_id": str, "run_stamp": str})
        for c in COLS:
            if c not in df:
                df[c] = np.nan
        return df[COLS]
    return pd.DataFrame(columns=COLS)


def save(df):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    df[COLS].to_csv(LEDGER, index=False)


def cmd_log(args):
    picks = pd.read_csv(args.picks, dtype={"event_id": str})
    need = {"player", "market", "side", "line", "price", "book", "claude_verdict", "claude_reason"}
    miss = need - set(picks.columns)
    if miss:
        sys.exit(f"picks file lacks columns {sorted(miss)}; review every row first (claude_verdict keep|veto, claude_reason)")
    v = picks.claude_verdict.astype(str).str.strip().str.lower()
    bad = ~v.isin(["keep", "veto"])
    if bad.any() or (picks.claude_reason.astype(str).str.strip().str.len() < 5).any():
        sys.exit(f"{int(bad.sum())} rows without a keep/veto verdict or without a reason; review them before logging")
    picks["claude_verdict"] = v
    for c in COLS:
        if c not in picks:
            picks[c] = np.nan
    if "commence" in picks and picks.commence.notna().any():
        picks["game_date"] = pd.to_datetime(picks.commence, utc=True).dt.tz_convert("US/Eastern").dt.strftime("%Y-%m-%d")
    elif picks.game_date.isna().all():
        picks["game_date"] = args.date or datetime.now().strftime("%Y-%m-%d")
    picks["source"] = picks.source.fillna("rule")
    picks["dec"] = picks.dec.where(picks.dec.notna(), picks.price.map(american_to_dec))
    picks["logged_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    picks["run_stamp"] = args.run_stamp or pathlib.Path(args.picks).stem.replace("live_bets_", "")
    picks["result"] = "open"
    led = load()
    merged = picks.merge(led[KEY].drop_duplicates().assign(_dup=1), on=KEY, how="left")
    new = picks[merged._dup.isna().values]
    led = pd.concat([led, new[COLS]], ignore_index=True)
    save(led)
    print(f"logged {len(new)} new picks ({len(picks) - len(new)} already present); ledger now {len(led)} rows, "
          f"{int((led.claude_verdict == 'keep').sum())} kept / {int((led.claude_verdict == 'veto').sum())} vetoed")


def box_scores(date):
    """Player box-score lines for one game date (ET), cached. Returns DataFrame with PLAYER_ID, nm, PTS, REB, AST, FG3M, MIN."""
    DAILY.mkdir(parents=True, exist_ok=True)
    p = DAILY / f"{date}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    from nba_api.stats.endpoints import playergamelogs
    d = pd.Timestamp(date)
    season = f"{d.year}-{str(d.year + 1)[2:]}" if d.month >= 9 else f"{d.year - 1}-{str(d.year)[2:]}"
    frames = []
    for stype in ["Regular Season", "Playoffs", "PlayIn"]:
        for attempt in range(3):
            try:
                df = playergamelogs.PlayerGameLogs(season_nullable=season, season_type_nullable=stype,
                                                   date_from_nullable=d.strftime("%m/%d/%Y"), date_to_nullable=d.strftime("%m/%d/%Y"),
                                                   timeout=60).get_data_frames()[0]
                frames.append(df); break
            except Exception as e:  # noqa
                print("  retry", attempt, type(e).__name__, str(e)[:80]); time.sleep(5 * (attempt + 1))
        time.sleep(1.0)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if df.empty:
        return df
    df = df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN", "PTS", "REB", "AST", "FG3M"]].copy()
    df["nm"] = df.PLAYER_NAME.map(norm_name)
    # only cache once the day is over (games finished) so a partial pull is not frozen
    if pd.Timestamp.now(tz="US/Eastern").tz_localize(None) > d + pd.Timedelta(hours=30):
        df.to_parquet(p, index=False)
    return df


def settle_rows(picks, stats):
    """Settle picks against box scores. picks: player, market, side, line, price[, player_id]. Returns actual, result, profit_units."""
    out = picks.copy()
    st = stats.copy()
    st["PRA"] = st.PTS + st.REB + st.AST
    by_id = st.set_index("PLAYER_ID") if "PLAYER_ID" in st else None
    by_nm = st.drop_duplicates("nm").set_index("nm")
    actual, result, profit = [], [], []
    for _, r in out.iterrows():
        row = None
        pid = r.get("player_id")
        if by_id is not None and pd.notna(pid) and int(pid) in by_id.index:
            row = by_id.loc[int(pid)]
            row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
        elif norm_name(r.player) in by_nm.index:
            row = by_nm.loc[norm_name(r.player)]
        if row is None or pd.isna(row.get("MIN")) or float(row.get("MIN") or 0) <= 0:
            actual.append(np.nan); result.append("void"); profit.append(0.0); continue
        a = float(row[r.market]) if r.market != "PRA" else float(row.PRA)
        actual.append(a)
        if a == r.line:
            result.append("push"); profit.append(0.0)
        elif (a > r.line) == (r.side == "over"):
            result.append("win"); profit.append(american_profit(r.price))
        else:
            result.append("loss"); profit.append(-1.0)
    out["actual"], out["result"], out["profit_units"] = actual, result, profit
    return out


def cmd_settle(args):
    led = load()
    if led.empty:
        print("ledger empty"); return
    dates = [args.date] if args.date else sorted(led.loc[led.result.isin(["open"]) | led.result.isna(), "game_date"].dropna().unique())
    today = datetime.now().strftime("%Y-%m-%d")
    for date in dates:
        if date >= today:
            print(f"{date}: games not finished, skipping"); continue
        st = box_scores(date)
        m = (led.game_date == date) & (led.result.isin(["open"]) | led.result.isna())
        if st.empty:
            print(f"{date}: no box scores returned yet"); continue
        s = settle_rows(led.loc[m], st)
        led.loc[m, ["actual", "result", "profit_units"]] = s[["actual", "result", "profit_units"]].values
        print(f"{date}: settled {int(m.sum())} picks: " + ", ".join(f"{k}={v}" for k, v in s.result.value_counts().items()))
    save(led)


def latest_pretip_snapshot(date_et, commence, after=None):
    """Newest cached run folder whose snapshot time is before tip (and after `after`, the pick's own run time)."""
    c = pd.Timestamp(commence)
    runs = []
    for day in {date_et, c.tz_convert("UTC").strftime("%Y-%m-%d")}:
        dd = ODDS / day
        if not dd.exists():
            continue
        for r in dd.glob("*"):
            if r.is_dir() and r.name.isdigit():
                ts = pd.Timestamp(f"{day} {r.name[:2]}:{r.name[2:]}", tz="UTC")
                if ts <= c and (after is None or ts > after):
                    runs.append((ts, r))
    return sorted(runs)[-1][1] if runs else None


def cmd_close(args):
    led = load()
    m = led.closing_line.isna() & led.event_id.notna() & led.commence.notna()
    n = 0
    for i, r in led[m].iterrows():
        try:
            after = pd.Timestamp(datetime.strptime(str(r.run_stamp), "%Y%m%d_%H%M")).tz_localize("US/Eastern").tz_convert("UTC")
        except ValueError:
            after = None
        snap = latest_pretip_snapshot(r.game_date, r.commence, after)
        if snap is None:
            continue
        f = snap / f"{r.event_id}.json"
        if not f.exists():
            continue
        e = json.loads(f.read_text())
        for b in e.get("bookmakers", []):
            if b["key"] != r.book:
                continue
            for mk in b.get("markets", []):
                if mk["key"] != MARKET_KEYS.get(r.market):
                    continue
                for o in mk.get("outcomes", []):
                    if o["name"].lower() == r.side and norm_name(o.get("description", "")) == norm_name(r.player):
                        led.loc[i, "closing_line"] = o.get("point"); led.loc[i, "closing_price"] = o["price"]
                        # positive = the line moved in our favour (under: line went up; over: line went down)
                        led.loc[i, "clv_line"] = (o.get("point") - r.line) * (1 if r.side == "under" else -1)
                        n += 1
    save(led)
    print(f"filled closing lines for {n} picks (closing snapshot = latest cached pre-tip run; run the live script again near tip to capture it)")


def boot_roi(pnl, dates, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"pnl": pnl, "g": dates}).groupby("g").pnl.agg(["sum", "count"])
    if len(d) == 0:
        return np.nan, np.nan
    idx = rng.integers(0, len(d), size=(n, len(d)))
    s, c = d["sum"].values[idx].sum(1), d["count"].values[idx].sum(1)
    r = s / np.where(c == 0, 1, c)
    return np.percentile(r, 2.5), np.percentile(r, 97.5)


def cmd_report(args):
    led = load()
    if args.since:
        led = led[led.game_date >= args.since]
    if led.empty:
        print("ledger empty"); return
    led["verdict"] = led.claude_verdict.fillna("keep")
    led["stake"] = 1.0
    print(f"ledger: {len(led)} picks, {led.game_date.nunique()} dates ({led.game_date.min()}..{led.game_date.max()}), "
          f"{int(led.result.isin(['win', 'loss', 'push']).sum())} settled, {int((led.result == 'void').sum())} void, {int((led.result == 'open').sum())} open")
    rows = []
    groups = [("all sources", led)] + [(f"source={s}", g) for s, g in led.groupby("source")]
    for name, g in groups:
        for v, gg in [("kept", g[g.verdict == "keep"]), ("vetoed", g[g.verdict == "veto"]), ("all", g)]:
            s = gg[gg.result.isin(["win", "loss", "push"])]
            if len(gg) == 0:
                continue
            lo, hi = boot_roi(s.profit_units.values, s.game_date.values) if len(s) else (np.nan, np.nan)
            rows.append(dict(group=name, verdict=v, picks=len(gg), settled=len(s), W=int((s.result == "win").sum()), L=int((s.result == "loss").sum()),
                             P=int((s.result == "push").sum()), void=int((gg.result == "void").sum()), units=round(s.profit_units.sum(), 2),
                             roi=f"{s.profit_units.sum() / max(len(s), 1):+.1%}" if len(s) else "", ci=f"[{lo:+.1%},{hi:+.1%}]" if len(s) else "",
                             clv=round(gg.clv_line.mean(), 2) if gg.clv_line.notna().any() else np.nan, n_clv=int(gg.clv_line.notna().sum())))
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).to_string(index=False))
    print("ROI = units / settled picks (flat 1 unit; pushes count as settled at 0); ci = 95% bootstrap over game dates; "
          "clv = mean closing-line move in our favour (points), n_clv = picks with a closing snapshot.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("log"); a.add_argument("--picks", required=True); a.add_argument("--run-stamp"); a.add_argument("--date")
    a = sub.add_parser("settle"); a.add_argument("--date")
    sub.add_parser("close")
    a = sub.add_parser("report"); a.add_argument("--since")
    args = ap.parse_args()
    {"log": cmd_log, "settle": cmd_settle, "close": cmd_close, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
