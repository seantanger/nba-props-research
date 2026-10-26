"""Parse the cached official injury-report PDFs into one table:
report_ts (ET), game_date, game_time, matchup, team, player, status, reason.
Column assignment uses word x-positions against the header row, so it survives the 2023+ layout where the text layer drops spaces.
Output: data/processed/injury_reports.parquet. Usage: .venv/bin/python src/parse_injury.py [--workers 8]
"""
import argparse, pathlib, re, sys
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "injury" / "pdf"
OUT = ROOT / "data" / "processed" / "injury_reports.parquet"
COLS = ["game_date", "game_time", "matchup", "team", "player", "status", "reason"]
HEADER_FIRST = {"Game": None, "Matchup": "matchup", "Team": "team", "Player": "player", "Current": "status", "Reason": "reason"}
STATUSES = {"Out", "Questionable", "Doubtful", "Probable", "Available"}


def header_bounds(words):
    """[(col, x0)] sorted by x for the header row on a page; None if not found. Handles 'Game Date' as one word
    (keep_blank_chars, old layout) or two words (new layout): the first 'Game' is the date column, the second the time."""
    xs, games = {}, []
    for w in words:
        if w["top"] > 130:
            continue
        first = w["text"].split()[0] if w["text"].strip() else ""
        if first == "Game":
            games.append(w["x0"])
        elif first in HEADER_FIRST and HEADER_FIRST[first] not in xs:
            xs[HEADER_FIRST[first]] = w["x0"]
    if len(games) >= 2:
        xs["game_date"], xs["game_time"] = sorted(games)[:2]
    if len(xs) < 7:
        return None
    return sorted(xs.items(), key=lambda kv: kv[1])


def col_of(x0, bounds):
    col = bounds[0][0]
    for c, x in bounds:
        if x0 >= x - 4:
            col = c
    return col


def parse_pdf(path):
    import pdfplumber
    m = re.search(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})(?:_(\d{2}))?(AM|PM)", path.name)
    d, hh, mm, ap = m.group(1), int(m.group(2)), int(m.group(3) or 0), m.group(4)
    hour = hh % 12 + (12 if ap == "PM" else 0)
    report_ts = pd.Timestamp(f"{d} {hour:02d}:{mm:02d}")
    rows, bounds, hdr_ts = [], None, None
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                words = page.extract_words(keep_blank_chars=True, x_tolerance=1.5, y_tolerance=2)
                if hdr_ts is None:  # the report's own timestamp, e.g. "Injury Report: 12/23/20 05:30 PM" (file _05PM = 5:30 PM report)
                    hm = re.search(r"(\d{2}/\d{2}/\d{2}) (\d{1,2}:\d{2}) ?(AM|PM)", " ".join(w["text"] for w in words if w["top"] < 60))
                    if hm:
                        hdr_ts = pd.to_datetime(f"{hm.group(1)} {hm.group(2)} {hm.group(3)}", format="%m/%d/%y %I:%M %p", errors="coerce")
                b = header_bounds(words) or bounds
                if b is None:
                    continue
                bounds = b
                hdr_top = max([w["top"] for w in words if w["top"] <= 130 and w["text"].split()[:1] in (["Game"], ["Reason"])] or [0])
                body = [w for w in words if w["top"] > hdr_top + 5 and not w["text"].startswith("Page")]
                for w in body:
                    w["col"] = col_of(w["x0"], b)
                # a row is anchored by its status word; every other word attaches to the nearest anchor vertically
                anchors = sorted([w for w in body if w["col"] == "status" and w["text"].strip() in STATUSES], key=lambda w: w["top"])
                if not anchors:
                    continue
                tops = [a["top"] for a in anchors]
                recs = [{c: [] for c, _ in b} for _ in anchors]
                for w in sorted(body, key=lambda w: (w["top"], w["x0"])):
                    i = min(range(len(tops)), key=lambda k: abs(tops[k] - w["top"]))
                    tol = 12 if w["col"] == "reason" else 4
                    if abs(tops[i] - w["top"]) <= tol:
                        recs[i][w["col"]].append(w["text"])
                for r in recs:
                    rows.append({c: " ".join(v).strip() for c, v in r.items()})
    except Exception as e:  # corrupt PDF etc.
        return pd.DataFrame(), f"{path.name}: {e}"
    df = pd.DataFrame(rows)
    if df.empty:
        return df, f"{path.name}: no rows"
    for c in COLS:
        if c not in df:
            df[c] = ""
    for c in ["game_date", "game_time", "matchup", "team"]:
        df[c] = df[c].replace("", pd.NA).ffill()
    df["status"] = df.status.str.replace(" ", "")
    bad = ~df.status.isin(STATUSES) | (df.player.str.len() == 0)
    note = f"{path.name}: {bad.sum()}/{len(df)} bad rows" if bad.mean() > 0.02 else None
    df = df[~bad].copy()
    df["report_ts"] = hdr_ts if hdr_ts is not None and not pd.isna(hdr_ts) else report_ts + pd.Timedelta(minutes=30 if mm == 0 and m.group(3) is None else 0)
    df["file_ts"] = report_ts
    df["player"] = df.player.str.replace(r"\s+", " ", regex=True)
    df["game_date"] = pd.to_datetime(df.game_date, format="%m/%d/%Y", errors="coerce")
    return df[["report_ts", "file_ts"] + COLS], note


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(PDF.glob("Injury-Report_*.pdf"))
    if a.limit:
        files = files[: a.limit]
    out, notes = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (df, note) in enumerate(ex.map(parse_pdf, files, chunksize=20)):
            if len(df):
                out.append(df)
            if note:
                notes.append(note)
            if i % 1000 == 0:
                print(i, len(files), flush=True)
    res = pd.concat(out, ignore_index=True)
    res.to_parquet(OUT, index=False)
    print("rows", len(res), "reports", res.report_ts.nunique(), "notes", len(notes))
    (ROOT / "logs" / "parse_injury_notes.txt").write_text("\n".join(notes))


if __name__ == "__main__":
    main()
