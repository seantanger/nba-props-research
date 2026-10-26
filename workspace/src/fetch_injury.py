"""Download the NBA's official injury-report PDFs (hourly snapshots) for every game date in the logs.

URL pattern (from the `nbainjuries` package): https://ak-static.cms.nba.com/referee/injury/Injury-Report_<YYYY-MM-DD>_<HH><AM|PM>.pdf
until 2025-12-19 (hourly), and Injury-Report_<date>_<HH>_<MM><AM|PM>.pdf from 2025-12-22 (15-minute). We take the :00 file each hour.
Only hours 11AM..09PM ET are fetched (the latest report at least 60 minutes before any tip-off falls in that range).
Every PDF is cached under data/raw/injury/pdf/ and never fetched twice; 404s are recorded in data/raw/injury/missing.txt.
Usage: .venv/bin/python src/fetch_injury.py [--since YYYY-MM-DD] [--until YYYY-MM-DD] [--workers 3]
"""
import argparse, time, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd, requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "injury" / "pdf"
MISSING = ROOT / "data" / "raw" / "injury" / "missing.txt"
BASE = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_"
HOURS = ["11AM", "12PM", "01PM", "02PM", "03PM", "04PM", "05PM", "06PM", "07PM", "08PM", "09PM"]
HDR = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
NEWFMT = pd.Timestamp("2025-12-22")


def stem(date, hour):
    d = pd.Timestamp(date)
    return f"{d:%Y-%m-%d}_{hour[:2]}_00{hour[2:]}" if d >= NEWFMT else f"{d:%Y-%m-%d}_{hour}"


def fetch_one(s):
    out = PDF / f"Injury-Report_{s}.pdf"
    if out.exists():
        return "cached"
    for attempt in range(3):
        try:
            r = requests.get(BASE + s + ".pdf", headers=HDR, timeout=30)
        except requests.RequestException:
            time.sleep(2); continue
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            out.write_bytes(r.content); time.sleep(0.3); return "ok"
        if r.status_code in (403, 404):
            with open(MISSING, "a") as f:
                f.write(s + "\n")
            time.sleep(0.3); return "missing"
        time.sleep(2)
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2020-12-01"); ap.add_argument("--until", default="2027-12-31"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    PDF.mkdir(parents=True, exist_ok=True)
    pg = pd.read_parquet(ROOT / "data" / "processed" / "player_games.parquet", columns=["GAME_DATE"])
    dates = sorted(d for d in pg.GAME_DATE.drop_duplicates() if a.since <= str(d.date()) <= a.until)
    missing = set(MISSING.read_text().split()) if MISSING.exists() else set()
    todo = [stem(d, h) for d in dates for h in HOURS]
    todo = [s for s in todo if s not in missing and not (PDF / f"Injury-Report_{s}.pdf").exists()]
    print(f"{len(dates)} game dates, {len(todo)} files to fetch", flush=True)
    counts = {}
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, res in enumerate(ex.map(fetch_one, todo)):
            counts[res] = counts.get(res, 0) + 1
            if i % 200 == 0:
                print(i, counts, f"{time.time()-t0:.0f}s", flush=True)
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
