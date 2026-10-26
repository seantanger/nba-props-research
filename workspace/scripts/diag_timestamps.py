"""Diagnostic: how the availability-tier inputs (FD update age, book update age, book-after-FD) are distributed, dev vs holdout.
Counts only; no outcomes are used."""
import pandas as pd, numpy as np
from src.common import *
for which in ["dev", "holdout"]:
    df = pd.read_parquet(RESULTS / f"bp_eval_{which}.parquet", columns=["season", "game_date", "book_id", "line_fd", "mins_before_tip", "mbt_fd", "book_after_fd", "p_anc_fd"])
    df = df[df.line_fd.notna() & (df.mins_before_tip >= 0)]
    print(f"\n=== {which} (rows with an FD line) ===")
    for s, g in df.groupby("season"):
        fd = g.drop_duplicates(["game_date"])  # not needed; use row-level
        q = g.mbt_fd.quantile([.1, .25, .5, .75, .9]).round(0).tolist()
        qb = g.mins_before_tip.quantile([.1, .25, .5, .75, .9]).round(0).tolist()
        print(f"{s}: n={len(g):>9,}  FD mins-before-tip q10/25/50/75/90={q}  book q={qb}  "
              f"FD<=120: {(g.mbt_fd<=120).mean():.1%}  book_after_fd: {g.book_after_fd.mean():.1%}  clean120: {((g.mbt_fd<=120)&(g.mbt_fd>=0)&g.book_after_fd).mean():.2%}")
    # by month for holdout
    if which == "holdout":
        g = df.copy(); g["ym"] = g.game_date.astype(str).str[:7]
        for ym, h in g.groupby("ym"):
            print(f"  {ym}: n={len(h):>8,} FD median mbt={h.mbt_fd.median():.0f} book median mbt={h.mins_before_tip.median():.0f} FD<=120 {(h.mbt_fd<=120).mean():.1%} book_after_fd {h.book_after_fd.mean():.1%} clean120 {((h.mbt_fd<=120)&(h.mbt_fd>=0)&h.book_after_fd).mean():.2%}")
        print("\nholdout by book: median book mbt, share book_after_fd")
        print(df.assign(book=df.book_id.map(BOOK_NAMES)).groupby("book").agg(n=("book_id", "size"), med_mbt=("mins_before_tip", "median"), after_fd=("book_after_fd", "mean")).round(3).to_string())
