#!/bin/bash
# Full tracking backfill (stats.nba.com only, no Odds API). Order = cheapest / most useful first, so an interruption still
# leaves the important data: (1) shot charts by month 2016-17..2022-23, (2) passing/rebounding/possessions 2016-17..2022-23,
# (3) drives/catch-and-shoot/pull-up for every season. Each step is resumable and is re-run until it prints "done"
# (fetch_tracking._call gives up after ~40 retries during long throttle episodes).
cd /Users/seantang/git/nba-ai/workspace
OLD="2016-17 2017-18 2018-19 2019-20 2020-21 2021-22 2022-23"
ALL="$OLD 2023-24 2024-25 2025-26 2026-27"
LOG=logs/fetch_tracking_backfill.log

step() {  # step <label> <args...>
  local label=$1; shift
  for attempt in 1 2 3 4 5 6; do
    echo "=== $label attempt $attempt $(date '+%F %T')" >> $LOG
    .venv/bin/python -u src/fetch_tracking.py --sleep 3.0 "$@" >> $LOG 2>&1
    tail -1 $LOG | grep -q '^done$' && return 0
    echo "=== $label stopped early; sleeping 15 min before resuming" >> $LOG
    sleep 900
  done
  echo "=== $label FAILED after 6 attempts" >> $LOG
}

step shots_month --seasons $OLD --kinds shots_month
step core_old    --seasons $OLD --kinds passing rebounding possessions
step scoring_all --seasons $ALL --kinds drives catchshoot pullup
echo "=== ALLDONE $(date '+%F %T')" >> $LOG
