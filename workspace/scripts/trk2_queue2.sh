#!/bin/bash
# runs variants sequentially on a foldset once fewer than 2 run_trk2 processes are active (keeps total LightGBM threads at 8)
cd /Users/seantang/git/nba-ai/workspace
fs=$1; th=$2; shift 2
for v in "$@"; do
  while [ "$(pgrep -f 'scripts/run_trk2.py' | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done
  .venv/bin/python -u scripts/run_trk2.py $v $fs $th > logs/trk2/${fs}_${v}.log 2>&1 &
  sleep 120
done
wait
