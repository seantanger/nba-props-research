#!/bin/bash
# sequential queue of (foldset variant) pairs; each waits until fewer than 2 run_trk2 processes are active
cd /Users/seantang/git/nba-ai/workspace
th=$1; shift
while [ $# -gt 0 ]; do
  fs=$1; v=$2; shift 2
  while [ "$(pgrep -f 'scripts/run_trk2.py' | wc -l | tr -d ' ')" -ge 2 ]; do sleep 60; done
  .venv/bin/python -u scripts/run_trk2.py $v $fs $th > logs/trk2/${fs}_${v}.log 2>&1 &
  sleep 120
done
wait
