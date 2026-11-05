#!/bin/bash
# usage: scripts/trk2_queue.sh <foldset> <threads> <variant> [<variant> ...]  (runs sequentially; waits for A/B screening to finish first)
cd /Users/seantang/git/nba-ai/workspace
fs=$1; th=$2; shift 2
while pgrep -f "run_trk2.py [AB] screen" > /dev/null; do sleep 30; done
for v in "$@"; do
  .venv/bin/python -u scripts/run_trk2.py $v $fs $th > logs/trk2/${fs}_${v}.log 2>&1
done
echo QUEUE_DONE >> logs/trk2/${fs}_queue_$$.log
