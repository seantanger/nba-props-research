#!/bin/bash
# wait for the dev injury training, then evaluate, then start the holdout run (reused holdout, labeled) in the background
cd /Users/seantang/git/nba-ai/workspace
while pgrep -f "run_bp.py dev inj" > /dev/null; do sleep 30; done
.venv/bin/python src/eval_bp.py dev inj > logs/eval_bp_dev_inj.log 2>&1
.venv/bin/python scripts/eval_inj.py inj "" > logs/eval_inj.out 2>&1
LGB_THREADS=10 nohup .venv/bin/python src/run_bp.py holdout inj > logs/run_bp_holdout_inj.log 2>&1 &
echo "holdout started $(date)" >> logs/eval_inj.out
