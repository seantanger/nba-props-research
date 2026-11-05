#!/bin/bash
# P2-10 reused-holdout runs: baseline (injury model, current data) then tracking, same folds
cd /Users/seantang/git/nba-ai/workspace
LGB_THREADS=12 .venv/bin/python -u src/run_bp.py holdout base > logs/run_bp_holdout_base.log 2>&1
PLAYER_GAMES=data/processed/player_games_trk.parquet FEATURE_EXTRA='^trk_' LGB_THREADS=12 .venv/bin/python -u src/run_bp.py holdout trk > logs/run_bp_holdout_trk.log 2>&1
.venv/bin/python src/eval_bp.py holdout base > logs/eval_bp_holdout_base.log 2>&1
.venv/bin/python src/eval_bp.py holdout trk > logs/eval_bp_holdout_trk.log 2>&1
.venv/bin/python scripts/eval_trk.py holdout trk base > /dev/null 2>&1
echo ALLDONE >> logs/run_bp_holdout_trk.log
