#!/bin/bash
# Re-run the dev2 and (reused) holdout walk-forward with the lagged_roll fix (tag `fix`) and compare against the pre-fix
# injury-aware baseline (tag `base`, P2-10). eval_trk.py writes logs/eval_trk_<which>.log, so P2-10's logs are restored after.
cd /Users/seantang/git/nba-ai/workspace
for w in dev2 holdout; do
  LGB_THREADS=12 .venv/bin/python -u src/run_bp.py $w fix > logs/run_bp_${w}_fix.log 2>&1
  .venv/bin/python src/eval_bp.py $w fix > logs/eval_bp_${w}_fix.log 2>&1
  cp logs/eval_trk_$w.log logs/eval_trk_$w.p210.bak
  .venv/bin/python scripts/eval_trk.py $w fix base > /dev/null 2>&1
  mv logs/eval_trk_$w.log logs/eval_rollfix_$w.log
  mv logs/eval_trk_$w.p210.bak logs/eval_trk_$w.log
done
echo ALLDONE >> logs/eval_rollfix_holdout.log
