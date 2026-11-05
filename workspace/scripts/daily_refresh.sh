#!/bin/bash
# Daily data refresh before the live run (stats.nba.com + NBA injury PDFs only; no Odds API calls). Run after ~1 AM ET so last
# night's games are final; safe to re-run. Steps:
#   1. current-season box scores, re-downloaded (cached files never update otherwise: the P2-8 frozen-logs bug)
#   2. injury-report PDFs for the last 3 days -> incremental parse -> decision-time statuses
#   3. tracking (7 stats.nba.com calls per new game date) -> tracking tables for the assists model
#   4. production feature file (player_games.parquet; the live script also rebuilds features with tonight's stubs)
set -euo pipefail
cd "$(dirname "$0")/.."
Y=$(date +%Y); M=$(date +%-m)
if [ "$M" -ge 9 ]; then S0=$Y; else S0=$((Y - 1)); fi
SEASON="$S0-$(printf %02d $(( (S0 + 1) % 100 )))"
SINCE=$(date -v-3d +%F)
LOG=logs/daily_refresh_$(date +%Y%m%d).log
run() { echo "=== $* $(date '+%T')" | tee -a "$LOG"; uv run python -u "$@" 2>&1 | tee -a "$LOG" | tail -3; }
run src/fetch_nba.py --refresh "$SEASON"
run src/fetch_injury.py --since "$SINCE"
run src/parse_injury.py --since "$SINCE"
run src/injury_status.py
run src/fetch_tracking.py --seasons "$SEASON"
run src/fetch_tracking.py --build
run src/features.py
uv run python - <<'EOF' | tee -a "$LOG"
import pandas as pd
pg = pd.read_parquet("data/processed/player_games.parquet", columns=["GAME_DATE"])
tr = pd.read_parquet("data/processed/tracking_player_games.parquet", columns=["GAME_DATE"])
st = pd.read_parquet("data/processed/injury_status.parquet", columns=["GAME_DATE"])
print(f"REFRESH OK: box scores through {pg.GAME_DATE.max().date()}, tracking through {tr.GAME_DATE.max().date()}, "
      f"injury statuses through {st.GAME_DATE.max().date()}")
EOF
