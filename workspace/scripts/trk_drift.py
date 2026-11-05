"""League-level drift in stats.nba.com tracking definitions, by season (P2-11 diagnosis).
Writes logs/trk2/drift.log. Reads only processed parquet files; no network."""
import pandas as pd, numpy as np, sys
t = pd.read_parquet('data/processed/tracking_player_games.parquet')
t['GAME_DATE'] = pd.to_datetime(t['GAME_DATE'])
t['season'] = np.where(t.GAME_DATE.dt.month >= 10, t.GAME_DATE.dt.year, t.GAME_DATE.dt.year - 1)
t = t[t.MIN > 0]
g = t.groupby('season')
out = pd.DataFrame({
    'player_games': g.size(),
    'reb_per_chance': g.REB.sum() / g.REB_CHANCES.sum(),
    'chances_per_min': g.REB_CHANCES.sum() / g.MIN.sum(),
    'contest_share': g.REB_CONTEST.sum() / g.REB.sum(),
    'drives_per_min': g.DRIVES.sum() / g.MIN.sum(),
    'touches_per_min': g.TOUCHES.sum() / g.MIN.sum(),
    'poss_time_per_min': g.TIME_OF_POSS.sum() / g.MIN.sum(),
    'pot_ast_per_min': g.POTENTIAL_AST.sum() / g.MIN.sum(),
    'ast_per_pot_ast': g.AST.sum() / g.POTENTIAL_AST.sum(),
    'cs_fga_per_min': g.CATCH_SHOOT_FGA.sum() / g.MIN.sum(),
    'pu_fga_per_min': g.PULL_UP_FGA.sum() / g.MIN.sum(),
})
lines = ['League tracking rates by season (MIN>0 player-dates):', out.round(3).to_string()]
# Holdout REB projection means, A vs ALL, by fold cutoff
try:
    a = pd.read_parquet('results/bp_dist_holdout_t2A.parquet'); b = pd.read_parquet('results/bp_dist_holdout_t2ALL.parquet')
    m = a.merge(b, on=['PLAYER_ID', 'GAME_ID', 'market', 'cutoff'], suffixes=('_a', '_b'))
    for mk in ['REB', 'AST', 'PTS']:
        r = m[m.market == mk]
        lines.append(f'holdout {mk}: mean mu A={r.mu_a.mean():.3f} ALL={r.mu_b.mean():.3f}  var/mu A={(r.var_a/r.mu_a).mean():.3f} ALL={(r.var_b/r.mu_b).mean():.3f}  n={len(r):,}')
    r = m[m.market == 'REB'].groupby('cutoff').agg(mu_a=('mu_a', 'mean'), mu_b=('mu_b', 'mean'), n=('mu_a', 'size'))
    lines.append('holdout REB by fold cutoff:'); lines.append(r.round(3).to_string())
except Exception as e:
    lines.append(f'(holdout dist files not read: {e})')
txt = '\n'.join(lines)
print(txt)
open('logs/trk2/drift.log', 'w').write(txt + '\n')
