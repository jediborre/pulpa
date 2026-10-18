"""
tmp/m27_v5_2_audit/a10_team_stats_inventory.py
Inventario de team_statistics (por periodo) y cobertura por año, para proponer features de stats Q1-Q3
(rebotes, perdidas, TL, % tiro, biggest lead, max run, fast break, paint) disponibles al fin de Q3 en vivo.
Tambien cobertura por año de spread/total en match_odds y matches con graph_points.
"""
import sys, sqlite3, pandas as pd
sys.stdout.reconfigure(encoding="utf-8"); pd.set_option("display.width",220); pd.set_option("display.max_columns",40); pd.set_option("display.max_rows",100)
c = sqlite3.connect("file:matches.db?mode=ro", uri=True, timeout=120)
print(pd.read_sql("select period, count(distinct match_id) m from team_statistics group by 1 order by m desc", c))
print(pd.read_sql("select group_name, stat_key, count(distinct match_id) m from team_statistics where period='Q3' group by 1,2 order by m desc", c))
cov = pd.read_sql("""select substr(m.date,1,4) yr, count(*) n_matches,
  sum(exists(select 1 from team_statistics t where t.match_id=m.match_id and t.period='Q3')) with_q3_stats,
  sum(exists(select 1 from match_odds o where o.match_id=m.match_id and o.odds_type='Point spread')) with_spread,
  sum(exists(select 1 from match_odds o where o.match_id=m.match_id and o.odds_type='Over/Under')) with_total,
  sum(exists(select 1 from match_odds o where o.match_id=m.match_id and o.odds_type='Home/Away' and o.market_period='Match')) with_ml
  from matches m where m.status_type='finished' and m.date>='2023-01-01' group by 1""", c)
print(cov)
