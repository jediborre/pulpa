# =====================================================================
# ¿LOS VALORES POR CUARTO SON REALES O LAS FALTAS VIENEN A CERO?
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("PARTIDOS CON DESGLOSE POR CUARTO (Q3), con el agregado ALL al lado:")
n = 0
for mid in [r["match_id"] for r in conn.execute(
        "SELECT DISTINCT match_id FROM team_stats_v3 WHERE period='Q3' LIMIT 6")]:
    print(f"   --- {mid}")
    for r in conn.execute("""SELECT period, stat_name, home_value, away_value FROM team_stats_v3
                             WHERE match_id=? AND (stat_name IN ('Fouls','Points','Free throws made',
                             'Turnovers','Timeouts'))
                             ORDER BY period, stat_name""", (mid,)):
        print(f"      {r['period']:<4} {r['stat_name']:<20} {r['home_value']} - {r['away_value']}")
    n += 1
    if n >= 3:
        break

print()
print("RESUMEN: las faltas por cuarto historicas son utilizables?")
tot = conn.execute("""SELECT COUNT(*) n FROM team_stats_v3
                       WHERE period LIKE 'Q%' AND stat_name='Fouls'""").fetchone()["n"]
con_f = conn.execute("""SELECT COUNT(*) n FROM team_stats_v3
                        WHERE period LIKE 'Q%' AND stat_name='Fouls'
                        AND (home_value > 0 OR away_value > 0)""").fetchone()["n"]
tot_pt = conn.execute("""SELECT COUNT(*) n FROM team_stats_v3
                          WHERE period LIKE 'Q%' AND stat_name='Points'""").fetchone()["n"]
con_pt = conn.execute("""SELECT COUNT(*) n FROM team_stats_v3
                          WHERE period LIKE 'Q%' AND stat_name='Points'
                          AND (home_value > 0 OR away_value > 0)""").fetchone()["n"]
print(f"   filas de faltas por cuarto: {tot:,} | con valor > 0: {con_f:,} "
      f"({con_f * 100 / max(tot, 1):.1f} %)")
print(f"   filas de puntos por cuarto: {tot_pt:,} | con valor > 0: {con_pt:,} "
      f"({con_pt * 100 / max(tot_pt, 1):.1f} %)")
conn.close()
