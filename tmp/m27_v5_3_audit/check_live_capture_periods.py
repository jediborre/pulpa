# =====================================================================
# ¿LA CAPTURA VIVA YA GUARDA LOS STATS POR CUARTO?
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

filas = conn.execute("""SELECT id, match_id, game_minute, team_stats_json, created_at
                         FROM live_snapshots_v3
                         WHERE team_stats_json IS NOT NULL AND team_stats_json != '[]'
                         ORDER BY id DESC LIMIT 200""").fetchall()
n_con_periodos = 0
n_all = 0
muestra = None
for r in filas:
    try:
        d = json.loads(r["team_stats_json"])
    except Exception:
        continue
    periodos = {x[0] for x in d if isinstance(x, list) and x}
    if periodos - {"ALL"}:
        n_con_periodos += 1
        if muestra is None:
            muestra = (r, d)
    if "ALL" in periodos:
        n_all += 1
print(f"snapshots con stats: {len(filas):,} | con agregado ALL: {n_all:,} "
      f"| CON desglose por cuarto: {n_con_periodos:,}")
if muestra:
    r, d = muestra
    print(f"\nejemplo (match {r['match_id']}, min {r['game_minute']}):")
    for x in d:
        if len(x) >= 4:
            print(f"   {x[0]:<5} {x[1]:<22} {x[2]} - {x[3]}")
    print()
    # el snapshot mas reciente con el formato nuevo
    nuevo = None
    for r2 in filas:
        try:
            d2 = json.loads(r2["team_stats_json"])
        except Exception:
            continue
        if d2 and isinstance(d2[0], list) and len(d2[0]) >= 4:
            nuevo = (r2, d2)
            break
    if nuevo:
        r2, d2 = nuevo
        print(f"formato nuevo (match {r2['match_id']}, min {r2['game_minute']}, {r2['created_at'][:19]}):")
        for x in d2:
            print(f"   {x[0]:<5} {x[1]:<22} {x[2]} - {x[3]}")
else:
    print("\n todavia no hay snapshots con desglose por cuarto")
conn.close()
