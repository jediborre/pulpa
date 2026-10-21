# =====================================================================
# 1) ¿QUE EVENTOS HAY EN LA DATA? (el PBP del play_by_play)
# 2) ¿LAS CONDICIONES DE MOMENTUM DE m27_v3 SEPARAN EL ACIERTO?
#    - condicion medida tick a tick en el minuto 27
#    - WR de m27_v5_3 a 0.75 con la condicion presente vs ausente
#    - frecuencia de cada condicion en vivo
# =====================================================================

import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) ESQUEMA Y TIPOS DE EVENTO EN play_by_play")
cols = [r["name"] for r in conn.execute("PRAGMA table_info(play_by_play)")]
print("   columnas:", cols)
for col in ("type", "event_type", "comment", "description"):
    if col in cols:
        print(f"   tipos en '{col}':")
        for r in conn.execute(f"SELECT {col} t, COUNT(*) n FROM play_by_play "
                               f"GROUP BY {col} ORDER BY n DESC LIMIT 18"):
            print(f"      {str(r['t'])[:46]:<46} {r['n']:>9,}")
        break
else:
    for r in conn.execute("SELECT * FROM play_by_play LIMIT 3"):
        print("   ejemplo:", dict(r))

print()
print("2) ¿LA DATA DE PARTIDOS EN VIVO TIENE LOS MISMOS EVENTOS? (cobertura)")
for r in conn.execute("""
    SELECT COUNT(DISTINCT match_id) n, COUNT(*) filas FROM play_by_play
    WHERE match_id IN (SELECT match_id FROM model_inference_v3)"""):
    print(f"   partidos de la serie viva con PBP: {r['n']} | filas: {r['filas']:,}")

print()
print("3) ¿QUE HAY EN LAS FEATURES DE live_snapshots_v3 (players_json)?")
r = conn.execute("SELECT players_json FROM live_snapshots_v3 "
                 "WHERE players_json IS NOT NULL AND players_json != '[]' LIMIT 1").fetchone()
if r:
    import json
    d = json.loads(r["players_json"])
    print(f"   tipo: {type(d).__name__} | registros: {len(d)}")
    print("   primero:", d[0] if isinstance(d, list) else list(d)[:5])
    # cobertura de faltas en los registros de jugadores
    n_foul = 0
    n_tot = 0
    for rr in conn.execute("SELECT players_json FROM live_snapshots_v3 "
                           "WHERE players_json IS NOT NULL AND players_json != '[]'"):
        try:
            dd = json.loads(rr["players_json"])
        except Exception:
            continue
        for p in (dd if isinstance(dd, list) else []):
            n_tot += 1
            if isinstance(p, dict) and p.get("fouls") is not None:
                n_foul += 1
    print(f"   registros de jugador totales: {n_tot:,} | con FALTAS: {n_foul:,} "
          f"({n_foul * 100 / max(n_tot, 1):.1f} %)")

print()
print("4) ¿EL API DA FALTAS EN VIVO? (endpoint player_stats)")
print("   (monitor_v3/models/evaluator.py mapea 'fouls' desde player_stats)")
print("   columnas de live_snapshots_v3: team_stats_json ->", end=" ")
r2 = conn.execute("SELECT team_stats_json FROM live_snapshots_v3 "
                  "WHERE team_stats_json IS NOT NULL AND team_stats_json != '[]' LIMIT 1").fetchone()
if r2:
    import json
    d2 = json.loads(r2["team_stats_json"])
    print(json.dumps(d2[:2], ensure_ascii=False)[:400])
else:
    print("(vacio)")
conn.close()
