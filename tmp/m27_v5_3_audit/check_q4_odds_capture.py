# =====================================================================
# ¿TENEMOS LA CUOTA DEL Q4 EN VIVO? (para medir EV por minuto)
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

cols = [r["name"] for r in conn.execute("PRAGMA table_info(live_snapshots_v3)")]
print("columnas de live_snapshots_v3:", cols)
print()
for c in cols:
    if "q4" in c or "odds" in c or "ml" in c:
        n = conn.execute(f"SELECT COUNT({c}) n, COUNT(*) t FROM live_snapshots_v3").fetchone()
        print(f"   {c:<20} con dato {n['n']:>6,} / {n['t']:>6,}")

print()
print("filas con cuota Q4 (ejemplo):")
r = conn.execute("""
    SELECT * FROM live_snapshots_v3
    WHERE q4_ml_home IS NOT NULL AND q4_ml_away IS NOT NULL
    ORDER BY id DESC LIMIT 8""").fetchall()
if not r:
    print("   NINGUNA fila con cuota Q4 capturada todavia")
else:
    for x in r:
        d = dict(x)
        print("   ", {k: v for k, v in d.items() if k in (
            "match_id", "game_minute", "q4_ml_home", "q4_ml_away", "created_at",
            "q4_ml_home_1", "q4_ml_away_1", "q4_odds_json")})

print()
print("ESTADO DEL RESTO DE LA SERIE:")
for r in conn.execute("""
    SELECT COUNT(*) filas, COUNT(DISTINCT match_id) partidos,
           MIN(created_at) desde, MAX(created_at) hasta
    FROM live_snapshots_v3"""):
    print("  ", dict(r))
for r in conn.execute("""
    SELECT COUNT(*) filas, COUNT(DISTINCT match_id) partidos FROM model_inference_v3"""):
    print("   model_inference_v3:", dict(r))
conn.close()
