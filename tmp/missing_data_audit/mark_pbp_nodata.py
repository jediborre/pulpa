"""
tmp/missing_data_audit/mark_pbp_nodata.py
Proposito: Marcar en detail_404 los partidos sin play_by_play que genuinamente no
tienen jugadas de anotacion (match_events sin incident_type='goal'), para no
re-descargarlos en el backfill. Operacion puntual (una sola vez).
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "matches.db")
con.execute(
    "CREATE TABLE IF NOT EXISTS detail_404 ("
    "match_id TEXT, table_name TEXT, PRIMARY KEY (match_id, table_name))"
)

# Partidos con match_events pero SIN ninguna jugada de anotacion -> pbp vacio real.
rows = con.execute("""
    SELECT match_id FROM match_events
    GROUP BY match_id
    HAVING SUM(CASE WHEN incident_type = 'goal' THEN 1 ELSE 0 END) = 0
""").fetchall()
con.executemany(
    "INSERT OR IGNORE INTO detail_404 (match_id, table_name) VALUES (?, 'play_by_play')",
    [(r[0],) for r in rows],
)
con.commit()
print(f"marcados play_by_play sin anotaciones: {len(rows)}")
con.close()
