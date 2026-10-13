"""
tmp/missing_data_audit/check_all.py
Proposito: Cuantificar cuantos partidos tienen team_statistics por cuarto (Q1..) pero
les falta el total de partido (period='ALL'), por el bug de borrado.
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "matches.db")
con_a = con.execute(
    "SELECT COUNT(*) FROM matches m WHERE EXISTS "
    "(SELECT 1 FROM team_statistics x WHERE x.match_id=m.match_id) AND NOT EXISTS "
    "(SELECT 1 FROM team_statistics x WHERE x.match_id=m.match_id AND x.period='ALL')"
).fetchone()[0]
print("con stats pero SIN periodo ALL:", con_a)
con_all = con.execute("SELECT COUNT(DISTINCT match_id) FROM team_statistics WHERE period='ALL'").fetchone()[0]
print("con periodo ALL:", con_all)
con.close()
