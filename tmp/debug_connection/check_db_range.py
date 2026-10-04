"""tmp/debug_connection/check_db_range.py - Cuenta partidos por fecha en matches.db."""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "matches.db")
expr = "date(datetime(date || ' ' || time, '-6 hours'))"
rows = con.execute(
    f"select {expr} d, count(*) from matches group by d order by d desc limit 8"
).fetchall()
print("ultimas fechas en matches:")
for d, n in rows:
    print(f"  {d}: {n}")
n = con.execute(
    f"select count(*) from matches where {expr} between '2026-09-01' and '2026-10-03'"
).fetchone()[0]
print("matches en rango 2026-09-01..2026-10-03:", n)
print("total matches:", con.execute("select count(*) from matches").fetchone()[0])
