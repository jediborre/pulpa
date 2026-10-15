"""
tmp/missing_data_audit/recent_gaps.py
Proposito: Listar los dias de los ultimos N dias que NO tienen partidos en matches.db
(para re-descargarlos con fetch-range).
"""
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "matches.db")
today = datetime.now().date()
existing = {
    r[0] for r in con.execute(
        "SELECT DISTINCT date(datetime(date || ' ' || time, '-6 hours')) FROM matches"
    )
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
missing = []
for i in range(N):
    d = (today - timedelta(days=i)).isoformat()
    if d not in existing:
        missing.append(d)
print(f"ultimos {N} dias | sin datos: {len(missing)}")
for d in missing:
    print("  ", d)
con.close()
