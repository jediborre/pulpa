"""
tmp/missing_data_audit/status2.py
Proposito: Diagnostico tras la re-corrida: cuantos 404 registrados por tabla, cuantos
partidos siguen marcados con details_checked_at, y si partidos concretos ya tienen datos.
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "matches.db")
con.row_factory = sqlite3.Row

print("=== detail_404 por tabla ===")
for r in con.execute("SELECT table_name, COUNT(*) n FROM detail_404 GROUP BY table_name ORDER BY n DESC"):
    print(f"  {r['table_name']:<18} {r['n']}")

checked = con.execute("SELECT COUNT(*) n FROM matches WHERE details_checked_at IS NOT NULL").fetchone()["n"]
print(f"\npartidos con details_checked_at (marcado viejo): {checked}")

print("\n=== muestra (partidos que antes daban 200) ===")
for mid in ("16732074", "16732071", "17162090", "17203539"):
    lu = con.execute("SELECT COUNT(*) n FROM lineups WHERE match_id=?", (mid,)).fetchone()["n"]
    ts = con.execute("SELECT COUNT(*) n FROM team_statistics WHERE match_id=?", (mid,)).fetchone()["n"]
    d404 = [r["table_name"] for r in con.execute("SELECT table_name FROM detail_404 WHERE match_id=?", (mid,))]
    print(f"  {mid}: lineups={lu} team_statistics={ts} detail_404={d404}")

con.close()
