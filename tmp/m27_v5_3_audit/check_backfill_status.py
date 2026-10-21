# =====================================================================
# ESTADO DEL BACKFILL DE team_stats_v3 (cuanto falta)
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) ESTADO POR STATUS:")
for r in conn.execute("""SELECT status, COUNT(*) n, SUM(n_rows) filas FROM team_stats_backfill_state
                         GROUP BY status ORDER BY n DESC"""):
    print(f"   {r['status']:<18} {r['n']:>7,} partidos | {r['filas'] or 0:>9,} filas")
tot = conn.execute("SELECT COUNT(*) n FROM team_stats_v3").fetchone()
print(f"   team_stats_v3: {tot['n']:,} filas")

print()
print("2) COBERTURA POR PERIODO:")
for r in conn.execute("""SELECT period, COUNT(*) n, COUNT(DISTINCT match_id) partidos
                         FROM team_stats_v3 GROUP BY period ORDER BY period"""):
    print(f"   {r['period']:<5} {r['n']:>8,} filas | {r['partidos']:>6,} partidos")

print()
print("3) CUANTO FALTA (universo de las ligas rentables):")
import json
cal = json.loads((ROOT / "models" / "m27_v5_3" / "model_outputs" /
                  "league_calibration.json").read_text(encoding="utf-8"))
ligas = [c["league"] for c in cal
         if (c.get("wr") or 0) >= 0.714 and (c.get("bets") or 0) >= 30]
n_univ = conn.execute(
    "SELECT COUNT(*) n FROM matches WHERE status_type='finished' AND league IN ("
    + ",".join("?" * len(ligas)) + ") AND date >= '2019-01-01'", ligas).fetchone()["n"]
hechos = conn.execute("SELECT COUNT(DISTINCT match_id) n FROM team_stats_v3").fetchone()["n"]
print(f"   universo: {n_univ:,} | ya descargados: {hechos:,} | faltan: {max(0, n_univ - hechos):,}")
print(f"   tiempo restante a 2 req/s (2 requests por partido): "
      f"{max(0, n_univ - hechos) * 2 / 2 / 60:.0f} min")
conn.close()
