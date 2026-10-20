# =====================================================================
# ¿FALTA DATOS O SOLO HAY QUE MIGRAR? (quarter_scores vs quarter_scores_v2)
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


def block(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


block("1. ESQUEMA DE LAS DOS TABLAS")
for t in ("quarter_scores", "quarter_scores_v2"):
    print(f"\n  {t}:")
    for r in conn.execute(f"PRAGMA table_info({t})"):
        print(f"     {r['name']:<12} {r['type']:<8} notnull={r['notnull']} pk={r['pk']}")

block("2. COBERTURA")
def one(sql, args=()):
    return conn.execute(sql, args).fetchone()[0]

fin = one("SELECT COUNT(*) FROM matches WHERE status_type='finished'")
print(f"  partidos terminados                       : {fin:,}")
print(f"  quarter_scores   (legacy) filas          : {one('SELECT COUNT(*) FROM quarter_scores'):,}")
print(f"  quarter_scores   partidos distintos      : {one('SELECT COUNT(DISTINCT match_id) FROM quarter_scores'):,}")
print(f"  quarter_scores_v2 (canónica) filas       : {one('SELECT COUNT(*) FROM quarter_scores_v2'):,}")
q_legacy_fin = one(
    "SELECT COUNT(*) FROM (SELECT DISTINCT match_id FROM quarter_scores WHERE match_id IN "
    "(SELECT match_id FROM matches WHERE status_type='finished'))")
print(f"  partidos con quarters en la legacy       : {q_legacy_fin:,}")
print(f"  partidos con quarters en v2              : {one('SELECT COUNT(*) FROM quarter_scores_v2'):,}")
print(f"  partidos en v2 que NO estan en legacy    : {one('SELECT COUNT(*) FROM quarter_scores_v2 WHERE match_id NOT IN (SELECT match_id FROM quarter_scores)'):,}")
q_incompletos = one(
    "SELECT COUNT(*) FROM (SELECT match_id FROM quarter_scores GROUP BY match_id "
    "HAVING COUNT(*)=4 AND SUM(CASE WHEN home IS NULL OR away IS NULL THEN 1 ELSE 0 END)=0)")
print(f"  partidos legacy con las 4 filas y sin nulos: {q_incompletos:,}")

block("3. ¿QUÉ FALTA REALMENTE? (partidos terminados sin quarters en ninguna tabla)")
sin = one("""
  SELECT COUNT(*) FROM matches m WHERE m.status_type='finished'
  AND m.match_id NOT IN (SELECT match_id FROM quarter_scores)
  AND m.match_id NOT IN (SELECT match_id FROM quarter_scores_v2)
""")
print(f"  partidos terminados SIN quarters en ninguna tabla: {sin:,}")
print("  (el resto de los 'terminados' no tieneource de datos de API)")

block("4. LA TABLA LEGACY: ¿GUARDA UN FIL POR CUARTO O UNO POR PARTIDO?")
for r in conn.execute("SELECT * FROM quarter_scores LIMIT 3"):
    print("   ", dict(r))
print()
print("  ¿la legacy tiene las columnas de overtime (ot_home/ot_away)?")
cols_legacy = {r["name"] for r in conn.execute("PRAGMA table_info(quarter_scores)")}
cols_v2 = {r["name"] for r in conn.execute("PRAGMA table_info(quarter_scores_v2)")}
print(f"     solo en legacy: {sorted(cols_legacy - cols_v2)}")
print(f"     solo en v2    : {sorted(cols_v2 - cols_legacy)}")

block("5. CANTIDAD DE PARTIDOS CON OVERTIME (para el fill de ot_*)")
print(f"  con OT en la legacy (4 filas, alguna con ot): ", one("""
  SELECT COUNT(*) FROM (
    SELECT match_id FROM quarter_scores GROUP BY match_id
    HAVING SUM(CASE WHEN quarter='OT' THEN 1 ELSE 0 END)>0
  )"""))
print(f"  v2 con OT informado: ", one(
    "SELECT COUNT(*) FROM quarter_scores_v2 WHERE ot_home IS NOT NULL OR ot_away IS NOT NULL"))
print(f"  v2 filas totales: ", one("SELECT COUNT(*) FROM quarter_scores_v2"))

block("6. ¿QUÉ ESCRIBE EN CADA UNA? (codigo)")
import subprocess
for pat in ("save_quarter_scores", "quarter_scores_v2", "INSERT INTO quarter_scores"):
    out = subprocess.run(["git", "grep", "-n", "--", pat], cwd=str(ROOT),
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [ln for ln in out.stdout.splitlines() if ln.strip()][:12]
    print(f"\n  grep '{pat}':")
    for ln in lines:
        print("   ", ln[:150])
conn.close()
