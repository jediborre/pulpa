# =====================================================================
# COBERTURA REAL DE LOS CUARTOS (¿de dónde los saca el entrenamiento?)
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

def one(sql):
    return conn.execute(sql).fetchone()[0]

print("quarter_scores_v2:")
print("  filas totales              :", one("SELECT COUNT(*) FROM quarter_scores_v2"))
print("  con Q1 completo             :", one(
    "SELECT COUNT(*) FROM quarter_scores_v2 WHERE q1_home IS NOT NULL AND q1_away IS NOT NULL"))
print("  con Q1..Q4 completos        :", one(
    "SELECT COUNT(*) FROM quarter_scores_v2 WHERE q1_home IS NOT NULL AND q1_away IS NOT NULL "
    "AND q2_home IS NOT NULL AND q2_away IS NOT NULL AND q3_home IS NOT NULL AND q3_away IS NOT NULL "
    "AND q4_home IS NOT NULL AND q4_away IS NOT NULL"))
print("  Q1..Q4 completos y sin empate:", one(
    "SELECT COUNT(*) FROM quarter_scores_v2 WHERE q1_home IS NOT NULL AND q1_away IS NOT NULL "
    "AND q2_home IS NOT NULL AND q2_away IS NOT NULL AND q3_home IS NOT NULL AND q3_away IS NOT NULL "
    "AND q4_home IS NOT NULL AND q4_away IS NOT NULL "
    "AND q1_home<>q1_away AND q2_home<>q2_away AND q3_home<>q3_away AND q4_home<>q4_away"))
print()
print("  desglose de la perdida (Q1 no completo):", one(
    "SELECT COUNT(*) FROM quarter_scores_v2 WHERE q1_home IS NULL OR q1_away IS NULL"))
print()
print("matches terminados            :", one(
    "SELECT COUNT(*) FROM matches WHERE status_type='finished'"))
print("  con alguna fila en quarter_scores_v2:", one(
    "SELECT COUNT(DISTINCT m.match_id) FROM matches m JOIN quarter_scores_v2 q ON q.match_id=m.match_id "
    "WHERE m.status_type='finished'"))
print("  con Q1..Q4 completos        :", one(
    "SELECT COUNT(*) FROM matches m JOIN quarter_scores_v2 q ON q.match_id=m.match_id "
    "WHERE m.status_type='finished' AND q.q1_home IS NOT NULL AND q.q1_away IS NOT NULL "
    "AND q.q2_home IS NOT NULL AND q.q2_away IS NOT NULL AND q.q3_home IS NOT NULL AND q.q3_away IS NOT NULL "
    "AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL"))
print()
print("EJEMPLO de una fila con cuartos parciales:")
for r in conn.execute(
    "SELECT * FROM quarter_scores_v2 WHERE q1_home IS NOT NULL AND q4_home IS NULL LIMIT 3"
):
    print("   ", dict(r))
print()
print("EJEMPLO de una fila completa:")
for r in conn.execute(
    "SELECT * FROM quarter_scores_v2 WHERE q4_home IS NOT NULL AND q1_home IS NOT NULL LIMIT 3"
):
    print("   ", dict(r))
conn.close()
