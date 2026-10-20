# =====================================================================
# VERIFICA que el dataset se construye igual con la tabla canonica
# =====================================================================

import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

sys.path.insert(0, str(ROOT))
from models.m27_v5_3.features import is_excluded_stage, is_toxic_league  # noqa: E402

t0 = time.time()
nuevo = conn.execute(
    "SELECT m.match_id, m.date, m.league, m.home_team, m.away_team, "
    "q.q1_home, q.q1_away, q.q2_home, q.q2_away, q.q3_home, q.q3_away, "
    "q.q4_home, q.q4_away "
    "FROM matches m JOIN quarter_scores_v2 q ON q.match_id = m.match_id "
    "WHERE m.status_type='finished' AND m.date IS NOT NULL "
    "  AND q.q1_home IS NOT NULL AND q.q1_away IS NOT NULL "
    "  AND q.q2_home IS NOT NULL AND q.q2_away IS NOT NULL "
    "  AND q.q3_home IS NOT NULL AND q.q3_away IS NOT NULL "
    "  AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL "
    "ORDER BY m.date ASC, m.match_id ASC"
).fetchall()
t_nuevo = time.time() - t0

t0 = time.time()
viejo = conn.execute(
    "SELECT m.match_id, m.date, m.league, m.home_team, m.away_team, "
    "MAX(CASE WHEN q.quarter='Q1' THEN q.home END), MAX(CASE WHEN q.quarter='Q1' THEN q.away END), "
    "MAX(CASE WHEN q.quarter='Q2' THEN q.home END), MAX(CASE WHEN q.quarter='Q2' THEN q.away END), "
    "MAX(CASE WHEN q.quarter='Q3' THEN q.home END), MAX(CASE WHEN q.quarter='Q3' THEN q.away END), "
    "MAX(CASE WHEN q.quarter='Q4' THEN q.home END), MAX(CASE WHEN q.quarter='Q4' THEN q.away END) "
    "FROM matches m JOIN quarter_scores q ON q.match_id = m.match_id "
    "WHERE m.status_type='finished' AND m.date IS NOT NULL "
    "GROUP BY m.match_id ORDER BY m.date ASC, m.match_id ASC"
).fetchall()
t_viejo = time.time() - t0

print(f"  consulta nueva (v2, sin pivot): {len(nuevo):,} partidos en {t_nuevo:.1f}s")
print(f"  consulta vieja (legacy, con pivot): {len(viejo):,} partidos en {t_viejo:.1f}s")

set_nuevo = {r["match_id"] for r in nuevo}
set_viejo = {r["match_id"] for r in viejo}
print(f"  solo en la nueva: {len(set_nuevo - set_viejo):,}")
print(f"  solo en la vieja: {len(set_viejo - set_nuevo):,}")
q_dur = {r["league"]: r["qd"] for r in conn.execute(
    "SELECT league, quarter_duration_minutes qd FROM leagues_classification")}

# Los valores de los cuartos deben coincidir
idx_viejo = {r["match_id"]: tuple(r[5:13]) for r in viejo}
diferentes = 0
for r in nuevo:
    a = tuple(r[5:13])
    b = idx_viejo.get(r["match_id"])
    if b is not None and a != b:
        diferentes += 1
print(f"  partidos con valores de cuarto DISTINTOS entre las dos fuentes: {diferentes:,}")

# Elegibles con EXACTAMENTE el mismo criterio que el pipeline:
#   None not in r[5:13]  AND  q4_home != q4_away  AND  q_dur != 12
#   AND  not toxic  AND  not excluded_stage
def elegible(r) -> bool:
    if any(v is None for v in tuple(r[5:13])):
        return False
    if int(r[11]) == int(r[12]):
        return False
    if int(q_dur.get(r["league"], 10)) == 12:
        return False
    if is_toxic_league(r["league"]) or is_excluded_stage(r["league"]):
        return False
    return True


eleg_new = [r for r in nuevo if elegible(r)]
eleg_old = [r for r in viejo if elegible(r)]
print(f"\n  ELEGIBLES con el filtro exacto del pipeline:")
print(f"     nueva (v2)   : {len(eleg_new):,}")
print(f"     vieja (legacy): {len(eleg_old):,}")
print(f"     diferencia   : {len(eleg_new) - len(eleg_old):+,}")
print(f"     (el dataset anterior se construyo con 184.173 elegibles)")

set_n = {r["match_id"] for r in eleg_new}
set_o = {r["match_id"] for r in eleg_old}
print(f"     solo en la nueva: {len(set_n - set_o):,} | solo en la vieja: {len(set_o - set_n):,}")
conn.close()
