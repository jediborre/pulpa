# =====================================================================
# VERIFICA: (a) el OT ya esta en v2, (b) la liquidacion pendiente de AET
#           se resuelve, (c) no queda ninguna apuesta bloqueada sin motivo
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print("1) PROLONGACION (OT) ya informada en v2:")
print(f"   partidos con OT en v2: {conn.execute('SELECT COUNT(*) FROM quarter_scores_v2 WHERE ot_home IS NOT NULL OR ot_away IS NOT NULL').fetchone()[0]}")
for r in conn.execute(
        "SELECT v.match_id, m.status_description, m.home_score, m.away_score, "
        "v.q4_home, v.q4_away, v.ot_home, v.ot_away FROM quarter_scores_v2 v "
        "JOIN matches m USING(match_id) WHERE v.ot_home IS NOT NULL LIMIT 8"):
    print("   ", dict(r))

print()
print("2) APUESTAS PENDIENTES y por que siguen pendientes:")
for b in conn.execute(
        "SELECT id, match_id, model_version, signal_type, picked_side, result, created_at "
        "FROM bet_monitor_log_v3 WHERE result='pending' ORDER BY id"):
    q = conn.execute(
        "SELECT q1_home,q1_away,q2_home,q2_away,q3_home,q3_away,q4_home,q4_away,ot_home,ot_away "
        "FROM quarter_scores_v2 WHERE match_id=?", (b["match_id"],)).fetchone()
    m = conn.execute(
        "SELECT home_score, away_score, status_description, status_type FROM matches WHERE match_id=?",
        (b["match_id"],)).fetchone()
    detalle = "sin cuartos en v2"
    if q and m and m["home_score"] is not None:
        partes = [q["q1_home"], q["q1_away"], q["q2_home"], q["q2_away"],
                  q["q3_home"], q["q3_away"], q["q4_home"], q["q4_away"]]
        if all(p is not None for p in partes):
            st = f"{m['status_description'] or ''} {m['status_type'] or ''}".upper()
            aet = "AET" in st or "EXTRA TIME" in st or "OVERTIME" in st
            detalle = (f"suma={sum(partes)} final={m['home_score'] + m['away_score']} "
                       f"status={m['status_description']!r} prorroga={aet} "
                       f"ot_en_v2={q['ot_home']}")
        else:
            detalle = "cuartos incompletos en v2"
    print(f"   id={b['id']:<5} {b['match_id']} {b['model_version']:<10} {detalle}")
conn.close()
