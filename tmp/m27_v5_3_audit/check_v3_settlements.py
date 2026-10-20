# =====================================================================
# ¿El log v3 (m27_v5_2 / m27_v5_3) tiene liquidaciones hechas con Q4
#    PROVISIONAL (dato en vivo) en vez del definitivo (legacy)?
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

conn.execute("DROP TABLE IF EXISTS temp._pivot")
conn.execute("""
CREATE TEMP TABLE _pivot AS
SELECT q.match_id,
  MAX(CASE WHEN q.quarter='Q1' THEN q.home END) q1h, MAX(CASE WHEN q.quarter='Q1' THEN q.away END) q1a,
  MAX(CASE WHEN q.quarter='Q2' THEN q.home END) q2h, MAX(CASE WHEN q.quarter='Q2' THEN q.away END) q2a,
  MAX(CASE WHEN q.quarter='Q3' THEN q.home END) q3h, MAX(CASE WHEN q.quarter='Q3' THEN q.away END) q3a,
  MAX(CASE WHEN q.quarter='Q4' THEN q.home END) q4h, MAX(CASE WHEN q.quarter='Q4' THEN q.away END) q4a
FROM quarter_scores q GROUP BY q.match_id""")

for tabla in ("bet_monitor_log_v3", "bet_monitor_log_v2"):
    print(f"\n=== {tabla} ===")
    bets = list(conn.execute(
        f"SELECT match_id, model_version, signal_type, picked_side, result, created_at "
        f"FROM {tabla} WHERE signal_type LIKE '%BET%' AND signal_type NOT LIKE '%NO_BET%'"))
    mal = []
    for b in bets:
        row = conn.execute(
            "SELECT p.q4h, p.q4a, v.q4_home vq4h, v.q4_away vq4a FROM _pivot p "
            "LEFT JOIN quarter_scores_v2 v ON v.match_id=p.match_id WHERE p.match_id=?",
            (b["match_id"],)).fetchone()
        if not row or row["q4h"] is None or row["vq4h"] is None:
            continue
        pick_home = b["picked_side"] == "HOME"
        # resultado con el dato DEFINITIVO (legacy)
        if row["q4h"] == row["q4a"]:
            esperado = "push"
        else:
            esperado = "win" if pick_home == (row["q4h"] > row["q4a"]) else "loss"
        # resultado con el dato que se USO (v2, provisional)
        if row["vq4h"] == row["vq4a"]:
            usado = "push"
        else:
            usado = "win" if pick_home == (row["vq4h"] > row["vq4a"]) else "loss"
        guardado = b["result"]
        if esperado != guardado and usado == guardado:
            mal.append((b, esperado, usado, (row["q4h"], row["q4a"]), (row["vq4h"], row["vq4a"])))
    print(f"  apuestas con señal BET: {len(bets):,} | liquidadas con dato PROVISIONAL: {len(mal):,}")
    for b, esp, uso, leg, v2 in mal:
        print(f"     {b['match_id']} {b['model_version']:<12} guardado={b['result']:<5} "
              f"definitivo={esp:<5} | legacy Q4={leg} v2 Q4={v2} | {b['created_at'][:10]}")

print()
print("=== RESUMEN DEL LOG v3 (la base del analisis en vivo) ===")
for mv in ("m27_v5_2", "m27_v5_3"):
    rs = [dict(r) for r in conn.execute(
        "SELECT result, COUNT(*) n FROM bet_monitor_log_v3 WHERE model_version=? "
        "AND signal_type LIKE '%BET%' AND signal_type NOT LIKE '%NO_BET%' GROUP BY result", (mv,))]
    print(f"  {mv}: {rs}")
conn.close()
