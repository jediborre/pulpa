# =====================================================================
# VERIFICACION DE LAS 5 APUESTAS REALES DE HOY DE m27_v5_3
#   1) ¿La liquidacion (result=win/loss) coincide con el Q4 real?
#   2) ¿Que vio el modelo en el momento de apostar (Q3, margen, minuto)?
#   3) ¿Son bets "de garbage" (partido ya decidido)?
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
HOY = "2026-10-20"

print("1) LAS APUESTAS REALES (signal_type LIKE 'BET%') CON SU Q4 REAL:")
q = """
SELECT b.id, b.match_id, b.picked_side, b.confidence, b.result, b.created_at,
       b.inference_minute, b.raw_json, m.home_team, m.away_team, m.league,
       m.home_score, m.away_score
FROM bet_monitor_log_v3 b JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id
WHERE substr(b.created_at,1,10)=? AND b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%'
ORDER BY b.id
"""
for r in conn.execute(q, (HOY,)):
    mid = r["match_id"]
    qs = conn.execute(
        "SELECT * FROM quarter_scores_v2 WHERE match_id=?", (mid,)).fetchone()
    cols = qs.keys() if qs else []
    q4h = qs["q4_home"] if qs and "q4_home" in cols else None
    q4a = qs["q4_away"] if qs and "q4_away" in cols else None
    q3h = qs["q3_home"] if qs and "q3_home" in cols else None
    q3a = qs["q3_away"] if qs and "q3_away" in cols else None
    raw = json.loads(r["raw_json"]) if r["raw_json"] else {}
    print()
    print(f"   id {r['id']} | {r['home_team']} vs {r['away_team']} | {r['league']}")
    print(f"     apuesta: {r['picked_side']} conf {r['confidence']:.4f} | resultado registrado: {r['result']}")
    print(f"     FINAL {r['home_score']}-{r['away_score']} | Q3 {q3h}-{q3a} | Q4 {q4h}-{q4a}"
          f" -> margen Q4 {(q4h - q4a) if (q4h is not None and q4a is not None) else '?'}")
    print(f"     inferencia en el minuto {r['inference_minute']} (creado {r['created_at'][11:19]})")
    for k in ("q3_score", "q3_diff", "margin", "expected_q4_margin", "p_cal", "p_raw",
              "snapshot_minute_used", "q4_ml_home", "q4_ml_away", "block_reason", "reason"):
        if k in raw:
            print(f"       raw.{k} = {raw[k]}")
    if not qs:
        print("     !! NO HAY FILA EN quarter_scores_v2 para este partido")

print()
print("2) SALUD DE LA LIQUIDACION: distribucion de resultados en todo el log")
for r in conn.execute(
        "SELECT result, COUNT(*) n FROM bet_monitor_log_v3 GROUP BY result ORDER BY n DESC"):
    print(f"   {str(r['result']):<10} {r['n']:>6,}")

print()
print("3) PARTIDOS LIQUIDADOS HOY: ¿el Q4 suma con el marcador final?")
n_ok = n_push = n_bad = 0
for r in conn.execute(
        "SELECT DISTINCT b.match_id FROM bet_monitor_log_v3 b "
        "WHERE substr(b.created_at,1,10)=? AND b.result IS NOT NULL AND b.result<>''", (HOY,)):
    mid = r["match_id"]
    qs = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id=?", (mid,)).fetchone()
    m = conn.execute("SELECT home_score, away_score, status_description FROM matches "
                      "WHERE CAST(match_id AS INTEGER)=?", (mid,)).fetchone()
    if not qs or not m:
        n_bad += 1
        continue
    cols = qs.keys()
    q4h = qs["q4_home"] if "q4_home" in cols else 0
    q4a = qs["q4_away"] if "q4_away" in cols else 0
    suma_h = sum(int(qs[c] or 0) for c in ("q1_home", "q2_home", "q3_home", "q4_home") if c in cols)
    suma_a = sum(int(qs[c] or 0) for c in ("q1_away", "q2_away", "q3_away", "q4_away") if c in cols)
    if (q4h or 0) == 0 and (q4a or 0) == 0:
        n_push += 1
    elif suma_h == m["home_score"] and suma_a == m["away_score"]:
        n_ok += 1
    else:
        n_bad += 1
print(f"   Q4 con dato y cuadra con el final : {n_ok}")
print(f"   Q4 = 0-0 (sin dato)                : {n_push}")
print(f"   no cuadra                          : {n_bad}")
conn.close()
