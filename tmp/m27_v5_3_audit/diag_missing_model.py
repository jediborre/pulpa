# =====================================================================
# DIAGNOSTICO: por que m27_v5_3 no sale en la alerta de apuesta
#   Partido: Falco KC-Szombathely vs Salon Vilpas Vikings (Champions League)
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

partidos = [r["match_id"] for r in conn.execute(
    "SELECT match_id FROM matches WHERE home_team LIKE '%Falco%' OR away_team LIKE '%Falco%' "
    "OR home_team LIKE '%Vilpas%' OR away_team LIKE '%Vilpas%' ORDER BY date DESC LIMIT 3")]
print(f"partidosLocalized: {partidos}")

for mid in partidos:
    print()
    print("=" * 100)
    print(f"MATCH {mid}")
    r = conn.execute("SELECT * FROM matches WHERE match_id=?", (mid,)).fetchone()
    print(f"  {r['home_team']} vs {r['away_team']} | {r['league']} | {r['date']} | {r['status_type']}")
    print("  --- bet_monitor_log_v3 ---")
    for b in conn.execute(
            "SELECT id, model_version, signal_type, picked_side, confidence, result, "
            "inference_minute, created_at, inference_json FROM bet_monitor_log_v3 "
            "WHERE match_id=? ORDER BY id", (mid,)):
        raw = {}
        try:
            raw = (json.loads(b["inference_json"] or "{}").get("raw_json") or {})
        except Exception:
            pass
        print(f"    id={b['id']:<4} {b['model_version']:<12} {b['signal_type']:<16} "
              f"pick={b['picked_side']:<5} conf={b['confidence']} res={b['result']:<6} "
              f"min={b['inference_minute']} p_cal={raw.get('p_cal')} thr={raw.get('min_conf_threshold')} "
              f"block={raw.get('block_reason')}")
    print("  --- model_inference_v3 ---")
    for b in conn.execute(
            "SELECT game_minute, model_version, p_cal, signal, confidence, block_reason "
            "FROM model_inference_v3 WHERE match_id=? ORDER BY game_minute, model_version", (mid,)):
        print(f"    min {b['game_minute']:>5} {b['model_version']:<12} p_cal={b['p_cal']} "
              f"signal={b['signal']} conf={b['confidence']} block={b['block_reason']}")

print()
print("=" * 100)
print("FASE DETECTADA PARA CADA LIGA DE LAS APUESTAS RECIENTES")
sys.path.insert(0, str(ROOT))
from models.common.competition_stage import classify_stage  # noqa: E402
for b in conn.execute(
        "SELECT DISTINCT s.league FROM bet_monitor_log_v3 l "
        "LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id=l.match_id "
        "WHERE l.model_version='m27_v5_3' ORDER BY l.created_at DESC LIMIT 25"):
    cat, motivo = classify_stage(b["league"] or "")
    print(f"  [{cat:<14}] {str(b['league'])[:60]:<62} {motivo}")
conn.close()
