# =====================================================================
# PRUEBA DEL FORMATO CON EL CASO REAL QUE REPORTASTE
#   Falco KC-Szombathely vs Salon Vilpas Vikings (Champions League, Group C)
#   m27_v5_3 NO aparecia en la alerta de apuesta pero SI en la de resultado.
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.notifications.bet_image import build_q4_alert_image  # noqa: E402
from monitor_v3.notifications.telegram_bot import (  # noqa: E402
    _display_reason, _is_pending, format_combined_bet_message, format_combined_final_message,
)

MID = "17230818"
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

logs = [dict(r) for r in conn.execute(
    "SELECT * FROM bet_monitor_log_v3 WHERE match_id=? ORDER BY id", (MID,))]
m = conn.execute("SELECT * FROM matches WHERE match_id=?", (MID,)).fetchone()
conn.close()
home, away = m["home_team"], m["away_team"]

print("=" * 96)
print("CASO REAL: alerta de APUESTA en el min 27 (v6_2 apostaba, m27_v5_3 vigilaba)")
print("=" * 96)
predicciones = {
    "m27_v5_3": {"signal": "UNAVAILABLE", "reason": "pendiente_zona_verde", "pick": "HOME",
                  "confidence": 0.6723, "p_home_win": 0.6723},
    "m27_v5_2": {"signal": "PASS", "raw_model_signal": "PASS", "pick": "HOME",
                  "confidence": 0.5875, "p_home_win": 0.5875},
    "v6_2": {"signal": "BET_HOME", "raw_model_signal": "BET", "pick": "HOME",
              "confidence": 0.28435, "p_home_win": 0.28435},
}
print(format_combined_bet_message(predicciones, home, away))

print()
print("=" * 96)
print("CASO REAL: tarjeta de RESULTADO FT (sale del log, con los 4 modelos)")
print("=" * 96)
print(format_combined_final_message(logs, home, away))

print()
print("=" * 96)
print("MOTIVOS NORMALIZADOS")
print("=" * 96)
for k, p in predicciones.items():
    print(f"  {k:<12} motivo={_display_reason(p)!r} pendiente={_is_pending(p)}")
for lg in logs:
    from monitor_v3.notifications.telegram_bot import _log_block_reason
    print(f"  log {lg['model_version']:<12} motivo={_log_block_reason(lg)!r}")

print()
print("=" * 96)
print("IMAGEN (alerta de apuesta con el modelo pendiente visible)")
print("=" * 96)
bets = []
for model, pred in predicciones.items():
    is_home = str(pred.get("pick") or "").upper() == "HOME"
    bets.append({
        "model": model, "signal": pred.get("signal"),
        "picked_team": home if is_home else away, "is_home": is_home,
        "confidence": pred.get("confidence"),
        "probability": pred.get("p_home_win"),
        "decimal_odds": 1.85 if model == "v6_2" else None,
        "inference_json": None,
        "reason": _display_reason(pred),
    })
match_data = {
    "match": {"home_team": home, "away_team": away, "league": m["league"],
              "status_description": "4th quarter"},
    "score": {"home": 48, "away": 44, "quarters": {"4": {"home": 23, "away": 16}}},
    "odds_live": {"ml_home": 1.42, "ml_away": 2.85},
}
photo = build_q4_alert_image(match_data, bets, minute=27, q_key="4", include_chart=False, seq=23)
if photo:
    out = ROOT / "tmp" / "m27_v5_3_audit" / "_tarjeta_falco.jpg"
    out.write_bytes(photo)
    print(f"  imagen: {out} ({len(photo):,} bytes)")
else:
    print("  ERROR: no se genero la imagen")
