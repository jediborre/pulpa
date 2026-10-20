# =====================================================================
# PRUEBA END-TO-END: el motivo del descarte debe verse en el MENSAJE de
#                    Telegram y en la TARJETA (imagen).
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
    _display_reason, format_combined_bet_message, format_bet_message,
)

OUT_DIR = ROOT / "tmp" / "m27_v5_3_audit"

# Caso 1: m27_v5_3 DESCARTADO por formato de torneo; m27_v5_2 apuesta.
preds = {
    "m27_v5_2": {
        "signal": "BET_HOME", "raw_model_signal": "BET", "pick": "HOME",
        "confidence": 0.83, "p_home_win": 0.83,
    },
    "m27_v5_3": {
        "signal": "NO_BET_BLOCKED", "raw_model_signal": "PASS", "pick": "HOME",
        "confidence": 0.77, "p_home_win": 0.77,
        "block_reason": "descartada: torneo amistoso o invitacion (categoria AMISTOSOS)",
    },
}

# Caso 2: m27_v5_3 NO PUEDE INFERIR (fuera de la ventana de entrenamiento).
preds2 = {
    "m27_v5_2": {
        "signal": "BET_AWAY", "raw_model_signal": "BET", "pick": "AWAY",
        "confidence": 0.76, "p_home_win": 0.24,
    },
    "m27_v5_3": {
        "signal": "UNAVAILABLE", "pick": "NONE", "confidence": 0.0,
        "reason": "Fuera de la ventana de entrenamiento (min 32.0 > 30.0); "
                  "m27_v5_3 solo infiere en [25.5, 30.0]",
    },
}

print("=" * 92)
print("1. MENSAJE COMBINADO (Telegram)")
print("=" * 92)
print(format_combined_bet_message(preds, "Local FC", "Visitante FC"))
print()
print(format_combined_bet_message(preds2, "Local FC", "Visitante FC"))

print()
print("=" * 92)
print("2. MOTIVO NORMALIZADO (_display_reason)")
print("=" * 92)
for k, p in list(preds.items()) + list(preds2.items()):
    print(f"   {k:<12} -> {_display_reason(p)}")
print(f"   {'interno':<12} -> {_display_reason({'reason': 'monitoring_q3_window_for_green'})}")

print()
print("=" * 92)
print("3. MENSAJE SIMPLE (un solo modelo)")
print("=" * 92)
print(format_bet_message("m27_v5_3", "Local FC", True, "NO_BET",
                         confidence=0.77,
                         reason=_display_reason(preds["m27_v5_3"])))
print(format_bet_message("m27_v5_3", "Local FC", True, "NO_BET", confidence=0.77))

print()
print("=" * 92)
print("4. TARJETA (imagen): debe incluir el motivo en la fila NO BET")
print("=" * 92)

bets = []
for model, pred in preds.items():
    is_home = str(pred.get("pick") or "").upper() == "HOME"
    bets.append({
        "model": model,
        "signal": pred.get("signal"),
        "picked_team": "Local FC" if is_home else "Visitante FC",
        "is_home": is_home,
        "confidence": pred.get("confidence"),
        "probability": pred.get("p_home_win") if is_home else None,
        "decimal_odds": 1.85 if pred.get("signal", "").startswith("BET") else None,
        "inference_json": pred.get("inference_json"),
        "reason": _display_reason(pred),
    })

match_data = {
    "match": {"home_team": "Local FC", "away_team": "Visitante FC", "league": "Liga Test",
              "status_description": "4th quarter", "home_score": 62, "away_score": 58},
    "score": {"home": 62, "away": 58, "quarters": {"4": {"home": 12, "away": 9}}},
    "odds_live": {"ml_home": 1.42, "ml_away": 2.85},
}
photo = build_q4_alert_image(match_data, bets, minute=28, q_key="4",
                             include_chart=False, seq=7)
if photo:
    path = OUT_DIR / "_tarjeta_con_motivo.jpg"
    path.write_bytes(photo)
    print(f"   imagen generada: {len(photo):,} bytes -> {path}")
    print("   (abrirla para ver la fila 'no operó: <motivo>')")
else:
    print("   ERROR: no se genero la imagen")
