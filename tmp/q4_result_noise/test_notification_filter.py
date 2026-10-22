# Test temporal de la politica de notificaciones de monitor_v3.
#
# Hipotesis/problema que resuelve:
#   El 2026-10-22 el bot envio "RESULTADO Q4" con la fila de m27_v5_2 al 56 %
#   (modelo en MODO SOMBRA y por debajo de su umbral). Causa: (a) el filtro de
#   logs en main.py caia al modelo sombra si no habia log visible, y (b) la
#   comprobacion `"BET" in signal_type` contaba "NO_BET" como apuesta porque
#   "BET" es subcadena de "NO_BET".
#
# Valida que select_notifiable_logs / select_notifiable_predictions:
#   - descartan modelos en modo sombra,
#   - descartan NO_BET / pendientes,
#   - dejan pasar unicamente BET reales de m27_v5_3.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIRD = ROOT.parents[1] if False else ROOT
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3.notifications.telegram_bot import (  # noqa: E402
    select_notifiable_logs,
    select_notifiable_predictions,
)

# --- Caso real del 22-oct: solo log de m27_v5_2 NO_BET al 55.91 % ---
efes_logs = [{
    "model_version": "m27_v5_2",
    "signal_type": "NO_BET",
    "confidence": 0.5591,
    "picked_side": "HOME",
}]
assert select_notifiable_logs(efes_logs) == [], "caso Efes: no debe notificarse nada"

# --- Apuesta real de v5_3: pasa ---
bet_v53 = [{
    "model_version": "m27_v5_3",
    "signal_type": "BET_HOME",
    "confidence": 0.81,
    "picked_side": "HOME",
}]
assert select_notifiable_logs(bet_v53) == bet_v53, "apuesta valida de v5_3 debe pasar"

# --- v5_3 NO_BET: no se manda tarjeta ---
assert select_notifiable_logs([{
    "model_version": "m27_v5_3",
    "signal_type": "NO_BET",
    "confidence": 0.6,
}]) == []

# --- Mixto: BET v5_3 + NO_BET v5_2 -> solo v5_3 ---
mixed = bet_v53 + efes_logs
assert select_notifiable_logs(mixed) == bet_v53

# --- Predicciones del tick ---
preds = {
    "m27_v5_3": {"signal": "BET_HOME", "confidence": 0.8},
    "m27_v5_2": {"signal": "BET_AWAY", "confidence": 0.9},
}
assert set(select_notifiable_predictions(preds)) == {"m27_v5_3"}
assert select_notifiable_predictions({"m27_v5_3": {"signal": "UNAVAILABLE"}}) == {}
assert select_notifiable_predictions({"m27_v5_3": {"signal": "NO_BET"}}) == {}

print("OK: politica de notificaciones (solo BET con umbral valido de v5_3)")
