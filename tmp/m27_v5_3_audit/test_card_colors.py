# =====================================================================
# TARJETA DE RESULTADO: colores por lado y del resultado hipotetico
#   - equipo local SIEMPRE en verde, visitante SIEMPRE en azul
#   - HUBIERA GANADO en verde, HUBIERA PERDIDO en rojo
#   - la fila NO BET se marca con la palabra "NO BET" en gris
# =====================================================================

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.notifications.bet_image import build_q4_alert_image  # noqa: E402

MATCH = {
    "match": {
        "home_team": "Salvadoreños BC",
        "away_team": "San Salvador BC",
        "league": "LMB, Clausura",
        "country": "El Salvador",
        "status_type": "finished",
        "status_description": "Ended",
        "home_score": 85,
        "away_score": 80,
    },
    "score": {
        "current": 85, "home": 85, "away": 80,
        "quarters": {"Q1": {"home": 22, "away": 19}, "Q2": {"home": 20, "away": 18},
                     "Q3": {"home": 19, "away": 24}, "Q4": {"home": 24, "away": 19}},
    },
    "graph_points": [],
}

bets = [
    # NO BET que HABRIA GANADO (local en verde, etiqueta en verde)
    {"model": "m27_v5_3", "signal": "NO_BET", "picked_team": "Salvadoreños BC",
     "is_home": True, "confidence": 0.59, "result": "would_win", "threshold": 0.75},
    # NO BET que HABRIA PERDIDO (visitante en azul, etiqueta en rojo)
    {"model": "m27_v5_2", "signal": "NO_BET", "picked_team": "San Salvador BC",
     "is_home": False, "confidence": 0.53, "result": "would_loss", "threshold": 0.75},
    # Apuesta real PERDIDA
    {"model": "m27_v5_4", "signal": "BET_HOME", "picked_team": "Salvadoreños BC",
     "is_home": True, "confidence": 0.81, "result": "loss", "decimal_odds": 1.90},
]

out = ROOT / "tmp" / "m27_v5_3_audit" / "_tarjeta_colores.jpg"
data = build_q4_alert_image(MATCH, bets, is_result=True, include_chart=False, seq=217)
out.write_bytes(data)
print("imagen:", out, out.stat().st_size, "bytes")
