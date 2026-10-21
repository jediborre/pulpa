# =====================================================================
# Genera la IMAGEN de la tarjeta de RESULTADO con un NO BET y una
# apuesta real, para verificar que el NO BET ya no se ve en rojo.
# =====================================================================

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.notifications.bet_image import build_q4_alert_image  # noqa: E402

MATCH = {
    "match": {
        "home_team": "Falco KC-Szombathely",
        "away_team": "Salon Vilpas Vikings",
        "league": "Champions League, Group C",
        "country": "International",
        "status_type": "finished",
        "status_description": "Ended",
        "home_score": 85,
        "away_score": 72,
    },
    "score": {
        "current": 85, "home": 85, "away": 72,
        "quarters": {"Q1": {"home": 24, "away": 18}, "Q2": {"home": 22, "away": 20},
                     "Q3": {"home": 23, "away": 19}, "Q4": {"home": 16, "away": 15}},
    },
    "graph_points": [],
}

bets = [
    {"model": "m27_v5_3", "signal": "NO_BET", "picked_team": "Falco KC-Szombathely",
     "is_home": True, "confidence": 0.62, "result": "would_loss", "threshold": 0.75},
    {"model": "m27_v5_2", "signal": "NO_BET", "picked_team": "Falco KC-Szombathely",
     "is_home": True, "confidence": 0.59, "result": "would_loss", "threshold": 0.75},
    {"model": "v6_2", "signal": "BET_HOME", "picked_team": "Falco KC-Szombathely",
     "is_home": True, "confidence": 0.28, "result": "win", "decimal_odds": 1.83},
    {"model": "m27_v5_3", "signal": "BET_HOME", "picked_team": "Falco KC-Szombathely",
     "is_home": True, "confidence": 0.81, "result": "loss", "decimal_odds": 1.90},
]

out = ROOT / "tmp" / "m27_v5_3_audit" / "_tarjeta_resultado.jpg"


async def main() -> None:
    data = build_q4_alert_image(
        MATCH, bets, is_result=True, include_chart=False, seq=23,
    )
    if data is None:
        print("no se genero la imagen")
        return
    out.write_bytes(data)
    print("imagen:", out, out.stat().st_size, "bytes")


asyncio.run(main())
