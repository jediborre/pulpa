# Test temporal del fix "resultado sin curva de momentum" (monitor_v3).
#
# Aclaración del usuario (2026-10-22): por "gráfico" se refiere a TODO el
# conjunto visual del mensaje (la tarjeta), no solo a la curva de momentum.
# Por tanto, un cierre sin `graph_points` debe enviar la tarjeta igual (sin
# curva), y NUNCA quedarse sin notificar.
#
# Caso real: 17330673 Vasto Verde Blumenau vs AJAB Jaraguá — m27_v5_3 BET_HOME
# 84.1 %, Q4 14-7, resultado win. El FT llegó sin graph_points y el aviso nunca
# salió.
#
# Valida:
#   1. `_notify_final_result` envía el resultado de ese partido (transporte mock).
#   2. La tarjeta se puede generar sin graph_points (JPEG válido, card only).
import asyncio
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3 import main as mon  # noqa: E402
from monitor_v3.notifications.bet_image import build_q4_alert_image  # noqa: E402

sent: list[dict] = []


async def fake_final(**kwargs):
    sent.append(kwargs)
    return {"ok": True}


async def fake_combined(**kwargs):
    sent.append({"combined": kwargs})
    return {"ok": True}


mon.send_final_confirmation = fake_final
mon.send_combined_final_confirmation = fake_combined

MID = "17330673"
conn = sqlite3.connect(ROOT / "matches.db")
conn.row_factory = sqlite3.Row
qs = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id = ?", (MID,)).fetchone()
conn.close()
assert qs, "falta quarter_scores_v2 del partido de prueba"
quarters = {
    "Q1": {"home": qs["q1_home"], "away": qs["q1_away"]},
    "Q2": {"home": qs["q2_home"], "away": qs["q2_away"]},
    "Q3": {"home": qs["q3_home"], "away": qs["q3_away"]},
    "Q4": {"home": qs["q4_home"], "away": qs["q4_away"]},
}
data = {
    "match": {
        "status_type": "finished",
        "home_team": "Vasto Verde Blumenau",
        "away_team": "AJAB Jaraguá",
        "league": "Catarinense, Women, Playoffs",
        "country": "Brazil",
    },
    "score": {
        "quarters": quarters,
        "home": sum(q["home"] for q in quarters.values()),
        "away": sum(q["away"] for q in quarters.values()),
    },
    "graph_points": [],
}

# 1) Notificación del resultado
asyncio.run(mon._notify_final_result(MID, data, "Vasto Verde Blumenau", "AJAB Jaraguá", quarters))
assert len(sent) == 1, f"deberia enviarse 1 tarjeta, se enviaron {len(sent)}"
call = sent[0]
assert call.get("model") == "m27_v5_3", call
assert call.get("signal") == "BET_HOME", call
assert call.get("outcome") == "win", call
assert call.get("q_home") == qs["q4_home"] and call.get("q_away") == qs["q4_away"], call
print(f"OK notificacion: m27_v5_3 BET_HOME win (Q4 {qs['q4_home']}-{qs['q4_away']})")

# 2) La tarjeta visual se genera sin curva de momentum
bets = [{
    "model": "m27_v5_3",
    "signal": "BET_HOME",
    "picked_team": "Vasto Verde Blumenau",
    "is_home": True,
    "confidence": 0.841,
    "result": "win",
    "inference_json": {"raw_json": {"min_conf_threshold": 0.75}},
}]
img = build_q4_alert_image(
    data, bets, minute=28, q_key="Q4", is_result=True, include_chart=True, seq=88
)
assert img and len(img) > 10_000, f"tarjeta no generada: {len(img) if img else None}"
print(f"OK tarjeta: {len(img)} bytes (JPEG sin curva de momentum)")
print("OK: cierre sin gráfica => tarjeta de resultado enviada")
