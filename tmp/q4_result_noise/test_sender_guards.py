# Test temporal end-to-end de los guards de notificacion (monitor_v3).
#
# Problema que reproduce: el 2026-10-22 llego a Telegram "🟢 RESULTADO Q4 /
# ❌ m27_v5_2 56% -> Anadolu Efes" para un partido donde el unico log era
# m27_v5_2 NO_BET (modelo en sombra, 55.91 %, por debajo del umbral 75 %).
#
# Se mockean el transporte (_send_bet_photo / broadcast_message) y se verifica
# que los senders reales no emiten nada cuando no hay apuesta con umbral valido,
# y que si emiten cuando m27_v5_3 tiene un BET real.
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3.notifications import telegram_bot as tb  # noqa: E402

calls: list[tuple[str, str]] = []


async def fake_photo(text, bets, match_data, match_url, **kwargs):
    calls.append(("photo", text))
    return [{"ok": True}]


async def fake_broadcast(text, **kwargs):
    calls.append(("broadcast", text))
    return [{"ok": True}]


tb._send_bet_photo = fake_photo
tb.broadcast_message = fake_broadcast

HOME, AWAY = "Anadolu Efes", "Beşiktaş JK"

# --- 1. Caso real del 22-oct: m27_v5_2 NO_BET al 56 % -> NADA ---
calls.clear()
res = asyncio.run(tb.send_final_confirmation(
    model="m27_v5_2",
    picked_team=HOME,
    is_home=True,
    signal="NO_BET",
    outcome="loss",
    match_id=None,
    home_team=HOME,
    away_team=AWAY,
    minute=27,
    q_key="Q4",
    q_home=21,
    q_away=20,
    confidence=0.5591,
))
assert res.get("skipped") is True, res
assert calls == [], f"no debio enviarse nada, se envio: {calls}"

# --- 2. Combinada con solo NO_BET (visible) -> NADA ---
calls.clear()
res = asyncio.run(tb.send_combined_final_confirmation(
    logs=[{"model_version": "m27_v5_3", "signal_type": "NO_BET",
           "picked_side": "HOME", "confidence": 0.62, "result": "push",
           "target_quarter": 4, "inference_minute": 27}],
    home_team=HOME,
    away_team=AWAY,
    minute=27,
))
assert res.get("skipped") is True, res
assert calls == [], f"no debio enviarse nada, se envio: {calls}"

# --- 3. Alerta combinada sin apuesta visible -> NADA ---
calls.clear()
res = asyncio.run(tb.send_combined_bet_alert(
    predictions={
        "m27_v5_3": {"signal": "UNAVAILABLE", "reason": "pendiente_zona_verde", "confidence": 0.6},
        "m27_v5_2": {"signal": "BET_HOME", "pick": "HOME", "confidence": 0.8},
    },
    home_team=HOME,
    away_team=AWAY,
    minute=27,
))
assert res.get("skipped") is True, res
assert calls == [], f"no debio enviarse nada, se envio: {calls}"

# --- 4. Apuesta valida de m27_v5_3 -> SI se envia ---
calls.clear()
res = asyncio.run(tb.send_final_confirmation(
    model="m27_v5_3",
    picked_team=HOME,
    is_home=True,
    signal="BET_HOME",
    outcome="win",
    home_team=HOME,
    away_team=AWAY,
    minute=27,
    q_key="Q4",
    q_home=21,
    q_away=20,
    confidence=0.81,
))
assert not res.get("skipped"), res
assert len(calls) == 1, calls
print("Enviado (valido):\n" + calls[0][1])

print("\nOK: no se notifica nada sin umbral valido; si con BET real de m27_v5_3")
