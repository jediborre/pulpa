# Diagnóstico temporal: latencia de la Bot API de Telegram para texto vs subida
# de imagen. Se usa chat_id=1 (inválido) para que Telegram responda un error sin
# entregar nada a nadie: mide solo la capa de red/subida.
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3.config.constants import TELEGRAM_BOT_TOKEN  # noqa: E402
from monitor_v3.notifications.bet_image import build_q4_alert_image  # noqa: E402
import requests  # noqa: E402

print("token configurado:", bool(TELEGRAM_BOT_TOKEN))
print("proxies env:", {k: v for k, v in os.environ.items() if "proxy" in k.lower()})

# 1) getMe
t0 = time.time()
try:
    r = requests.get(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getMe", timeout=20)
    print(f"getMe: {r.status_code} en {time.time() - t0:.2f}s")
except Exception as e:
    print(f"getMe ERROR en {time.time() - t0:.2f}s: {e}")

# 2) sendMessage a chat inválido (no entrega nada)
t0 = time.time()
try:
    r = requests.post(
        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
        data={"chat_id": "1", "text": "ping"}, timeout=20,
    )
    print(f"sendMessage(chat=1): {r.status_code} en {time.time() - t0:.2f}s -> {r.text[:90]}")
except Exception as e:
    print(f"sendMessage ERROR en {time.time() - t0:.2f}s: {e}")

# 3) sendPhoto a chat inválido con la tarjeta real
data = {
    "match": {"status_type": "finished", "home_team": "Vasto Verde Blumenau",
              "away_team": "AJAB Jaraguá", "league": "Catarinense, Women, Playoffs"},
    "score": {"quarters": {"Q4": {"home": 18, "away": 1}}, "home": 84, "away": 43},
    "graph_points": [],
}
bets = [{"model": "m27_v5_3", "signal": "BET_HOME", "picked_team": "Vasto Verde Blumenau",
         "is_home": True, "confidence": 0.791, "result": "win"}]
img = build_q4_alert_image(data, bets, minute=27, q_key="Q4", is_result=True, include_chart=False, seq=88)
print("imagen de prueba:", len(img) if img else None, "bytes")

for label, timeout in (("20s", 20), ("60s", 60)):
    t0 = time.time()
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
            data={"chat_id": "1", "caption": "test"},
            files={"photo": ("t.jpg", img, "image/jpeg")},
            timeout=timeout,
        )
        print(f"sendPhoto(chat=1, timeout={label}): {r.status_code} en {time.time() - t0:.2f}s -> {r.text[:90]}")
    except Exception as e:
        print(f"sendPhoto(chat=1, timeout={label}) ERROR en {time.time() - t0:.2f}s: {e}")
