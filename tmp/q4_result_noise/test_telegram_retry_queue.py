# Test temporal de la cola de reintentos de Telegram (monitor_v3).
#
# Caída de Telegram 2026-10-23 18:16-19:00 (HTTP 502 Bad Gateway): los envíos
# se perdían porque no había reintentos. Ahora:
#   - cada envío reintenta 3 veces con backoff ante 502/timeout,
#   - si sigue fallando, el aviso queda en una cola en memoria que se vacía
#     cada 60 s durante 30 min (hasta que Telegram vuelva).
#
# Este test simula la caída y la recuperación sin red real.
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3.notifications import telegram_bot as tb  # noqa: E402

tb._resolve_recipients = lambda signal_type_filter="all": [111, 222]
tb._retry_delay = lambda attempt: 0  # sin esperas reales en el test

# ── 1. Telegram caído: 502 en todos los intentos ──────────────────────
tb._send_photo_request = lambda *a, **k: {"ok": False, "description": "Bad Gateway"}
tb._send_request = lambda *a, **k: {"ok": False, "description": "HTTP Error 502: Bad Gateway"}

asyncio.run(tb.broadcast_photo(b"jpeg-fake", "🟢 APUESTA Q4 m27_v5_3 84%", match_url="https://x"))
assert tb.pending_notifications_count() == 1, tb.pending_notifications_count()
print("OK: imagen fallida quedó en cola")

asyncio.run(tb.broadcast_message("🟢 RESULTADO Q4 ✅ m27_v5_3 84%"))
assert tb.pending_notifications_count() == 2, tb.pending_notifications_count()
print("OK: texto fallido quedó en cola")

asyncio.run(tb._flush_pending_once())
assert tb.pending_notifications_count() == 2, "sin red los avisos deben seguir en cola"
print("OK: reintento sin red mantiene los avisos pendientes")

# ── 2. Telegram se recupera ───────────────────────────────────────────
entregas: list[str] = []


def photo_ok(chat_id, photo, caption, match_url=None):
    entregas.append(f"foto:{chat_id}")
    return {"ok": True}


def text_ok(url, payload):
    entregas.append(f"texto:{payload['chat_id']}")
    return {"ok": True}


tb._send_photo_request = photo_ok
tb._send_request = text_ok
asyncio.run(tb._flush_pending_once())
assert tb.pending_notifications_count() == 0, tb.pending_notifications_count()
assert len(entregas) == 4, entregas
print(f"OK: cola vaciada al recuperar Telegram ({entregas})")

# ── 3. Fallo no transitorio (HTML inválido) NO se reintenta ───────────
tb._send_request = lambda *a, **k: {"ok": False, "description": "Bad Request: can't parse entities"}
asyncio.run(tb.broadcast_message("<b>roto"))
assert tb.pending_notifications_count() == 0, "un 400 no debe encolarse"
print("OK: error no transitorio no entra en la cola")

print("\nOK: reintentos + cola de pendientes funcionando")
