# Test temporal del bug "el resumen no se envía" (2026-10-23).
#
# Síntoma: el bot enviaba "⏳ Calculando el resumen..." y borraba el aviso, pero
# el mensaje final nunca llegaba. Causa: el texto incluía `<0.60` sin escapar;
# Telegram lo interpretaba como etiqueta HTML y devolvía
# "Can't parse entities: unsupported start tag \"0.60)\"" (HTTP 400), que
# `_send_with_retry` trataba como fallo de red y agotaba los 3 intentos.
#
# Valida:
#   1. El texto renderizado no contiene `<` fuera de las etiquetas permitidas.
#   2. `_reply_long` cae a texto plano si Telegram rechaza el HTML.
import asyncio
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from telegram.error import BadRequest  # noqa: E402

from monitor_v3.models.live_summary import (  # noqa: E402
    compute_live_summary,
    render_live_summary_text,
)
from monitor_v3.notifications import bot_runner as br  # noqa: E402

conn = sqlite3.connect(ROOT / "matches.db")
conn.row_factory = sqlite3.Row
text = render_live_summary_text(compute_live_summary(conn, "m27_v5_3"))
conn.close()

# 1) Solo etiquetas HTML válidas
allowed = {"<b>", "</b>", "<i>", "</i>", "<code>", "</code>", "<a ", "</a>"}
for match in re.finditer(r"<[^>]*>?", text):
    tag = match.group(0)
    assert tag in allowed, f"etiqueta/`<` inválido en el resumen: {tag!r} (offset {match.start()})"
assert "<0.60" not in text, "quedó un < sin escapar"
assert "&lt;0.60" in text or "<0.60" not in text
print(f"OK HTML: {len(text)} caracteres, sin etiquetas inválidas")

# 2) Fallback a texto plano si Telegram rechaza el HTML
class FakeMessage:
    def __init__(self):
        self.sent: list[tuple[str, str | None]] = []

    async def reply_text(self, chunk, parse_mode=None, **kwargs):
        if parse_mode == "HTML":
            raise BadRequest('Can\'t parse entities: unsupported start tag "0.60)"')
        self.sent.append((chunk, parse_mode))
        return "ok"


async def scenario():
    msg = FakeMessage()
    br.SEND_BACKOFF_BASE_SECS = 0
    await br._reply_long(msg, text)
    return msg


msg = asyncio.run(scenario())
assert msg.sent, "no se envió nada en texto plano"
assert all(pm is None for _c, pm in msg.sent), msg.sent
assert "0.60" in msg.sent[0][0]
print(f"OK fallback: enviado en texto plano ({len(msg.sent[0][0])} caracteres)")
print("\nOK: el resumen ya se entrega aunque el HTML sea rechazado")
