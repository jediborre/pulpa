# Verificación temporal: el texto del /resumen debe ser aceptado por el parser
# HTML de Telegram. Se envía a chat_id=1 (inválido): si el HTML fuera incorrecto
# devolvería "can't parse entities"; si es correcto responde "chat not found"
# y no entrega nada a nadie.
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import requests  # noqa: E402

from monitor_v3.config.constants import TELEGRAM_BOT_TOKEN  # noqa: E402
from monitor_v3.models.live_summary import (  # noqa: E402
    compute_live_summary,
    render_live_summary_text,
)

conn = sqlite3.connect(ROOT / "matches.db")
conn.row_factory = sqlite3.Row
text = render_live_summary_text(compute_live_summary(conn, "m27_v5_3"))
conn.close()

r = requests.post(
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
    data={"chat_id": "1", "text": text, "parse_mode": "HTML"},
    timeout=30,
)
print("HTTP", r.status_code, "->", r.text[:160])
if "can't parse entities" in r.text.lower():
    raise SystemExit("ERROR: Telegram sigue rechazando el HTML del resumen")
print("OK: el HTML del resumen es válido para Telegram (solo falló el chat de prueba)")
