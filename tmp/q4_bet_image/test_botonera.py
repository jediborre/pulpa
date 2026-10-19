# =====================================================================
# PRUEBA END-TO-END DEL FLUJO DE APUESTAS POR BOTONES
# ---------------------------------------------------------------------
# QUÉ HACE:
#   Simula pulsar "🎯 Apuestas" (selector de días) y elegir un día, enviando
#   los mensajes REALES al chat indicado con el mismo código de producción
#   (`_show_date_picker`, `_send_day_view`, `_send_match_view`).
#
#   Sirve para validar teclados inline, imagen del día, botones de partido y
#   la imagen de detalle sin tener que tocar el móvil.
#
# USO:
#   .venv\Scripts\python.exe tmp\q4_bet_image\test_botonera.py --chat-id 123456789
#   .venv\Scripts\python.exe tmp\q4_bet_image\test_botonera.py --chat-id ... --date 2026-10-19
# =====================================================================

import argparse
import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from telegram import Bot  # noqa: E402

from monitor_v3.config.constants import TELEGRAM_BOT_TOKEN  # noqa: E402
from monitor_v3.notifications.bot_runner import (  # noqa: E402
    _send_day_view,
    _send_match_view,
    _show_date_picker,
    fetch_day_matches,
)


class _Shim:
    """Emula un `Message` con los métodos que usan los handlers."""

    def __init__(self, bot: Bot, chat_id: int):
        self.bot = bot
        self.chat_id = chat_id

    async def reply_text(self, text, **kwargs):
        return await self.bot.send_message(self.chat_id, text, **kwargs)

    async def reply_photo(self, photo, **kwargs):
        return await self.bot.send_photo(self.chat_id, photo, **kwargs)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat-id", type=int, required=True)
    ap.add_argument("--date", help="día a abrir (YYYY-MM-DD); si se omite, solo el selector")
    ap.add_argument("--match-index", type=int, default=0, help="índice del partido a abrir")
    args = ap.parse_args()

    async with Bot(TELEGRAM_BOT_TOKEN) as bot:
        shim = _Shim(bot, args.chat_id)

        print("→ Enviando selector de días...")
        await _show_date_picker(shim)

        if args.date:
            print(f"→ Abriendo día {args.date}...")
            await _send_day_view(shim, args.date)

            matches = fetch_day_matches(args.date)
            if matches:
                idx = min(args.match_index, len(matches) - 1)
                mid = matches[idx]["match_id"]
                print(f"→ Abriendo partido {mid} (índice {idx})...")
                await _send_match_view(shim, mid)

    print("✅ Flujo enviado")


if __name__ == "__main__":
    asyncio.run(main())
