# =====================================================================
# PRUEBA DEL CAMINO COMPLETO DE /resumen (como lo llama el bot)
#   - la conexion se abre y se usa en el mismo hilo
#   - el texto sale bien formado
# =====================================================================

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.notifications import bot_runner  # noqa: E402


async def main() -> None:
    data = await asyncio.to_thread(bot_runner._compute_live_summary)
    from monitor_v3.models.live_summary import render_live_summary_text
    text = render_live_summary_text(data)
    print(text)
    print()
    print(f"longitud: {len(text)} | filas del resumen: {len(data)} secciones")


asyncio.run(main())
