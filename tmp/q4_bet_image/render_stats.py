# =====================================================================
# PRUEBA DE LAS IMÁGENES DE ESTADÍSTICAS (status y resumen diario)
# ---------------------------------------------------------------------
# QUÉ HACE:
#   1. Construye la imagen del `/status` con los datos reales de
#      `bot_runner.fetch_v3_stats()` (tablas _v3).
#   2. Construye la imagen del resumen diario con `_daily_summary_data()`
#      de telegram_bot (mismas cifras que el texto que ya se enviaba).
#   3. Guarda ambos JPEG en tmp/q4_bet_image/ para revisión visual y, con
#      --send, los envía por Telegram usando el camino real de producción.
#
# HIPÓTESIS QUE ABORDA:
#   Verificar layout, colores y barras de progreso antes de que el bot
#   interactivo y el daemon los envíen en producción.
#
# USO:
#   .venv\Scripts\python.exe tmp\q4_bet_image\render_stats.py
#   .venv\Scripts\python.exe tmp\q4_bet_image\render_stats.py --send
# =====================================================================

import argparse
import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.config.constants import ACTIVE_MODELS  # noqa: E402
from monitor_v3.notifications import stats_image  # noqa: E402
from monitor_v3.notifications.bot_runner import fetch_v3_stats  # noqa: E402
from monitor_v3.notifications.telegram_bot import (  # noqa: E402
    _daily_summary_data,
    _render_daily_summary_text,
)

OUT_DIR = Path(__file__).resolve().parent


def main_sync(send: bool) -> None:
    print("=== 1. Imagen de /status ===")
    stats = fetch_v3_stats()
    total = stats.get("total") or {}
    print(f"  total: {total}")
    photo = stats_image.status_image(stats, ACTIVE_MODELS, footer="Bet Monitor V3 · datos de bet_monitor_log_v3")
    if photo:
        path = OUT_DIR / "stats_status.jpg"
        path.write_bytes(photo)
        print(f"  ✅ {len(photo)} bytes -> {path.name}")
    else:
        print("  ❌ no se generó")

    print("\n=== 2. Imagen del resumen diario ===")
    data = _daily_summary_data()
    if not data:
        print("  (sin datos para hoy)")
    else:
        print(f"  día: {data['day']} | total: {data['total']} | categorías: {data['categories']}")
        print("  --- texto equivalente ---")
        print("  " + _render_daily_summary_text(data).replace("\n", "\n  "))
        photo2 = stats_image.summary_image(
            data["total"],
            data["per_model"],
            day=data["day"],
            categories=data["categories"],
            footer="Bet Monitor V3 · resumen diario (tablas _v3)",
        )
        if photo2:
            path2 = OUT_DIR / "stats_resumen.jpg"
            path2.write_bytes(photo2)
            print(f"  ✅ {len(photo2)} bytes -> {path2.name}")

            if send:
                from monitor_v3.notifications.telegram_bot import broadcast_photo

                res = asyncio.run(
                    broadcast_photo(photo2, _render_daily_summary_text(data))
                )
                print(f"  Telegram: {[r.get('ok') for r in res]}")
        else:
            print("  ❌ no se generó")


def _demo_summary() -> None:
    """Resumen sintético: para validar el layout cuando el día aún no tiene datos."""
    total = {"bets": 42, "win": 24, "loss": 14, "push": 3, "pending": 1, "nobet": 118}
    per_model = {
        "m27_v5_3": {"bets": 18, "win": 11, "loss": 6, "pending": 1},
        "m27_v5_2": {"bets": 12, "win": 7, "loss": 4, "pending": 0},
        "v6_2": {"bets": 12, "win": 6, "loss": 4, "pending": 0},
    }
    categories = [
        {"key": "🟢", "win": 15, "loss": 7},
        {"key": "🟡", "win": 6, "loss": 4},
        {"key": "⚪️", "win": 2, "loss": 2},
        {"key": "🟡⚪️", "win": 1, "loss": 1},
    ]
    photo = stats_image.summary_image(
        total, per_model, day="2026-10-19 (demo)", categories=categories,
        footer="Bet Monitor V3 · resumen diario (DEMO con datos sintéticos)",
    )
    if photo:
        path = OUT_DIR / "stats_resumen_demo.jpg"
        path.write_bytes(photo)
        print(f"  ✅ demo {len(photo)} bytes -> {path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="enviar el resumen por Telegram")
    ap.add_argument("--demo", action="store_true", help="solo renderizar un resumen sintético")
    args = ap.parse_args()
    if args.demo:
        _demo_summary()
    else:
        main_sync(args.send)
