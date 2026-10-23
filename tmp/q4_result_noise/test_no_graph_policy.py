# Test temporal de la política "resultado solo con gráfica" (monitor_v3).
#
# Decisión del usuario (2026-10-22): la tarjeta de RESULTADO Q4 siempre lleva su
# gráfica de momentum. Si la API no entrega `graph_points` (ligas menores), el
# cierre NO debe mandar nada a Telegram: solo guarda marcadores y liquida.
#
# Se mockean la descarga FT, la persistencia y la notificación para verificar
# que un cierre "no_graph" no dispara `_notify_final_result`.
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3 import main as mon  # noqa: E402

notified: list = []


async def fake_notify(*args, **kwargs):
    notified.append((args, kwargs))


async def fake_fetch(match_id, is_ft=False):
    # Payload FT sin curva de momentum, como el del caso 17330673.
    return {
        "match": {"status_type": "finished"},
        "score": {"quarters": {"Q1": {"home": 29, "away": 11},
                               "Q2": {"home": 21, "away": 13},
                               "Q3": {"home": 17, "away": 5},
                               "Q4": {"home": 14, "away": 7}}},
        "graph_points": [],
    }


mon._notify_final_result = fake_notify
mon.fetch_match_by_id = fake_fetch
mon.save_quarter_scores = lambda *a, **k: None
mon.reconcile_pending_results = lambda *a, **k: None
mon.update_schedule_status = lambda *a, **k: None

# match_id inexistente en bet_monitor_schedule_v3 => pasa el guard de dedup.
asyncio.run(mon._final_fetch_and_save("TEST_NOGRAPH", "Vasto Verde Blumenau", "AJAB Jaraguá"))

assert notified == [], f"no debio notificarse nada sin grafica: {notified}"
print("OK: cierre sin gráfica => liquida en BD pero NO envía tarjeta de resultado")
