# =====================================================================
# PRUEBA DE LOS ARREGLOS DEL WATCHER Y DEL BOT
# ---------------------------------------------------------------------
# QUÉ VALIDA:
#   1. `fetch_event_snapshot` clasifica un HTTP 404 real como
#      EventNotFoundError (fallo PERMANENTE) y NO como error transitorio.
#   2. Un error transitorio (p. ej. HTTP 500 simulado) NO se confunde con
#      404, para que el contador de fallos consecutivos sea el que actúe.
#   3. `_send_with_retry` del bot reintenta ante fallos de red y devuelve
#      None (sin propagar excepción) al agotar los intentos.
#
# HIPÓTESIS QUE ABORDA:
#   El watcher del partido 17353606 quedó colgado 10 h sondeando cada 45 s un
#   evento que la API ya devolvía como 404 (SofaScore lo re-creó con el id
#   17356027). Con estos cambios el watcher debe abandonar de inmediato.
#
# USO: .venv\Scripts\python.exe tmp\probe_404_audit\test_watcher_fixes.py
# =====================================================================

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DELETED_EVENT_ID = "17353606"


async def check_404_classification() -> None:
    from monitor_v3.scrapers.live_scraper import (
        EventNotFoundError,
        fetch_event_snapshot,
    )

    print("\n[1] 404 real → EventNotFoundError")
    try:
        await fetch_event_snapshot(DELETED_EVENT_ID)
        print("    ⚠️  inesperado: el evento respondió 200 (¿reapareció?)")
    except EventNotFoundError as e:
        print(f"    ✅ clasificado como EventNotFoundError ({e})")
    except Exception as e:  # noqa: BLE001
        print(f"    ❌ clasificado MAL como {type(e).__name__}: {e}")


async def check_transient_not_misclassified() -> None:
    """Simula un 500 parcheando el cliente móvil."""
    from monitor_v3.scrapers import live_scraper

    class _FakeClient:
        async def get_event_snapshot(self, match_id):
            raise RuntimeError("HTTP 500")

    original = live_scraper.get_mobile_client
    live_scraper.get_mobile_client = lambda: _FakeClient()
    try:
        print("\n[2] HTTP 500 simulado → NO debe ser EventNotFoundError")
        try:
            await live_scraper.fetch_event_snapshot("1")
            print("    ⚠️  inesperado: no lanzó excepción")
        except live_scraper.EventNotFoundError:
            print("    ❌ clasificado MAL como EventNotFoundError")
        except RuntimeError as e:
            print(f"    ✅ sigue siendo error transitorio (RuntimeError: {e})")
    finally:
        live_scraper.get_mobile_client = original


async def check_send_retry() -> None:
    from telegram.error import NetworkError
    from monitor_v3.notifications import bot_runner as br

    print("\n[3] _send_with_retry: 2 fallos y luego éxito")
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise NetworkError("httpx.ReadError")
        return "enviado"

    br.SEND_BACKOFF_BASE_SECS = 0.01  # acelerar la prueba
    res = await br._send_with_retry(flaky, description="prueba-flaky")
    print(f"    resultado={res!r} intentos={calls['n']} "
          f"{'✅' if res == 'enviado' and calls['n'] == 3 else '❌'}")

    print("\n[4] _send_with_retry: fallo permanente (no propaga excepción)")
    async def always_fail():
        raise NetworkError("httpx.ReadError")

    res2 = await br._send_with_retry(always_fail, description="prueba-fallo-total", attempts=2)
    print(f"    resultado={res2!r} {'✅ devolvió None' if res2 is None else '❌'}")


async def main() -> None:
    await check_404_classification()
    await check_transient_not_misclassified()
    await check_send_retry()


if __name__ == "__main__":
    asyncio.run(main())
