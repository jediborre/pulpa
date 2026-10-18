# =====================================================================
# PRUEBA DE ABANDONO DEL WATCHER (monitor_v3.main._watch_match)
# ---------------------------------------------------------------------
# QUÉ VALIDA (sin tocar la red ni la BD):
#   A) Si el sondeo devuelve HTTP 404 (evento eliminado/re-creado), el watcher
#      abandona de inmediato y marca el partido como done / evento_404_eliminado,
#      tanto en el modo sonda prestart como en el bucle previo a la ventana Q4.
#   B) Si el evento nunca arranca y se superan las 3.5 h (WATCH_OBSOLETE_SECS),
#      el watcher abandona en lugar de sondear para siempre
#      (discarded / obsoleto_sin_arranque).
#
# HIPÓTESIS QUE ABORDA:
#   El watcher de 17353606 quedó 10 h emitiendo "[PROBE] Error sondeo ...: HTTP
#   404" cada 45 s porque el bucle previo a la ventana Q4 no tenía guarda de
#   obsolescencia ni distinguía 404 (permanente) de fallos transitorios.
#
# USO: .venv\Scripts\python.exe tmp\probe_404_audit\test_watch_match_abort.py
# =====================================================================

import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import monitor_v3.main as m  # noqa: E402
from monitor_v3.scrapers.live_scraper import EventNotFoundError  # noqa: E402


def _make_row(seconds_ago: int) -> dict:
    import time

    return {
        "match_id": "17353606",
        "home_team": "Ieliecmaisā.lv",
        "away_team": "Selga",
        "league": "Jūrmalas Čempionats",
        "scheduled_utc_ts": int(time.time()) - seconds_ago,
    }


async def _run_case(name: str, row: dict, fetch, fake_time=None, skip_real_sleep=False) -> str:
    """Ejecuta un caso y devuelve el skip_reason registrado.

    `row` se construye con el reloj REAL antes de simular `fake_time`; si no, la
    fila heredaría el reloj simulado y el caso de prueba no tendría sentido.
    `fake_time` puede ser un dict {"t": ...} mutable para avanzar el reloj desde
    un `asyncio.sleep` simulado (`skip_real_sleep=True`).
    """
    calls: list[tuple[str, str, str]] = []
    original = (m.fetch_event_snapshot, m.update_schedule_status, m._capture_pregame_odds,
                m._final_fetch_and_save, m.time.time, m.asyncio.sleep)

    async def _noop_capture(*a, **k):
        return None

    async def _noop_final(*a, **k):
        return None

    m.fetch_event_snapshot = fetch
    m.update_schedule_status = lambda mid, status, skip_reason="": calls.append((mid, status, skip_reason))
    m._capture_pregame_odds = _noop_capture
    m._final_fetch_and_save = _noop_final
    if fake_time is not None:
        m.time.time = (lambda: fake_time["t"]) if isinstance(fake_time, dict) else (lambda: fake_time)
    if skip_real_sleep:
        _real_sleep = original[-1]  # asyncio.sleep original (evita recursión)

        async def _fake_sleep(secs, *a, **k):
            if isinstance(fake_time, dict):
                fake_time["t"] += secs
            await _real_sleep(0)
        m.asyncio.sleep = _fake_sleep

    try:
        await asyncio.wait_for(m._watch_match("17353606", row, asyncio.Event()), timeout=10)
    except asyncio.TimeoutError:
        print(f"  ❌ [{name}] el watcher NO abandonó (timeout)")
        return "TIMEOUT"
    finally:
        (m.fetch_event_snapshot, m.update_schedule_status, m._capture_pregame_odds,
         m._final_fetch_and_save, m.time.time, m.asyncio.sleep) = original

    if not calls:
        print(f"  ❌ [{name}] el watcher terminó sin registrar estado")
        return "SIN REGISTRO"
    mid, status, reason = calls[-1]
    print(f"  ✅ [{name}] status={status} skip_reason={reason}")
    return reason


async def _fetch_404(match_id):
    raise EventNotFoundError("HTTP 404")


async def _fetch_ok_sin_reloj(match_id):
    # Partido "vivo" pero sin reloj jugado y sin cuartos 3/4: obliga al bucle a
    # seguir iterando (es el caso que quedaba colgado indefinidamente).
    return {"status_type": "notstarted", "status_description": "Not started", "game_seconds_played": None}


async def main() -> None:
    import time as _time

    real_now = _time.time()
    print("\n[A] 404 durante el modo sonda prestart (partido de hace 1 h)")
    r_a = await _run_case("probe-404", _make_row(3600), _fetch_404)

    print("\n[B] 404 después del Drift Exit (partido de hace 2 h)")
    r_b = await _run_case("ventana-404", _make_row(7200), _fetch_404)

    print("\n[C] Evento que nunca arranca: cruza las 3.5 h estando ya en el bucle previo a Q4")
    row_c = _make_row(7200)
    clock = {"t": row_c["scheduled_utc_ts"] + 7200}
    r_c = await _run_case(
        "obsoleto-sin-arranque", row_c, _fetch_ok_sin_reloj,
        fake_time=clock, skip_real_sleep=True,
    )

    esperado = {
        "probe-404": "evento_404_eliminado",
        "ventana-404": "evento_404_eliminado",
        "obsoleto-sin-arranque": "obsoleto_sin_arranque",
    }
    obtenido = {"probe-404": r_a, "ventana-404": r_b, "obsoleto-sin-arranque": r_c}
    print("\nRESULTADO GLOBAL:", "✅ OK" if obtenido == esperado else f"❌ {obtenido} != {esperado}")


if __name__ == "__main__":
    asyncio.run(main())
