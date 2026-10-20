# =====================================================================
# PRUEBA END-TO-END DEL EVALUADOR REAL (no el formateador)
#   -> ¿el predictions dict incluye al modelo que todavia no ha confirmado
#      su NO BET, con su pick y su confianza?
# =====================================================================

import asyncio
import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.models.evaluator import GameWatcherState, evaluate_match_q4  # noqa: E402


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    client = get_mobile_client()
    try:
        events = await client.get_live_events()
    except Exception as exc:
        print("no se pudieron listar los eventos en vivo:", exc)
        return
    print(f"eventos en vivo: {len(events)}")

    for ev in events[:6]:
        mid = str(ev.get("id") or "")
        lg = (((ev.get("tournament") or {}).get("category") or {}).get("name") or "?")
        if not mid:
            continue
        try:
            payload = await client.fetch_full_match(mid)
        except Exception as exc:
            print(f"  {mid}: ERROR {exc}")
            continue
        m = payload.get("match") or {}
        st = m.get("status_description") or m.get("status_type")
        print()
        print("=" * 92)
        print(f"  {mid} | {m.get('home_team')} vs {m.get('away_team')} | {lg} | {st}")
        state = GameWatcherState(mid)
        try:
            res = await evaluate_match_q4(mid, payload, state)
        except Exception as exc:
            print(f"    ERROR del evaluador: {type(exc).__name__}: {exc}")
            continue
        if not res.get("ok"):
            print(f"    el evaluador no evaluo: {res.get('reason')}")
            continue
        preds = res.get("predictions") or {}
        for modelo, p in preds.items():
            print(f"    {modelo:<12} signal={str(p.get('signal')):<18} pick={str(p.get('pick')):<5} "
                  f"conf={p.get('confidence')} reason={p.get('reason')!r} "
                  f"block={p.get('block_reason')!r}")

        # Mismo filtro que aplica main.py para la alerta
        INTERNOS = ("awaiting_no_bet_confirmations", "no_data")

        def visible(m):
            if m in ("m27_v5_3",) and False:
                return False
            p = preds.get(m) or {}
            sig = p.get("signal")
            if sig in (None,):
                return False
            if sig == "UNAVAILABLE":
                r = str(p.get("reason") or "")
                return bool(r) and r not in INTERNOS
            return True

        print(f"    -> modelos visibles en la ALERTA: {sorted([m for m in preds if visible(m)])}")


if __name__ == "__main__":
    asyncio.run(main())
