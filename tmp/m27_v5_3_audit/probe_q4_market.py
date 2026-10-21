# =====================================================================
# ¿EXISTE EL MERCADO "GANADOR DEL Q4" (4th quarter) Y CUANDO SE PUEDE VER?
# ---------------------------------------------------------------------
#   1. ¿Qué hay YA guardado en match_odds (mercado por periodo)?
#   2. ¿Qué mercados devuelve la API para un partido EN VIVO? (grupos x
#      periodos, live y no-live) -> ¿existe el 4th quarter in-play?
# =====================================================================

import asyncio
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client  # noqa: E402


async def main() -> None:
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row

    print("1) QUE HAY YA EN match_odds (mercado por periodo):")
    for r in conn.execute(
        "SELECT market_period, market_name, COUNT(*) n FROM match_odds "
        "GROUP BY market_period, market_name ORDER BY n DESC"
    ):
        print(f"   {str(r['market_period']):<16} {str(r['market_name']):<24} {r['n']:>8,}")

    client = get_mobile_client()

    print()
    print("2) PARTIDOS EN VIVO: que mercados devuelve la API")
    events = await client.get_live_events()
    for ev in events[:5]:
        mid = str(ev.get("id") or "")
        cat = (((ev.get("tournament") or {}).get("category") or {}).get("name") or "?")
        if not mid:
            continue
        try:
            resp = await client.request("GET", f"event/{mid}/odds/1/all")
            if hasattr(resp, "json"):
                resp = resp.json()
        except Exception as exc:
            print(f"   {mid}: ERROR {exc}")
            continue
        markets = (resp or {}).get("markets") or []
        print(f"   {mid} ({cat}): {len(markets)} mercados")
        pares = Counter()
        for m in markets:
            if m.get("suspended"):
                continue
            flag = "LIVE" if m.get("isLive") else "pre"
            pares[(flag, str(m.get("marketGroup")), str(m.get("marketPeriod")))] += 1
        for (flag, grp, per), n in sorted(pares.items()):
            print(f"      {flag:<4} {grp:<16} {per:<24} x{n}")

    print()
    print("3) ¿ALGUN MERCADO CON PERIODO '4th quarter' EN TODO EL HISTORICO?")
    for r in conn.execute(
        "SELECT market_period, COUNT(*) n FROM match_odds "
        "WHERE lower(market_period) LIKE '%4th%' OR lower(market_period) LIKE '%quarter 4%' "
        "GROUP BY market_period"
    ):
        print("   ", dict(r))
    print("   (vacio = nunca se ha capturado ese mercado)")

    print()
    print("4) RESUMEN DE PERIODOS PRESENTES:")
    for r in conn.execute(
        "SELECT market_period, COUNT(DISTINCT match_id) n FROM match_odds GROUP BY market_period"
    ):
        print(f"   {str(r['market_period']):<16} {r['n']:>8,} partidos")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
