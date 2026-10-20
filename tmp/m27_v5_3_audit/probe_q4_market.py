# =====================================================================
# ¿EXISTE EL MERCADO "GANADOR DEL Q4" (4th quarter) Y CUANDO SE PUEDE VER?
# ---------------------------------------------------------------------
# Preguntas:
#   1. ¿El payload de cuotas PRE-partido incluye el mercado de cuarto 4?
#   2. ¿El payload IN-PLAY lo incluye (y con que margen)?
#   3. ¿Cuanto margen tiene la cuota del mercado Q4 frente a la del partido?
#      (si paga mucho mejor que el ML del partido, es informacion que el
#       modelo NO tiene y deberia tener)
#   4. ¿Que hay YA guardado en match_odds (market_period)?
# =====================================================================

import asyncio
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client  # noqa: E402


def frac(v):
    if not v or not isinstance(v, str) or "/" not in v:
        return None
    a, b = v.split("/")
    try:
        return round(int(a) / int(b) + 1.0, 3)
    except (ValueError, ZeroDivisionError):
        return None


async def main() -> None:
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row

    print("1) QUE HAY YA EN match_odds (market_period) — es el historico descargado:")
    for r in conn.execute(
        "SELECT market_period, market_name, COUNT(*) n FROM match_odds "
        "GROUP BY market_period, market_name ORDER BY n DESC"
    ):
        print(f"   {str(r['market_period']):<16} {str(r['market_name']):<22} {r['n']:>8,}")

    print()
    print("2) ¿QUE DEVUELVE LA API PARA UN PARTIDO? (grupos x periodos, live y no-live)")
    client = get_mobile_client()
    mids = [r["match_id"] for r in conn.execute(
        "SELECT match_id FROM matches WHERE status_type='finished' ORDER BY date DESC LIMIT 5")]
    for mid in mids:
        try:
            resp = await client.request("GET", f"event/{mid}/odds/1/all")
            if hasattr(resp, "json"):
                resp = resp.json()
        except Exception as exc:
            print(f"   {mid}: ERROR {exc}")
            continue
        markets = (resp or {}).get("markets") or []
        if not markets:
            print(f"   {mid}: 0 mercados")
            continue
        pares = Counter()
        for m in markets:
            if m.get("suspended"):
                continue
            flag = "LIVE" if m.get("isLive") else "pre"
            pares[(flag, str(m.get("marketGroup")), str(m.get("marketPeriod")))] += 1
        print(f"   {mid}: {len(markets)} mercados")
        for (flag, grp, per), n in sorted(pares.items()):
            print(f"      {flag:<4} {grp:<16} {per:<22} x{n}")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
