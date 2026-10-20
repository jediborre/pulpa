# =====================================================================
# ¿LA API OFRECE EL MERCADO "4th quarter winner"? Y ¿que hace el parser?
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


async def main() -> None:
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row

    print("1) ¿QUE MERCADOS POR PERIODO HAY GUARDADOS Y EN QUE LIGAS?")
    for r in conn.execute(
        "SELECT o.market_period, o.market_name, m.league, COUNT(*) n "
        "FROM match_odds o JOIN matches m USING(match_id) "
        "WHERE o.market_period != 'Match' "
        "GROUP BY o.market_period, o.market_name ORDER BY n DESC LIMIT 12"
    ):
        print(f"   {str(r['market_period']):<14} {str(r['market_name']):<22} "
              f"{r['n']:>7,}  {str(r['league'])[:40]}")

    print()
    print("2) PARTIDOS QUE TIENEN CUOTAS DE 1st quarter: ¿que devuelve la API HOY?")
    from monitor_v3.core.mobile_client import get_mobile_client
    client = get_mobile_client()
    mids = [r["match_id"] for r in conn.execute(
        "SELECT DISTINCT o.match_id FROM match_odds o JOIN matches m USING(match_id) "
        "WHERE o.market_period='1st quarter' AND m.date >= '2024-01-01' "
        "ORDER BY m.date DESC LIMIT 6")]
    for mid in mids:
        try:
            resp = await client.request("GET", f"event/{mid}/odds/1/all")
            if hasattr(resp, "json"):
                resp = resp.json()
        except Exception as exc:
            print(f"   {mid}: ERROR {exc}")
            continue
        markets = (resp or {}).get("markets") or []
        pares = Counter()
        for m in markets:
            if m.get("suspended"):
                continue
            flag = "LIVE" if m.get("isLive") else "pre"
            pares[(flag, str(m.get("marketGroup")), str(m.get("marketPeriod")))] += 1
        print(f"   {mid}: {len(markets)} mercados -> "
              + ", ".join(f"{f}/{g}/{p}" for (f, g, p) in sorted(pares)))
        if not markets:
            print("      (la API ya no devuelve cuotas de este partido)")

    print()
    print("3) ¿QUE GUARDA EL PARSER DE CUOTAS? (match/scraper.py::_parse_odds)")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
