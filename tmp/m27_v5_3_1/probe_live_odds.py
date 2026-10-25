"""
tmp/m27_v5_3_1/probe_live_odds.py
Prueba odds en vivo (event/{id}/odds/1/all) en partidos con mercado (KBL, B League, Euroleague, DBBL) para ver:
 si el mercado cambia a in-play (isLive), si conserva initialFractionalValue, y que mercados existen (spread/total/quarter).
"""
import asyncio, json, sys
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from monitor_v3.core.mobile_client import get_mobile_client
async def main():
    c = get_mobile_client()
    evs = await c.get_live_events()
    for e in evs:
        mid = e["id"]; st = (e.get("status") or {}).get("description")
        r = await c.request("GET", f"event/{mid}/odds/1/all")
        if r.status_code != 200: continue
        j = r.json()
        print("\n#####", mid, (e.get("tournament") or {}).get("name"), st, (e.get("homeScore") or {}).get("current"), (e.get("awayScore") or {}).get("current"))
        for m in j.get("markets", []):
            print(" ", m.get("marketName"), "|", m.get("marketGroup"), "|", m.get("marketPeriod"), "| live:", m.get("isLive"), "| susp:", m.get("suspended"),
                  "|", [(ch.get("name"), ch.get("initialFractionalValue"), ch.get("fractionalValue")) for ch in m.get("choices", [])])
asyncio.run(main())
