"""
tmp/m27_v5_3_4/probe_odds_shapes.py
Forma de los mercados de odds: (a) partido terminado con spread en BD, (b) partido PRE-PARTIDO proximo (ML + spread + total con linea),
para ver si la linea (handicap) esta disponible y si se puede capturar pre-tip.
"""
import asyncio, json, sys, datetime as dt
sys.path.insert(0, "."); sys.stdout.reconfigure(encoding="utf-8")
from monitor_v3.core.mobile_client import get_mobile_client
async def main():
    c = get_mobile_client()
    r = await c.request("GET", "event/16014405/odds/1/all"); print("FIN 16014405", r.status_code); print(json.dumps(r.json(), ensure_ascii=False)[:1800] if r.status_code==200 else "")
    today = dt.datetime.utcnow().strftime("%Y-%m-%d")
    r = await c.request("GET", f"sport/basketball/scheduled-events/{today}")
    print("sched", r.status_code)
    evs = r.json().get("events", []) if r.status_code == 200 else []
    ns = [e for e in evs if (e.get("status") or {}).get("type") == "notstarted"][:60]
    print(len(evs), len(ns))
    shown = 0
    for e in ns:
        rr = await c.request("GET", f"event/{e['id']}/odds/1/all")
        if rr.status_code != 200: continue
        j = rr.json()
        kinds = [(m.get("marketName"), m.get("marketPeriod"), m.get("isLive")) for m in j.get("markets", [])]
        print(e["id"], (e.get("tournament") or {}).get("name"), kinds)
        if len(kinds) > 2 and shown < 1:
            print(json.dumps(j, ensure_ascii=False)[:3000]); shown += 1
asyncio.run(main())
