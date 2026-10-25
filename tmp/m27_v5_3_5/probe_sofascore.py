"""
tmp/m27_v5_3_5/probe_sofascore.py
Sondeo en vivo de la API movil de SofaScore (cliente monitor_v3) para m27_v5_3_5:
 - lista eventos de baloncesto en vivo y su minuto/estado
 - estructura REAL de event/{id}/odds/1/all (mercados, initialFractionalValue vs fractionalValue) en vivo y en un partido terminado
 - estructura de lineups (statistics por jugador en vivo: secondsPlayed, fouls, plusMinus)
 - endpoints candidatos (statistics, odds/1/featured, votes) y su disponibilidad
Hipotesis: la cuota 'actual' en vivo es in-play; initialFractionalValue permanece como apertura.
"""
import asyncio, json, sys
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")
from monitor_v3.core.mobile_client import get_mobile_client

async def main():
    c = get_mobile_client()
    evs = await c.get_live_events()
    print("live events:", len(evs))
    rows = []
    for e in evs:
        st = e.get("status", {}); t = e.get("time", {})
        rows.append((e["id"], (e.get("tournament") or {}).get("name"), st.get("description"), t.get("played"), t.get("periodLength"),
                     (e.get("homeScore") or {}).get("current"), (e.get("awayScore") or {}).get("current")))
    for r in rows[:25]: print(r)
    tgt = [r for r in rows if r[2] and ("3rd" in r[2] or "4th" in r[2] or "2nd" in r[2])]
    print("candidatos Q2-Q4:", tgt[:5])
    pick = (tgt or rows)[:1]
    for r in pick:
        mid = r[0]
        for ep in (f"event/{mid}/odds/1/all", f"event/{mid}/odds/1/featured", f"event/{mid}/lineups", f"event/{mid}/statistics", f"event/{mid}/winning-odds"):
            res = await c.request("GET", ep)
            print("\n=====", ep, res.status_code)
            if res.status_code == 200:
                txt = json.dumps(res.json(), ensure_ascii=False)
                print(txt[:2500])
    # partido terminado reciente
    fin = await c.request("GET", "event/12451859/odds/1/all")
    print("\n===== FINISHED odds", fin.status_code); print(json.dumps(fin.json(), ensure_ascii=False)[:2500] if fin.status_code == 200 else fin.text[:200])
asyncio.run(main())
