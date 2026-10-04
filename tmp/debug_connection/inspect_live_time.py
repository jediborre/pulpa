"""
tmp/debug_connection/inspect_live_time.py
Proposito: Inspeccionar que campos de tiempo expone SofaScore para un partido en vivo
(event.time, event.status, graph.periodTime, minute de graphPoints, max timeSeconds de
incidents) y compararlos con el minuto inferido actualmente. Objetivo: hallar la fuente
fiable del minuto para no disparar el FT antes de tiempo.
"""
import asyncio
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client


async def main() -> None:
    mc = get_mobile_client()
    live = await mc.get_live_events()
    print(f"partidos en vivo: {len(live)}")
    if not live:
        print("No hay partidos en vivo ahora; probando un partido terminado.")
        mids = ["16988998"]
    else:
        mids = [str(ev.get("id")) for ev in live[:4]]

    for mid in mids:
        try:
            r = await mc.request("GET", f"event/{mid}")
            ev = r.json().get("event", {})
            status = ev.get("status") or {}
            time_obj = ev.get("time") or {}
            print(f"\n=== {mid} {ev.get('homeTeam',{}).get('name')} vs {ev.get('awayTeam',{}).get('name')} ===")
            print("status.type        :", status.get("type"))
            print("status.description :", status.get("description"))
            print("status.code        :", status.get("code"))
            print("time               :", json.dumps(time_obj))
            print("score current      :", (ev.get("homeScore") or {}).get("current"),
                  "-", (ev.get("awayScore") or {}).get("current"))
            # graph
            g = await mc.request("GET", f"event/{mid}/graph")
            gj = g.json() if g.status_code == 200 else {}
            pts = gj.get("graphPoints", [])
            print("graph periodTime   :", gj.get("periodTime"), "overtimeLength:", gj.get("overtimeLength"),
                  "periodCount:", gj.get("periodCount"))
            if pts:
                print("graphPoints count  :", len(pts))
                print("  primero:", {k: pts[0].get(k) for k in ("minute", "value", "period")})
                print("  ultimo :", {k: pts[-1].get(k) for k in ("minute", "value", "period")})
                print("  minutos unicos   :", sorted({p.get("minute") for p in pts})[:5], "...",
                      sorted({p.get("minute") for p in pts})[-5:])
            # incidents
            inc = await mc.request("GET", f"event/{mid}/incidents")
            ij = inc.json() if inc.status_code == 200 else {}
            incs = ij.get("incidents", [])
            if incs:
                max_ts = max((i.get("timeSeconds") or 0) for i in incs)
                print("incidents max timeSeconds:", max_ts, "=> min ~", round(max_ts / 60, 1))
        except Exception as exc:
            print(f"ERR {mid}: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
