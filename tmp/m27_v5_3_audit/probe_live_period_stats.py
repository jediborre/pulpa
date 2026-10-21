# =====================================================================
# ¿EL ENDPOINT 'statistics' TRAE EL DESGLOSE POR CUARTO EN VIVO?
#   (si solo trae 'ALL', las faltas no sirven como feature en vivo:
#    el agregado del partido solo se conoce al final)
# =====================================================================

import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    from match.scraper import _parse_team_statistics
    client = get_mobile_client()
    events = await client.get_live_events()
    n = 0
    for ev in events:
        mid = str(ev.get("id") or "")
        st = str((ev.get("status") or {}).get("description") or "")
        if not mid or "quarter" not in st.lower():
            continue
        try:
            resp = await client.request("GET", f"event/{mid}/statistics")
            resp = resp.json() if hasattr(resp, "json") else resp
        except Exception as exc:
            print(f"   {mid}: ERROR {exc}")
            continue
        rows = _parse_team_statistics(mid, resp)
        periodos = sorted({r["period"] for r in rows})
        print(f"   {mid} [{st}] periodos: {periodos}")
        for r in rows:
            if r["stat_name"] in ("Fouls", "Free throws", "Turnovers", "Time spent in lead"):
                print(f"      {r['period']:<4} {r['stat_name']:<20} {r['home_value']} - {r['away_value']}")
        n += 1
        if n >= 4:
            break
    if n == 0:
        print("   no hay partidos en vivo con cuarto en curso")


asyncio.run(main())
