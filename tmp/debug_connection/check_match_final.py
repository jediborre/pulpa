"""
tmp/debug_connection/check_match_final.py
Proposito: Consultar el estado real de un partido (status, marcador final, reloj
played, cuartos) para verificar si el FT se disparo antes de tiempo.
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

MID = sys.argv[1] if len(sys.argv) > 1 else "16830132"


async def main() -> None:
    mc = get_mobile_client()
    r = await mc.request("GET", f"event/{MID}")
    ev = r.json().get("event", {})
    st = ev.get("status") or {}
    tm = ev.get("time") or {}
    hs = ev.get("homeScore") or {}
    as_ = ev.get("awayScore") or {}
    print(f"{MID} {ev.get('homeTeam',{}).get('name')} vs {ev.get('awayTeam',{}).get('name')}")
    print("status:", st.get("type"), "|", st.get("description"))
    print("time  :", json.dumps(tm))
    print("final :", hs.get("current"), "-", as_.get("current"))
    print("cuartos:", {f"Q{i}": (hs.get(f"period{i}"), as_.get(f"period{i}")) for i in range(1, 6)})


if __name__ == "__main__":
    asyncio.run(main())
