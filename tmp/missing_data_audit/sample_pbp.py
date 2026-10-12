"""
tmp/missing_data_audit/sample_pbp.py
Proposito: Ver, para partidos sin play_by_play, que devuelve el endpoint incidents
(200 vacio vs 404) para decidir como marcarlos.
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client


async def main() -> None:
    con = sqlite3.connect(ROOT / "matches.db")
    ids = [str(r[0]) for r in con.execute(
        "SELECT m.match_id FROM matches m WHERE NOT EXISTS "
        "(SELECT 1 FROM play_by_play p WHERE p.match_id=m.match_id) "
        "AND EXISTS (SELECT 1 FROM match_events e WHERE e.match_id=m.match_id) LIMIT 8"
    )]
    mc = get_mobile_client()
    for mid in ids:
        r = await mc.request("GET", f"event/{mid}/incidents")
        n = 0
        try:
            n = len(r.json().get("incidents", []))
        except Exception:
            pass
        print(f"{mid}: incidents HTTP {r.status_code} ({len(r.content)}b) incidents={n}")


if __name__ == "__main__":
    asyncio.run(main())
