"""
tmp/missing_data_audit/sample_marked.py
Proposito: Tomar una muestra de partidos MARCADOS como revisados a los que les falta
lineups y comprobar si el API realmente devuelve 404 (dato inexistente) o si fue un
fallo transitorio (el dato si existe).
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
        "SELECT m.match_id FROM matches m WHERE m.details_checked_at IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM lineups x WHERE x.match_id=m.match_id) LIMIT 10"
    )]
    mc = get_mobile_client()
    for mid in ids:
        r = await mc.request("GET", f"event/{mid}/lineups")
        r2 = await mc.request("GET", f"event/{mid}/statistics")
        print(f"{mid}: lineups HTTP {r.status_code} ({len(r.content)}b) | statistics HTTP {r2.status_code} ({len(r2.content)}b)")


if __name__ == "__main__":
    asyncio.run(main())
