"""
tmp/missing_data_audit/test_save_one.py
Proposito: Descargar un partido pendiente de team_statistics y guardarlo, para ver si
el guardado realmente persiste team_statistics (aislar el bug).
"""
import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "match")):
    if p not in sys.path:
        sys.path.insert(0, p)

import db as db_mod
from monitor_v3.core.mobile_client import get_mobile_client

MID = sys.argv[1] if len(sys.argv) > 1 else "10049297"


async def main() -> None:
    mc = get_mobile_client()
    data = await mc.fetch_full_match(MID)
    print("team_statistics parseadas:", len(data.get("team_statistics") or []))
    print("period_stats parseadas:", len(data.get("period_stats") or []))
    print("_endpoint_status:", data.get("_endpoint_status"))
    conn = db_mod.get_conn(str(ROOT / "matches.db"))
    db_mod.init_db(conn)
    db_mod.save_match(conn, MID, data)
    n = conn.execute("SELECT COUNT(*) FROM team_statistics WHERE match_id=?", (MID,)).fetchone()[0]
    print("team_statistics en DB:", n)
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
