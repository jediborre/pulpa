"""
tmp/missing_data_audit/sample_stats.py
Proposito: Para partidos pendientes de team_statistics (sin datos y sin detail_404),
ver que devuelve el endpoint statistics y que parsea fetch_full_match.
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
        "(SELECT 1 FROM team_statistics x WHERE x.match_id=m.match_id) AND NOT EXISTS "
        "(SELECT 1 FROM detail_404 d WHERE d.match_id=m.match_id AND d.table_name='team_statistics') LIMIT 5"
    )]
    mc = get_mobile_client()
    for mid in ids:
        r = await mc.request("GET", f"event/{mid}/statistics")
        body = {}
        try:
            body = r.json()
        except Exception:
            pass
        parsed = None
        try:
            from match.scraper import _parse_team_statistics
            parsed = _parse_team_statistics(mid, body)
        except Exception as e:
            parsed = f"err {e}"
        n_stats = len(body.get("statistics", [])) if isinstance(body, dict) else "?"
        print(f"{mid}: statistics HTTP {r.status_code} ({len(r.content)}b) statistics_groups={n_stats} parsed_rows={len(parsed) if isinstance(parsed, list) else parsed}")


if __name__ == "__main__":
    asyncio.run(main())
