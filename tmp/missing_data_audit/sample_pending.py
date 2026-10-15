"""
tmp/missing_data_audit/sample_pending.py
Proposito: Ver por que los 63 pendientes fallan: consultar el endpoint event de una
muestra y ver si devuelve 404 (partido inexistente) u otro error.
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

TABLES = ["play_by_play", "graph_points", "match_events", "match_h2h",
          "team_statistics", "lineups", "player_stats", "match_odds", "team_strength"]


def _missing(t):
    if t == "team_statistics":
        return "NOT EXISTS (SELECT 1 FROM team_statistics x WHERE x.match_id=m.match_id AND x.period='ALL')"
    return f"NOT EXISTS (SELECT 1 FROM {t} x WHERE x.match_id=m.match_id)"


async def main() -> None:
    con = sqlite3.connect(ROOT / "matches.db")
    cond = " OR ".join(
        f"({_missing(t)} AND NOT EXISTS (SELECT 1 FROM detail_404 d WHERE d.match_id=m.match_id AND d.table_name='{t}'))"
        for t in TABLES
    )
    ids = [str(r[0]) for r in con.execute(f"SELECT m.match_id FROM matches m WHERE ({cond}) LIMIT 10")]
    print("pendientes (muestra):", ids)
    mc = get_mobile_client()
    for mid in ids:
        r = await mc.request("GET", f"event/{mid}")
        print(f"  {mid}: event HTTP {r.status_code} ({len(r.content)}b)")


if __name__ == "__main__":
    asyncio.run(main())
