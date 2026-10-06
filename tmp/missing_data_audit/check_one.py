"""
tmp/missing_data_audit/check_one.py
Proposito: Descargar un partido y ver que endpoints devuelven datos (lineups,
player_stats, team_statistics, odds, h2h) para entender por que siguen faltando.
"""
import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client

MID = sys.argv[1] if len(sys.argv) > 1 else "16940786"


async def main() -> None:
    mc = get_mobile_client()
    data = await mc.fetch_full_match(MID)
    print(f"{MID}: {data['match']['home_team']} vs {data['match']['away_team']} | {data['match'].get('status_type')}")
    for k in ("lineups", "player_stats", "team_statistics", "period_stats", "odds", "h2h", "graph_points"):
        v = data.get(k) or []
        print(f"  {k:<16}: {len(v)}")
    # probar endpoints crudos
    for ep in ("lineups", "statistics", "odds/1/all"):
        r = await mc.request("GET", f"event/{MID}/{ep}")
        print(f"  raw {ep:<12}: HTTP {r.status_code} len={len(r.content)}")


if __name__ == "__main__":
    asyncio.run(main())
