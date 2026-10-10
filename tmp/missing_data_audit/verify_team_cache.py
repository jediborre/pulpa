"""
tmp/missing_data_audit/verify_team_cache.py
Proposito: Comprobar empiricamente que team_strength es identico para todos los
partidos de un mismo equipo (por lo que cachear por equipo es valido).
Descarga fetch_full_match(fetch_team_strength=True) para dos partidos distintos del
mismo equipo y compara la fila de team_strength de ese equipo.
"""
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client

TEAM_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 3543  # Barça Basket


async def main() -> None:
    con = sqlite3.connect(ROOT / "matches.db")
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT match_id FROM matches WHERE home_team_id = ? OR away_team_id = ? "
        "ORDER BY date DESC LIMIT 2",
        (TEAM_ID, TEAM_ID),
    ).fetchall()
    if len(rows) < 2:
        print("no hay 2 partidos para ese equipo")
        return
    mid1, mid2 = str(rows[0]["match_id"]), str(rows[1]["match_id"])
    print(f"equipo {TEAM_ID} | partidos: {mid1}, {mid2}")

    mc = get_mobile_client()
    d1 = await mc.fetch_full_match(mid1, fetch_team_strength=True)
    d2 = await mc.fetch_full_match(mid2, fetch_team_strength=True)

    def row_for(d):
        for r in d.get("team_strength", []):
            if r.get("team_id") == TEAM_ID:
                return r
        return None

    r1, r2 = row_for(d1), row_for(d2)
    print("\npartido 1 ->", json.dumps(r1, ensure_ascii=False)[:300])
    print("partido 2 ->", json.dumps(r2, ensure_ascii=False)[:300])
    print("\nIGUALES?:", r1 == r2)
    if r1 and r2:
        print("  position igual:", r1.get("position") == r2.get("position"))
        print("  wins/losses igual:", (r1.get("wins"), r1.get("losses")) == (r2.get("wins"), r2.get("losses")))
        print("  form igual:", r1.get("form") == r2.get("form"))
        print("  perf_points igual:", r1.get("perf_points") == r2.get("perf_points"))


if __name__ == "__main__":
    asyncio.run(main())
