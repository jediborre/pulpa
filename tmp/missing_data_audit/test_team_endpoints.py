"""
tmp/missing_data_audit/test_team_endpoints.py
Proposito: Verificar si la API movil (huella OkHttp + UA firmado) puede traer los
endpoints de team_strength directamente: /team/{id} (pregameForm) y
/team/{id}/performance (points), sin navegador.
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
    row = con.execute(
        "SELECT home_team_id, away_team_id, home_team FROM matches "
        "WHERE home_team_id IS NOT NULL ORDER BY date DESC LIMIT 1"
    ).fetchone()
    tid = row[0]
    print(f"team_id de prueba: {tid} ({row[2]})")

    mc = get_mobile_client()
    r = await mc.request("GET", f"team/{tid}")
    print(f"GET team/{tid}: HTTP {r.status_code} len={len(r.content)}")
    if r.status_code == 200:
        body = r.json()
        print("  keys:", list(body.keys())[:12])
        pf = body.get("pregameForm")
        print("  pregameForm:", pf)

    r2 = await mc.request("GET", f"team/{tid}/performance")
    print(f"GET team/{tid}/performance: HTTP {r2.status_code} len={len(r2.content)}")
    if r2.status_code == 200:
        body2 = r2.json()
        pts = body2.get("points") or {}
        print("  points keys:", list(pts)[:5], "count:", len(pts))


if __name__ == "__main__":
    asyncio.run(main())
