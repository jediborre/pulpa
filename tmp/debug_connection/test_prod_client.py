"""
tmp/debug_connection/test_prod_client.py
Proposito: Smoke test del MobileClient de produccion (monitor_v3.core.mobile_client)
con la nueva huella OkHttp + UA firmado: descubrir categorias de una fecha y descargar
un partido completo. Verifica el pipeline real sin tocar la base de datos.
"""
import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client


async def main() -> None:
    mc = get_mobile_client()
    cats = await mc.get_categories_for_date("2026-09-01")
    print(f"categorias con partidos: {len(cats)}")
    if cats:
        c0 = cats[0]["category"]
        print(f"  primera categoria: {c0.get('id')} {c0.get('name')}")
        events = await mc.get_category_scheduled_events(c0["id"], "2026-09-01")
        print(f"  eventos en esa categoria: {len(events)}")
    print("\ndescargando partido 15935071...")
    data = await mc.fetch_full_match("15935071")
    print(f"  match: {data['match']['home_team']} vs {data['match']['away_team']}")
    print(f"  quarters: {list(data['score']['quarters'].keys())}")
    print(f"  pbp periods: {list(data['play_by_play'].keys())}")
    print(f"  graph_points: {len(data['graph_points'])}")
    print(f"  h2h: {len(data.get('h2h', []))}")


if __name__ == "__main__":
    asyncio.run(main())
