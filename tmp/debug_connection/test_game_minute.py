"""
tmp/debug_connection/test_game_minute.py
Proposito: Verificar que el MobileClient/parser expone match.game_seconds_played y que
el minuto calculado (played//60) coincide con el cuarto real de un partido en vivo.
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
    live = await mc.get_live_events()
    for ev in live[:5]:
        mid = str(ev.get("id"))
        data = await mc.fetch_full_match(mid)
        m = data.get("match", {})
        played = m.get("game_seconds_played")
        minute = int(played // 60) if played is not None else None
        print(f"{mid} | {m.get('status_description')} | played={played}s -> MIN {minute} "
              f"| period_length={m.get('period_length')} | clock_running={m.get('clock_running')} "
              f"| {m.get('home_team')} {data['score']['home']}-{data['score']['away']} {m.get('away_team')}")


if __name__ == "__main__":
    asyncio.run(main())
