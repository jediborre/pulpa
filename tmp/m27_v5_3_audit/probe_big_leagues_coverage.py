# =====================================================================
# ¿QUE LIGAS TIENEN DESGLOSE POR CUARTO? (muestra de las ligas grandes)
#   Es donde el detector de faltas/tiros libres seria aplicable.
# =====================================================================

import asyncio
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    client = get_mobile_client()
    ligas = [r["league"] for r in conn.execute(
        "SELECT league, COUNT(*) n FROM matches WHERE status_type='finished' "
        "AND date >= '2024-01-01' GROUP BY league ORDER BY n DESC LIMIT 40")]
    print(f"{'liga':<46}{'partidos':>10}{'muestra':>9}{'con 1Q-3Q':>12}")
    cubiertas = []
    t0 = time.time()
    for liga in ligas:
        mids = [r["match_id"] for r in conn.execute(
            "SELECT match_id FROM matches WHERE league=? AND status_type='finished' "
            "AND date >= '2024-06-01' ORDER BY date DESC LIMIT 2", (liga,))]
        if not mids:
            continue
        n_1q = 0
        for mid in mids:
            try:
                resp = await client.request("GET", f"event/{mid}/statistics")
                resp = resp.json() if hasattr(resp, "json") else resp
            except Exception:
                continue
            periods = {s.get("period") for s in ((resp or {}).get("statistics") or [])}
            if {"1Q", "2Q", "3Q"} <= periods:
                n_1q += 1
            await asyncio.sleep(0.45)
        n_tot = conn.execute("SELECT COUNT(*) n FROM matches WHERE league=? "
                             "AND status_type='finished' AND date >= '2024-01-01'",
                             (liga,)).fetchone()["n"]
        if n_1q:
            cubiertas.append((liga, n_tot))
        print(f"{liga[:45]:<46}{n_tot:>10,}{len(mids):>9}{n_1q:>12}")
    print()
    print(f"ligas con desglose por cuarto: {len(cubiertas)} de {len(ligas)} muestreadas")
    print("partidos acumulados en esas ligas (2024+): "
          f"{sum(n for _l, n in cubiertas):,}")
    print(f"tiempo: {time.time() - t0:.0f}s")


asyncio.run(main())
conn.close()
