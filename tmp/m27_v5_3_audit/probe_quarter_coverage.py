# =====================================================================
# ¿EN QUE LIGAS EXISTE EL DESGLOSE POR CUARTO?
#   (necesario para saber donde el detector de faltas seria aplicable)
#   Muestreo de 3 partidos por liga de la calibracion.
# =====================================================================

import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
cal = json.loads((ROOT / "models" / "m27_v5_3" / "model_outputs" /
                  "league_calibration.json").read_text(encoding="utf-8"))


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    client = get_mobile_client()
    print(f"{'liga':<44}{'bets':>6}{'WR':>7}{'muestra':>9}{'con 3Q':>9}{'con 1Q-3Q':>12}")
    con_q = []
    sin_q = []
    t0 = time.time()
    for c in sorted(cal, key=lambda x: -x.get("bets", 0)):
        liga = c["league"]
        mids = [r["match_id"] for r in conn.execute(
            "SELECT match_id FROM matches WHERE league=? AND status_type='finished' "
            "AND date >= '2024-01-01' ORDER BY date DESC LIMIT 3", (liga,))]
        if not mids:
            continue
        n_ok = 0
        n_3q = 0
        n_1q = 0
        for mid in mids:
            try:
                resp = await client.request("GET", f"event/{mid}/statistics")
                resp = resp.json() if hasattr(resp, "json") else resp
            except Exception:
                continue
            periods = {s.get("period") for s in ((resp or {}).get("statistics") or [])}
            if periods:
                n_ok += 1
            if "3Q" in periods:
                n_3q += 1
            if {"1Q", "2Q", "3Q"} <= periods:
                n_1q += 1
            await asyncio.sleep(0.5)
        etiq = "SI" if n_1q else "no"
        print(f"{liga[:43]:<44}{c.get('bets', 0):>6}{(c.get('wr') or 0):>6.1f}%"
              f"{len(mids):>9}{n_3q:>9}{n_1q:>12}")
        if n_1q:
            con_q.append((liga, c.get("bets", 0), c.get("wr")))
        else:
            sin_q.append((liga, c.get("bets", 0), c.get("wr")))
    print()
    print(f"con desglose 1Q-3Q: {len(con_q)} ligas | bets {sum(x[1] for x in con_q):,} | "
          f"WR medio {sum(x[2] * x[1] for x in con_q) / max(sum(x[1] for x in con_q), 1):.1f} %")
    print(f"sin desglose     : {len(sin_q)} ligas | bets {sum(x[1] for x in sin_q):,} | "
          f"WR medio {sum(x[2] * x[1] for x in sin_q) / max(sum(x[1] for x in sin_q), 1):.1f} %")
    print(f"tiempo: {time.time() - t0:.0f}s")


asyncio.run(main())
conn.close()
