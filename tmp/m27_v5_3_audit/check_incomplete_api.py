# =====================================================================
# ¿LOS PARTIDOS "INCOMPLETOS" TIENEN REALMENTE 4 CUARTOS EN LA API?
#   (si la API tampoco tiene el Q4, NO hay nada que descargar)
# =====================================================================

import asyncio
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row

    print("A) Partidos con menos de 4 filas en la legacy: ¿qué dice la API?")
    mids = [r["match_id"] for r in conn.execute(
        "SELECT match_id FROM quarter_scores GROUP BY match_id HAVING COUNT(*)<4 "
        "ORDER BY match_id LIMIT 25")]
    client = get_mobile_client()
    dist = Counter()
    ejemplos = []
    for mid in mids:
        try:
            resp = await client.request("GET", f"event/{mid}")
            if hasattr(resp, "json"):
                resp = resp.json()
        except Exception as exc:
            dist["error"] += 1
            continue
        if not isinstance(resp, dict):
            dist["error"] += 1
            continue
        ev = (resp or {}).get("event") or {}
        hs = ev.get("homeScore") or {}
        n = sum(1 for k in ("period1", "period2", "period3", "period4")
                if hs.get(k) is not None)
        dist[n] += 1
        if len(ejemplos) < 6:
            ejemplos.append((mid, {k: hs.get(k) for k in
                                   ("period1", "period2", "period3", "period4")},
                             ev.get("statusDescription") or ev.get("statusType")))
    print(f"   muestra de {len(mids)} partidos -> periodos en la API: {dict(sorted(dist.items()))}")
    for mid, per, st in ejemplos:
        print(f"     {mid}: {per}  status={st}")

    print()
    print("B) Para los que SÍ tienen 4 periodos en la API pero menos de 4 en la legacy:")
    print("   -> esos son los únicos que se pueden recuperar descargando.")
    n_recuperables = dist.get(4, 0)
    total_incompletos = conn.execute(
        "SELECT COUNT(*) FROM (SELECT match_id FROM quarter_scores GROUP BY match_id HAVING COUNT(*)<4)"
    ).fetchone()[0]
    print(f"   muestra: {n_recuperables}/{len(mids)} recuperables "
          f"({n_recuperables * 100 / max(len(mids), 1):.0f} %)")
    print(f"   extrapolado al total de {total_incompletos:,} incompletos: "
          f"~{total_incompletos * n_recuperables // max(len(mids), 1):,} recuperables")

    print()
    print("C) Mismo control para un partido COMPLETO (sanidad):")
    ok_mid = conn.execute(
        "SELECT match_id FROM quarter_scores GROUP BY match_id HAVING COUNT(*)=4 LIMIT 1").fetchone()[0]
    resp = await client.request("GET", f"event/{ok_mid}")
    if hasattr(resp, "json"):
        resp = resp.json()
    hs = ((resp or {}).get("event") or {}).get("homeScore") or {}
    print(f"   {ok_mid}: {[(k, hs.get(k)) for k in ('period1','period2','period3','period4')]}")
    conn.close()


if __name__ == "__main__":
    asyncio.run(main())
