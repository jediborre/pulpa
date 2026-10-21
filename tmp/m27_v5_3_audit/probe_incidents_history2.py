# =====================================================================
# ¿EL FEED 'incidents' TIENE FALTAS/TIEMPOS MUERTOS EN HISTORICO?
#   (con las claves correctas: incidentType / isHome / timeSeconds)
#   Si funciona en las ligas de 10 min, el backfill del feed da el
#   desglose POR CUARTO y con el tiempo exacto -> misma feature que vivo.
# =====================================================================

import asyncio
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    from match.scraper import _parse_period_stats
    client = get_mobile_client()

    print(f"{'liga':<42}{'año':>6}{'incid':>7}{'fouls':>7}{'timeouts':>10}"
          f"{'turnovers':>11}{'3Q fouls':>10}")
    tot_inc = tot_fouls = tot_to = 0
    muestras = 0
    for liga, anio in (("NCAA Women's Division I, Regular season", "2024"),
                       ("Poland 2nd Basketball League, Runda Zasadnicza", "2024"),
                       ("ZBL, Women", "2025"),
                       ("LNB, Women", "2024"),
                       ("Brazil NBB", "2023"),
                       ("Spain Liga ACB", "2023"),
                       ("China CBA", "2024"),
                       ("Turkey Basketball Super League", "2024"),
                       ("B League One", "2024"),
                       ("France Pro A", "2023")):
        r = conn.execute("SELECT match_id FROM matches WHERE league=? AND status_type='finished' "
                         "AND substr(date,1,4)=? ORDER BY date DESC LIMIT 1", (liga, anio)).fetchone()
        if not r:
            continue
        mid = r["match_id"]
        try:
            resp = await client.request("GET", f"event/{mid}/incidents")
            resp = resp.json() if hasattr(resp, "json") else resp
        except Exception as exc:
            print(f"{liga[:41]:<42}{anio:>6}  ERROR {exc}")
            continue
        inc = (resp or {}).get("incidents") or []
        tipos = Counter(i.get("incidentType") or "?" for i in inc)
        n_fouls = tipos.get("foul", 0)
        n_to = tipos.get("timeout", 0)
        n_turn = tipos.get("turnover", 0)
        # desglose por cuarto con el parser canonico
        filas = _parse_period_stats(str(mid), inc, period_seconds=600)
        q3 = {f["stat_name"]: (f["home_value"], f["away_value"]) for f in filas
              if f["period"] == "Q3"}
        tot_inc += len(inc)
        tot_fouls += n_fouls
        tot_to += n_to
        muestras += 1
        print(f"{liga[:41]:<42}{anio:>6}{len(inc):>7}{n_fouls:>7}{n_to:>10}{n_turn:>11}"
              f"{str(q3.get('Fouls')):>10}")
        await asyncio.sleep(0.5)
    print()
    print(f"muestras {muestras} | incidentes {tot_inc} | faltas {tot_fouls} | timeouts {tot_to}")
    print("tipos (ultima muestra):", dict(tipos))


asyncio.run(main())
conn.close()
