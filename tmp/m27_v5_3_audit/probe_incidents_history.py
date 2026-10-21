# =====================================================================
# ¿EL ENDPOINT 'incidents' TIENE FALTAS (Y MAS) EN HISTORICO?
#   Es el play-by-play completo: faltas, tiros libres, tiempos muertos,
#   sustituciones, con marca de tiempo. Si funciona en las ligas de 10
#   min, permite reconstruir historicamente "faltas hasta el min 27" =
#   la MISMA feature que se captura en vivo -> paridad.
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

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    client = get_mobile_client()
    print(f"{'liga':<40}{'año':>6}{'incid':>7}{'con fouls':>11}{'con ft':>9}"
          f"{'con to':>9}{'con sub':>10}")
    muestras = 0
    total_inc = 0
    con_fouls = 0
    for liga, anio in (("NCAA Women's Division I, Regular season", "2024"),
                       ("Poland 2nd Basketball League, Runda Zasadnicza", "2024"),
                       ("ZBL, Women", "2025"),
                       ("LNB, Women", "2024"),
                       ("Brazil NBB", "2023"),
                       ("Federal Basketball League, Australia", "2024"),
                       ("Spain Liga ACB", "2023")):
        r = conn.execute("SELECT match_id FROM matches WHERE league=? AND status_type='finished' "
                         "AND substr(date,1,4)=? ORDER BY date DESC LIMIT 1", (liga, anio)).fetchone()
        if not r:
            continue
        mid = r["match_id"]
        try:
            resp = await client.request("GET", f"event/{mid}/incidents")
            resp = resp.json() if hasattr(resp, "json") else resp
        except Exception as exc:
            print(f"{liga[:39]:<40}{anio:>6}  ERROR {exc}")
            continue
        inc = (resp or {}).get("incidents") or []
        tipos = Counter()
        for i in inc:
            t = (i.get("type") or {})
            tipos[t.get("name") or t.get("key") or "?"] += 1
        n_fouls = sum(v for k, v in tipos.items() if "Foul" in k)
        n_ft = sum(v for k, v in tipos.items() if "Free Throw" in k)
        n_to = sum(v for k, v in tipos.items() if "Timeout" in k)
        n_sub = sum(v for k, v in tipos.items() if "Substitution" in k)
        total_inc += len(inc)
        con_fouls += 1 if n_fouls else 0
        muestras += 1
        print(f"{liga[:39]:<40}{anio:>6}{len(inc):>7}{n_fouls:>11}{n_ft:>9}{n_to:>9}{n_sub:>10}")
        await asyncio.sleep(0.5)
    print()
    print(f"muestras: {muestras} | incidentes totales: {total_inc} | con faltas: {con_fouls}")
    print("tipos de incidente (muestra):", dict(tipos))

    print()
    print("ESTRUCTURA DE UN INCIDENTE CON FALTA (para ver el timestamp):")
    for i in inc[:400]:
        t = (i.get("type") or {})
        if "Foul" in (t.get("name") or ""):
            print(json.dumps({k: i.get(k) for k in
                              ("time", "incident", "period", "id", "participants")},
                             ensure_ascii=False)[:600])
            break


asyncio.run(main())
conn.close()
