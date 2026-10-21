# =====================================================================
# ¿SE PUEDEN DESCARGAR LAS FALTAS (Y OTROS EVENTOS) EN EL HISTORICO?
#   1) ¿Los endpoints 'lineups' (stats por jugador) y 'statistics' (stats
#      de equipo por periodo) traen datos en partidos ANTIGUOS?
#   2) ¿Hasta qué fecha llega el detalle?
#   3) Coste de un backfill SELECTIVO por ligas rentables.
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


async def main() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client
    client = get_mobile_client()

    print("1) DISPONIBILIDAD DEL DETALLE POR AÑO (partidos ya descargados):")
    for r in conn.execute("""
        SELECT substr(m.date,1,4) yr, COUNT(*) n FROM matches m
        WHERE m.status_type='finished' AND m.date >= '2019-01-01'
        GROUP BY yr ORDER BY yr"""):
        print(f"   {r['yr']}: {r['n']:,} partidos acabados")

    print()
    print("2) PRUEBA DE LOS ENDPOINTS EN PARTIDOS HISTORICOS (1 request c/u):")
    for yr in ("2026", "2025", "2024", "2022", "2020"):
        r = conn.execute("""
            SELECT m.match_id, m.home_team, m.away_team, m.league
            FROM matches m WHERE m.status_type='finished' AND substr(m.date,1,4)=?
            ORDER BY m.date DESC LIMIT 1""", (yr,)).fetchone()
        if not r:
            continue
        mid = r["match_id"]
        t0 = time.time()
        try:
            lin = await client.request("GET", f"event/{mid}/lineups")
            lin = lin.json() if hasattr(lin, "json") else lin
        except Exception as exc:
            print(f"   {yr} {mid}: ERROR lineups {exc}")
            continue
        try:
            st = await client.request("GET", f"event/{mid}/statistics")
            st = st.json() if hasattr(st, "json") else st
        except Exception as exc:
            print(f"   {yr} {mid}: ERROR statistics {exc}")
            continue
        dt = time.time() - t0

        # faltas por jugador
        n_jug = 0
        con_faltas = 0
        for team in (lin or {}).get("lineups") or []:
            for p in team.get("players") or []:
                n_jug += 1
                st_ = p.get("statistics") or {}
                if st_.get("fouls") is not None:
                    con_faltas += 1
        # stats de equipo por periodo
        stats = (st or {}).get("statistics") or []
        periods = []
        for s in stats:
            periods.append(s.get("period"))
            groups = s.get("groups") or []
            for g in groups:
                names = [it.get("name") for it in (g.get("statisticsItems") or [])]
                if "Fouls" in names or "fouls" in names:
                    break
        tiene_faltas_equipo = any(
            any(str(it.get("name", "")).lower() == "fouls"
                for it in (g.get("statisticsItems") or []))
            for s in stats for g in (s.get("groups") or []))
        print(f"   {yr} {mid} | {str(r['home_team'])[:18]} vs {str(r['away_team'])[:18]:<18} "
              f"| {dt:.2f}s")
        print(f"      lineups: {n_jug} jugadores, {con_faltas} con FALTAS | "
              f"statistics: {len(stats)} periodos {periods} | faltas de equipo: {tiene_faltas_equipo}")

    print()
    print("3) LIGAS RENTABLES (calibracion del modelo) Y COSTE DE UN BACKFILL SELECTIVO:")
    cal_path = ROOT / "models" / "m27_v5_3" / "model_outputs" / "league_calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text(encoding="utf-8"))
        print(f"   ligas en la calibracion: {len(cal)}")
        muestra = cal[:3]
        print(f"   ejemplo: {muestra}")
    conn.close()


asyncio.run(main())
