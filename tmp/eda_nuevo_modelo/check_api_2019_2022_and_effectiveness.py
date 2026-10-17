"""
tmp/eda_nuevo_modelo/check_api_2019_2022_and_effectiveness.py
=============================================================
Propósito:
1. Auditar exactamente qué rangos de fechas faltan en `matches.db` por arquetipo
   (`fiba_men_10m`, `fiba_women_10m`, `nba_12m`).
2. Consultar en vivo la API móvil de SofaScore (`MobileClient`) en los años 2018-Q4,
   2019, 2020, 2021 y 2022 para verificar que los partidos FIBA existen en la API y
   qué endpoints de detalle tienen cobertura (`incidents`/PBP, `graph`, `lineups`,
   `statistics`, `h2h`).
3. Calcular la efectividad teórica exacta en corridas del modelo campeón `models/m27_v4`.
"""
import asyncio
import json
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monitor_v3.core.mobile_client import get_mobile_client


def check_db_date_boundaries():
    print("=" * 80)
    print("0. FRONTERAS EXACTAS DE FECHAS EN matches.db POR ARQUETIPO")
    print("=" * 80)
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT
            substr(m.date, 1, 7) AS ym,
            SUM(CASE WHEN lc.quarter_duration_minutes = 12 THEN 1 ELSE 0 END) AS nba_12m,
            SUM(CASE WHEN lc.quarter_duration_minutes = 10 AND lc.is_women = 0 THEN 1 ELSE 0 END) AS fiba_men,
            SUM(CASE WHEN lc.quarter_duration_minutes = 10 AND lc.is_women = 1 THEN 1 ELSE 0 END) AS fiba_women
        FROM matches m
        LEFT JOIN leagues_classification lc ON m.league = lc.league
        WHERE m.date >= '2018-08' AND m.date <= '2023-03'
        GROUP BY ym
        ORDER BY ym
    """).fetchall()
    for ym, n_nba, n_fm, n_fw in rows:
        if n_fm > 0 or n_fw > 0 or ym in ("2018-09", "2018-10", "2019-01", "2020-01", "2021-01", "2022-01", "2022-12", "2023-01"):
            print(f"  {ym}: fiba_men_10m={n_fm:5d} | fiba_women_10m={n_fw:4d} | nba_12m={n_nba:4d}")
    conn.close()


async def probe_sofascore_2019_2022():
    print("\n" + "=" * 80)
    print("1. SONDEO EN VIVO A LA API DE SOFASCORE PARA EL HUECO FIBA (2018-10 a 2022-12)")
    print("=" * 80)

    client = get_mobile_client()
    sample_dates = [
        ("2019-11-15", "Temporada 2019-20"),
        ("2020-12-18", "Temporada 2020-21"),
        ("2021-11-19", "Temporada 2021-22"),
        ("2022-11-18", "Temporada 2022-23"),
    ]

    for dt_str, label in sample_dates:
        events = await client.get_all_scheduled_events_for_date(dt_str)
        status = 200 if events else 404
        finished = [
            e for e in events
            if str((e.get("status") or {}).get("type", "")).lower() == "finished"
        ]
        nba_cnt = 0
        fiba_leagues = {}
        sample_fiba_event = None
        for e in finished:
            tourn = (e.get("tournament") or {}).get("name", "")
            cat = ((e.get("tournament") or {}).get("category") or {}).get("name", "")
            full_lg = f"{cat} - {tourn}" if cat else tourn
            if any(x in tourn.upper() for x in ["NBA", "CBA", "PBA", "NBL"]):
                nba_cnt += 1
            else:
                fiba_leagues[full_lg] = fiba_leagues.get(full_lg, 0) + 1
                if sample_fiba_event is None and any(
                    k in full_lg.lower()
                    for k in ["euroleague", "acb", "liga", "bbl", "lnb", "serie a", "eurocup", "turk"]
                ):
                    sample_fiba_event = (e.get("id"), full_lg, e)

        if sample_fiba_event is None and finished:
            for e in finished:
                tourn = (e.get("tournament") or {}).get("name", "")
                if "NBA" not in tourn.upper():
                    sample_fiba_event = (e.get("id"), tourn, e)
                    break

        print(f"\nFecha {dt_str} ({label}) -> HTTP {status}:")
        print(f"  Total eventos terminados en API: {len(finished)} | NBA/12m: {nba_cnt} | FIBA 10m disponibles: {len(finished) - nba_cnt} en {len(fiba_leagues)} ligas")
        top_lgs = sorted(fiba_leagues.items(), key=lambda x: -x[1])[:6]
        print(f"  Top ligas FIBA ese día: {top_lgs}")

        if sample_fiba_event:
            ev_id, lg_name, ev_obj = sample_fiba_event
            await asyncio.sleep(0.8)
            r_inc = await client.request("GET", f"/event/{ev_id}/incidents")
            await asyncio.sleep(0.8)
            r_grp = await client.request("GET", f"/event/{ev_id}/graph")
            await asyncio.sleep(0.8)
            r_lin = await client.request("GET", f"/event/{ev_id}/lineups")
            await asyncio.sleep(0.8)
            r_sta = await client.request("GET", f"/event/{ev_id}/statistics")
            await asyncio.sleep(0.8)
            r_h2h = await client.request("GET", f"/event/{ev_id}/h2h")

            inc = r_inc.json() if (r_inc and r_inc.status_code == 200) else {}
            grp = r_grp.json() if (r_grp and r_grp.status_code == 200) else {}
            lin = r_lin.json() if (r_lin and r_lin.status_code == 200) else {}
            sta = r_sta.json() if (r_sta and r_sta.status_code == 200) else {}

            n_inc = len((inc or {}).get("incidents", []))
            n_grp = len((grp or {}).get("graphPoints", []))
            n_home_p = len(((lin or {}).get("home") or {}).get("players", []))
            periods_sta = [p.get("period") for p in (sta or {}).get("statistics", [])]
            has_q_scores = all(
                k in (ev_obj.get("homeScore") or {})
                for k in ("period1", "period2", "period3", "period4")
            )
            print(
                f"  Muestra FIBA ID={ev_id} ({lg_name}):\n"
                f"    - quarter_scores (Q1..Q4): {'SÍ' if has_q_scores else 'NO'}\n"
                f"    - /incidents (PBP): HTTP {r_inc.status_code if r_inc else 0} ({n_inc} eventos)\n"
                f"    - /graph (Curva): HTTP {r_grp.status_code if r_grp else 0} ({n_grp} puntos)\n"
                f"    - /lineups (player_stats): HTTP {r_lin.status_code if r_lin else 0} ({n_home_p} jugadores local)\n"
                f"    - /statistics (team_stats): HTTP {r_sta.status_code if r_sta else 0} (periodos: {periods_sta})\n"
                f"    - /h2h: HTTP {r_h2h.status_code if r_h2h else 0}"
            )


def analyze_theoretical_effectiveness():
    print("\n" + "=" * 80)
    print("2. EFECTIVIDAD TEÓRICA EN CORRIDAS DEL MODELO m27_v4 (TEST OUT-OF-TIME)")
    print("=" * 80)
    rep_path = ROOT / "models" / "m27_v4" / "model_outputs" / "v4_training_report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    check_db_date_boundaries()
    asyncio.run(probe_sofascore_2019_2022())
    analyze_theoretical_effectiveness()
