"""
Script de Auditoría y Prueba de Concepto: Descarga Histórica con Mobile JWT

Propósito:
- Verificar si la API móvil de SofaScore con autenticación JWT permite descargar
  partidos históricos del pasado (fechas sin datos en matches.db entre junio y octubre 2026).
- Probar la detección de partidos programados y finalizados para fechas pasadas (ej. 2026-06-12 y 2026-10-03).
- Probar la extracción completa de endpoints (event, incidents, graph, h2h, stats, lineups)
  con MobileClient.
- Verificar la inserción limpia en matches.db usando save_match().
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Configurar entorno
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "match"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from monitor_v3.core.mobile_client import get_mobile_client
import db as db_mod

TEST_DATES = ["2026-10-03", "2026-06-12"]

async def run_test():
    mc = get_mobile_client()
    conn = db_mod.get_conn(str(ROOT / "matches.db"))
    db_mod.init_db(conn)

    print("=" * 70)
    print("AUDITORÍA DE DESCARGA HISTÓRICA CON JWT MÓVIL")
    print("=" * 70)

    for test_date in TEST_DATES:
        print(f"\n--- Probando fecha histórica: {test_date} ---")
        t0 = time.perf_counter()

        # 1. Probar categorías y eventos programados
        events = await mc.get_all_scheduled_events_for_date(test_date)
        dur_sched = time.perf_counter() - t0
        print(f"[{test_date}] Total eventos encontrados: {len(events)} en {dur_sched:.2f}s")

        if not events:
            # Probar también el endpoint directo sport/basketball/scheduled-events/{date}
            print(f"[{test_date}] Probando endpoint alternativo sport/basketball/scheduled-events/{test_date}...")
            res_alt = await mc.request("GET", f"sport/basketball/scheduled-events/{test_date}")
            if res_alt.status_code == 200:
                alt_events = res_alt.json().get("events", [])
                print(f"[{test_date}] Endpoint alternativo retorno: {len(alt_events)} eventos!")
                events = alt_events
            else:
                print(f"[{test_date}] Endpoint alternativo HTTP: {res_alt.status_code}")

        # Filtrar finalizados
        finished_events = [e for e in events if (e.get("status") or {}).get("type") == "finished"]
        print(f"[{test_date}] Eventos 'finished': {len(finished_events)} de {len(events)}")

        if not finished_events:
            print(f"[{test_date}] No hay eventos finalizados para probar descarga.")
            continue

        # Seleccionar muestra de hasta 3 partidos
        sample = finished_events[:3]
        for ev in sample:
            mid = str(ev.get("id"))
            h_name = ev.get("homeTeam", {}).get("name", "Home")
            a_name = ev.get("awayTeam", {}).get("name", "Away")
            t_name = ev.get("tournament", {}).get("name", "Liga")

            print(f"\n  -> Descargando partido muestra: ID {mid} | {h_name} vs {a_name} ({t_name})...")
            t_match = time.perf_counter()
            try:
                full_data = await mc.fetch_full_match(mid)
                dur_match = time.perf_counter() - t_match

                pbp_count = sum(len(v) for v in full_data.get("play_by_play", {}).values())
                gp_count = len(full_data.get("graph_points", []))
                h2h_count = len(full_data.get("h2h", []))
                stats_count = len(full_data.get("team_statistics", []))
                lineup_count = len(full_data.get("lineups", []))
                q_scores = full_data.get("score", {}).get("quarters", {})

                print(f"     [OK] Descarga completa en {dur_match:.2f}s:")
                print(f"          - Quarters: {q_scores}")
                print(f"          - Play-by-play scoring: {pbp_count} jugadas")
                print(f"          - Graph points: {gp_count} puntos")
                print(f"          - H2H: {h2h_count} registros")
                print(f"          - Team Stats: {stats_count} métricas")
                print(f"          - Lineups: {lineup_count} registros")

                # Guardar en matches.db
                print(f"     Guardando en matches.db...")
                db_mod.save_match(conn, mid, full_data)
                conn.commit()

                # Verificar lectura de vuelta desde matches.db
                cur = conn.cursor()
                cur.execute("SELECT home_team, away_team, home_score, away_score, date FROM matches WHERE match_id = ?", (mid,))
                db_row = cur.fetchone()
                cur.execute("SELECT count(*) FROM quarter_scores WHERE match_id = ?", (mid,))
                db_qs = cur.fetchone()[0]
                cur.execute("SELECT count(*) FROM graph_points WHERE match_id = ?", (mid,))
                db_gps = cur.fetchone()[0]

                print(f"     [VERIFICADO EN DB] {db_row[0]} {db_row[2]} - {db_row[3]} {db_row[1]} ({db_row[4]})")
                print(f"                        quarter_scores={db_qs}, graph_points={db_gps}")

            except Exception as ex:
                print(f"     [ERROR] Falló descarga de match {mid}: {ex}")

    conn.close()
    await mc.close()
    print("\n" + "=" * 70)
    print("AUDITORÍA COMPLETADA EXITOSAMENTE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_test())
