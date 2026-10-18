# =====================================================================
# AUDITORÍA: Error recurrente HTTP 404 en la sonda pre-partido (PROBE)
# ---------------------------------------------------------------------
# PROBLEMA QUE ABORDA:
#   El log logs/monitor_v3_YYYY-MM-DD.log muestra cientos de líneas
#   "[PROBE] Error sondeo <hh:mm> <match_id> ...: HTTP 404" cada ~45s para
#   el partido 17353606 (Ieliecmaisā.lv vs Selga, Jūrmalas Čempionāts),
#   indefinidamente incluso horas después de su horario programado.
#
# OBJETIVO DE ESTE SCRIPT (solo lectura, NO escribe en matches.db):
#   1. Confirmar si la API móvil realmente devuelve 404 para ese evento
#      (¿evento eliminado/pospuesto en SofaScore?) y guardar el cuerpo.
#   2. Buscar el evento por fecha/liga para ver si reapareció con otro id.
#   3. Listar cuántos partidos de bet_monitor_schedule_v3 quedaron colgados
#      (status='pending', horario ya pasado) para dimensionar el problema.
#
# USO:  .venv\Scripts\python.exe tmp\probe_404_audit\inspect_event_404.py
# =====================================================================

import asyncio
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # permitir "import monitor_v3" al ejecutar desde tmp/
DB = ROOT / "matches.db"
TARGET_ID = "17353606"
TARGET_SIBLING_ID = "17356027"  # mismo cruce (Ieliecmaisā.lv vs Selga) con otro id
STALE_THRESHOLD_SECS = 12600  # 3.5 h, mismo umbral de "obsoleto" del watcher
MAX_DB_ROWS_PRINTED = 25


async def probe_api() -> None:
    from monitor_v3.core.mobile_client import get_mobile_client

    mc = get_mobile_client()
    print(f"\n=== 1. GET /event/{TARGET_ID} ===")
    res = await mc.request("GET", f"event/{TARGET_ID}")
    print("status_code:", res.status_code)
    body = res.text or ""
    print("body[:600]:", body[:600])

    print("\n=== 2. Búsqueda del evento por fecha (schedule) ===")
    try:
        res2 = await mc.request("GET", "sport/basketball/scheduled-events/2026-10-18")
        print("status_code:", res2.status_code)
        if res2.status_code == 200:
            data = res2.json()
            events = data.get("events", [])
            hits = [
                e for e in events
                if TARGET_ID in (str(e.get("id")), str(e.get("customId", "")))
                or "Selga" in json.dumps(e, ensure_ascii=False)
            ]
            print("eventos totales hoy:", len(events))
            print(f"coincidencias con id {TARGET_ID} o 'Selga':", len(hits))
            for e in hits[:10]:
                print(json.dumps(
                    {
                        "id": e.get("id"),
                        "home": (e.get("homeTeam") or {}).get("name"),
                        "away": (e.get("awayTeam") or {}).get("name"),
                        "tournament": (e.get("tournament") or {}).get("name"),
                        "startTimestamp": e.get("startTimestamp"),
                        "status": (e.get("status") or {}).get("type"),
                    },
                    ensure_ascii=False,
                ))
    except Exception as e:
        print("fallo en schedule:", type(e).__name__, e)


def audit_db() -> None:
    print("\n=== 3. Partidos colgados en bet_monitor_schedule_v3 ===")
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    now = int(time.time())
    rows = conn.execute(
        """
        SELECT match_id, home_team, away_team, scheduled_utc_ts, status,
               skip_reason, final_fetched, updated_at
        FROM bet_monitor_schedule_v3
        WHERE status NOT IN ('done') AND scheduled_utc_ts < ?
        ORDER BY scheduled_utc_ts
        """,
        (now - STALE_THRESHOLD_SECS,),
    ).fetchall()
    print(f"filas no-done con horario vencido >3.5h: {len(rows)}")

    buckets = {"3.5-12h": 0, "12-48h": 0, ">48h": 0}
    for r in rows:
        drift_h = (now - int(r["scheduled_utc_ts"])) / 3600.0
        if drift_h > 48:
            buckets[">48h"] += 1
        elif drift_h > 12:
            buckets["12-48h"] += 1
        else:
            buckets["3.5-12h"] += 1
    print("distribución por drift:", buckets)

    print(f"\nprimeras {MAX_DB_ROWS_PRINTED} por antigüedad:")
    for r in rows[:MAX_DB_ROWS_PRINTED]:
        drift_h = (now - int(r["scheduled_utc_ts"])) / 3600.0
        print(
            f"  {r['match_id']} drift={drift_h:7.1f}h status={r['status']:<10} "
            f"upd={r['updated_at']} | {r['home_team']} vs {r['away_team']}"
        )

    print("\nfila del objetivo y de su hermano:")
    for mid in (TARGET_ID, TARGET_SIBLING_ID):
        r = conn.execute(
            "SELECT match_id, home_team, away_team, scheduled_utc_ts, status, updated_at "
            "FROM bet_monitor_schedule_v3 WHERE match_id=?",
            (mid,),
        ).fetchone()
        if r:
            drift_h = (now - int(r["scheduled_utc_ts"])) / 3600.0
            print(
                f"  {r['match_id']} drift={drift_h:.1f}h status={r['status']} "
                f"upd={r['updated_at']} | {r['home_team']} vs {r['away_team']}"
            )
        else:
            print(f"  {mid}: no está en la tabla")
    conn.close()


async def main() -> None:
    try:
        await probe_api()
    except Exception as e:
        print("ERROR consultando la API:", type(e).__name__, e)
    audit_db()


if __name__ == "__main__":
    asyncio.run(main())
