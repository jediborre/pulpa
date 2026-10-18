# =====================================================================
# SANEAMIENTO DEL BACKLOG DE bet_monitor_schedule_v3 (BACKFILL FT)
# ---------------------------------------------------------------------
# PROBLEMA QUE ABORDA:
#   Tras corregir el watcher (404 permanente + guarda de obsolescencia) se
#   auditó la tabla y quedaron 2 082 partidos en status 'pending'/'in_progress'
#   con horario vencido hace más de 3.5 h (WATCH_OBSOLETE_SECS), 1 161 de ellos
#   con más de 48 h. El planificador del daemon ya no los toca, pero son ruido y
#   muchos nunca tuvieron su descarga FT (final_fetched = 0), lo que deja huecos
#   en el histórico de matches.db.
#
# QUÉ HACE (por partido colgado, a ritmo configurable):
#   1. Sondeo ligero (/event/{id}):
#        - HTTP 404  → el evento fue eliminado/re-creado en SofaScore
#                      → discarded / 'depuracion_backlog_404'
#        - no finalizado → nunca se jugó (pospuesto/cancelado)
#                      → discarded / 'depuracion_backlog_no_jugado'
#   2. Si está finalizado, reutiliza EXACTAMENTE la misma ruta del daemon
#      (monitor_v3.main._final_fetch_and_save): guarda cuartos en
#      quarter_scores_v2, persiste el partido en matches.db histórica, liquida
#      apuestas y respeta la deduplicación por final_fetched.
#        - datos válidos (gp >= FINAL_FETCH_MIN_GP) → done (datos recuperados)
#        - sin gráfica → discarded / 'depuracion_backlog_no_graph' con
#          final_fetched=0 para permitir un reintento futuro
#   3. Errores de red/403 se contabilizan; si hay N 403 consecutivos el script
#      ABORTA para no quemar el pool de JWT (interés: no bloquear el token).
#
# ADVERTENCIA DE RITMO:
#   El ritmo por defecto es 2 partidos/segundo (0.5 s), pedido explícitamente.
#   AGENTS.md recomienda 1.0–1.5 s entre partidos para no provocar el WAF de
#   Fastly; cada partido puede implicar 2 peticiones, así que el ritmo efectivo
#   de red es ~4 req/s. Si aparece 'reason: challenge' (403), bajar --rate a 0.5-1.0.
#
# USO:
#   # 1) Simulación sobre 20 partidos (no escribe nada):
#   .venv\Scripts\python.exe tmp\probe_404_audit\cleanup_stale_schedule.py --dry-run --limit 20
#   # 2) Backfill real completo a 2 partidos/s:
#   .venv\Scripts\python.exe tmp\probe_404_audit\cleanup_stale_schedule.py --rate 2.0
#
# SALIDA: informe CSV en tmp\probe_404_audit\cleanup_report_<fecha>.csv
# =====================================================================

import argparse
import asyncio
import csv
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.config.constants import WATCH_OBSOLETE_SECS  # noqa: E402
from monitor_v3.database.connection import get_db_connection, get_real_db_path  # noqa: E402
from monitor_v3.database.repository import update_schedule_status  # noqa: E402
from monitor_v3.scrapers.live_scraper import (  # noqa: E402
    EventNotFoundError,
    fetch_event_snapshot,
)
from monitor_v3.scrapers.match_detail_scraper import fetch_match_by_id  # noqa: E402
from monitor_v3.utils.logger import log_error, log_info, log_warning  # noqa: E402

# Import perezoso de la ruta oficial de descarga FT del daemon.
_final_fetch_and_save = None


def _get_final_fetch():
    global _final_fetch_and_save
    if _final_fetch_and_save is None:
        from monitor_v3.main import _final_fetch_and_save as fn

        _final_fetch_and_save = fn
    return _final_fetch_and_save


def _fetch_candidates() -> list[dict]:
    """Partidos en pending/in_progress con horario vencido más allá del umbral de obsolescencia."""
    cutoff = int(time.time()) - WATCH_OBSOLETE_SECS
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT match_id, home_team, away_team, league, scheduled_utc_ts,
                   status, skip_reason, final_fetched
            FROM bet_monitor_schedule_v3
            WHERE status IN ('pending', 'in_progress') AND scheduled_utc_ts < ?
            ORDER BY scheduled_utc_ts
            """,
            (cutoff,),
        ).fetchall()
    return [dict(r) for r in rows]


def _reset_for_retry(match_id: str, skip_reason: str) -> None:
    """Descarta el partido dejando final_fetched=0 para permitir un backfill futuro."""
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        with conn:
            conn.execute(
                """
                UPDATE bet_monitor_schedule_v3
                SET status = 'discarded', skip_reason = ?, final_fetched = 0,
                    final_fetch_at = '', updated_at = ?
                WHERE match_id = ?
                """,
                (skip_reason, now, match_id),
            )


def _row_state(match_id: str) -> tuple[str, str, int]:
    with get_db_connection() as conn:
        r = conn.execute(
            "SELECT status, skip_reason, final_fetched FROM bet_monitor_schedule_v3 WHERE match_id = ?",
            (match_id,),
        ).fetchone()
    if not r:
        return ("?", "?", 0)
    return (r["status"], r["skip_reason"] or "", int(r["final_fetched"] or 0))


async def _process_one(row: dict, dry_run: bool) -> tuple[str, str]:
    """Devuelve (categoria, detalle). Categorías: backfilled/404/no_jugado/no_graph/error_403/error."""
    mid = str(row["match_id"])
    home, away = row["home_team"], row["away_team"]

    # 1. Sondeo ligero para clasificar el evento.
    try:
        snap = await fetch_event_snapshot(mid)
    except EventNotFoundError:
        if not dry_run:
            update_schedule_status(mid, status="discarded", skip_reason="depuracion_backlog_404")
        return ("404", "evento eliminado/re-creado en SofaScore")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        cat = "error_403" if "403" in msg else "error"
        return (cat, f"sondeo falló: {msg}")

    status_type = (snap.get("status_type") or "").lower()
    if status_type != "finished":
        if not dry_run:
            update_schedule_status(mid, status="discarded", skip_reason="depuracion_backlog_no_jugado")
        return ("no_jugado", f"status_type={status_type or '?'}")

    # 2. Partido finalizado: ruta oficial de descarga FT del daemon.
    if dry_run:
        return ("backfilled", "simulado (dry-run)")

    try:
        await _get_final_fetch()(mid, home, away)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        cat = "error_403" if "403" in msg else "error"
        return (cat, f"FT falló: {msg}")

    status, skip_reason, final_fetched = _row_state(mid)
    if skip_reason in ("no_graph", "final_fetch_failed"):
        _reset_for_retry(mid, f"depuracion_backlog_{skip_reason}")
        return ("no_graph", f"sin datos válidos ({skip_reason})")
    return ("backfilled", f"status={status} final_fetched={final_fetched}")


async def _main_async(args) -> None:
    rows = _fetch_candidates()
    total_avail = len(rows)
    if args.limit:
        rows = rows[: args.limit]

    spacing = 1.0 / args.rate if args.rate > 0 else 0.0
    print(f"Backlog detectado: {total_avail} partidos | procesando: {len(rows)}")
    print(f"Ritmo: {args.rate:g} partidos/s ({spacing:.2f}s entre partidos) | "
          f"dry-run: {args.dry_run} | abortar tras {args.max_consecutive_403} 403 seguidos")
    print(f"BD: {get_real_db_path()}\n")

    report_path = Path(__file__).resolve().parent / f"cleanup_report_{datetime.now():%Y%m%d_%H%M%S}.csv"
    counters: dict[str, int] = {}
    consecutive_403 = 0

    with open(report_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["match_id", "home", "away", "scheduled_utc_ts", "categoria", "detalle"])

        for idx, row in enumerate(rows, start=1):
            t0 = time.monotonic()
            try:
                cat, detail = await _process_one(row, args.dry_run)
            except Exception as e:  # noqa: BLE001
                cat, detail = "error", f"excepción no controlada: {e}"

            counters[cat] = counters.get(cat, 0) + 1
            writer.writerow([row["match_id"], row["home_team"], row["away_team"],
                             row["scheduled_utc_ts"], cat, detail])
            fh.flush()

            consecutive_403 = consecutive_403 + 1 if cat == "error_403" else 0
            if consecutive_403 >= args.max_consecutive_403:
                log_error("DESCARGA", f"[BACKFILL] {consecutive_403} HTTP 403 consecutivos → abordo para no bloquear el token")
                print(f"\n⛔ Abortado por {consecutive_403} HTTP 403 consecutivos (posible challenge del WAF).")
                break

            if idx % 50 == 0 or idx == len(rows):
                resumen = " ".join(f"{k}={v}" for k, v in sorted(counters.items()))
                print(f"[{idx}/{len(rows)}] {resumen}")

            elapsed = time.monotonic() - t0
            if spacing > elapsed:
                await asyncio.sleep(spacing - elapsed)

    print("\n=== RESUMEN ===")
    for k, v in sorted(counters.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<12} {v}")
    print(f"Informe CSV: {report_path}")
    log_info("DESCARGA", f"[BACKFILL] Saneamiento finalizado: " +
             " ".join(f"{k}={v}" for k, v in sorted(counters.items())))


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill FT y depuración del backlog de _schedule_v3")
    ap.add_argument("--rate", type=float, default=2.0, help="partidos por segundo (default 2.0)")
    ap.add_argument("--limit", type=int, default=0, help="máximo de partidos a procesar (0 = todos)")
    ap.add_argument("--dry-run", action="store_true", help="no escribe en la BD ni descarga FT")
    ap.add_argument("--max-consecutive-403", type=int, default=10,
                    help="abortar tras N HTTP 403 consecutivos (default 10)")
    args = ap.parse_args()

    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario (el informe parcial ya está en CSV).")


if __name__ == "__main__":
    main()
