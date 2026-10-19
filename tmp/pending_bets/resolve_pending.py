# =====================================================================
# RESOLVER APUESTAS PENDIENTES SIN CUARTOS (backfill de Q1..Q4)
# ---------------------------------------------------------------------
# PROBLEMA QUE ABORDA:
#   `bet_monitor_log_v3` tiene apuestas en result='pending' que nunca se
#   liquidan porque `quarter_scores_v2` no tiene q4_home/q4_away. Origen:
#   cuando un partido no traía gráfica de momentum, `_final_fetch_and_save`
#   descartaba el FT ANTES de guardar los cuartos (bug corregido en main.py).
#
# QUÉ HACE (solo para partidos con apuestas pendientes y sin Q4):
#   1. Re-descarga el payload FT del partido (event/incidents/graph/...).
#   2. Si el API trae el desglose por cuartos, lo guarda en quarter_scores_v2.
#   3. Al final ejecuta `reconcile_pending_results()` para liquidar.
#   4. Reporta cuántas apuestas se resolvieron y cuántas quedan sin datos.
#
# RITMO: 2 partidos/s por defecto (--rate). `fetch_match_by_id` ya añade ~1s
#   interno por petición FT, así que el ritmo real es ≈1/s.
#
# USO:
#   .venv\Scripts\python.exe tmp\pending_bets\resolve_pending.py --dry-run --limit 5
#   .venv\Scripts\python.exe tmp\pending_bets\resolve_pending.py --rate 2.0
# =====================================================================

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.database.connection import get_db_connection  # noqa: E402
from monitor_v3.database.repository import (  # noqa: E402
    reconcile_pending_results,
    save_quarter_scores,
)
from monitor_v3.scrapers.match_detail_scraper import fetch_match_by_id  # noqa: E402


def _pending_matches() -> list[dict]:
    """Partidos con apuestas pendientes y sin Q4 guardado."""
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT l.match_id,
                   COALESCE(s.home_team, '?') AS home_team,
                   COALESCE(s.away_team, '?') AS away_team,
                   COALESCE(s.event_date, '') AS event_date
            FROM bet_monitor_log_v3 l
            LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
            LEFT JOIN quarter_scores_v2 q ON q.match_id = l.match_id
            WHERE l.result = 'pending'
              AND (q.q4_home IS NULL OR q.q4_away IS NULL)
            ORDER BY s.event_date, l.match_id
            """
        ).fetchall()
    return [dict(r) for r in rows]


def _pending_count() -> int:
    with get_db_connection() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM bet_monitor_log_v3 WHERE result = 'pending'"
        ).fetchone()[0]


async def main_async(args) -> None:
    matches = _pending_matches()
    total_avail = len(matches)
    if args.limit:
        matches = matches[: args.limit]

    print(f"Partidos con apuestas pendientes sin Q4: {total_avail} | procesando: {len(matches)}")
    print(f"Apuestas pendientes antes: {_pending_count()}")
    print(f"Ritmo: {args.rate:g} partidos/s | dry-run: {args.dry_run}\n")

    spacing = 1.0 / args.rate if args.rate > 0 else 0.0
    con_q4 = sin_q4 = errores = 0

    for index, match in enumerate(matches, start=1):
        mid = str(match["match_id"])
        label = f"{match['home_team']} vs {match['away_team']}"
        t0 = time.monotonic()
        try:
            data = await fetch_match_by_id(mid, is_ft=True)
            quarters = (data.get("score") or {}).get("quarters") or {}
            q4 = quarters.get("Q4") or {}
            if q4.get("home") is not None and q4.get("away") is not None:
                con_q4 += 1
                if not args.dry_run:
                    save_quarter_scores(mid, quarters)
                print(f"  [{index}/{len(matches)}] ✅ Q4 {q4['home']}-{q4['away']} | {mid} {label}")
            else:
                sin_q4 += 1
                print(f"  [{index}/{len(matches)}] ⚠️  sin cuartos en el API | {mid} {label}")
        except Exception as e:  # noqa: BLE001
            errores += 1
            print(f"  [{index}/{len(matches)}] ❌ {type(e).__name__}: {e} | {mid} {label}")

        elapsed = time.monotonic() - t0
        if spacing > elapsed:
            await asyncio.sleep(spacing - elapsed)

    if not args.dry_run:
        reconcile_pending_results()

    print("\n=== RESUMEN ===")
    print(f"  con cuartos recuperables: {con_q4}")
    print(f"  sin cuartos en el API   : {sin_q4}")
    print(f"  errores                 : {errores}")
    print(f"  apuestas pendientes después: {_pending_count()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=2.0, help="partidos por segundo (default 2.0)")
    ap.add_argument("--limit", type=int, default=0, help="máximo de partidos (0 = todos)")
    ap.add_argument("--dry-run", action="store_true", help="descarga pero no guarda nada")
    asyncio.run(main_async(ap.parse_args()))
