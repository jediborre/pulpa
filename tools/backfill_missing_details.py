"""
tools/backfill_missing_details.py
=================================
Detecta y re-descarga la informacion de detalle faltante de partidos ya guardados en
matches.db (lineups, player_stats, team_statistics, odds, h2h, events, pbp, graph).

Motivo: muchos partidos se descargaron antes de que existieran esas tablas/features,
asi que tienen filas en `matches` pero no en las tablas de detalle.

Uso:
  python tools/backfill_missing_details.py audit
  python tools/backfill_missing_details.py run [--tables t1,t2] [--since YYYY-MM-DD]
      [--until YYYY-MM-DD] [--limit N] [--concurrency 3] [--dry-run]

- audit: cuenta partidos sin datos por tabla.
- run:   re-descarga los partidos que les falta AL MENOS una de las tablas indicadas
         (por defecto lineups, player_stats, team_statistics, match_odds).
         Usa la API movil (huella OkHttp + UA firmado), rota tokens cada 10 partidos
         y auto-regenera ante 403. Es resumible (omite los que ya estan completos).
"""
import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "match")):
    if p not in sys.path:
        sys.path.insert(0, p)

import db as db_mod
from monitor_v3.core.mobile_client import get_mobile_client

DB_PATH = str(ROOT / "matches.db")

DETAIL_TABLES = [
    "quarter_scores", "play_by_play", "graph_points", "match_events",
    "match_h2h", "player_stats", "lineups", "team_statistics",
    "match_odds", "team_strength",
]
# Tablas que la API móvil SÍ puede traer (se rellenan con este backfill).
# team_strength se obtiene con /team/{id} y /team/{id}/performance (2 por equipo).
DEFAULT_REQUIRED = [
    "play_by_play", "graph_points", "match_events", "match_h2h",
    "team_statistics", "lineups", "player_stats", "match_odds", "team_strength",
]


def _open_db():
    conn = db_mod.get_conn(DB_PATH)
    db_mod.init_db(conn)
    return conn


def cmd_audit(_args) -> None:
    conn = _open_db()
    all_ids = {r[0] for r in conn.execute("SELECT match_id FROM matches")}
    total = len(all_ids)
    print(f"matches totales: {total}\n")
    print(f"{'tabla':<18} {'con datos':>10} {'sin datos':>10} {'% falta':>8}")
    print("-" * 50)
    for t in DETAIL_TABLES:
        try:
            present = {r[0] for r in conn.execute(f"SELECT DISTINCT match_id FROM {t}")}
        except Exception as e:
            print(f"{t:<18} ERROR: {e}")
            continue
        missing = len(all_ids - present)
        pct = f"{int(round(missing * 100 / total))}%" if total else "-"
        print(f"{t:<18} {len(present):>10} {missing:>10} {pct:>8}")
    conn.close()


def _select_missing(conn, tables: list[str], since: str | None, until: str | None,
                    limit: int | None, force: bool = False) -> list[str]:
    conds = " OR ".join(
        f"NOT EXISTS (SELECT 1 FROM {t} x WHERE x.match_id = m.match_id)" for t in tables
    )
    q = f"SELECT m.match_id FROM matches m WHERE ({conds})"
    params: list = []
    if not force:
        # Omitir partidos ya revisados por el backfill (p. ej. el API devolvio 404
        # para lineups/statistics: esos datos no existen y no deben reintentarse).
        q += " AND m.details_checked_at IS NULL"
    if since:
        q += " AND m.date >= ?"
        params.append(since)
    if until:
        q += " AND m.date <= ?"
        params.append(until)
    q += " ORDER BY m.date DESC, m.time DESC, m.match_id DESC"
    if limit:
        q += f" LIMIT {int(limit)}"
    return [str(r[0]) for r in conn.execute(q, params)]


async def _run(args) -> None:
    tables = [t.strip() for t in args.tables.split(",") if t.strip()]
    bad = [t for t in tables if t not in DETAIL_TABLES]
    if bad:
        print(f"[backfill] tablas invalidas: {bad}. Validas: {DETAIL_TABLES}")
        return

    conn = _open_db()
    ids = _select_missing(conn, tables, args.since, args.until, args.limit, force=args.force)
    total = len(ids)
    print(f"[backfill] partidos a re-descargar (faltan {tables}): {total}")
    if not total:
        conn.close()
        return
    if args.dry_run:
        for mid in ids[:20]:
            print("  ", mid)
        if total > 20:
            print(f"   ... y {total - 20} mas")
        conn.close()
        return

    mc = get_mobile_client()
    sem = asyncio.Semaphore(max(1, args.concurrency))
    stats = {"ok": 0, "fail": 0}
    done = 0
    t0 = time.perf_counter()

    async def worker(mid: str) -> None:
        nonlocal done
        async with sem:
            try:
                data = await mc.fetch_full_match(mid, fetch_team_strength=("team_strength" in tables))
            except Exception:
                stats["fail"] += 1
                done += 1
                return
        try:
            db_mod.save_match(conn, mid, data)
            stats["ok"] += 1
        except Exception:
            stats["fail"] += 1
        # Marcar como revisado: aunque falten datos que el API no tiene (404),
        # no se debe reintentar en futuras corridas.
        try:
            conn.execute(
                "UPDATE matches SET details_checked_at = ? WHERE match_id = ?",
                (datetime.now().isoformat(), mid),
            )
            conn.commit()
        except Exception:
            pass
        done += 1
        if done % 10 == 0 or done == total:
            el = time.perf_counter() - t0
            rate = done / el if el else 0
            eta = (total - done) / rate if rate else 0
            width = 28
            filled = int((done / total) * width) if total else 0
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\r\x1b[2K[backfill] [{bar}] {done}/{total} "
                f"ok={stats['ok']} fail={stats['fail']} {rate:.1f}/s "
                f"tok=...{mc.token_pool.current_token_suffix()} "
                f"eta {int(eta//60)}m{int(eta%60):02d}s",
                end="", flush=True,
            )
        mc.token_pool.notify_match_done(silent=True)

    await asyncio.gather(*[worker(m) for m in ids])
    print()
    conn.close()
    print(f"[backfill] listo: ok={stats['ok']} fail={stats['fail']} de {total}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill de datos de detalle faltantes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit", help="Contar partidos sin datos por tabla")
    p_audit.set_defaults(func=cmd_audit)

    p_run = sub.add_parser("run", help="Re-descargar partidos con datos faltantes")
    p_run.add_argument("--tables", default=",".join(DEFAULT_REQUIRED),
                       help=f"Tablas requeridas separadas por coma (default: {','.join(DEFAULT_REQUIRED)})")
    p_run.add_argument("--since", default=None, help="Fecha minima YYYY-MM-DD")
    p_run.add_argument("--until", default=None, help="Fecha maxima YYYY-MM-DD")
    p_run.add_argument("--limit", type=int, default=None, help="Maximo de partidos")
    p_run.add_argument("--concurrency", type=int, default=3, help="Partidos en paralelo (default 3)")
    p_run.add_argument("--force", action="store_true", help="Re-procesar partidos ya marcados como revisados")
    p_run.add_argument("--dry-run", action="store_true", help="Solo listar, no descargar")
    p_run.set_defaults(func=lambda a: asyncio.run(_run(a)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
