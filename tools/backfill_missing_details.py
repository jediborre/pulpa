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


def cmd_complete(_args) -> None:
    """Auditoría de COMPLETITUD: partidos con datos parciales (no solo ausentes)."""
    conn = _open_db()
    total = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    print(f"matches totales: {total}\n")
    print("=== Datos PARCIALES (existen pero incompletos) ===")

    gp_partial = conn.execute(
        "SELECT COUNT(*) FROM (SELECT match_id FROM graph_points GROUP BY match_id HAVING COUNT(*) < 30)"
    ).fetchone()[0]
    pbp_no_q12 = conn.execute(
        "SELECT COUNT(*) FROM matches m WHERE NOT EXISTS ("
        "SELECT 1 FROM play_by_play p WHERE p.match_id=m.match_id AND p.quarter IN ('Q1','Q2'))"
    ).fetchone()[0]
    lu_partial = conn.execute(
        "SELECT COUNT(*) FROM (SELECT match_id FROM lineups GROUP BY match_id HAVING COUNT(*) < 10)"
    ).fetchone()[0]
    ps_partial = conn.execute(
        "SELECT COUNT(*) FROM (SELECT match_id FROM player_stats GROUP BY match_id HAVING COUNT(*) < 10)"
    ).fetchone()[0]

    print(f"  graph_points < 30 puntos : {gp_partial}")
    print(f"  play_by_play sin Q1/Q2   : {pbp_no_q12}")
    print(f"  lineups < 10 jugadores   : {lu_partial}")
    print(f"  player_stats < 10 filas  : {ps_partial}")
    print("\n=== Datos AUSENTES ===")
    for t in ("lineups", "player_stats", "team_statistics", "match_odds",
              "play_by_play", "graph_points", "team_strength"):
        missing = conn.execute(
            f"SELECT COUNT(*) FROM matches m WHERE NOT EXISTS "
            f"(SELECT 1 FROM {t} x WHERE x.match_id=m.match_id)"
        ).fetchone()[0]
        print(f"  sin {t:<16}: {missing}")
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


async def _run_team_strength(args) -> None:
    """
    Backfill de team_strength OPTIMIZADO por equipo único.
    Descarga /team/{id} y /team/{id}/performance UNA sola vez por equipo y aplica el
    snapshot a todos sus partidos. (La forma del equipo es 'actual' en el API, así que
    cachear por equipo da el mismo resultado que pedirlo por partido, pero ~20-40x más rápido.)
    """
    from collections import defaultdict

    conn = _open_db()
    team_matches: dict[int, list[str]] = defaultdict(list)
    team_names: dict[int, str] = {}
    for r in conn.execute(
        "SELECT match_id, home_team_id, away_team_id, home_team, away_team FROM matches"
    ):
        if r["home_team_id"]:
            team_matches[r["home_team_id"]].append(r["match_id"])
            team_names[r["home_team_id"]] = r["home_team"]
        if r["away_team_id"]:
            team_matches[r["away_team_id"]].append(r["match_id"])
            team_names[r["away_team_id"]] = r["away_team"]

    covered = {r[0] for r in conn.execute("SELECT DISTINCT match_id FROM team_strength")}
    teams = [tid for tid, mids in team_matches.items() if any(m not in covered for m in mids)]
    if args.limit:
        teams = teams[: args.limit]
    print(f"[team_strength] equipos únicos: {len(team_matches)} | a descargar: {len(teams)}")
    if not teams:
        conn.close()
        return
    if args.dry_run:
        for tid in teams[:20]:
            print(f"   {tid} ({team_names.get(tid, '?')}) -> {len(team_matches[tid])} partidos")
        if len(teams) > 20:
            print(f"   ... y {len(teams) - 20} mas")
        conn.close()
        return

    mc = get_mobile_client()
    sem = asyncio.Semaphore(max(1, args.concurrency))
    stats = {"ok": 0, "fail": 0}
    done = 0
    t0 = time.perf_counter()

    async def worker(tid: int) -> None:
        nonlocal done
        async with sem:
            try:
                rows = await mc.fetch_team_strength(tid, None, team_names.get(tid, ""), "")
            except Exception:
                stats["fail"] += 1
                done += 1
                return
        data = rows[0] if rows else {}
        for mid in team_matches[tid]:
            try:
                conn.execute("DELETE FROM team_strength WHERE match_id=? AND team_id=?", (mid, tid))
                conn.execute(
                    "INSERT INTO team_strength (team_id, team_name, match_id, position, wins, losses, form, perf_points) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (int(tid), team_names.get(tid, ""), mid, data.get("position"),
                     data.get("wins"), data.get("losses"), data.get("form"), data.get("perf_points")),
                )
            except Exception:
                pass
        conn.commit()
        stats["ok"] += 1
        done += 1
        if done % 10 == 0 or done == len(teams):
            el = time.perf_counter() - t0
            rate = done / el if el else 0
            eta = (len(teams) - done) / rate if rate else 0
            width = 28
            filled = int((done / len(teams)) * width) if teams else 0
            bar = "#" * filled + "-" * (width - filled)
            print(
                f"\r\x1b[2K[team_strength] [{bar}] {done}/{len(teams)} "
                f"ok={stats['ok']} fail={stats['fail']} {rate:.1f}/s "
                f"tok=...{mc.token_pool.current_token_suffix()} eta {int(eta//60)}m{int(eta%60):02d}s",
                end="", flush=True,
            )
        mc.token_pool.notify_match_done(silent=True)

    await asyncio.gather(*[worker(t) for t in teams])
    print()
    conn.close()
    print(f"[team_strength] listo: equipos ok={stats['ok']} fail={stats['fail']} de {len(teams)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill de datos de detalle faltantes")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit", help="Contar partidos sin datos por tabla")
    p_audit.set_defaults(func=cmd_audit)

    p_complete = sub.add_parser("complete", help="Auditoria de completitud (datos parciales)")
    p_complete.set_defaults(func=cmd_complete)

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

    p_ts = sub.add_parser("team-strength", help="Backfill de team_strength por equipo único (optimizado)")
    p_ts.add_argument("--limit", type=int, default=None, help="Maximo de equipos")
    p_ts.add_argument("--concurrency", type=int, default=4, help="Equipos en paralelo (default 4)")
    p_ts.add_argument("--dry-run", action="store_true", help="Solo listar, no descargar")
    p_ts.set_defaults(func=lambda a: asyncio.run(_run_team_strength(a)))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
