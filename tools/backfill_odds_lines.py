"""
tools/backfill_odds_lines.py
Backfill ESPECIFICO de cuotas pre-partido para m27_v5_3 (sin re-descargar el detalle completo de los partidos).

Que arregla (ver docs/findings_m27_v5_3.md):
  * match_odds no guardaba la LINEA de handicap ni la linea del total (solo el precio ~1.83) -> el prior de mercado
    mas informativo (spread/total) era inutilizable. Este script descarga event/{id}/odds/1/all y guarda
    line / home_initial / away_initial solo con mercados PRE-PARTIDO (isLive=False).
  * Partidos monitoreados en vivo cuyo match_odds pudo quedar contaminado con cuotas in-play.
  * Partidos terminados sin ninguna cuota ML (solo en ligas con cobertura de cuotas real).

Modos de `run`:
  lines      partidos con filas Point spread / Over/Under sin `line` (cuotas ya existentes, falta la linea)
  monitored  partidos presentes en bet_monitor_schedule(_v2/_v3) / bet_monitor_log(_v2/_v3) (re-fetch pre-partido limpio)
  missing    partidos terminados desde --since sin cuota ML, restringido a ligas con cobertura >= --min-league-cov

Comandos:
  audit      conteos por modo / año (no descarga)
  probe      muestrea N partidos pendientes y mide cuantos SI tienen cuotas en linea (decide si vale la pena el modo)
  run        ejecuta (reanudable; estado en tabla odds_backfill_state; respeta 1 req/s por defecto)

Ejemplos (siempre con el venv):
  .venv\\Scripts\\python.exe tools\\backfill_odds_lines.py audit
  .venv\\Scripts\\python.exe tools\\backfill_odds_lines.py probe --mode missing --since 2024-01-01 --n 60
  .venv\\Scripts\\python.exe tools\\backfill_odds_lines.py run --mode monitored
  .venv\\Scripts\\python.exe tools\\backfill_odds_lines.py run --mode lines --limit 3000
  .venv\\Scripts\\python.exe tools\\backfill_odds_lines.py run --mode missing --since 2024-01-01 --limit 5000
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / "match")):
    if p not in sys.path:
        sys.path.insert(0, p)

import db as db_mod  # noqa: E402  (match/db.py)
from match.scraper import _parse_odds  # noqa: E402
from monitor_v3.core.mobile_client import get_mobile_client  # noqa: E402

DB_PATH = str(ROOT / "matches.db")
MONITOR_TABLES = ("bet_monitor_schedule", "bet_monitor_schedule_v2", "bet_monitor_schedule_v3",
                  "bet_monitor_log", "bet_monitor_log_v2", "bet_monitor_log_v3")


def _open():
    conn = db_mod.get_conn(DB_PATH)
    db_mod.init_db(conn)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS odds_backfill_state ("
        "match_id TEXT PRIMARY KEY, mode TEXT, status TEXT, n_markets INTEGER, checked_at TEXT)"
    )
    conn.commit()
    return conn


def _select(conn: sqlite3.Connection, mode: str, since: str | None, until: str | None,
            min_cov: float, limit: int | None) -> list[str]:
    done = "SELECT match_id FROM odds_backfill_state WHERE status IN ('ok','404','empty')"
    date_sql = ""
    params: list = []
    if since:
        date_sql += " AND m.date >= ?"
        params.append(since)
    if until:
        date_sql += " AND m.date <= ?"
        params.append(until)

    if mode == "lines":
        sql = (
            "SELECT DISTINCT o.match_id FROM match_odds o JOIN matches m ON m.match_id = o.match_id "
            "WHERE o.odds_type IN ('Point spread','Over/Under') AND o.line IS NULL "
            f"AND o.match_id NOT IN ({done}) {date_sql} ORDER BY m.date DESC"
        )
    elif mode == "monitored":
        ids: set[str] = set()
        for t in MONITOR_TABLES:
            try:
                ids.update(str(r[0]) for r in conn.execute(f"SELECT DISTINCT match_id FROM {t}"))
            except sqlite3.Error:
                continue
        fin = {r[0] for r in conn.execute("SELECT match_id FROM matches WHERE status_type='finished'")}
        dn = {r[0] for r in conn.execute(f"{done}")}
        out = sorted((ids & fin) - dn)
        return out[:limit] if limit else out
    elif mode == "missing":
        sql = (
            "WITH lg AS (SELECT m.league lg, 1.0*SUM(EXISTS(SELECT 1 FROM match_odds o WHERE o.match_id=m.match_id "
            "AND o.odds_type='Home/Away' AND o.market_period='Match'))/COUNT(*) cov, COUNT(*) n "
            "FROM matches m WHERE m.status_type='finished' AND m.date >= '2023-01-01' GROUP BY m.league) "
            "SELECT m.match_id FROM matches m JOIN lg ON lg.lg = m.league "
            "WHERE m.status_type='finished' AND lg.cov >= ? AND lg.n >= 30 "
            "AND NOT EXISTS (SELECT 1 FROM match_odds o WHERE o.match_id=m.match_id AND o.odds_type='Home/Away' AND o.market_period='Match') "
            f"AND m.match_id NOT IN ({done}) {date_sql} ORDER BY m.date DESC"
        )
        params = [min_cov] + params
    else:
        raise SystemExit(f"modo desconocido: {mode}")
    if limit:
        sql += f" LIMIT {int(limit)}"
    return [r[0] for r in conn.execute(sql, params)]


def cmd_audit(args) -> None:
    conn = _open()
    q = lambda s, *p: conn.execute(s, p).fetchone()[0]
    print(f"partidos terminados: {q('SELECT COUNT(*) FROM matches WHERE status_type=?', 'finished'):,}")
    print("\n-- cobertura de cuotas (terminados) por año --")
    print(f"{'año':<6}{'partidos':>10}{'con ML':>10}{'spread c/linea':>16}{'spread s/linea':>16}{'total c/linea':>15}{'total s/linea':>15}")
    for yr in range(2023, 2027):
        a, b = f"{yr}-01-01", f"{yr}-12-31"
        n = q("SELECT COUNT(*) FROM matches WHERE status_type='finished' AND date BETWEEN ? AND ?", a, b)
        ml = q("SELECT COUNT(DISTINCT o.match_id) FROM match_odds o JOIN matches m USING(match_id) WHERE o.odds_type='Home/Away' AND o.market_period='Match' AND m.date BETWEEN ? AND ?", a, b)
        def sp(t, has):
            return q(f"SELECT COUNT(DISTINCT o.match_id) FROM match_odds o JOIN matches m USING(match_id) WHERE o.odds_type=? AND o.line IS {'NOT ' if has else ''}NULL AND m.date BETWEEN ? AND ?", t, a, b)
        print(f"{yr:<6}{n:>10,}{ml:>10,}{sp('Point spread', True):>16,}{sp('Point spread', False):>16,}{sp('Over/Under', True):>15,}{sp('Over/Under', False):>15,}")
    for mode in ("lines", "monitored", "missing"):
        ids = _select(conn, mode, args.since, args.until, args.min_league_cov, None)
        print(f"\nmodo {mode:<10}: {len(ids):>7,} partidos pendientes  (~{len(ids)*args.spacing/3600:.1f} h a {args.spacing}s/req)")
    st = conn.execute("SELECT status, COUNT(*) FROM odds_backfill_state GROUP BY status").fetchall()
    print("\nestado previo:", dict(st))


async def _fetch_one(client, match_id: str) -> tuple[str, list[dict]]:
    r = await client.request("GET", f"event/{match_id}/odds/1/all")
    if r.status_code == 404:
        return "404", []
    if r.status_code != 200:
        return f"http_{r.status_code}", []
    rows = _parse_odds(match_id, r.json())
    return ("ok" if rows else "empty"), rows


async def _run(args, probe: bool) -> None:
    conn = _open()
    ids = _select(conn, args.mode, args.since, args.until, args.min_league_cov, None if probe else args.limit)
    if probe:
        random.seed(7)
        random.shuffle(ids)
        ids = ids[: args.n]
    print(f"{'PROBE' if probe else 'RUN'} modo={args.mode} partidos={len(ids):,}")
    if args.dry_run or not ids:
        return
    client = get_mobile_client()
    stats = {"ok": 0, "404": 0, "empty": 0, "err": 0, "ml": 0, "spread_line": 0, "total_line": 0}
    t0 = time.time()
    for i, mid in enumerate(ids, 1):
        try:
            status, rows = await _fetch_one(client, mid)
        except Exception as exc:  # red: no marcar como revisado, reintentable
            stats["err"] += 1
            print(f"  [{mid}] error: {exc}")
            await asyncio.sleep(args.spacing * 3)
            continue
        if status.startswith("http_"):
            stats["err"] += 1
            print(f"  [{mid}] {status}")
            await asyncio.sleep(args.spacing * 3)
            continue
        stats[status] = stats.get(status, 0) + 1
        if rows:
            stats["ml"] += any(r["odds_type"] == "Home/Away" and r["market_period"] == "Match" for r in rows)
            stats["spread_line"] += any(r["odds_type"] == "Point spread" and r.get("line") is not None for r in rows)
            stats["total_line"] += any(r["odds_type"] == "Over/Under" and r.get("line") is not None for r in rows)
        if not probe:
            if rows:
                db_mod.save_match_odds(conn, mid, rows)
            conn.execute(
                "INSERT OR REPLACE INTO odds_backfill_state VALUES (?,?,?,?,datetime('now'))",
                (mid, args.mode, status, len(rows)),
            )
            if i % 25 == 0:
                conn.commit()
        if i % 50 == 0 or i == len(ids):
            el = time.time() - t0
            print(f"  {i:>6,}/{len(ids):,} | {stats} | {el/60:.1f} min | ETA {(len(ids)-i)*el/i/60:.1f} min", flush=True)
        await asyncio.sleep(args.spacing)
    conn.commit()
    if probe and ids:
        n = len(ids)
        print(f"\nDisponibles en linea: ML {stats['ml']}/{n} ({stats['ml']/n:.0%}) | spread con linea {stats['spread_line']}/{n} | "
              f"total con linea {stats['total_line']}/{n} | 404 {stats['404']} | vacios {stats['empty']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill de cuotas pre-partido (ML, spread/total con linea) para m27_v5_3")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "probe", "run"):
        p = sub.add_parser(name)
        p.add_argument("--mode", choices=["lines", "monitored", "missing"], default="lines")
        p.add_argument("--since", default=None, help="YYYY-MM-DD")
        p.add_argument("--until", default=None, help="YYYY-MM-DD")
        p.add_argument("--min-league-cov", type=float, default=0.20, help="cobertura minima de cuotas de la liga (modo missing)")
        p.add_argument("--spacing", type=float, default=1.1, help="segundos entre peticiones (>=1.0 recomendado)")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--n", type=int, default=60, help="tamaño de muestra del probe")
        p.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "audit":
        cmd_audit(args)
    else:
        asyncio.run(_run(args, probe=(args.cmd == "probe")))


if __name__ == "__main__":
    main()
