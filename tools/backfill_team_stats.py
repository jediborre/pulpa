# =====================================================================
# BACKFILL DE ESTADISTICAS DE EQUIPO POR CUARTO (proceso independiente)
# ---------------------------------------------------------------------
# Descarga GET /event/{id}/statistics (1 request por partido) y guarda las
# stats por cuarto (faltas, tiros libres, perdidas, tiempo en ventaja...) en
# team_stats_v3, con la MISMA seleccion canonica que captura el monitor en
# vivo (models/common/team_stats.py) -> paridad entrenamiento/inferencia.
#
# Alcance por defecto: las ligas que el modelo considera rentables (WR por
# encima del equilibrio a 1.40 con muestra minima), leidas de
# models/m27_v5_3/model_outputs/league_calibration.json. Con --all se
# extiende a todo el dataset elegible.
#
# Uso:
#   .venv\Scripts\python.exe tools\backfill_team_stats.py
#   .venv\Scripts\python.exe tools\backfill_team_stats.py --all --from 2019
#   .venv\Scripts\python.exe tools\backfill_team_stats.py --rate 2 --limit 500
#
# Es reanudable: el estado va a team_stats_backfill_state.
# =====================================================================

import argparse
import asyncio
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.common.team_stats import compact_team_stats  # noqa: E402
from monitor_v3.database.connection import get_db_connection  # noqa: E402

# Ritmo seguro: el WAF de SofaScore (Fastly) tolera ~1.6 req/s medidos; el
# limite documentado es 1 req/s. 2 req/s con 0.5 s de separacion es lo que
# aguanta sin 403 "challenge".
RATE_DEFAULT = 2.0
BREAKEVEN = 0.714  # equilibrio a 1.40
MIN_BETS_POR_LIGA = 30


def ligas_rentables(min_bets: int = MIN_BETS_POR_LIGA) -> list[str]:
    path = ROOT / "models" / "m27_v5_3" / "model_outputs" / "league_calibration.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [c["league"] for c in data
            if (c.get("wr") or 0) >= BREAKEVEN and (c.get("bets") or 0) >= min_bets]


def partidos_objetivo(conn, ligas: list[str], desde: str | None, todos: bool) -> list[tuple[int, str]]:
    if todos:
        q = ("SELECT CAST(match_id AS INTEGER), league FROM matches "
             "WHERE status_type='finished' AND match_id IS NOT NULL")
        params: list = []
        if desde:
            q += " AND date >= ?"
            params.append(desde)
    else:
        if not ligas:
            return []
        q = ("SELECT CAST(match_id AS INTEGER), league FROM matches "
             "WHERE status_type='finished' AND match_id IS NOT NULL AND league IN ("
             + ",".join("?" * len(ligas)) + ")")
        params = list(ligas)
        if desde:
            q += " AND date >= ?"
            params.append(desde)
    return conn.execute(q + " ORDER BY date DESC", params).fetchall()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="todo el dataset elegible")
    ap.add_argument("--from", dest="desde", default="2020-01-01")
    ap.add_argument("--rate", type=float, default=RATE_DEFAULT, help="requests por segundo")
    ap.add_argument("--limit", type=int, default=0, help="maximo de partidos (0 = todos)")
    ap.add_argument("--min-bets", type=int, default=MIN_BETS_POR_LIGA)
    ap.add_argument("--sleep", type=float, default=0.0, help="pausa extra entre requests")
    args = ap.parse_args()

    conn = get_db_connection()
    ligas = [] if args.all else ligas_rentables(args.min_bets)
    objs = partidos_objetivo(conn, ligas, args.desde, args.all)
    ya = {str(r[0]) for r in conn.execute("SELECT match_id FROM team_stats_backfill_state "
                                          "WHERE status IN ('ok','empty')")}
    pendientes = [(m, l) for m, l in objs if str(m) not in ya]
    if args.limit:
        pendientes = pendientes[:args.limit]

    print(f"[backfill] ligas: {'TODAS' if args.all else f'{len(ligas)} rentables'}")
    print(f"[backfill] partidos en el universo: {len(objs):,} | ya procesados: {len(ya):,} "
          f"| pendientes: {len(pendientes):,}")
    print(f"[backfill] ritmo: {args.rate} req/s | desde {args.desde}")
    if not pendientes:
        print("[backfill] nada que hacer")
        return

    from monitor_v3.core.mobile_client import get_mobile_client
    client = get_mobile_client()

    # Largo de cuarto por liga (para reconstruir los periodos desde incidents)
    periodo_por_liga = {
        r["league"]: (r["quarter_duration_minutes"] or 10)
        for r in conn.execute(
            "SELECT league, quarter_duration_minutes FROM leagues_classification")
        if r["league"]
    }
    print(f"[backfill] ligas con largo de cuarto conocido: {len(periodo_por_liga):,}")

    t0 = time.time()
    ok = empty = fail = 0
    pausa = 1.0 / max(args.rate, 0.1)
    for i, (mid, liga) in enumerate(pendientes, 1):
        status = "fail"
        n_rows = 0
        try:
            # incidents -> reconstruccion por cuarto (faltas, tiros libres,
            # tiempos muertos, perdidas) ; statistics -> agregado por periodo.
            # Son las MISMAS dos fuentes que usa el monitor en vivo -> paridad.
            inc = None
            try:
                resp = await client.request("GET", f"event/{mid}/incidents")
                resp = resp.json() if hasattr(resp, "json") else resp
                inc = (resp or {}).get("incidents") or []
            except Exception:
                inc = None
            stat = None
            try:
                resp = await client.request("GET", f"event/{mid}/statistics")
                stat = resp.json() if hasattr(resp, "json") else resp
            except Exception:
                stat = None

            from match.scraper import _parse_period_stats, _parse_team_statistics
            per = 60 * int(periodo_por_liga.get(liga, 10))
            filas = compact_team_stats(
                _parse_team_statistics(str(mid), stat)
                + (_parse_period_stats(str(mid), inc, period_seconds=per) if inc else [])
            )
            if filas:
                with conn:
                    conn.executemany(
                        "INSERT OR REPLACE INTO team_stats_v3 "
                        "(match_id, period, stat_name, home_value, away_value, created_at) "
                        "VALUES (?,?,?,?,?,?)",
                        [(str(mid), f[0], f[1], f[2], f[3], datetime.now().isoformat())
                         for f in filas],
                    )
                n_rows = len(filas)
                status = "ok"
                ok += 1
            else:
                empty += 1
                status = "empty"
        except Exception as exc:  # noqa: BLE001
            fail += 1
            status = f"error:{type(exc).__name__}"
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO team_stats_backfill_state "
                "(match_id, status, n_rows, checked_at) VALUES (?,?,?,?)",
                (str(mid), status, n_rows, datetime.now().isoformat()),
            )
        if i % 25 == 0 or i == len(pendientes):
            dt = time.time() - t0
            vel = i / dt if dt else 0
            falta = (len(pendientes) - i) / max(vel, 0.01) / 60
            print(f"   [{i:,}/{len(pendientes):,}] ok {ok:,} empty {empty:,} fail {fail:,} "
                  f"| {vel:.2f} req/s | faltan {falta:.0f} min", flush=True)
        await asyncio.sleep(pausa + args.sleep)

    dt = time.time() - t0
    print(f"[backfill] terminado en {dt / 60:.1f} min | ok {ok:,} empty {empty:,} fail {fail:,}")
    with conn:
        conn.execute("INSERT OR REPLACE INTO team_stats_backfill_state "
                     "(match_id, status, n_rows, checked_at) VALUES (?,?,?,?)",
                     ("__meta__", f"done:{ok}:{empty}:{fail}", 0,
                      datetime.now().isoformat()))


if __name__ == "__main__":
    asyncio.run(main())
