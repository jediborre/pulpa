#!/usr/bin/env python3
"""
tools/smart_historical_backfill.py
==================================
Herramienta de descarga histórica inteligente (Backfill Estratégico) para SofaScore Basketball.

Permite recabar partidos históricos distinguiendo entre dos modalidades operativas:
1. Modo 'full_ml' (2018-2025): Exige 4 cuartos + PBP + Graph Points para entrenamiento de m27/m34.
2. Modo 'elo_h2h' (2015-2018): Exige solo marcadores de 4 cuartos para sembrar Elo y H2H histórico.
3. Modo 'auto' (Recomendado): Guarda completo si hay PBP/GP, o como Elo/H2H si solo hay cuartos.

Soporta filtrado declarativo por clúster de ligas:
- --cluster fiba_men
- --cluster nba_12m
- --cluster fiba_women
- --skip-blacklist (descarta juveniles, college y amistosos)
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ajustar PYTHONPATH a la raíz
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import match.db as db_mod
import match.scraper as scraper_mod
from monitor_v3.core.token_manager import get_token_pool
from monitor_v3.utils.logger import log_info, log_warning, log_error

DB_PATH = ROOT / "matches.db"


def load_league_taxonomies(conn) -> dict:
    """Carga la clasificación de ligas de leagues_classification para filtrado en memoria."""
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT league, is_women, is_youth, is_college, competition_type, quarter_duration_minutes
        FROM leagues_classification
    """).fetchall()
    return {r["league"]: dict(r) for r in rows}


def is_match_eligible_for_cluster(league_name: str, cluster: str, taxonomy: dict, skip_blacklist: bool) -> bool:
    """Verifica si la liga del partido califica para el clúster objetivo."""
    tax = taxonomy.get(league_name)
    
    # Si no está catalogada aún, permitimos si no hay filtro estricto
    if not tax:
        return cluster == "all"

    # Filtrar blacklist
    if skip_blacklist:
        if tax.get("is_youth") == 1 or tax.get("is_college") == 1 or tax.get("competition_type") == "friendly":
            return False

    q_dur = tax.get("quarter_duration_minutes", 10)
    is_w = tax.get("is_women", 0)

    if cluster == "all":
        return True
    elif cluster == "fiba_men":
        return is_w == 0 and q_dur == 10
    elif cluster == "nba_12m":
        return q_dur == 12
    elif cluster == "fiba_women":
        return is_w == 1 and q_dur == 10
    return True


def check_data_usability(data: dict, mode: str) -> tuple[bool, str]:
    """Evalúa la completitud del partido según el modo seleccionado."""
    if not data or not isinstance(data, dict):
        return False, "payload_vacio"

    match_meta = data.get("match", {})
    status_type = str(match_meta.get("status_type", "")).lower()
    if status_type != "finished":
        return False, f"not_finished({status_type})"

    score = data.get("score", {})
    quarters = score.get("quarters", {})
    if not quarters or not isinstance(quarters, dict):
        return False, "sin_cuartos"

    missing_q = [q for q in ("Q1", "Q2", "Q3", "Q4") if q not in quarters]
    if missing_q:
        return False, f"faltan_cuartos({','.join(missing_q)})"

    pbp = data.get("play_by_play", {})
    has_pbp = bool(isinstance(pbp, dict) and (pbp.get("Q1") or pbp.get("Q2")))
    gp = data.get("graph_points", [])
    has_gp = bool(isinstance(gp, list) and len(gp) > 0)

    if mode == "full_ml":
        if not has_pbp:
            return False, "sin_pbp_q1_q2"
        if not has_gp:
            return False, "sin_graph_points"
        return True, "ok_full_ml"
    elif mode == "elo_h2h":
        return True, "ok_elo_h2h"
    elif mode == "auto":
        if has_pbp and has_gp:
            return True, "ok_full_ml"
        return True, "ok_elo_h2h"

    return False, "modo_desconocido"


def run_historical_backfill(
    start_date: str,
    end_date: str,
    cluster: str = "all",
    mode: str = "auto",
    skip_blacklist: bool = True,
    delay_between_matches: float = 0.9,
    limit_per_date: int | None = None,
):
    """Bucle principal de ingesta histórica adaptativa."""
    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)
    taxonomy = load_league_taxonomies(conn)
    token_pool = get_token_pool()

    s_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    e_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if s_dt > e_dt:
        s_dt, e_dt = e_dt, s_dt

    curr = s_dt
    dates = []
    while curr <= e_dt:
        dates.append(curr.isoformat())
        curr += timedelta(days=1)

    print("=" * 70)
    print("🚀 SMART HISTORICAL BACKFILL — SOFASCORE BASKETBALL")
    print(f"  • Rango: {start_date} al {end_date} ({len(dates)} fechas)")
    print(f"  • Clúster: {cluster.upper()} | Modo: {mode.upper()}")
    print(f"  • Excluir Blacklist: {'SÍ' if skip_blacklist else 'NO'}")
    print(f"  • Delay entre partidos: {delay_between_matches}s (Jitter activo)")
    print("=" * 70)

    total_ingested_ml = 0
    total_ingested_elo = 0
    total_skipped = 0
    total_errors = 0

    for d_idx, dt_str in enumerate(dates, 1):
        print(f"\n[{d_idx}/{len(dates)}] 📅 Procesando fecha: {dt_str} ...")
        
        try:
            matches_day = scraper_mod.fetch_finished_match_ids_for_date(dt_str, backend="mobile")
        except Exception as exc:
            print(f"  ❌ Error consultando itinerario para {dt_str}: {exc}")
            time.sleep(3.0)
            continue

        if not matches_day:
            print(f"  ℹ️ Sin partidos finalizados en {dt_str}.")
            continue

        # Filtrar por clúster si es posible desde los metadatos de discovery
        eligible_matches = []
        for m in matches_day:
            league = m.get("league", "")
            if is_match_eligible_for_cluster(league, cluster, taxonomy, skip_blacklist):
                eligible_matches.append(m)

        print(f"  • Encontrados: {len(matches_day)} | Elegibles para clúster '{cluster}': {len(eligible_matches)}")

        if limit_per_date:
            eligible_matches = eligible_matches[:limit_per_date]

        for m_idx, m_info in enumerate(eligible_matches, 1):
            mid = str(m_info.get("match_id", ""))
            if not mid:
                continue

            # Revisar si ya existe en DB
            existing = db_mod.get_match(conn, mid)
            if existing:
                total_skipped += 1
                continue

            # Jitter de cortesía para proteger tokens de Varnish/Fastly
            jitter = random.uniform(delay_between_matches * 0.8, delay_between_matches * 1.3)
            time.sleep(jitter)

            try:
                data = scraper_mod.fetch_match_by_id(mid, backend="mobile")
                usable, reason = check_data_usability(data, mode)
                
                if not usable:
                    total_skipped += 1
                    continue

                db_mod.save_match(conn, mid, data)
                db_mod.mark_discovered_processed(conn, mid)

                if reason == "ok_full_ml":
                    total_ingested_ml += 1
                    tag = "🟢 [FULL ML]"
                else:
                    total_ingested_elo += 1
                    tag = "🔵 [ELO/H2H]"

                match_lbl = f"{data.get('match', {}).get('home_team', 'Home')} vs {data.get('match', {}).get('away_team', 'Away')}"
                league_lbl = data.get('match', {}).get('league', 'Unknown')
                print(f"    {tag} ID {mid} ({league_lbl}): {match_lbl}")

                if token_pool:
                    token_pool.notify_match_done()

            except KeyboardInterrupt:
                print("\n⚠️ Descarga pausada por el usuario. Guardando estado y saliendo...")
                conn.close()
                return
            except Exception as e:
                total_errors += 1
                print(f"    ❌ Error en ID {mid}: {e}")

        # Pausa de cortesía entre fechas
        time.sleep(2.0)

    conn.close()
    print("\n" + "=" * 70)
    print("🏁 BACKFILL COMPLETADO")
    print(f"  • Total Guardados Full ML: {total_ingested_ml}")
    print(f"  • Total Guardados Elo/H2H: {total_ingested_elo}")
    print(f"  • Omitidos (ya en DB o no elegibles): {total_skipped}")
    print(f"  • Errores: {total_errors}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Descargador histórico inteligente para SofaScore Basketball.")
    parser.add_argument("--start-date", required=True, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--cluster", choices=["all", "fiba_men", "nba_12m", "fiba_women"], default="all",
                        help="Clúster de ligas a descargar (default: all)")
    parser.add_argument("--mode", choices=["auto", "full_ml", "elo_h2h"], default="auto",
                        help="Modo de completitud: auto (recom), full_ml (exige PBP/GP) o elo_h2h (solo cuartos)")
    parser.add_argument("--include-blacklist", action="store_true",
                        help="Si se activa, incluye juveniles, college y amistosos (por defecto se omiten)")
    parser.add_argument("--delay", type=float, default=0.9,
                        help="Segundos base de espera entre partidos (default: 0.9s)")
    parser.add_argument("--limit-per-date", type=int, default=None,
                        help="Límite máximo de partidos por día (opcional)")

    args = parser.parse_args()

    run_historical_backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        cluster=args.cluster,
        mode=args.mode,
        skip_blacklist=not args.include_blacklist,
        delay_between_matches=args.delay,
        limit_per_date=args.limit_per_date,
    )


if __name__ == "__main__":
    main()
