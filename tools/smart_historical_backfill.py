#!/usr/bin/env python3
"""
tools/smart_historical_backfill.py
==================================
Herramienta de descarga histórica inteligente (Smart Backfill) para SofaScore Basketball.

Características:
- Barra de progreso dinámica por día (con conteo de FullML, Elo y estado del token JWT).
- Jitter optimizado y reducido para descargas más veloces y seguras.
- Guardado automático de progreso (checkpoint) al presionar Ctrl+C con mensaje resumen.
- Soporte para reanudación instantánea (--resume) desde la última fecha procesada.
- Filtrado por clústeres: fiba_men, nba_12m, fiba_women, all.
- Modos: auto (recomendado), full_ml, elo_h2h.
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configurar encoding UTF-8 en consola Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ajustar PYTHONPATH a la raíz
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import match.db as db_mod
import match.scraper as scraper_mod
from monitor_v3.core.token_manager import get_token_pool

DB_PATH = ROOT / "matches.db"
STATE_FILE = ROOT / "tools" / "smart_backfill_state.json"


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
    if not tax:
        return cluster == "all"

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
        if not has_pbp or not has_gp:
            return False, "incompleto_ml"
        return True, "ok_full_ml"
    elif mode == "elo_h2h":
        return True, "ok_elo_h2h"
    elif mode == "auto":
        if has_pbp and has_gp:
            return True, "ok_full_ml"
        return True, "ok_elo_h2h"

    return False, "modo_desconocido"


def get_jwt_status(token_pool) -> str:
    """Devuelve una etiqueta compacta sobre el estado del pool de tokens JWT."""
    if not token_pool or not token_pool.tokens:
        return "JWT:N/A"
    try:
        cur_idx = token_pool._current_index + 1
        total = len(token_pool.tokens)
        reqs = getattr(token_pool, "_matches_on_current_token", 0)
        return f"JWT:#{cur_idx}/{total}({reqs})"
    except Exception:
        return "JWT:OK"


def save_checkpoint(
    last_date: str,
    start_date: str,
    end_date: str,
    cluster: str,
    mode: str,
    tot_ml: int,
    tot_elo: int,
    tot_skip: int,
    tot_err: int,
    status: str = "in_progress",
):
    """Guarda el cursor de descarga en disco para reanudación posterior."""
    payload = {
        "last_processed_date": last_date,
        "status": status,
        "start_date": start_date,
        "end_date": end_date,
        "cluster": cluster,
        "mode": mode,
        "total_ingested_ml": tot_ml,
        "total_ingested_elo": tot_elo,
        "total_skipped": tot_skip,
        "total_errors": tot_err,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"\n[AVISO] No se pudo escribir checkpoint: {e}")


def load_checkpoint() -> dict | None:
    """Carga el último checkpoint guardado si existe."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def print_pause_banner(
    last_date: str,
    tot_ml: int,
    tot_elo: int,
    tot_skip: int,
    tot_err: int,
):
    """Imprime un banner limpio y formateado cuando el usuario interrumpe con Ctrl+C."""
    print("\n")
    print("╔═════════════════════════════════════════════════════════════════════════╗")
    print("║                   ⏸️   DESCARGA PAUSADA POR EL USUARIO                  ║")
    print("╠═════════════════════════════════════════════════════════════════════════╣")
    print(f"║  📅 Última fecha procesada : {last_date:<43}║")
    print(f"║  🟢 Partidos Full ML        : {tot_ml:<43,}║")
    print(f"║  🔵 Partidos Elo / H2H      : {tot_elo:<43,}║")
    print(f"║  ⚪ Omitidos (ya en DB)     : {tot_skip:<43,}║")
    print(f"║  ❌ Errores                 : {tot_err:<43}║")
    print(f"║  💾 Archivo de checkpoint   : {str(STATE_FILE.name):<43}║")
    print("╠═════════════════════════════════════════════════════════════════════════╣")
    print("║  💡 Para retomar desde donde te quedaste:                               ║")
    print("║     python tools\\smart_historical_backfill.py --resume                   ║")
    print("║     o selecciona la opción 35 (Submenú Smart Backfill) en menu.bat       ║")
    print("╚═════════════════════════════════════════════════════════════════════════╝")
    print()


def render_progress_bar(
    date_str: str,
    date_idx: int,
    total_dates: int,
    current: int,
    total: int,
    day_ml: int,
    day_elo: int,
    day_skip: int,
    jwt_status: str,
    bar_length: int = 14,
):
    """Dibuja una barra de progreso limpia en una sola línea usando \\r."""
    pct = int((current / total) * 100) if total else 100
    filled = int((pct / 100.0) * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    line = (
        f"\r  [{date_idx}/{total_dates}] {date_str} [{bar}] {pct:>3}% ({current}/{total}) "
        f"| 🟢ML:{day_ml} 🔵Elo:{day_elo} ⚪Skip:{day_skip} | 🔑{jwt_status}"
    )
    # Rellenar con espacios para limpiar caracteres sobrantes en consola
    sys.stdout.write(f"{line:<95}")
    sys.stdout.flush()


def run_historical_backfill(
    start_date: str | None = None,
    end_date: str | None = None,
    cluster: str = "all",
    mode: str = "auto",
    skip_blacklist: bool = True,
    delay_between_matches: float = 0.60,
    limit_per_date: int | None = None,
    resume: bool = False,
):
    """Bucle principal de ingesta histórica adaptativa."""
    total_ingested_ml = 0
    total_ingested_elo = 0
    total_skipped = 0
    total_errors = 0

    # 1. Gestionar reanudación desde Checkpoint
    if resume:
        cp = load_checkpoint()
        if not cp:
            print("[SMART BACKFILL] No se encontró ningún checkpoint guardado. Especifica fechas.")
            return
        last_dt = cp.get("last_processed_date", "")
        status = cp.get("status", "")
        end_date = cp.get("end_date")
        cluster = cp.get("cluster", cluster)
        mode = cp.get("mode", mode)
        total_ingested_ml = cp.get("total_ingested_ml", 0)
        total_ingested_elo = cp.get("total_ingested_elo", 0)
        total_skipped = cp.get("total_skipped", 0)
        total_errors = cp.get("total_errors", 0)

        if status in ("completed_date", "completed_all") and last_dt and end_date:
            try:
                next_dt = (datetime.strptime(last_dt, "%Y-%m-%d").date() + timedelta(days=1)).isoformat()
                if datetime.strptime(next_dt, "%Y-%m-%d").date() > datetime.strptime(end_date, "%Y-%m-%d").date():
                    print(f"✨ [RESUME] La descarga hasta {end_date} ya se había completado en su totalidad.")
                    return
                start_date = next_dt
            except Exception:
                start_date = last_dt
        else:
            start_date = last_dt if last_dt else cp.get("start_date")

        print(f"🔄 [RESUME] Reanudando desde {start_date} hasta {end_date} (Clúster: {cluster}, Modo: {mode})")
    
    if not start_date or not end_date:
        print("[ERROR] Debes proporcionar --start-date y --end-date (o usar --resume).")
        return

    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)
    taxonomy = load_league_taxonomies(conn)
    token_pool = get_token_pool()

    try:
        s_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        e_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as e:
        print(f"[ERROR] Formato de fecha inválido: {e}")
        return

    if s_dt > e_dt:
        s_dt, e_dt = e_dt, s_dt

    curr = s_dt
    dates = []
    while curr <= e_dt:
        dates.append(curr.isoformat())
        curr += timedelta(days=1)

    print("=" * 75)
    print("🚀 SMART HISTORICAL BACKFILL — SOFASCORE BASKETBALL")
    print(f"  • Rango Fechas : {start_date} al {end_date} ({len(dates)} días)")
    print(f"  • Clúster      : {cluster.upper()} | Modo: {mode.upper()}")
    print(f"  • Blacklist    : {'OMITIDA (Juveniles/NCAA/Amistosos fuera)' if skip_blacklist else 'INCLUIDA'}")
    print(f"  • Delay Jitter : {delay_between_matches}s ± 0.15s (Rápido y Seguro)")
    if resume:
        print(f"  • Acumulado ML : {total_ingested_ml:,} | Acumulado Elo: {total_ingested_elo:,}")
    print("=" * 75)

    last_active_date = start_date
    day_completed = False

    try:
        for d_idx, dt_str in enumerate(dates, 1):
            last_active_date = dt_str
            day_completed = False
            
            try:
                matches_day = scraper_mod.fetch_finished_match_ids_for_date(dt_str, backend="mobile")
            except Exception as exc:
                print(f"\n  ❌ Error en itinerario para {dt_str}: {exc}")
                time.sleep(2.0)
                continue

            if not matches_day:
                print(f"\n  ℹ️ [{d_idx}/{len(dates)}] {dt_str}: Sin partidos finalizados.")
                day_completed = True
                save_checkpoint(
                    dt_str, start_date, end_date, cluster, mode,
                    total_ingested_ml, total_ingested_elo, total_skipped, total_errors,
                    status="completed_date",
                )
                continue

            # Filtrar por clúster
            eligible_matches = [
                m for m in matches_day
                if is_match_eligible_for_cluster(m.get("league", ""), cluster, taxonomy, skip_blacklist)
            ]

            if limit_per_date:
                eligible_matches = eligible_matches[:limit_per_date]

            total_day = len(eligible_matches)
            if total_day == 0:
                print(f"\n  ⚪ [{d_idx}/{len(dates)}] {dt_str}: {len(matches_day)} partidos (0 elegibles para '{cluster}').")
                day_completed = True
                save_checkpoint(
                    dt_str, start_date, end_date, cluster, mode,
                    total_ingested_ml, total_ingested_elo, total_skipped, total_errors,
                    status="completed_date",
                )
                continue

            day_ml = 0
            day_elo = 0
            day_skip = 0

            # Render inicial
            render_progress_bar(
                dt_str, d_idx, len(dates), 0, total_day,
                day_ml, day_elo, day_skip, get_jwt_status(token_pool)
            )

            for m_idx, m_info in enumerate(eligible_matches, 1):
                mid = str(m_info.get("match_id", ""))
                if not mid:
                    continue

                # Revisar si ya existe en SQLite
                existing = db_mod.get_match(conn, mid)
                if existing:
                    day_skip += 1
                    total_skipped += 1
                    render_progress_bar(
                        dt_str, d_idx, len(dates), m_idx, total_day,
                        day_ml, day_elo, day_skip, get_jwt_status(token_pool)
                    )
                    continue

                # Jitter reducido y seguro (0.50s a 0.75s por defecto)
                jitter = random.uniform(delay_between_matches * 0.85, delay_between_matches * 1.25)
                time.sleep(jitter)

                try:
                    data = scraper_mod.fetch_match_by_id(mid, backend="mobile")
                    usable, reason = check_data_usability(data, mode)
                    
                    if not usable:
                        day_skip += 1
                        total_skipped += 1
                    else:
                        db_mod.save_match(conn, mid, data)
                        db_mod.mark_discovered_processed(conn, mid)

                        if reason == "ok_full_ml":
                            day_ml += 1
                            total_ingested_ml += 1
                        else:
                            day_elo += 1
                            total_ingested_elo += 1

                        if token_pool:
                            token_pool.notify_match_done(silent=True)

                except KeyboardInterrupt:
                    raise  # Propagar al bloque exterior para manejo limpio
                except Exception:
                    total_errors += 1

                render_progress_bar(
                    dt_str, d_idx, len(dates), m_idx, total_day,
                    day_ml, day_elo, day_skip, get_jwt_status(token_pool)
                )

            # Línea final del día completado con salto de línea
            jwt_tag = get_jwt_status(token_pool)
            sys.stdout.write(
                f"\r  ✅ [{d_idx}/{len(dates)}] {dt_str} completado: {total_day}/{total_day} "
                f"| 🟢+{day_ml} ML 🔵+{day_elo} Elo ⚪+{day_skip} Skip | 🔑{jwt_tag}\n"
            )
            sys.stdout.flush()

            # Guardar checkpoint al cierre de cada fecha
            day_completed = True
            save_checkpoint(
                dt_str, start_date, end_date, cluster, mode,
                total_ingested_ml, total_ingested_elo, total_skipped, total_errors,
                status="completed_date",
            )

            # Breve pausa entre días
            time.sleep(1.0)

    except KeyboardInterrupt:
        # Guardar estado y banner bonito
        final_status = "completed_date" if day_completed else "interrupted_in_day"
        save_checkpoint(
            last_active_date, start_date, end_date, cluster, mode,
            total_ingested_ml, total_ingested_elo, total_skipped, total_errors,
            status=final_status,
        )
        print_pause_banner(
            last_active_date, total_ingested_ml, total_ingested_elo, total_skipped, total_errors
        )
        conn.close()
        return

    # Checkpoint de finalización total
    save_checkpoint(
        end_date, start_date, end_date, cluster, mode,
        total_ingested_ml, total_ingested_elo, total_skipped, total_errors,
        status="completed_all",
    )
    conn.close()
    print("\n" + "=" * 75)
    print("🏁 DESCARGA HISTÓRICA COMPLETADA")
    print(f"  • Total Guardados Full ML : {total_ingested_ml:,}")
    print(f"  • Total Guardados Elo/H2H : {total_ingested_elo:,}")
    print(f"  • Total Omitidos (ya en DB): {total_skipped:,}")
    print(f"  • Total Errores           : {total_errors}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Descargador histórico inteligente para SofaScore Basketball.")
    parser.add_argument("--start-date", default=None, help="Fecha inicio (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="Fecha fin (YYYY-MM-DD)")
    parser.add_argument("--cluster", choices=["all", "fiba_men", "nba_12m", "fiba_women"], default="all",
                        help="Clúster de ligas a descargar (default: all)")
    parser.add_argument("--mode", choices=["auto", "full_ml", "elo_h2h"], default="auto",
                        help="Modo de completitud: auto (recom), full_ml (exige PBP/GP) o elo_h2h (solo cuartos)")
    parser.add_argument("--include-blacklist", action="store_true",
                        help="Si se activa, incluye juveniles, college y amistosos (por defecto se omiten)")
    parser.add_argument("--delay", type=float, default=0.60,
                        help="Segundos base de espera entre partidos (default: 0.60s)")
    parser.add_argument("--limit-per-date", type=int, default=None,
                        help="Límite máximo de partidos por día (opcional)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanuda la descarga desde el último checkpoint guardado")

    args = parser.parse_args()

    run_historical_backfill(
        start_date=args.start_date,
        end_date=args.end_date,
        cluster=args.cluster,
        mode=args.mode,
        skip_blacklist=not args.include_blacklist,
        delay_between_matches=args.delay,
        limit_per_date=args.limit_per_date,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
