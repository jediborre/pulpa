# =====================================================================
# REGLAS DE ARQUITECTURA, FORMATEO Y LOGGING DE PRODUCCIÓN (MANTENER):
# 1. ESTRUCTURA: Estrictamente modularizado (config, core, database, scrapers, 
#    models, notifications, utils) coordinados asíncronamente por main.py.
# 2. INFRAESTRUCTURA DB: El archivo base SQLite se localiza exclusivamente en 
#    matches.db (en la raíz del proyecto) y todas las tablas de V3 finalizan
#    con el sufijo '_v3'.
# 3. TABLA DE LOGS: 'bet_monitor_log_v3' se particiona por modelo y contiene 
#    obligatoriamente los campos 'raw_json' (TEXT), 'inference_minute' (INT), 
#    y 'graph_points_count' (INT) junto con marcadores reales del juego.
# 4. CONFIGURACIÓN DE LIGAS: Consumida declarativamente desde config/leagues.yaml.
# 5. Formato Log: {Fecha Hora} [INFO/WARNING/ERROR] [COMPONENTE]
#    - Colores ANSI: INFO=Azul, WARNING=Amarillo, ERROR=Rojo.
# 6. Formato Matches en Log: {horario_match} {match_id} {home} vs {away}
# 7. Errores de red críticos: Imprimir explícitamente "HTTP 403/404" en ROJO.
# 8. Monitoreo avanzado en progreso de cuarto final: Usar obligatoriamente "Q4 🟠".
# 9. Telegram Prefijos de Apuestas: 🟢 (Bettable), 🟡 (No Bettable), ⚪ (Tardía).
# 10. Telegram Resultados FT: Prefijar con ✅ (Ganada) o ❌ (Perdida).
# 11. Conversión de tiempos siempre legibles en formato humano.
# =====================================================================

import json
import sqlite3
import time
from datetime import datetime
from pathlib import Path
import yaml

from monitor_v3.database.connection import get_db_connection
from monitor_v3.config.constants import LEAGUES_CONFIG_PATH
from monitor_v3.utils.logger import log_info, log_error

def init_tables() -> None:
    """
    Crea las tablas con el sufijo '_v3' y sus índices asociados de forma transaccional.
    """
    with get_db_connection() as conn:
        with conn:
            # 1. bet_monitor_schedule_v3
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bet_monitor_schedule_v3 (
                    match_id TEXT PRIMARY KEY,
                    home_team TEXT,
                    away_team TEXT,
                    league TEXT,
                    event_date TEXT,
                    scheduled_utc_ts INTEGER,
                    scheduled_utc TEXT,
                    status_type TEXT,
                    status TEXT,
                    q4_checked INTEGER DEFAULT 0,
                    q4_signal TEXT DEFAULT 'UNAVAILABLE',
                    q4_notified INTEGER DEFAULT 0,
                    q4_model TEXT DEFAULT '',
                    final_fetched INTEGER DEFAULT 0,
                    final_fetch_at TEXT DEFAULT '',
                    skip_reason TEXT DEFAULT '',
                    daily_seq INTEGER,
                    updated_at TEXT
                );
            """)
            # Migración: numeración diaria de partidos (1..N por event_date)
            existing_cols = {
                row["name"] for row in conn.execute("PRAGMA table_info(bet_monitor_schedule_v3)").fetchall()
            }
            if "daily_seq" not in existing_cols:
                conn.execute("ALTER TABLE bet_monitor_schedule_v3 ADD COLUMN daily_seq INTEGER;")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_schedule_date_status_v3 ON bet_monitor_schedule_v3 (event_date, status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_schedule_utc_ts_v3 ON bet_monitor_schedule_v3 (scheduled_utc_ts);")

            # 2. bet_monitor_log_v3
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bet_monitor_log_v3 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    model_version TEXT,
                    target_quarter INTEGER,
                    inference_minute INTEGER,
                    graph_points_count INTEGER,
                    raw_json TEXT,
                    signal_type TEXT,
                    picked_side TEXT,
                    confidence REAL,
                    actual_home_score INTEGER,
                    actual_away_score INTEGER,
                    result TEXT,
                    created_at TEXT,
                    inference_json TEXT,
                    h2h_available INTEGER
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_match_model_v3 ON bet_monitor_log_v3 (match_id, model_version);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_result_v3 ON bet_monitor_log_v3 (result);")

            # 3. eval_match_results_v3
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_match_results_v3 (
                    match_id TEXT PRIMARY KEY,
                    available INTEGER
                );
            """)

            # 4. leagues_config_v3
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leagues_config_v3 (
                    league_name_pattern TEXT PRIMARY KEY,
                    filter_mode TEXT,
                    updated_at TEXT
                );
            """)

            # 6. live_snapshots_v3: telemetria en vivo por minuto (jugadores, faltas, +/-, cuotas in-play)
            #    para entrenar a futuro con informacion que hoy no se puede backtestear.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_snapshots_v3 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    game_minute REAL,
                    home_score INTEGER,
                    away_score INTEGER,
                    status_description TEXT,
                    odds_live_json TEXT,
                    players_json TEXT,
                    team_stats_json TEXT,
                    created_at TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_live_snap_match_v3 ON live_snapshots_v3 (match_id, game_minute);")

            # 7. model_inference_v3: inferencia del modelo POR MINUTO dentro de su
            #    ventana de entrenamiento valida. Es la serie que permite
            #    analizar calibracion y acierto por minuto (y no solo la
            #    apuesta final que queda en bet_monitor_log_v3, que es una fila
            #    por partido y modelo).
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_inference_v3 (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    match_id TEXT,
                    game_minute REAL,
                    model_version TEXT,
                    p_home REAL,
                    p_away REAL,
                    p_cal REAL,
                    margin REAL,
                    signal TEXT,
                    pick TEXT,
                    confidence REAL,
                    snapshot_minute REAL,
                    block_reason TEXT,
                    features_json TEXT,
                    created_at TEXT
                );
            """)
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_model_inf_uniq_v3 "
                         "ON model_inference_v3 (match_id, model_version, game_minute);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_inf_match_v3 "
                         "ON model_inference_v3 (match_id, model_version);")

            # 5. Asegurar existencia de quarter_scores_v2 para persistencia canónica de cuartos
            conn.execute("""
                CREATE TABLE IF NOT EXISTS quarter_scores_v2 (
                    match_id TEXT PRIMARY KEY,
                    q1_home INTEGER,
                    q1_away INTEGER,
                    q2_home INTEGER,
                    q2_away INTEGER,
                    q3_home INTEGER,
                    q3_away INTEGER,
                    q4_home INTEGER,
                    q4_away INTEGER,
                    ot_home INTEGER,
                    ot_away INTEGER
                );
            """)

    # Numeración diaria (1..N por fecha) para las filas que aún no la tengan.
    # Se ejecuta fuera de la transacción anterior para no bloquear la conexión.
    backfill_daily_seq()


def backfill_daily_seq() -> None:
    """
    Asigna `daily_seq` (1..N dentro de cada `event_date`) a los partidos del
    itinerario, en orden cronológico. Idempotente: solo procesa fechas que
    tengan alguna fila sin numerar.
    """
    try:
        with get_db_connection() as conn:
            with conn:
                dates = [
                    row["event_date"]
                    for row in conn.execute(
                        "SELECT DISTINCT event_date FROM bet_monitor_schedule_v3 "
                        "WHERE daily_seq IS NULL AND event_date IS NOT NULL AND event_date != ''"
                    ).fetchall()
                ]
                for date in dates:
                    rows = conn.execute(
                        "SELECT match_id FROM bet_monitor_schedule_v3 "
                        "WHERE event_date = ? ORDER BY scheduled_utc_ts, match_id",
                        (date,),
                    ).fetchall()
                    for index, row in enumerate(rows, start=1):
                        conn.execute(
                            "UPDATE bet_monitor_schedule_v3 SET daily_seq = ? WHERE match_id = ?",
                            (index, row["match_id"]),
                        )
    except Exception as e:
        log_error("DATABASE", f"Error asignando numeración diaria: {e}")


def get_daily_seq(match_id: str) -> int | None:
    """Número del partido dentro de su día (1..N), o None si no está numerado."""
    try:
        with get_db_connection() as conn:
            row = conn.execute(
                "SELECT daily_seq FROM bet_monitor_schedule_v3 WHERE match_id = ?",
                (str(match_id),),
            ).fetchone()
        if row and row["daily_seq"] is not None:
            return int(row["daily_seq"])
    except Exception:
        pass
    return None

def ensure_eval_match_results_columns(conn: sqlite3.Connection, model_versions: list[str]) -> None:
    """
    Asegura mediante introspección que existan las columnas de la tabla matricial
    eval_match_results_v3 para cada versión de modelo especificada.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(eval_match_results_v3);")
    columns = {row["name"] for row in cursor.fetchall()}

    for version in model_versions:
        for suffix in ["_signal", "_pick", "_confidence", "_outcome"]:
            col_name = f"q4{suffix}__{version}"
            if col_name not in columns:
                conn.execute(f"ALTER TABLE eval_match_results_v3 ADD COLUMN {col_name} TEXT;")

def sync_leagues_config() -> None:
    """
    Sincroniza la configuración declarativa del archivo leagues.yaml
    con el espejo transaccional de la tabla leagues_config_v3.
    """
    yaml_path = Path(LEAGUES_CONFIG_PATH)
    if not yaml_path.exists():
        return
        
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            
        excluded = cfg.get("excluded_leagues", {}).get("patterns", [])
        ft_only = cfg.get("ft_only_leagues", {}).get("patterns", [])
        now = datetime.now().isoformat()
        
        with get_db_connection() as conn:
            with conn:
                conn.execute("DELETE FROM leagues_config_v3;")
                for p in excluded:
                    conn.execute("""
                        INSERT OR REPLACE INTO leagues_config_v3 (league_name_pattern, filter_mode, updated_at)
                        VALUES (?, 'EXCLUDE', ?);
                    """, (p, now))
                for p in ft_only:
                    conn.execute("""
                        INSERT OR REPLACE INTO leagues_config_v3 (league_name_pattern, filter_mode, updated_at)
                        VALUES (?, 'FT_ONLY', ?);
                    """, (p, now))
    except Exception as e:
        log_error("DATABASE", f"Error sincronizando leagues_config_v3: {e}")

def get_leagues_config() -> dict:
    """Retorna la configuración activa de ligas categorizada."""
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT league_name_pattern, filter_mode FROM leagues_config_v3;")
        excluded = []
        ft_only = []
        for row in cursor.fetchall():
            if row["filter_mode"] == "EXCLUDE":
                excluded.append(row["league_name_pattern"].lower())
            elif row["filter_mode"] == "FT_ONLY":
                ft_only.append(row["league_name_pattern"].lower())
        return {"excluded": excluded, "ft_only": ft_only}

def get_league_mode(league_name: str, config: dict) -> str:
    """
    Evalúa el nombre de una liga contra los patrones configurados:
    Retorna 'EXCLUDE', 'FT_ONLY', o 'NORMAL'.
    """
    if not league_name:
        return "NORMAL"
    low = league_name.lower()
    for p in config.get("excluded", []):
        if p in low:
            return "EXCLUDE"
    for p in config.get("ft_only", []):
        if p in low:
            return "FT_ONLY"
    return "NORMAL"

def save_schedule_matches(matches: list[dict]) -> None:
    """Guarda o actualiza partidos del itinerario de forma atómica."""
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        with conn:
            for m in matches:
                conn.execute("""
                    INSERT OR IGNORE INTO bet_monitor_schedule_v3 (
                        match_id, home_team, away_team, league, event_date,
                        scheduled_utc_ts, scheduled_utc, status_type, status,
                        q4_checked, q4_signal, q4_notified, q4_model,
                        final_fetched, final_fetch_at, skip_reason, daily_seq, updated_at
                    ) VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, 'UNAVAILABLE', 0, '', 0, '', '',
                        (SELECT COALESCE(MAX(daily_seq), 0) + 1 FROM bet_monitor_schedule_v3 WHERE event_date = ?),
                        ?
                    )
                """, (
                    str(m["match_id"]), m.get("home_team", ""), m.get("away_team", ""),
                    m.get("league", ""), m.get("event_date", ""),
                    m.get("scheduled_utc_ts", 0), m.get("scheduled_utc", ""),
                    m.get("status_type", ""), m.get("event_date", ""), now
                ))

def update_schedule_status(match_id: str, status: str, skip_reason: str = "") -> None:
    """Actualiza el estado de un partido programado."""
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        with conn:
            if status == "done":
                conn.execute("""
                    UPDATE bet_monitor_schedule_v3
                    SET status = ?, skip_reason = ?, final_fetched = 1, final_fetch_at = ?, updated_at = ?
                    WHERE match_id = ?
                """, (status, skip_reason, now, now, match_id))
            else:
                conn.execute("""
                    UPDATE bet_monitor_schedule_v3
                    SET status = ?, skip_reason = ?, updated_at = ?
                    WHERE match_id = ?
                """, (status, skip_reason, now, match_id))

def update_schedule_q4(match_id: str, signal: str, model: str, notified: int = 1) -> None:
    """Actualiza la señal Q4 detectada para un match programado."""
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        with conn:
            conn.execute("""
                UPDATE bet_monitor_schedule_v3
                SET q4_checked = 1, q4_signal = ?, q4_notified = ?, q4_model = ?, updated_at = ?
                WHERE match_id = ?
            """, (signal, notified, model, now, match_id))

def get_schedule_matches(event_date: str) -> list[dict]:
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM bet_monitor_schedule_v3 WHERE event_date = ?", (event_date,))
        return [dict(r) for r in cursor.fetchall()]

def get_pending_schedule_matches() -> list[dict]:
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM bet_monitor_schedule_v3 WHERE status IN ('pending', 'in_progress')")
        return [dict(r) for r in cursor.fetchall()]


def get_stale_pending_bets(min_age_secs: int, limit: int = 40) -> list[dict]:
    """
    Partidos con apuestas en 'pending' y sin Q4 guardado, cuyo inicio ya pasó
    hace más de `min_age_secs` (deberían tener el FT publicado).
    """
    cutoff = int(time.time()) - int(min_age_secs)
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT l.match_id,
                   COALESCE(s.home_team, '?') AS home_team,
                   COALESCE(s.away_team, '?') AS away_team,
                   COALESCE(s.scheduled_utc_ts, 0) AS scheduled_utc_ts
            FROM bet_monitor_log_v3 l
            LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
            LEFT JOIN quarter_scores_v2 q ON q.match_id = l.match_id
            WHERE l.result = 'pending'
              AND (q.q4_home IS NULL OR q.q4_away IS NULL)
              AND COALESCE(s.scheduled_utc_ts, 0) < ?
            ORDER BY s.scheduled_utc_ts
            LIMIT ?
            """,
            (cutoff, int(limit)),
        ).fetchall()
    return [dict(r) for r in rows]


def void_pending_older_than(older_than_secs: int) -> int:
    """
    Anula ('void') las apuestas que siguen pendientes demasiado tiempo después
    del inicio del partido: no hay datos para liquidarlas, así que dejan de
    contar en las estadísticas en lugar de quedarse en 'pending' para siempre.
    Devuelve cuántas apuestas se anularon.
    """
    cutoff = int(time.time()) - int(older_than_secs)
    with get_db_connection() as conn:
        with conn:
            rows = conn.execute(
                """
                SELECT l.id
                FROM bet_monitor_log_v3 l
                LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
                WHERE l.result = 'pending'
                  AND COALESCE(s.scheduled_utc_ts, 0) < ?
                """,
                (cutoff,),
            ).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE bet_monitor_log_v3 SET result = 'void' WHERE id = ?", (row["id"],)
                )
    return len(rows)

def save_quarter_scores(match_id: str, scores: dict) -> None:
    """Guarda los marcadores por cuarto en quarter_scores_v2."""
    with get_db_connection() as conn:
        with conn:
            conn.execute("""
                INSERT OR REPLACE INTO quarter_scores_v2 (
                    match_id, q1_home, q1_away, q2_home, q2_away,
                    q3_home, q3_away, q4_home, q4_away, ot_home, ot_away
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id,
                scores.get("Q1", {}).get("home"), scores.get("Q1", {}).get("away"),
                scores.get("Q2", {}).get("home"), scores.get("Q2", {}).get("away"),
                scores.get("Q3", {}).get("home"), scores.get("Q3", {}).get("away"),
                scores.get("Q4", {}).get("home"), scores.get("Q4", {}).get("away"),
                scores.get("OT1", {}).get("home"), scores.get("OT1", {}).get("away")
            ))

def get_quarter_scores(match_id: str) -> dict | None:
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id = ?", (match_id,)).fetchone()
        return dict(row) if row else None

def save_bet_log(
    match_id: str, model_version: str, inference_minute: int, graph_points_count: int,
    raw_json: dict, signal_type: str, picked_side: str, confidence: float,
    actual_home_score: int, actual_away_score: int, result: str = "pending",
    inference_json: dict | None = None, h2h_available: bool | None = None
) -> int:
    """Registra una inferencia en la tabla bet_monitor_log_v3."""
    now = datetime.now().isoformat()
    raw_json_str = json.dumps(raw_json, ensure_ascii=False)
    inf_json_str = json.dumps(inference_json, ensure_ascii=False) if inference_json is not None else None
    
    with get_db_connection() as conn:
        with conn:
            cursor = conn.execute("""
                INSERT INTO bet_monitor_log_v3 (
                    match_id, model_version, target_quarter, inference_minute, graph_points_count,
                    raw_json, signal_type, picked_side, confidence,
                    actual_home_score, actual_away_score, result, created_at, inference_json,
                    h2h_available
                ) VALUES (?, ?, 4, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id, model_version, inference_minute, graph_points_count,
                raw_json_str, signal_type, picked_side, confidence,
                actual_home_score, actual_away_score, result, now, inf_json_str,
                1 if h2h_available else (0 if h2h_available is not None else None)
            ))
            return cursor.lastrowid

def save_live_snapshot(
    match_id: str, game_minute: float | None, home_score: int, away_score: int,
    status_description: str, odds_live: dict | None, players: list | None, team_stats: list | None,
) -> None:
    """Persiste una foto compacta del estado en vivo (best-effort; nunca debe romper el monitoreo)."""
    try:
        with get_db_connection() as conn:
            with conn:
                conn.execute("""
                    INSERT INTO live_snapshots_v3 (
                        match_id, game_minute, home_score, away_score, status_description,
                        odds_live_json, players_json, team_stats_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id, game_minute, home_score, away_score, status_description,
                    json.dumps(odds_live or {}, ensure_ascii=False),
                    json.dumps(players or [], ensure_ascii=False),
                    json.dumps(team_stats or [], ensure_ascii=False),
                    datetime.now().isoformat(),
                ))
    except Exception as exc:
        log_error("DB", f"live_snapshots_v3: {exc}")

def update_bet_log_result(log_id: int, result: str) -> None:
    with get_db_connection() as conn:
        with conn:
            conn.execute("UPDATE bet_monitor_log_v3 SET result = ? WHERE id = ?", (result, log_id))


def save_model_inference(
    match_id: str, game_minute: float | None, model_version: str,
    p_home: float | None, p_away: float | None, p_cal: float | None,
    margin: float | None, signal: str, pick: str, confidence: float | None,
    snapshot_minute: float | None, block_reason: str | None, features: dict | None,
) -> None:
    """Persiste la inferencia del modelo en UN minuto concreto (best-effort).

    Es la serie para analizar el modelo por minuto dentro de su ventana de
    entrenamiento valida: acierto y calibracion por minuto, evolucion de la
    confianza, etc. `bet_monitor_log_v3` solo guarda una fila por partido y
    modelo (la apuesta o el NO_BET definitivo), asi que no sirve para eso.
    """
    try:
        with get_db_connection() as conn:
            with conn:
                conn.execute("""
                    INSERT OR REPLACE INTO model_inference_v3 (
                        match_id, game_minute, model_version, p_home, p_away, p_cal, margin,
                        signal, pick, confidence, snapshot_minute, block_reason,
                        features_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    match_id, game_minute, model_version, p_home, p_away, p_cal, margin,
                    signal, pick, confidence, snapshot_minute, block_reason,
                    json.dumps(features or {}, ensure_ascii=False),
                    datetime.now().isoformat(),
                ))
    except Exception as exc:
        log_error("DB", f"model_inference_v3: {exc}")


def _save_eval_match_results_impl(conn: sqlite3.Connection, match_id: str, available: int, metrics: dict) -> None:
    columns = ["match_id", "available"]
    values = [match_id, available]
    
    for version, data in metrics.items():
        columns.append(f"q4_signal__{version}")
        values.append(data.get("signal", "UNAVAILABLE"))
        columns.append(f"q4_pick__{version}")
        values.append(data.get("pick", "NONE"))
        columns.append(f"q4_confidence__{version}")
        values.append(data.get("confidence", 0.0))
        columns.append(f"q4_outcome__{version}")
        values.append(data.get("outcome", "pending"))
        
    placeholders = ", ".join(["?"] * len(values))
    col_str = ", ".join(columns)
    
    conn.execute(f"""
        INSERT OR REPLACE INTO eval_match_results_v3 ({col_str})
        VALUES ({placeholders})
    """, values)

def save_eval_match_results(match_id: str, available: int, metrics: dict, conn: sqlite3.Connection = None) -> None:
    model_versions = list(metrics.keys())
    if conn is not None:
        ensure_eval_match_results_columns(conn, model_versions)
        _save_eval_match_results_impl(conn, match_id, available, metrics)
    else:
        with get_db_connection() as c:
            with c:
                ensure_eval_match_results_columns(c, model_versions)
                _save_eval_match_results_impl(c, match_id, available, metrics)

def reconcile_pending_results() -> None:
    """
    Rutina sincronizada ejecutada periódicamente que inspecciona logs 'pending'
    en bet_monitor_log_v3 y los liquida contra quarter_scores_v2.
    """
    with get_db_connection() as conn:
        with conn:
            cursor = conn.execute("""
                SELECT l.id, l.match_id, l.picked_side, l.model_version, l.confidence, l.signal_type
                FROM bet_monitor_log_v3 l
                WHERE l.result = 'pending'
            """)
            pending_logs = cursor.fetchall()
            
            for log in pending_logs:
                log_id = log["id"]
                match_id = log["match_id"]
                picked_side = log["picked_side"]
                model_ver = log["model_version"]
                confidence = log["confidence"]
                sig_type = log["signal_type"]
                
                qs = conn.execute("SELECT q4_home, q4_away FROM quarter_scores_v2 WHERE match_id = ?", (match_id,)).fetchone()
                if qs and qs["q4_home"] is not None and qs["q4_away"] is not None:
                    q4h = qs["q4_home"]
                    q4a = qs["q4_away"]
                    
                    if q4h == q4a:
                        # Criterio de negocio: un cuarto EMPATADO no gana la apuesta,
                        # así que se contabiliza como PERDIDA (no como push anulado).
                        real_winner = "tie"
                    elif q4h > q4a:
                        real_winner = "home"
                    else:
                        real_winner = "away"
                        
                    if real_winner == "tie":
                        outcome = "loss"
                    elif picked_side.lower() == real_winner:
                        outcome = "win"
                    else:
                        outcome = "loss"
                        
                    conn.execute("UPDATE bet_monitor_log_v3 SET result = ? WHERE id = ?", (outcome, log_id))
                    log_info("DATABASE", f"Apuesta liquidada {match_id} ({model_ver}): {outcome.upper()}")
