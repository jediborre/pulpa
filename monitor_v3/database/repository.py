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
                    updated_at TEXT
                );
            """)
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
                        final_fetched, final_fetch_at, skip_reason, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, 'UNAVAILABLE', 0, '', 0, '', '', ?)
                """, (
                    str(m["match_id"]), m.get("home_team", ""), m.get("away_team", ""),
                    m.get("league", ""), m.get("event_date", ""),
                    m.get("scheduled_utc_ts", 0), m.get("scheduled_utc", ""),
                    m.get("status_type", ""), now
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

def update_bet_log_result(log_id: int, result: str) -> None:
    with get_db_connection() as conn:
        with conn:
            conn.execute("UPDATE bet_monitor_log_v3 SET result = ? WHERE id = ?", (result, log_id))

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
                        real_winner = "push"
                    elif q4h > q4a:
                        real_winner = "home"
                    else:
                        real_winner = "away"
                        
                    if real_winner == "push":
                        outcome = "push"
                    elif picked_side.lower() == real_winner:
                        outcome = "win"
                    else:
                        outcome = "loss"
                        
                    conn.execute("UPDATE bet_monitor_log_v3 SET result = ? WHERE id = ?", (outcome, log_id))
                    log_info("DATABASE", f"Apuesta liquidada {match_id} ({model_ver}): {outcome.upper()}")
