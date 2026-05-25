# =====================================================================
# REGLAS DE ARQUITECTURA, FORMATEO Y LOGGING DE PRODUCCIÓN (MANTENER):
# 1. ESTRUCTURA: Estrictamente modularizado (config, database, scrapers, 
#    models, notifications, utils) coordinados asíncronamente por main.py.
#    Cualquier aproximación monolítica de archivo único viola esta especificación.
# 2. INFRAESTRUCTURA DB: El archivo base SQLite se localiza exclusivamente en 
#    /match/matches.db y todas las tablas sin excepción finalizan con el sufijo '_v2'.
# 3. TABLA DE LOGS: 'bet_monitor_log_v2' se particiona por modelo y contiene 
#    obligatoriamente los campos 'raw_json' (TEXT), 'inference_minute' (INT), 
#    y 'graph_points_count' (INT) junto con marcadores reales del juego.
# 4. CONFIGURACIÓN DE LIGAS: Prohibido hardcodear filtros o patrones de texto 
#    en las consultas SQL o lógica directa. Debe consumirse declarativamente 
#    desde config/leagues.yaml o cargarse dinámicamente desde la BD SQLite.
# 5. Formato Log: {Fecha Hora} [INFO/WARNING/ERROR] [COMPONENTE]
#    - Colores ANSI: INFO=Azul, WARNING=Amarillo, ERROR=Rojo.
# 6. Formato Matches en Log: {horario_match} {match_id} {home} vs {away}
#    - horario_match en Amarillo, match_id en Azul (sin texto UTC-6).
# 7. Errores de red críticos: Imprimir explícitamente "HTTP 403/404" en ROJO.
# 8. Monitoreo avanzado en progreso de cuarto final: Usar obligatoriamente "Q4 🟠".
# 9. Telegram Prefijos de Apuestas: 🟢 (Bettable), 🟡 (No Bettable), ⚪ (Tardía).
# 10. Telegram Resultados FT: Prefijar con ✅ (Ganada) o ❌ (Perdida) manteniendo emoji base.
# 11. Conversión de tiempos siempre legibles en formato humano (ej. 1 dia 2h 15min / 45s).
# =====================================================================

import json
import yaml
import sqlite3
from datetime import datetime
from pathlib import Path
from bet_monitor_v2.database.connection import get_db_connection
from bet_monitor_v2.config.constants import LEAGUES_CONFIG_PATH

def init_tables() -> None:
    """
    Crea las tablas con el sufijo '_v2' y sus índices asociados de forma transaccional.
    """
    with get_db_connection() as conn:
        with conn:
            # 1. bet_monitor_schedule_v2
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bet_monitor_schedule_v2 (
                    match_id TEXT PRIMARY KEY,
                    home_team TEXT,
                    away_team TEXT,
                    league TEXT,
                    event_date TEXT,
                    scheduled_utc_ts INTEGER,
                    scheduled_utc TEXT,
                    status_type TEXT,
                    status TEXT,
                    q4_checked INTEGER,
                    q4_signal TEXT,
                    q4_notified INTEGER,
                    q4_model TEXT,
                    final_fetched INTEGER,
                    final_fetch_at TEXT,
                    skip_reason TEXT,
                    updated_at TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_schedule_date_status_v2 ON bet_monitor_schedule_v2 (event_date, status);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_schedule_utc_ts_v2 ON bet_monitor_schedule_v2 (scheduled_utc_ts);")

            # 2. bet_monitor_log_v2
            conn.execute("""
                CREATE TABLE IF NOT EXISTS bet_monitor_log_v2 (
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
                    inference_json TEXT
                );
            """)
            try:
                conn.execute("ALTER TABLE bet_monitor_log_v2 ADD COLUMN inference_json TEXT;")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_match_model_v2 ON bet_monitor_log_v2 (match_id, model_version);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_log_result_v2 ON bet_monitor_log_v2 (result);")

            # 3. eval_match_results_v2
            conn.execute("""
                CREATE TABLE IF NOT EXISTS eval_match_results_v2 (
                    match_id TEXT PRIMARY KEY,
                    available INTEGER
                );
            """)

            # 4. quarter_scores_v2
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

            # 5. leagues_config_v2
            conn.execute("""
                CREATE TABLE IF NOT EXISTS leagues_config_v2 (
                    league_name_pattern TEXT PRIMARY KEY,
                    filter_mode TEXT,
                    updated_at TEXT
                );
            """)


def ensure_eval_match_results_columns(conn: sqlite3.Connection, model_versions: list[str]) -> None:
    """
    Asegura mediante introspección que existan las columnas de la tabla matricial
    eval_match_results_v2 para cada versión de modelo especificada.
    """
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(eval_match_results_v2);")
    columns = {row["name"] for row in cursor.fetchall()}

    for version in model_versions:
        for suffix in ["_signal", "_pick", "_confidence", "_outcome"]:
            col_name = f"q4{suffix}__{version}"
            if col_name not in columns:
                conn.execute(f"ALTER TABLE eval_match_results_v2 ADD COLUMN {col_name} TEXT;")


def sync_leagues_config() -> None:
    """
    Sincroniza la configuración declarativa del archivo leagues.yaml
    con el espejo transaccional de la tabla leagues_config_v2.
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
                # Primero limpiamos la tabla para sincronizar
                conn.execute("DELETE FROM leagues_config_v2;")
                
                for p in excluded:
                    conn.execute("""
                        INSERT OR REPLACE INTO leagues_config_v2 (league_name_pattern, filter_mode, updated_at)
                        VALUES (?, 'EXCLUDE', ?);
                    """, (p, now))
                    
                for p in ft_only:
                    conn.execute("""
                        INSERT OR REPLACE INTO leagues_config_v2 (league_name_pattern, filter_mode, updated_at)
                        VALUES (?, 'FT_ONLY', ?);
                    """, (p, now))
    except Exception as e:
        # Silenciamos o dejamos pasar el error si hay fallo al leer yaml en el arranque
        pass


def save_schedule_matches(matches: list[dict]) -> None:
    """
    Guarda los partidos de la programación diaria de forma atómica.
    """
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        with conn:
            for m in matches:
                conn.execute("""
                    INSERT OR IGNORE INTO bet_monitor_schedule_v2 (
                        match_id, home_team, away_team, league, event_date,
                        scheduled_utc_ts, scheduled_utc, status_type, status,
                        q4_checked, q4_signal, q4_notified, q4_model,
                        final_fetched, final_fetch_at, skip_reason, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, 'UNAVAILABLE', 0, '', 0, '', '', ?)
                """, (
                    str(m["match_id"]), m["home_team"], m["away_team"], m["league"], m["event_date"],
                    m["scheduled_utc_ts"], m["scheduled_utc"], m["status_type"], now
                ))


def update_schedule_status(match_id: str, status: str, skip_reason: str = "") -> None:
    """
    Actualiza el estado y el motivo de descarte de un partido programado.
    """
    now = datetime.now().isoformat()
    with get_db_connection() as conn:
        with conn:
            conn.execute("""
                UPDATE bet_monitor_schedule_v2
                SET status = ?, skip_reason = ?, updated_at = ?
                WHERE match_id = ?
            """, (status, skip_reason, now, match_id))


def get_schedule_matches(event_date: str) -> list[dict]:
    """
    Obtiene los partidos programados para una fecha específica.
    """
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM bet_monitor_schedule_v2
            WHERE event_date = ?
        """, (event_date,))
        return [dict(r) for r in cursor.fetchall()]


def get_pending_schedule_matches() -> list[dict]:
    """
    Obtiene los partidos programados con estado 'pending' o 'in_progress'.
    """
    with get_db_connection() as conn:
        cursor = conn.execute("""
            SELECT * FROM bet_monitor_schedule_v2
            WHERE status IN ('pending', 'in_progress')
        """)
        return [dict(r) for r in cursor.fetchall()]


def save_quarter_scores(match_id: str, scores: dict) -> None:
    """
    Guarda los marcadores por cuarto de un partido.
    """
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
    """
    Obtiene los marcadores por cuarto para un match_id.
    """
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id = ?", (match_id,)).fetchone()
        return dict(row) if row else None


def save_bet_log(
    match_id: str, model_version: str, inference_minute: int, graph_points_count: int,
    raw_json: dict, signal_type: str, picked_side: str, confidence: float,
    actual_home_score: int, actual_away_score: int, result: str = "pending",
    inference_json: dict | None = None
) -> int:
    """
    Registra una inferencia en la tabla bet_monitor_log_v2 de forma persistente.
    """
    now = datetime.now().isoformat()
    raw_json_str = json.dumps(raw_json, ensure_ascii=False)
    inf_json_str = json.dumps(inference_json, ensure_ascii=False) if inference_json is not None else None
    
    with get_db_connection() as conn:
        with conn:
            cursor = conn.execute("""
                INSERT INTO bet_monitor_log_v2 (
                    match_id, model_version, target_quarter, inference_minute, graph_points_count,
                    raw_json, signal_type, picked_side, confidence,
                    actual_home_score, actual_away_score, result, created_at, inference_json
                ) VALUES (?, ?, 4, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                match_id, model_version, inference_minute, graph_points_count,
                raw_json_str, signal_type, picked_side, confidence,
                actual_home_score, actual_away_score, result, now, inf_json_str
            ))
            return cursor.lastrowid


def update_bet_log_result(log_id: int, result: str) -> None:
    """
    Actualiza el resultado (win, loss, push) de un registro de log de apuestas.
    """
    with get_db_connection() as conn:
        with conn:
            conn.execute("UPDATE bet_monitor_log_v2 SET result = ? WHERE id = ?", (result, log_id))


def _save_eval_match_results_impl(conn: sqlite3.Connection, match_id: str, available: int, metrics: dict) -> None:
    # Construir query dinámico
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
        INSERT OR REPLACE INTO eval_match_results_v2 ({col_str})
        VALUES ({placeholders})
    """, values)


def save_eval_match_results(match_id: str, available: int, metrics: dict, conn: sqlite3.Connection = None) -> None:
    """
    Guarda los resultados del modelo de forma transaccional, expandiendo dinámicamente las columnas.
    """
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
    Rutina sincronizada ejecutada en el arranque que inspecciona logs 'pending'
    en bet_monitor_log_v2 y los liquida contra quarter_scores_v2.
    """
    with get_db_connection() as conn:
        with conn:
            # 1. Traer todos los logs pendientes
            cursor = conn.execute("""
                SELECT l.id, l.match_id, l.picked_side, l.model_version, l.confidence, l.signal_type
                FROM bet_monitor_log_v2 l
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
                
                # Buscar marcador del Q4 en quarter_scores_v2
                qs = conn.execute("""
                    SELECT q4_home, q4_away FROM quarter_scores_v2
                    WHERE match_id = ?
                """, (match_id,)).fetchone()
                
                if qs and qs["q4_home"] is not None and qs["q4_away"] is not None:
                    q4h = qs["q4_home"]
                    q4a = qs["q4_away"]
                    
                    # Determinar ganador real
                    if q4h == q4a:
                        real_winner = "push"
                    elif q4h > q4a:
                        real_winner = "home"
                    else:
                        real_winner = "away"
                        
                    # Computar resultado del pick
                    if real_winner == "push":
                        outcome = "push"
                    elif picked_side.lower() == real_winner:
                        outcome = "win"
                    else:
                        outcome = "loss"
                        
                    # Actualizar log
                    conn.execute("UPDATE bet_monitor_log_v2 SET result = ? WHERE id = ?", (outcome, log_id))
                    
                    # Actualizar eval_match_results_v2 dinámico si corresponde
                    metrics = {
                        model_ver: {
                            "signal": sig_type,
                            "pick": picked_side.upper(),
                            "confidence": confidence,
                            "outcome": outcome
                        }
                    }
                    ensure_eval_match_results_columns(conn, [model_ver])
                    
                    # Cargar datos anteriores para no pisar
                    old_eval = conn.execute("SELECT * FROM eval_match_results_v2 WHERE match_id = ?", (match_id,)).fetchone()
                    if old_eval:
                        old_dict = dict(old_eval)
                        # Agregar las otras métricas que ya estaban
                        for col, val in old_dict.items():
                            if col not in ["match_id", "available"] and val is not None:
                                # Extraer versión de la columna
                                if "__" in col:
                                    col_ver = col.split("__")[1]
                                    if col_ver != model_ver:
                                        if col_ver not in metrics:
                                            metrics[col_ver] = {}
                                        if "_signal" in col: metrics[col_ver]["signal"] = val
                                        elif "_pick" in col: metrics[col_ver]["pick"] = val
                                        elif "_confidence" in col: metrics[col_ver]["confidence"] = val
                                        elif "_outcome" in col: metrics[col_ver]["outcome"] = val
                                        
                    save_eval_match_results(match_id, available=1, metrics=metrics, conn=conn)
