"""
SQLite persistence layer for SofaScore match data.

Schema:
  matches        – one row per match (metadata + final score)
  quarter_scores – score per quarter, FK → matches
  play_by_play   – every scoring play, FK → matches
    graph_points   – match pressure/momentum graph points, FK → matches
    discovered_ft_matches – finished match IDs discovered by date crawl
    backfill_state – key/value state checkpoints for resumable jobs
        eval_match_results – per-date per-match eval outputs (+ dynamic model columns)
"""

import re
import sqlite3


def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            match_id    TEXT PRIMARY KEY,
            home_team   TEXT NOT NULL,
            away_team   TEXT NOT NULL,
            home_slug   TEXT,
            away_slug   TEXT,
            event_slug  TEXT,
            custom_id   TEXT,
            status_type TEXT,
            status_description TEXT,
            date        TEXT NOT NULL,
            time        TEXT NOT NULL,
            venue       TEXT,
            league      TEXT,
            home_record TEXT,
            away_record TEXT,
            home_score  INTEGER,
            away_score  INTEGER
        );

        CREATE TABLE IF NOT EXISTS quarter_scores (
            match_id TEXT NOT NULL,
            quarter  TEXT NOT NULL,
            home     INTEGER,
            away     INTEGER,
            PRIMARY KEY (match_id, quarter),
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        CREATE TABLE IF NOT EXISTS play_by_play (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id   TEXT NOT NULL,
            quarter    TEXT NOT NULL,
            seq        INTEGER NOT NULL,
            time       TEXT,
            player     TEXT,
            points     INTEGER,
            team       TEXT,
            home_score INTEGER,
            away_score INTEGER,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        CREATE TABLE IF NOT EXISTS graph_points (
            match_id TEXT NOT NULL,
            seq      INTEGER NOT NULL,
            minute   INTEGER NOT NULL,
            value    INTEGER NOT NULL,
            PRIMARY KEY (match_id, seq),
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        CREATE TABLE IF NOT EXISTS discovered_ft_matches (
            match_id     TEXT PRIMARY KEY,
            event_date   TEXT NOT NULL,
            status_type  TEXT,
            home_team    TEXT,
            away_team    TEXT,
            league       TEXT,
            collected_at TEXT NOT NULL DEFAULT (datetime('now')),
            processed    INTEGER NOT NULL DEFAULT 0,
            processed_at TEXT,
            last_error   TEXT
        );

        CREATE TABLE IF NOT EXISTS backfill_state (
            state_key  TEXT PRIMARY KEY,
            state_value TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

            CREATE TABLE IF NOT EXISTS eval_match_results (
                event_date      TEXT NOT NULL,
                match_id        TEXT NOT NULL,
                home_team       TEXT,
                away_team       TEXT,
                q3_home_score   INTEGER,
                q3_away_score   INTEGER,
                q3_winner       TEXT,
                q4_home_score   INTEGER,
                q4_away_score   INTEGER,
                q4_winner       TEXT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (event_date, match_id)
            );
    """)
    _ensure_match_columns(conn)
    conn.commit()


def _ensure_match_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(matches)").fetchall()
    }
    for col_name in (
        "home_slug",
        "away_slug",
        "event_slug",
        "custom_id",
        "status_type",
        "status_description",
    ):
        if col_name in existing:
            continue
        conn.execute(
            f"ALTER TABLE matches ADD COLUMN {_quote_ident(col_name)} TEXT"
        )
    conn.commit()


def _winner_from_scores(home: int | None, away: int | None) -> str | None:
    if home is None or away is None:
        return None
    if home == away:
        return "push"
    return "home" if home > away else "away"


def _sanitize_result_tag(tag: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]+", "_", (tag or "").strip().lower())
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = "default"
    if cleaned[0].isdigit():
        cleaned = f"m_{cleaned}"
    return cleaned[:48]


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _ensure_eval_result_columns(
    conn: sqlite3.Connection,
    safe_tag: str,
) -> dict[str, str]:
    col_map = {
        "q3_pick": f"q3_pick__{safe_tag}",
        "q3_signal": f"q3_signal__{safe_tag}",
        "q3_outcome": f"q3_outcome__{safe_tag}",
        "q3_available": f"q3_available__{safe_tag}",
        "q3_confidence": f"q3_confidence__{safe_tag}",
        "q3_threshold_lean": f"q3_threshold_lean__{safe_tag}",
        "q3_threshold_bet": f"q3_threshold_bet__{safe_tag}",
        "q4_pick": f"q4_pick__{safe_tag}",
        "q4_signal": f"q4_signal__{safe_tag}",
        "q4_outcome": f"q4_outcome__{safe_tag}",
        "q4_available": f"q4_available__{safe_tag}",
        "q4_confidence": f"q4_confidence__{safe_tag}",
        "q4_threshold_lean": f"q4_threshold_lean__{safe_tag}",
        "q4_threshold_bet": f"q4_threshold_bet__{safe_tag}",
    }
    column_types = {
        col_map["q3_pick"]: "TEXT",
        col_map["q3_signal"]: "TEXT",
        col_map["q3_outcome"]: "TEXT",
        col_map["q3_available"]: "INTEGER",
        col_map["q3_confidence"]: "REAL",
        col_map["q3_threshold_lean"]: "REAL",
        col_map["q3_threshold_bet"]: "REAL",
        col_map["q4_pick"]: "TEXT",
        col_map["q4_signal"]: "TEXT",
        col_map["q4_outcome"]: "TEXT",
        col_map["q4_available"]: "INTEGER",
        col_map["q4_confidence"]: "REAL",
        col_map["q4_threshold_lean"]: "REAL",
        col_map["q4_threshold_bet"]: "REAL",
    }

    existing = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(eval_match_results)").fetchall()
    }
    for col_name, col_type in column_types.items():
        if col_name in existing:
            continue
        conn.execute(
            f"ALTER TABLE eval_match_results ADD COLUMN {_quote_ident(col_name)} {col_type}"
        )
    conn.commit()
    return col_map


def save_eval_match_result(
    conn: sqlite3.Connection,
    *,
    event_date: str,
    match_id: str,
    home_team: str,
    away_team: str,
    q3_home_score: int | None,
    q3_away_score: int | None,
    q4_home_score: int | None,
    q4_away_score: int | None,
    result_tag: str,
    predictions: dict,
) -> str:
    safe_tag = _sanitize_result_tag(result_tag)
    col_map = _ensure_eval_result_columns(conn, safe_tag)

    pred_q3 = (predictions or {}).get("q3", {})
    pred_q4 = (predictions or {}).get("q4", {})

    def pred_fields(pred: dict) -> tuple[str | None, str, str, int, float | None, float | None, float | None]:
        available = 1 if pred.get("available") else 0
        if available:
            pick = pred.get("predicted_winner")
            signal = (
                pred.get("final_recommendation")
                or pred.get("bet_signal")
                or "NO_BET"
            )
            outcome = str(pred.get("result", "pending") or "pending")
            confidence = pred.get("confidence")
            threshold_lean = pred.get("threshold_lean")
            threshold_bet = pred.get("threshold_bet")
        else:
            pick = None
            signal = "NO_BET"
            outcome = str(pred.get("reason", "unavailable") or "unavailable")
            confidence = None
            threshold_lean = None
            threshold_bet = None
        return pick, signal, outcome, available, confidence, threshold_lean, threshold_bet

    q3_pick, q3_signal, q3_outcome, q3_available, q3_confidence, q3_threshold_lean, q3_threshold_bet = pred_fields(pred_q3)
    q4_pick, q4_signal, q4_outcome, q4_available, q4_confidence, q4_threshold_lean, q4_threshold_bet = pred_fields(pred_q4)

    q3_winner = _winner_from_scores(q3_home_score, q3_away_score)
    q4_winner = _winner_from_scores(q4_home_score, q4_away_score)

    row_values = {
        "event_date": event_date,
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "q3_home_score": q3_home_score,
        "q3_away_score": q3_away_score,
        "q3_winner": q3_winner,
        "q4_home_score": q4_home_score,
        "q4_away_score": q4_away_score,
        "q4_winner": q4_winner,
        col_map["q3_pick"]: q3_pick,
        col_map["q3_signal"]: q3_signal,
        col_map["q3_outcome"]: q3_outcome,
        col_map["q3_available"]: q3_available,
        col_map["q3_confidence"]: q3_confidence,
        col_map["q3_threshold_lean"]: q3_threshold_lean,
        col_map["q3_threshold_bet"]: q3_threshold_bet,
        col_map["q4_pick"]: q4_pick,
        col_map["q4_signal"]: q4_signal,
        col_map["q4_outcome"]: q4_outcome,
        col_map["q4_available"]: q4_available,
        col_map["q4_confidence"]: q4_confidence,
        col_map["q4_threshold_lean"]: q4_threshold_lean,
        col_map["q4_threshold_bet"]: q4_threshold_bet,
    }

    all_cols = list(row_values.keys())
    insert_cols = ", ".join(_quote_ident(c) for c in all_cols)
    placeholders = ", ".join("?" for _ in all_cols)

    update_cols = [c for c in all_cols if c not in ("event_date", "match_id")]
    update_clause = ", ".join(
        f"{_quote_ident(c)} = excluded.{_quote_ident(c)}"
        for c in update_cols
    )
    update_clause += ", updated_at = datetime('now')"

    sql = (
        f"INSERT INTO eval_match_results ({insert_cols}) "
        f"VALUES ({placeholders}) "
        "ON CONFLICT(event_date, match_id) DO UPDATE SET "
        f"{update_clause}"
    )

    conn.execute(sql, [row_values[c] for c in all_cols])
    conn.commit()
    return safe_tag


def save_match(conn: sqlite3.Connection, match_id: str, data: dict) -> None:
    """Upsert a full match (metadata + quarters + play-by-play + graph)."""
    m = data["match"]
    s = data["score"]

    _ensure_match_columns(conn)

    conn.execute(
        """
        INSERT OR REPLACE INTO matches
                    (match_id, home_team, away_team, home_slug, away_slug, event_slug, custom_id, status_type, status_description, date, time, venue, league,
           home_record, away_record, home_score, away_score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            match_id,
            m["home_team"], m["away_team"],
            m.get("home_slug", "unknown"), m.get("away_slug", "unknown"),
                        m.get("event_slug", "unknown"), m.get("custom_id", ""),
                        m.get("status_type", ""), m.get("status_description", ""),
            m["date"], m["time"],
            m.get("venue", ""), m.get("league", ""),
            m.get("home_record", ""), m.get("away_record", ""),
            s["home"], s["away"],
        ),
    )

    for quarter, scores in s.get("quarters", {}).items():
        conn.execute(
            "INSERT OR REPLACE INTO quarter_scores (match_id, quarter, home, away) VALUES (?,?,?,?)",
            (match_id, quarter, scores["home"], scores["away"]),
        )

    # Full replace for play-by-play to avoid duplicates on re-scrape
    conn.execute("DELETE FROM play_by_play WHERE match_id = ?", (match_id,))
    for quarter, plays in data.get("play_by_play", {}).items():
        for seq, play in enumerate(plays):
            conn.execute(
                """
                INSERT INTO play_by_play
                  (match_id, quarter, seq, time, player, points, team, home_score, away_score)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                (
                    match_id, quarter, seq,
                    play.get("time"), play.get("player"), play.get("points"),
                    play.get("team"), play.get("home_score"), play.get("away_score"),
                ),
            )

    # Full replace for graph points to avoid duplicates on re-scrape
    conn.execute("DELETE FROM graph_points WHERE match_id = ?", (match_id,))
    for seq, point in enumerate(data.get("graph_points", [])):
        conn.execute(
            """
            INSERT INTO graph_points (match_id, seq, minute, value)
            VALUES (?,?,?,?)
            """,
            (
                match_id,
                seq,
                int(point.get("minute", 0)),
                int(point.get("value", 0)),
            ),
        )

    conn.commit()


def get_match(conn: sqlite3.Connection, match_id: str) -> dict | None:
    """Reconstruct the full match dict from the DB (same shape as scraper output)."""
    _ensure_match_columns(conn)
    row = conn.execute(
        "SELECT * FROM matches WHERE match_id = ?", (match_id,)
    ).fetchone()
    if not row:
        return None

    quarters: dict[str, dict] = {}
    for qr in conn.execute(
        "SELECT quarter, home, away FROM quarter_scores WHERE match_id = ? ORDER BY quarter",
        (match_id,),
    ):
        quarters[qr["quarter"]] = {"home": qr["home"], "away": qr["away"]}

    pbp: dict[str, list] = {}
    for pr in conn.execute(
        "SELECT quarter, time, player, points, team, home_score, away_score "
        "FROM play_by_play WHERE match_id = ? ORDER BY quarter, seq",
        (match_id,),
    ):
        pbp.setdefault(pr["quarter"], []).append({
            "time": pr["time"],
            "player": pr["player"],
            "points": pr["points"],
            "team": pr["team"],
            "home_score": pr["home_score"],
            "away_score": pr["away_score"],
        })

    graph_points: list[dict] = []
    for gr in conn.execute(
        "SELECT minute, value FROM graph_points WHERE match_id = ? ORDER BY seq",
        (match_id,),
    ):
        graph_points.append({"minute": gr["minute"], "value": gr["value"]})

    return {
        "match_id": match_id,
        "match": {
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "home_slug": row["home_slug"] or "unknown",
            "away_slug": row["away_slug"] or "unknown",
            "event_slug": row["event_slug"] or "unknown",
            "custom_id": row["custom_id"] or "",
            "status_type": row["status_type"] or "",
            "status_description": row["status_description"] or "",
            "date": row["date"],
            "time": row["time"],
            "venue": row["venue"],
            "league": row["league"],
            "home_record": row["home_record"],
            "away_record": row["away_record"],
        },
        "score": {
            "home": row["home_score"],
            "away": row["away_score"],
            "quarters": quarters,
        },
        "play_by_play": pbp,
        "graph_points": graph_points,
    }


def list_matches(conn: sqlite3.Connection) -> list:
    """Return a summary list of all stored matches."""
    rows = conn.execute(
        """
        SELECT match_id, home_team, away_team, date, time,
               home_score, away_score, league, venue
        FROM matches
        ORDER BY date DESC, time DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def save_discovered_ft_matches(conn: sqlite3.Connection, rows: list[dict]) -> int:
    """Upsert discovered finished match IDs. Returns number of input rows."""
    for row in rows:
        conn.execute(
            """
            INSERT INTO discovered_ft_matches
              (match_id, event_date, status_type, home_team, away_team, league,
               collected_at, processed, processed_at, last_error)
            VALUES (?, ?, ?, ?, ?, ?, datetime('now'), 0, NULL, NULL)
            ON CONFLICT(match_id) DO UPDATE SET
              event_date = excluded.event_date,
              status_type = excluded.status_type,
              home_team = excluded.home_team,
              away_team = excluded.away_team,
              league = excluded.league,
              collected_at = datetime('now')
            """,
            (
                row.get("match_id"),
                row.get("event_date"),
                row.get("status_type"),
                row.get("home_team"),
                row.get("away_team"),
                row.get("league"),
            ),
        )
    conn.commit()
    return len(rows)


def list_pending_discovered_ft(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    limit: int | None = None,
) -> list[dict]:
    """List discovered finished matches pending individual ingestion."""
    sql = (
        "SELECT * FROM discovered_ft_matches "
        "WHERE processed = 0 AND event_date BETWEEN ? AND ? "
        "AND UPPER(COALESCE(last_error, '')) NOT LIKE '%HTTP 404%' "
        "ORDER BY event_date DESC, match_id DESC"
    )
    params: list = [date_from, date_to]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(sql, tuple(params)).fetchall()
    return [dict(r) for r in rows]


def mark_discovered_processed(conn: sqlite3.Connection, match_id: str) -> None:
    conn.execute(
        """
        UPDATE discovered_ft_matches
        SET processed = 1,
            processed_at = datetime('now'),
            last_error = NULL
        WHERE match_id = ?
        """,
        (match_id,),
    )
    conn.commit()


def mark_discovered_error(
    conn: sqlite3.Connection,
    match_id: str,
    error_text: str,
) -> None:
    err = (error_text or "")[:1000]
    upper_err = err.upper()
    is_non_retryable_404 = "HTTP 404" in upper_err

    if is_non_retryable_404:
        conn.execute(
            """
            UPDATE discovered_ft_matches
            SET processed = 1,
                processed_at = datetime('now'),
                last_error = ?
            WHERE match_id = ?
            """,
            (err, match_id),
        )
        conn.commit()
        return

    conn.execute(
        """
        UPDATE discovered_ft_matches
        SET processed = 0,
            processed_at = NULL,
            last_error = ?
        WHERE match_id = ?
        """,
        (err, match_id),
    )
    conn.commit()


def mark_http_404_errors_processed(conn: sqlite3.Connection) -> int:
    """Mark legacy pending rows with HTTP 404 errors as processed/non-retryable."""
    cur = conn.execute(
        """
        UPDATE discovered_ft_matches
        SET processed = 1,
            processed_at = datetime('now')
        WHERE processed = 0
          AND UPPER(COALESCE(last_error, '')) LIKE '%HTTP 404%'
        """
    )
    conn.commit()
    return int(cur.rowcount or 0)


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute(
        "SELECT state_value FROM backfill_state WHERE state_key = ?",
        (key,),
    ).fetchone()
    if not row:
        return None
    return row["state_value"]


def set_state(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO backfill_state (state_key, state_value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(state_key) DO UPDATE SET
          state_value = excluded.state_value,
          updated_at = datetime('now')
        """,
        (key, value),
    )
    conn.commit()


def get_discovered_stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute(
        "SELECT COUNT(*) AS n FROM discovered_ft_matches"
    ).fetchone()["n"]
    processed = conn.execute(
        "SELECT COUNT(*) AS n FROM discovered_ft_matches WHERE processed = 1"
    ).fetchone()["n"]
    pending = conn.execute(
        "SELECT COUNT(*) AS n FROM discovered_ft_matches WHERE processed = 0"
    ).fetchone()["n"]
    with_error = conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM discovered_ft_matches
        WHERE processed = 0
          AND COALESCE(last_error, '') <> ''
        """
    ).fetchone()["n"]

    bounds = conn.execute(
        "SELECT MIN(event_date) AS min_date, MAX(event_date) AS max_date "
        "FROM discovered_ft_matches"
    ).fetchone()

    return {
        "total": int(total),
        "processed": int(processed),
        "pending": int(pending),
        "pending_with_error": int(with_error),
        "min_date": bounds["min_date"],
        "max_date": bounds["max_date"],
    }
