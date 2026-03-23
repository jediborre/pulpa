"""
SQLite persistence layer for SofaScore match data.

Schema:
  matches        – one row per match (metadata + final score)
  quarter_scores – score per quarter, FK → matches
  play_by_play   – every scoring play, FK → matches
    graph_points   – match pressure/momentum graph points, FK → matches
    discovered_ft_matches – finished match IDs discovered by date crawl
    backfill_state – key/value state checkpoints for resumable jobs
"""

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
    """)
    conn.commit()


def save_match(conn: sqlite3.Connection, match_id: str, data: dict) -> None:
    """Upsert a full match (metadata + quarters + play-by-play + graph)."""
    m = data["match"]
    s = data["score"]

    conn.execute(
        """
        INSERT OR REPLACE INTO matches
          (match_id, home_team, away_team, date, time, venue, league,
           home_record, away_record, home_score, away_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            match_id,
            m["home_team"], m["away_team"],
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
    conn.execute(
        """
        UPDATE discovered_ft_matches
        SET processed = 0,
            processed_at = NULL,
            last_error = ?
        WHERE match_id = ?
        """,
        (error_text[:1000], match_id),
    )
    conn.commit()


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
