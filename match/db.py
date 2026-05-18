"""
SQLite persistence layer for SofaScore match data.

Schema:
  matches              – one row per match (metadata + final score)
  quarter_scores       – score per quarter, FK → matches
  play_by_play         – scoring plays only (backward compat), FK → matches
   match_events         – ALL incidents with classification (scoring plays + period markers only in basketball), FK → matches
  match_h2h            – head-to-head history from /event/{id}/h2h, FK → matches
  player_stats         – per-player statistics (TODO: confirm endpoint), FK → matches
  lineups              – starting lineups and substitutions (TODO: confirm endpoint), FK → matches
  match_odds           – betting odds / handicap (TODO: confirm endpoint), FK → matches
  graph_points         – match pressure/momentum graph points, FK → matches
  discovered_ft_matches – finished match IDs discovered by date crawl
  backfill_state       – key/value state checkpoints for resumable jobs
  eval_match_results   – per-date per-match eval outputs (+ dynamic model columns)
"""

import re
import sqlite3


def get_conn(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
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

        CREATE INDEX IF NOT EXISTS idx_pbp_match_id
            ON play_by_play (match_id);
        CREATE INDEX IF NOT EXISTS idx_pbp_match_quarter
            ON play_by_play (match_id, quarter, seq);
        CREATE INDEX IF NOT EXISTS idx_graph_points_match_id
            ON graph_points (match_id);
        CREATE INDEX IF NOT EXISTS idx_quarter_scores_match_id
            ON quarter_scores (match_id);
        CREATE INDEX IF NOT EXISTS idx_discovered_event_date
            ON discovered_ft_matches (event_date);
        CREATE INDEX IF NOT EXISTS idx_discovered_event_date_processed
            ON discovered_ft_matches (event_date, processed);
        CREATE INDEX IF NOT EXISTS idx_matches_date
            ON matches (date);

        -- All incidents (scoring + non-scoring) with classification
        CREATE TABLE IF NOT EXISTS match_events (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id      TEXT NOT NULL,
            quarter       TEXT NOT NULL,
            seq           INTEGER NOT NULL,
            time          TEXT,
            time_seconds  INTEGER,
            incident_type TEXT NOT NULL,
            subtype       TEXT,
            player        TEXT,
            player_id     TEXT,
            team          TEXT,
            points        INTEGER,
            home_score    INTEGER,
            away_score    INTEGER,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        -- Head-to-head history per match
        CREATE TABLE IF NOT EXISTS match_h2h (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id    TEXT NOT NULL,
            h2h_match_id TEXT NOT NULL,
            date        TEXT NOT NULL,
            timestamp   INTEGER,
            home_team   TEXT,
            away_team   TEXT,
            home_score  INTEGER,
            away_score  INTEGER,
            q1_home     INTEGER, q1_away INTEGER,
            q2_home     INTEGER, q2_away INTEGER,
            q3_home     INTEGER, q3_away INTEGER,
            q4_home     INTEGER, q4_away INTEGER,
            tournament  TEXT,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        -- Per-player statistics (SofaScore rating, pts, fouls, mins, +/-)
        CREATE TABLE IF NOT EXISTS player_stats (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id        TEXT NOT NULL,
            team            TEXT NOT NULL,
            player_name     TEXT NOT NULL,
            player_id       TEXT,
            sofascore_rating REAL,
            minutes_played  INTEGER,
            points          INTEGER,
            fouls           INTEGER,
            plus_minus      INTEGER,
            field_goals_made    INTEGER,
            field_goals_attempted INTEGER,
            three_made      INTEGER,
            three_attempted INTEGER,
            free_throws_made    INTEGER,
            free_throws_attempted INTEGER,
            rebounds        INTEGER,
            assists         INTEGER,
            steals          INTEGER,
            turnovers       INTEGER,
            blocks          INTEGER,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        -- Lineups / starting players per match
        CREATE TABLE IF NOT EXISTS lineups (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id    TEXT NOT NULL,
            team        TEXT NOT NULL,
            player_name TEXT NOT NULL,
            player_id   TEXT,
            is_starter  INTEGER NOT NULL DEFAULT 0,
            shirt_number INTEGER,
            position    TEXT,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        -- Team-level statistics (2PT%, 3PT%, rebounds, fouls, timeouts, etc.)
        CREATE TABLE IF NOT EXISTS team_statistics (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id     TEXT NOT NULL,
            period       TEXT NOT NULL DEFAULT 'ALL',
            group_name   TEXT,
            stat_key     TEXT NOT NULL,
            stat_name    TEXT,
            home_value   REAL,
            away_value   REAL,
            home_total   REAL,
            away_total   REAL,
            home_display TEXT,
            away_display TEXT,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        -- Betting odds / handicap
        CREATE TABLE IF NOT EXISTS match_odds (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id      TEXT NOT NULL,
            odds_type     TEXT NOT NULL,
            market_name   TEXT,
            market_period TEXT,
            home_value    REAL,
            away_value    REAL,
            draw_value    REAL,
            timestamp     TEXT,
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );

        CREATE TABLE IF NOT EXISTS settings (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        -- Team strength snapshot (pregameForm + performance points)
        CREATE TABLE IF NOT EXISTS team_strength (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            team_id      INTEGER NOT NULL,
            team_name    TEXT NOT NULL,
            match_id     TEXT NOT NULL,
            position     INTEGER,
            wins         INTEGER,
            losses       INTEGER,
            form         TEXT,
            perf_points  TEXT,
            fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (match_id) REFERENCES matches(match_id)
        );
        CREATE INDEX IF NOT EXISTS idx_team_strength_team
            ON team_strength (team_id);
        CREATE INDEX IF NOT EXISTS idx_team_strength_match
            ON team_strength (match_id);
    """)
    _ensure_match_columns(conn)
    _ensure_h2h_columns(conn)
    conn.commit()


def get_setting(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO settings (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
        """,
        (key, value),
    )
    conn.commit()


def _ensure_match_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(matches)").fetchall()
    }
    # Also migrate match_odds columns
    odds_existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(match_odds)").fetchall()
    }
    for col_name in ("market_name", "market_period"):
        if col_name not in odds_existing:
            try:
                conn.execute(f"ALTER TABLE match_odds ADD COLUMN {col_name} TEXT")
            except Exception:
                pass

    for col_name in (
        "home_slug",
        "away_slug",
        "event_slug",
        "custom_id",
        "status_type",
        "status_description",
        "home_team_id",
        "away_team_id",
        "home_rating",
        "away_rating",
    ):
        if col_name in existing:
            continue
        if col_name.endswith("_id"):
            col_type = "INTEGER"
        elif col_name.endswith("_rating"):
            col_type = "REAL"
        else:
            col_type = "TEXT"
        conn.execute(
            f"ALTER TABLE matches ADD COLUMN {_quote_ident(col_name)} {col_type}"
        )
    conn.commit()


def _ensure_h2h_columns(conn: sqlite3.Connection) -> None:
    """Add missing columns to match_h2h table for backward compat."""
    existing = {
        row["name"] for row in conn.execute("PRAGMA table_info(match_h2h)").fetchall()
    }
    for col_name in ("timestamp",):
        if col_name not in existing:
            try:
                conn.execute(f"ALTER TABLE match_h2h ADD COLUMN {col_name} INTEGER")
            except Exception:
                pass
    conn.commit()


def enrich_h2h_team_stats(
    conn: sqlite3.Connection,
    match_id: str,
) -> list[dict]:
    """
    Enrich H2H data for a match with team strength statistics.

    For each past matchup between the same teams, looks up
    team_statistics from the DB to include avg game stats.

    Returns list with enriched H2H rows.
    """
    h2h_rows = conn.execute(
        """
        SELECT h2h_match_id, date, home_team, away_team,
               home_score, away_score
        FROM match_h2h
        WHERE match_id = ?
          AND date != ''
        ORDER BY date DESC LIMIT 20
        """,
        (match_id,),
    ).fetchall()

    enriched: list[dict] = []
    for row in h2h_rows:
        r = dict(row)
        stat_avgs = conn.execute(
            """
            SELECT t.stat_key, t.stat_name,
                   ROUND(AVG(t.home_value), 1) as home_avg,
                   ROUND(AVG(t.away_value), 1) as away_avg
            FROM match_h2h h
            JOIN team_statistics t ON t.match_id = h.h2h_match_id
            WHERE h.match_id = ?
              AND h.h2h_match_id != ?
              AND t.period = 'ALL'
              AND t.home_value IS NOT NULL
              AND t.away_value IS NOT NULL
            GROUP BY t.stat_key, t.stat_name
            ORDER BY t.stat_key
            """,
            (match_id, row["h2h_match_id"]),
        ).fetchall()
        if stat_avgs:
            r["h2h_team_avgs"] = [dict(s) for s in stat_avgs]
        enriched.append(r)

    return enriched or [dict(r) for r in h2h_rows]


def get_h2h_team_averages(
    conn: sqlite3.Connection,
    match_id: str,
) -> list[dict]:
    """
    Compute rolling averages of team statistics across all H2H matches
    for this team pairing.

    Returns list of stat dicts with stat_key, stat_name,
    n_matches, home_avg, away_avg.
    """
    rows = conn.execute(
        """
        SELECT
            t.stat_key,
            t.stat_name,
            COUNT(*) AS n_matches,
            ROUND(AVG(t.home_value), 1) AS home_avg,
            ROUND(AVG(t.away_value), 1) AS away_avg
        FROM match_h2h h
        JOIN team_statistics t ON t.match_id = h.h2h_match_id
        WHERE h.match_id = ?
          AND t.period = 'ALL'
          AND t.home_value IS NOT NULL
          AND t.away_value IS NOT NULL
        GROUP BY t.stat_key, t.stat_name
        ORDER BY t.stat_key
        """,
        (match_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def compute_h2h_side_stats(
    conn: sqlite3.Connection,
    match_id: str,
) -> dict:
    """
    Split H2H matchups by home/away side for the current match's teams.

    Returns dict:
        home_team: {name, as_home: {wins, losses, q1_avg..q4_avg, pts_scored_avg, pts_conceded_avg},
                           as_away: {...}}
        away_team: {name, as_home: {...}, as_away: {...}}
    """
    match_row = conn.execute(
        "SELECT home_team, away_team FROM matches WHERE match_id = ?",
        (match_id,),
    ).fetchone()
    if not match_row:
        return {}
    home_team = match_row["home_team"]
    away_team = match_row["away_team"]

    h2h_rows = conn.execute(
        """
        SELECT home_team, away_team, home_score, away_score,
               q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
        FROM match_h2h
        WHERE match_id = ?
          AND date != ''
        """,
        (match_id,),
    ).fetchall()

    def _side_stats(team: str, h2h: list) -> dict:
        as_home = []
        as_away = []
        for r in h2h:
            if r["home_team"] == team:
                as_home.append(r)
            elif r["away_team"] == team:
                as_away.append(r)

        def _agg(matches: list, is_home: bool) -> dict:
            if not matches:
                return {"matches": 0, "wins": 0, "losses": 0}
            wins = 0
            q1s, q2s, q3s, q4s = [], [], [], []
            pts_scored, pts_conceded = [], []
            for m in matches:
                if is_home:
                    hs, aw = m["home_score"], m["away_score"]
                    q1s.append(m["q1_home"] or 0)
                    q2s.append(m["q2_home"] or 0)
                    q3s.append(m["q3_home"] or 0)
                    q4s.append(m["q4_home"] or 0)
                else:
                    hs, aw = m["away_score"], m["home_score"]
                    q1s.append(m["q1_away"] or 0)
                    q2s.append(m["q2_away"] or 0)
                    q3s.append(m["q3_away"] or 0)
                    q4s.append(m["q4_home"] or 0)  # q4_away is opponent
                if hs > aw:
                    wins += 1
                pts_scored.append(hs or 0)
                pts_conceded.append(aw or 0)

            n = len(matches)
            def avg(vals):
                return round(sum(vals) / n, 1) if vals else 0.0

            return {
                "matches": n,
                "wins": wins,
                "losses": n - wins,
                "win_pct": round(wins / n, 3) if n else 0.0,
                "q1_avg": avg(q1s),
                "q2_avg": avg(q2s),
                "q3_avg": avg(q3s),
                "q4_avg": avg(q4s),
                "pts_scored_avg": avg(pts_scored),
                "pts_conceded_avg": avg(pts_conceded),
                "pts_diff_avg": avg([s - c for s, c in zip(pts_scored, pts_conceded)]),
            }

        return {
            "name": team,
            "as_home": _agg(as_home, is_home=True),
            "as_away": _agg(as_away, is_home=False),
        }

    return {
        "home_team": _side_stats(home_team, h2h_rows),
        "away_team": _side_stats(away_team, h2h_rows),
    }


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
        "q3_reasoning": f"q3_reasoning__{safe_tag}",
        "q3_predicted_home": f"q3_predicted_home__{safe_tag}",
        "q3_predicted_away": f"q3_predicted_away__{safe_tag}",
        "q3_predicted_total": f"q3_predicted_total__{safe_tag}",
        "q3_mae": f"q3_mae__{safe_tag}",
        "q3_mae_home": f"q3_mae_home__{safe_tag}",
        "q3_mae_away": f"q3_mae_away__{safe_tag}",
        "q4_pick": f"q4_pick__{safe_tag}",
        "q4_signal": f"q4_signal__{safe_tag}",
        "q4_outcome": f"q4_outcome__{safe_tag}",
        "q4_available": f"q4_available__{safe_tag}",
        "q4_confidence": f"q4_confidence__{safe_tag}",
        "q4_threshold_lean": f"q4_threshold_lean__{safe_tag}",
        "q4_threshold_bet": f"q4_threshold_bet__{safe_tag}",
        "q4_reasoning": f"q4_reasoning__{safe_tag}",
        "q4_predicted_home": f"q4_predicted_home__{safe_tag}",
        "q4_predicted_away": f"q4_predicted_away__{safe_tag}",
        "q4_predicted_total": f"q4_predicted_total__{safe_tag}",
        "q4_mae": f"q4_mae__{safe_tag}",
        "q4_mae_home": f"q4_mae_home__{safe_tag}",
        "q4_mae_away": f"q4_mae_away__{safe_tag}",
    }
    column_types = {
        col_map["q3_pick"]: "TEXT",
        col_map["q3_signal"]: "TEXT",
        col_map["q3_outcome"]: "TEXT",
        col_map["q3_available"]: "INTEGER",
        col_map["q3_confidence"]: "REAL",
        col_map["q3_threshold_lean"]: "REAL",
        col_map["q3_threshold_bet"]: "REAL",
        col_map["q3_reasoning"]: "TEXT",
        col_map["q3_predicted_home"]: "REAL",
        col_map["q3_predicted_away"]: "REAL",
        col_map["q3_predicted_total"]: "REAL",
        col_map["q3_mae"]: "REAL",
        col_map["q3_mae_home"]: "REAL",
        col_map["q3_mae_away"]: "REAL",
        col_map["q4_pick"]: "TEXT",
        col_map["q4_signal"]: "TEXT",
        col_map["q4_outcome"]: "TEXT",
        col_map["q4_available"]: "INTEGER",
        col_map["q4_confidence"]: "REAL",
        col_map["q4_threshold_lean"]: "REAL",
        col_map["q4_threshold_bet"]: "REAL",
        col_map["q4_reasoning"]: "TEXT",
        col_map["q4_predicted_home"]: "REAL",
        col_map["q4_predicted_away"]: "REAL",
        col_map["q4_predicted_total"]: "REAL",
        col_map["q4_mae"]: "REAL",
        col_map["q4_mae_home"]: "REAL",
        col_map["q4_mae_away"]: "REAL",
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

    def pred_fields(pred: dict) -> dict:
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
        return {
            "pick": pick,
            "signal": signal,
            "outcome": outcome,
            "available": available,
            "confidence": confidence,
            "threshold_lean": threshold_lean,
            "threshold_bet": threshold_bet,
            # V12-specific fields
            "reasoning": pred.get("reasoning"),
            "predicted_home": pred.get("predicted_home"),
            "predicted_away": pred.get("predicted_away"),
            "predicted_total": pred.get("predicted_total"),
            "mae": pred.get("mae"),
            "mae_home": pred.get("mae_home"),
            "mae_away": pred.get("mae_away"),
        }

    q3_f = pred_fields(pred_q3)
    q4_f = pred_fields(pred_q4)

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
        col_map["q3_pick"]: q3_f["pick"],
        col_map["q3_signal"]: q3_f["signal"],
        col_map["q3_outcome"]: q3_f["outcome"],
        col_map["q3_available"]: q3_f["available"],
        col_map["q3_confidence"]: q3_f["confidence"],
        col_map["q3_threshold_lean"]: q3_f["threshold_lean"],
        col_map["q3_threshold_bet"]: q3_f["threshold_bet"],
        col_map["q4_pick"]: q4_f["pick"],
        col_map["q4_signal"]: q4_f["signal"],
        col_map["q4_outcome"]: q4_f["outcome"],
        col_map["q4_available"]: q4_f["available"],
        col_map["q4_confidence"]: q4_f["confidence"],
        col_map["q4_threshold_lean"]: q4_f["threshold_lean"],
        col_map["q4_threshold_bet"]: q4_f["threshold_bet"],
        # V12-specific fields
        col_map["q3_reasoning"]: q3_f["reasoning"],
        col_map["q3_predicted_home"]: q3_f["predicted_home"],
        col_map["q3_predicted_away"]: q3_f["predicted_away"],
        col_map["q3_predicted_total"]: q3_f["predicted_total"],
        col_map["q3_mae"]: q3_f["mae"],
        col_map["q3_mae_home"]: q3_f["mae_home"],
        col_map["q3_mae_away"]: q3_f["mae_away"],
        col_map["q4_reasoning"]: q4_f["reasoning"],
        col_map["q4_predicted_home"]: q4_f["predicted_home"],
        col_map["q4_predicted_away"]: q4_f["predicted_away"],
        col_map["q4_predicted_total"]: q4_f["predicted_total"],
        col_map["q4_mae"]: q4_f["mae"],
        col_map["q4_mae_home"]: q4_f["mae_home"],
        col_map["q4_mae_away"]: q4_f["mae_away"],
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
    """Upsert a full match (metadata + quarters + PBP + events + H2H + graph)."""
    m = data["match"]
    s = data["score"]

    _ensure_match_columns(conn)

    conn.execute(
        """
        INSERT OR REPLACE INTO matches
                    (match_id, home_team, away_team, home_slug, away_slug, event_slug,
                     custom_id, status_type, status_description, date, time, venue, league,
                     home_record, away_record, home_score, away_score,
                     home_team_id, away_team_id,
                     home_rating, away_rating)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
            m.get("home_team_id"), m.get("away_team_id"),
            m.get("home_rating"), m.get("away_rating"),
        ),
    )

    for quarter, scores in s.get("quarters", {}).items():
        conn.execute(
            "INSERT OR REPLACE INTO quarter_scores (match_id, quarter, home, away) VALUES (?,?,?,?)",
            (match_id, quarter, scores["home"], scores["away"]),
        )

    # Full replace for play-by-play (scoring plays only, backward compat)
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

    # Full replace for all events (includes scoring + non-scoring)
    conn.execute("DELETE FROM match_events WHERE match_id = ?", (match_id,))
    for quarter, events in data.get("events", {}).items():
        for seq, ev in enumerate(events):
            conn.execute(
                """
                INSERT INTO match_events
                  (match_id, quarter, seq, time, time_seconds, incident_type, subtype,
                   player, player_id, team, points, home_score, away_score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    match_id, quarter, seq,
                    ev.get("time"), ev.get("time_seconds"),
                    ev.get("incident_type"), ev.get("subtype"),
                    ev.get("player"), ev.get("player_id"),
                    ev.get("team"), ev.get("points"),
                    ev.get("home_score"), ev.get("away_score"),
                ),
            )

    # Full replace for H2H
    save_match_h2h(conn, match_id, data.get("h2h", []))

    # Lineups
    save_lineups(conn, match_id, data.get("lineups", []))

    # Player statistics (with SofaScore ratings)
    save_player_stats(conn, match_id, data.get("player_stats", []))

    # Team-level statistics (2PT%, 3PT%, rebounds, fouls, etc.)
    save_team_statistics(conn, match_id, data.get("team_statistics", []))

    # Per-period team statistics (computed from incidents)
    # Stored in the same team_statistics table with period='Q1','Q2',etc.
    save_team_statistics(conn, match_id, data.get("period_stats", []))

    # Pre-match betting odds
    save_match_odds(conn, match_id, data.get("odds", []))

    # Team strength snapshots (pregameForm + performance points)
    save_team_strength(conn, match_id, data.get("team_strength", []))

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


def save_match_h2h(conn: sqlite3.Connection, match_id: str, h2h_rows: list[dict]) -> None:
    """Replace H2H history for a match."""
    _ensure_h2h_columns(conn)
    conn.execute("DELETE FROM match_h2h WHERE match_id = ?", (match_id,))
    for row in h2h_rows:
        conn.execute(
            """
            INSERT INTO match_h2h
              (match_id, h2h_match_id, date, timestamp,
               home_team, away_team,
               home_score, away_score,
               q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away,
               tournament)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                match_id,
                row.get("h2h_match_id", ""),
                row.get("date", ""),
                row.get("timestamp"),
                row.get("home_team", ""),
                row.get("away_team", ""),
                row.get("home_score"),
                row.get("away_score"),
                row.get("q1_home"), row.get("q1_away"),
                row.get("q2_home"), row.get("q2_away"),
                row.get("q3_home"), row.get("q3_away"),
                row.get("q4_home"), row.get("q4_away"),
                row.get("tournament", ""),
            ),
        )


def save_player_stats(conn: sqlite3.Connection, match_id: str, stats_rows: list[dict]) -> None:
    """Replace player statistics for a match."""
    if not stats_rows:
        return
    conn.execute("DELETE FROM player_stats WHERE match_id = ?", (match_id,))
    for row in stats_rows:
        conn.execute(
            """
            INSERT INTO player_stats
              (match_id, team, player_name, player_id, sofascore_rating,
               minutes_played, points, fouls, plus_minus,
               field_goals_made, field_goals_attempted,
               three_made, three_attempted,
               free_throws_made, free_throws_attempted,
               rebounds, assists, steals, turnovers, blocks)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                match_id,
                row.get("team", ""),
                row.get("player_name", ""),
                row.get("player_id"),
                row.get("sofascore_rating"),
                row.get("minutes_played"),
                row.get("points"),
                row.get("fouls"),
                row.get("plus_minus"),
                row.get("field_goals_made"),
                row.get("field_goals_attempted"),
                row.get("three_made"),
                row.get("three_attempted"),
                row.get("free_throws_made"),
                row.get("free_throws_attempted"),
                row.get("rebounds"),
                row.get("assists"),
                row.get("steals"),
                row.get("turnovers"),
                row.get("blocks"),
            ),
        )


def save_lineups(conn: sqlite3.Connection, match_id: str, lineup_rows: list[dict]) -> None:
    """Replace lineups for a match."""
    if not lineup_rows:
        return
    conn.execute("DELETE FROM lineups WHERE match_id = ?", (match_id,))
    for row in lineup_rows:
        conn.execute(
            """
            INSERT INTO lineups
              (match_id, team, player_name, player_id, is_starter, shirt_number, position)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                match_id,
                row.get("team", ""),
                row.get("player_name", ""),
                row.get("player_id"),
                1 if row.get("is_starter") else 0,
                row.get("shirt_number"),
                row.get("position"),
            ),
        )


def save_team_statistics(conn: sqlite3.Connection, match_id: str, stat_rows: list[dict]) -> None:
    """Replace team-level statistics for a match."""
    conn.execute("DELETE FROM team_statistics WHERE match_id = ?", (match_id,))
    for row in stat_rows:
        conn.execute(
            """
            INSERT INTO team_statistics
              (match_id, period, group_name, stat_key, stat_name,
               home_value, away_value, home_total, away_total,
               home_display, away_display)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                match_id,
                row.get("period", "ALL"),
                row.get("group_name", ""),
                row.get("stat_key", ""),
                row.get("stat_name", ""),
                row.get("home_value"),
                row.get("away_value"),
                row.get("home_total"),
                row.get("away_total"),
                row.get("home_display", ""),
                row.get("away_display", ""),
            ),
        )


def save_match_odds(conn: sqlite3.Connection, match_id: str, odds_rows: list[dict]) -> None:
    """Replace betting odds for a match."""
    if not odds_rows:
        return
    conn.execute("DELETE FROM match_odds WHERE match_id = ?", (match_id,))
    for row in odds_rows:
        conn.execute(
            """
            INSERT INTO match_odds
              (match_id, odds_type, market_name, market_period,
               home_value, away_value, draw_value, timestamp)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                match_id,
                row.get("odds_type", ""),
                row.get("market_name", ""),
                row.get("market_period", ""),
                row.get("home_value"),
                row.get("away_value"),
                row.get("draw_value"),
                row.get("timestamp"),
            ),
        )
    conn.commit()


def save_team_strength(
    conn: sqlite3.Connection, match_id: str, rows: list[dict]
) -> None:
    """Replace team strength data for a match."""
    if not rows:
        return
    conn.execute("DELETE FROM team_strength WHERE match_id = ?", (match_id,))
    for row in rows:
        conn.execute(
            """
            INSERT INTO team_strength
              (team_id, team_name, match_id, position, wins, losses, form, perf_points)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                row["team_id"],
                row.get("team_name", ""),
                match_id,
                row.get("position"),
                row.get("wins"),
                row.get("losses"),
                row.get("form"),
                row.get("perf_points"),
            ),
        )
    conn.commit()


def get_match(
    conn: sqlite3.Connection,
    match_id: str,
    include_events: bool = False,
    include_h2h: bool = False,
) -> dict | None:
    """Reconstruct the full match dict from the DB.

    By default returns the same shape as the old scraper output (backward compat).
    Set include_events=True to also load the full event stream.
    Set include_h2h=True to also load H2H history.
    """
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

    out = {
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
            "home_team_id": row["home_team_id"] if "home_team_id" in row.keys() else None,
            "away_team_id": row["away_team_id"] if "away_team_id" in row.keys() else None,
            "home_rating": row["home_rating"] if "home_rating" in row.keys() else None,
            "away_rating": row["away_rating"] if "away_rating" in row.keys() else None,
            "date": row["date"],
            "time": row["time"],
            "venue": row["venue"],
            "league": row["league"],
            "home_record": row["home_record"],
            "away_record": row["away_record"],
            "scraped_at": row["updated_at"] if "updated_at" in row.keys() else None,
        },
        "score": {
            "home": row["home_score"],
            "away": row["away_score"],
            "quarters": quarters,
        },
        "play_by_play": pbp,
        "graph_points": graph_points,
    }

    if include_events:
        events: dict[str, list] = {}
        for er in conn.execute(
            """
            SELECT quarter, time, time_seconds, incident_type, subtype,
                   player, player_id, team, points, home_score, away_score
            FROM match_events WHERE match_id = ? ORDER BY quarter, seq
            """,
            (match_id,),
        ):
            events.setdefault(er["quarter"], []).append({
                "time": er["time"],
                "time_seconds": er["time_seconds"],
                "incident_type": er["incident_type"],
                "subtype": er["subtype"],
                "player": er["player"],
                "player_id": er["player_id"],
                "team": er["team"],
                "points": er["points"],
                "home_score": er["home_score"],
                "away_score": er["away_score"],
            })
        out["events"] = events

    if include_h2h:
        h2h_rows = conn.execute(
            """
            SELECT h2h_match_id, date, home_team, away_team,
                   home_score, away_score,
                   q1_home, q1_away, q2_home, q2_away,
                   q3_home, q3_away, q4_home, q4_away,
                   tournament
            FROM match_h2h WHERE match_id = ? ORDER BY date
            """,
            (match_id,),
        ).fetchall()
        out["h2h"] = [dict(r) for r in h2h_rows]

    return out


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
