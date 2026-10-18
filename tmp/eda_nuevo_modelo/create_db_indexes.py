"""
Creación de índices compuestos y de cobertura (covering indexes) en `matches.db`
para acelerar consultas analíticas y de inferencia sobre:
- `player_stats` (por jugador, por partido+equipo)
- `lineups` (por partido+equipo+titular, por jugador)
- `play_by_play` (índice parcial de jugadas con puntos por partido+equipo+cuarto+jugador)
- `match_events` (por partido+cuarto+jugador)
- `quarter_scores` (índice de cobertura por partido+cuarto+marcadores)
- `matches` (por fecha+id y por equipos local/visitante)
"""
import sys
import time
import sqlite3
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

INDEXES = [
    (
        "idx_player_stats_player_match",
        "CREATE INDEX IF NOT EXISTS idx_player_stats_player_match ON player_stats (player_id, match_id)",
    ),
    (
        "idx_player_stats_match_team",
        "CREATE INDEX IF NOT EXISTS idx_player_stats_match_team ON player_stats (match_id, team)",
    ),
    (
        "idx_lineups_match_team_starter",
        "CREATE INDEX IF NOT EXISTS idx_lineups_match_team_starter ON lineups (match_id, team, is_starter)",
    ),
    (
        "idx_lineups_player_match",
        "CREATE INDEX IF NOT EXISTS idx_lineups_player_match ON lineups (player_id, match_id)",
    ),
    (
        "idx_pbp_scoring_player_quarter",
        "CREATE INDEX IF NOT EXISTS idx_pbp_scoring_player_quarter ON play_by_play (match_id, team, quarter, player, points) WHERE points > 0 AND player IS NOT NULL",
    ),
    (
        "idx_match_events_match_quarter_player",
        "CREATE INDEX IF NOT EXISTS idx_match_events_match_quarter_player ON match_events (match_id, quarter, team, player_id, points)",
    ),
    (
        "idx_qs_covering",
        "CREATE INDEX IF NOT EXISTS idx_qs_covering ON quarter_scores (match_id, quarter, home, away)",
    ),
    (
        "idx_matches_date_id",
        "CREATE INDEX IF NOT EXISTS idx_matches_date_id ON matches (date, match_id)",
    ),
    (
        "idx_matches_teams",
        "CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches (home_team_id, away_team_id)",
    ),
]


def main():
    conn = sqlite3.connect(str(DB_PATH), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-256000;")  # ~256MB cache

    print(f"Creando/verificando {len(INDEXES)} índices de alto rendimiento en {DB_PATH}...")
    for name, ddl in INDEXES:
        t0 = time.time()
        print(f"  -> Creando {name} ...", end=" ", flush=True)
        conn.execute(ddl)
        conn.commit()
        print(f"OK ({time.time() - t0:.2f}s)", flush=True)

    conn.execute("ANALYZE;")
    conn.commit()
    conn.close()
    print("Todos los índices y estadísticas del optimizador (ANALYZE) actualizados correctamente.")


if __name__ == "__main__":
    main()
