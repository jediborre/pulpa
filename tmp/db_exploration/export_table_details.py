"""
Script auxiliar para inspeccionar columnas y muestras de datos de cada tabla clave.
"""
import sqlite3
import json
from pathlib import Path

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    key_tables = [
        'matches',
        'quarter_scores',
        'quarter_scores_v2',
        'play_by_play',
        'match_events',
        'graph_points',
        'match_h2h',
        'team_statistics',
        'player_stats',
        'lineups',
        'match_odds',
        'team_strength',
        'eval_match_results',
        'eval_match_results_v2',
        'bet_monitor_log_v2'
    ]

    result = {}
    for t in key_tables:
        cols = cur.execute(f"PRAGMA table_info({t})").fetchall()
        fks = cur.execute(f"PRAGMA foreign_key_list({t})").fetchall()
        idxs = cur.execute(f"PRAGMA index_list({t})").fetchall()
        sample = cur.execute(f"SELECT * FROM {t} LIMIT 1").fetchone()
        
        sample_dict = dict(sample) if sample else {}
        # Convert any binary/bytes to str representation
        for k, v in sample_dict.items():
            if isinstance(v, bytes):
                sample_dict[k] = "<bytes>"

        result[t] = {
            "columns": [dict(c) for c in cols],
            "foreign_keys": [dict(f) for f in fks],
            "indexes": [dict(i) for i in idxs],
            "sample": sample_dict
        }

    Path("tmp/db_exploration/table_details.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("table_details.json generado con exito.")

if __name__ == "__main__":
    main()
