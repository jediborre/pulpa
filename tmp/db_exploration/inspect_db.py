"""
Script temporal de auditoría y extracción de metadatos de matches.db.
Propósito: Extraer schemas exactos, índices, foreign keys y métricas globales
(conteos de filas, rangos de fechas, ligas principales, cobertura de datos)
para documentar SCHEMA_DATABASE.md y EXPLORACION_GLOBAL_DATOS.md.
"""

import sqlite3
import json
from pathlib import Path

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. Obtener todas las tablas
    tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
    
    schema_info = {}
    for table in tables:
        if table.startswith("sqlite_"):
            continue
        # Columnas
        cols = cur.execute(f"PRAGMA table_info({table})").fetchall()
        cols_data = [
            {
                "cid": c["cid"],
                "name": c["name"],
                "type": c["type"],
                "notnull": c["notnull"],
                "dflt_value": c["dflt_value"],
                "pk": c["pk"]
            }
            for c in cols
        ]
        
        # Foreign keys
        fks = cur.execute(f"PRAGMA foreign_key_list({table})").fetchall()
        fks_data = [
            {
                "id": f["id"],
                "table": f["table"],
                "from": f["from"],
                "to": f["to"]
            }
            for f in fks
        ]
        
        # Indices
        idxs = cur.execute(f"PRAGMA index_list({table})").fetchall()
        idxs_data = [
            {
                "name": idx["name"],
                "unique": idx["unique"]
            }
            for idx in idxs
        ]

        # Conteo aproximado / exacto
        try:
            count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except Exception as e:
            count = -1
            
        schema_info[table] = {
            "count": count,
            "columns": cols_data,
            "foreign_keys": fks_data,
            "indexes": idxs_data
        }

    # 2. Métricas de matches
    matches_stats = {}
    matches_stats["total_matches"] = cur.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    date_range = cur.execute("SELECT MIN(date), MAX(date) FROM matches WHERE date IS NOT NULL AND date != ''").fetchone()
    matches_stats["date_min"] = date_range[0]
    matches_stats["date_max"] = date_range[1]
    
    # Status breakdown
    status_counts = cur.execute("SELECT status_type, COUNT(*) FROM matches GROUP BY status_type ORDER BY COUNT(*) DESC").fetchall()
    matches_stats["status_counts"] = {r[0]: r[1] for r in status_counts}
    
    # Finished breakdown
    finished_desc = cur.execute("SELECT status_description, COUNT(*) FROM matches WHERE status_type='finished' GROUP BY status_description ORDER BY COUNT(*) DESC LIMIT 10").fetchall()
    matches_stats["finished_desc"] = {r[0]: r[1] for r in finished_desc}

    # Unique leagues
    num_leagues = cur.execute("SELECT COUNT(DISTINCT league) FROM matches").fetchone()[0]
    matches_stats["unique_leagues"] = num_leagues

    # Top 30 leagues
    top_leagues = cur.execute("SELECT league, COUNT(*) as cnt FROM matches GROUP BY league ORDER BY cnt DESC LIMIT 35").fetchall()
    matches_stats["top_leagues"] = [{"league": r["league"], "count": r["cnt"]} for r in top_leagues]

    # Partidos por año
    by_year = cur.execute("SELECT substr(date, 1, 4) as y, COUNT(*) as cnt FROM matches WHERE date IS NOT NULL AND date != '' GROUP BY y ORDER BY y").fetchall()
    matches_stats["by_year"] = {r["y"]: r["cnt"] for r in by_year}

    # Cobertura de relaciones clave
    # ¿Cuántos matches tienen quarter_scores?
    qs_matches = cur.execute("SELECT COUNT(DISTINCT match_id) FROM quarter_scores").fetchone()[0]
    # ¿Cuántos tienen play_by_play?
    pbp_matches = cur.execute("SELECT COUNT(DISTINCT match_id) FROM play_by_play").fetchone()[0]
    # ¿Cuántos tienen graph_points?
    gp_matches = cur.execute("SELECT COUNT(DISTINCT match_id) FROM graph_points").fetchone()[0]
    # ¿Cuántos tienen match_h2h?
    h2h_matches = cur.execute("SELECT COUNT(DISTINCT match_id) FROM match_h2h").fetchone()[0]
    # ¿Cuántos tienen team_statistics?
    ts_matches = cur.execute("SELECT COUNT(DISTINCT match_id) FROM team_statistics").fetchone()[0]
    # ¿Cuántos tienen lineups/player_stats?
    ps_matches = cur.execute("SELECT COUNT(DISTINCT match_id) FROM player_stats").fetchone()[0]

    coverage = {
        "quarter_scores": qs_matches,
        "play_by_play": pbp_matches,
        "graph_points": gp_matches,
        "match_h2h": h2h_matches,
        "team_statistics": ts_matches,
        "player_stats": ps_matches
    }

    out = {
        "tables": schema_info,
        "matches_stats": matches_stats,
        "coverage": coverage
    }

    out_file = Path("tmp/db_exploration/db_summary.json")
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Resumen generado exitosamente en {out_file}")

if __name__ == "__main__":
    main()
