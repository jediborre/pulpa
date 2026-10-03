"""
inspect_eval_tables.py — Detalle profundo de eval_match_results y eval_match_results_v2.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def inspect():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    
    print("=== COLUMNAS EN eval_match_results ===")
    cur.execute("PRAGMA table_info(eval_match_results)")
    cols_v1 = [c[1] for c in cur.fetchall()]
    print(f"Total columnas: {len(cols_v1)}")
    # Modelos identificados en las columnas
    models_v1 = set()
    for col in cols_v1:
        if "__" in col:
            parts = col.split("__")
            models_v1.add(parts[1])
    print(f"Modelos presentes en columnas de eval_match_results: {sorted(list(models_v1))}")
    
    print("\n=== COLUMNAS EN eval_match_results_v2 ===")
    cur.execute("PRAGMA table_info(eval_match_results_v2)")
    cols_v2 = [c[1] for c in cur.fetchall()]
    print(f"Total columnas: {len(cols_v2)}")
    models_v2 = set()
    for col in cols_v2:
        if "__" in col:
            parts = col.split("__")
            models_v2.add(parts[1])
    print(f"Modelos presentes en columnas de eval_match_results_v2: {sorted(list(models_v2))}")
    
    # Muestra de datos recientes
    print("\n=== ÚLTIMOS 3 REGISTROS eval_match_results_v2 ===")
    cur.execute("SELECT * FROM eval_match_results_v2 ORDER BY rowid DESC LIMIT 3")
    rows = cur.fetchall()
    for r in rows:
        print(dict(zip(cols_v2, r)))
        
    print("\n=== MODELOS EN bet_monitor_log_v2 ===")
    cur.execute("SELECT model_version, count(*), sum(case when result='WIN' then 1 else 0 end) as wins, sum(case when result='LOSS' then 1 else 0 end) as losses FROM bet_monitor_log_v2 GROUP BY model_version")
    for r in cur.fetchall():
        print(f"Modelo: {r[0]} | Señales: {r[1]} | Wins: {r[2]} | Losses: {r[3]}")
        
    conn.close()

if __name__ == "__main__":
    inspect()
