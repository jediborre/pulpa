"""
inspect_tables.py — Inspección de tablas y esquemas relacionados a predicciones en matches.db.
Creado en tmp/db_inspection/ para auditar dónde se guardan PredictionResults y logs de modelos.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def inspect():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    print(f"=== TOTAL TABLAS: {len(tables)} ===")
    
    # Filtrar tablas relacionadas a predicciones, logs, modelos, eval
    relevant = [t for t in tables if any(k in t.lower() for k in ["predict", "eval", "result", "log", "bet", "model"])]
    print("\n--- TABLAS RELEVANTES DE PREDICCIONES / RESULTADOS ---")
    for t in relevant:
        cur.execute(f"SELECT COUNT(*) FROM {t}")
        cnt = cur.fetchone()[0]
        cur.execute(f"PRAGMA table_info({t})")
        cols = [c[1] for c in cur.fetchall()]
        print(f"\nTabla: {t} ({cnt} filas)")
        print(f"Columnas: {', '.join(cols[:15])}{'...' if len(cols) > 15 else ''}")
        
    print("\n--- TODAS LAS TABLAS ---")
    print(", ".join(tables))
    conn.close()

if __name__ == "__main__":
    inspect()
