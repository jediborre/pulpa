"""
Script auxiliar: query_latest_date.py
Propósito: Consultar las fechas más recientes de partidos almacenados en matches.db (en la raíz),
identificar cuándo fue el último partido registrado antes del día de hoy y revisar
la distribución temporal de la base de datos histórica.
"""

import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[2]
db_path = ROOT / "matches.db"

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# 1. Total de partidos
total_matches = cursor.execute("SELECT count(*) FROM matches").fetchone()[0]
print(f"Total partidos en matches.db: {total_matches}")

# 2. Partidos con fecha
print("\n=== TOP 25 FECHAS MAS RECIENTES EN matches ===")
dates = cursor.execute("""
    SELECT date, count(*) as cnt 
    FROM matches 
    WHERE date IS NOT NULL AND length(date) >= 8
    GROUP BY substr(date, 1, 10) 
    ORDER BY substr(date, 1, 10) DESC 
    LIMIT 25
""").fetchall()

for d, cnt in dates:
    print(f"Fecha: {d[:10]} | Partidos: {cnt}")

# 3. Fecha máxima absoluta y fecha anterior a hoy
today_str = "2026-10-04"
cursor.execute("SELECT max(date) FROM matches WHERE substr(date, 1, 10) < ?", (today_str,))
last_before_today = cursor.execute("SELECT max(date) FROM matches WHERE substr(date, 1, 10) < ?", (today_str,)).fetchone()[0]
print(f"\n-> Último partido registrado ANTES de hoy ({today_str}): {last_before_today}")

# 4. Revisar si hay partidos de hoy
today_matches = cursor.execute("SELECT count(*) FROM matches WHERE substr(date, 1, 10) = ?", (today_str,)).fetchone()[0]
print(f"-> Partidos de hoy ({today_str}) en matches: {today_matches}")

# 5. Revisar bet_monitor_schedule_v2 / v3
for tbl in ["bet_monitor_schedule_v2", "bet_monitor_schedule_v3"]:
    try:
        r = cursor.execute(f"SELECT min(event_date), max(event_date), count(*) FROM {tbl}").fetchone()
        print(f"-> {tbl}: Min: {r[0]} | Max: {r[1]} | Total: {r[2]}")
    except Exception as e:
        print(f"-> {tbl}: {e}")

conn.close()
