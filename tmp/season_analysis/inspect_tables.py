"""
tmp/season_analysis/inspect_tables.py
Inspección rápida de tablas y conteos en matches.db
"""
import sqlite3

conn = sqlite3.connect('matches.db')
c = conn.cursor()
c.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = [r[0] for r in c.fetchall()]
print('Tablas encontradas:', tables)

for t in tables:
    c.execute(f"SELECT COUNT(*) FROM {t}")
    print(f"  {t}: {c.fetchone()[0]:,}")

conn.close()
