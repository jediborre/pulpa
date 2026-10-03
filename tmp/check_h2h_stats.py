"""
Ubicación original: temp_scripts/check_h2h_stats.py
Propósito / Qué hacía:
Calcula el número total de partidos con datos H2H completos en la base de datos.
"""

import sqlite3
conn = sqlite3.connect('matches.db')
conn.row_factory = sqlite3.Row

# Total de partidos con H2H
total = conn.execute('SELECT COUNT(DISTINCT match_id) FROM match_h2h WHERE q1_home IS NOT NULL').fetchone()[0]
print(f'Partidos con H2H completo: {total}')

# Total de filas H2H
rows = conn.execute('SELECT COUNT(*) FROM match_h2h WHERE q1_home IS NOT NULL').fetchone()[0]
print(f'Total de filas H2H: {rows}')

# Promedio de partidos H2H por match
avg = conn.execute('SELECT AVG(cnt) FROM (SELECT match_id, COUNT(*) as cnt FROM match_h2h WHERE q1_home IS NOT NULL GROUP BY match_id)').fetchone()[0]
print(f'Promedio partidos H2H por match: {avg:.1f}')

# Distribucion de cantidad de H2H
print(f'\nDistribucion de partidos H2H por match:')
dist = conn.execute('''
    SELECT cnt, COUNT(*) as matches
    FROM (SELECT match_id, COUNT(*) as cnt FROM match_h2h WHERE q1_home IS NOT NULL GROUP BY match_id)
    GROUP BY cnt
    ORDER BY cnt
''').fetchall()
for r in dist:
    bar = '#' * min(r['matches'], 50)
    print(f'  {r["cnt"]:>3} partidos H2H: {r["matches"]:>4} matches {bar}')

# Min/Max
minmax = conn.execute('SELECT MIN(cnt), MAX(cnt) FROM (SELECT match_id, COUNT(*) as cnt FROM match_h2h WHERE q1_home IS NOT NULL GROUP BY match_id)').fetchone()
print(f'\nMin: {minmax[0]} | Max: {minmax[1]}')

conn.close()
