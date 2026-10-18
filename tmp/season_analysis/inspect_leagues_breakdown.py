"""
tmp/season_analysis/inspect_leagues_breakdown.py
Inspección de volúmenes de partidos terminados por arquetipos de ligas candidatas a modelos dedicados.
"""
import sqlite3

conn = sqlite3.connect('matches.db')
c = conn.cursor()

keywords = [
    ('NBA General', 'NBA'),
    ('China CBA', 'CBA'),
    ('Filipinas PBA', 'PBA'),
    ('NCAA College', 'NCAA'),
    ('Euroleague Élite', 'Euroleague'),
    ('Australia NBL', 'NBL'),
    ('Corea KBL', 'KBL'),
    ('WNBA', 'WNBA'),
    ('Argentina LNB / Liga', 'Argentina'),
    ('Brasil NBB', 'NBB'),
    ('Eurocup', 'Eurocup'),
    ('Champions League', 'Champions League')
]

print("=== VOLUMEN HISTÓRICO EN matches.db POR LIGA CANDIDATA ===")
for label, kw in keywords:
    c.execute("SELECT COUNT(*) FROM matches WHERE league LIKE ? AND status_type='finished'", (f'%{kw}%',))
    cnt = c.fetchone()[0]
    print(f"{label:22s}: {cnt:6,d} partidos terminados")

conn.close()
