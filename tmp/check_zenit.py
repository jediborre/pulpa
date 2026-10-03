"""
Ubicación original: scratch/check_zenit.py
Propósito / Qué hacía:
Diagnóstico específico del partido y datos del equipo Zenit.
"""

import sqlite3

def main():
    conn = sqlite3.connect('match/matches.db')
    conn.row_factory = sqlite3.Row
    
    r = conn.execute("""
        SELECT *
        FROM bet_monitor_log_v2
        WHERE match_id = '16220497'
    """).fetchall()
    
    for row in r:
        d = dict(row)
        clean_d = {}
        for k, v in d.items():
            if isinstance(v, str):
                clean_d[k] = v.encode('ascii', 'ignore').decode('ascii')
            else:
                clean_d[k] = v
        print(clean_d)

if __name__ == '__main__':
    main()
