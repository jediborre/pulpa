"""
tmp/season_analysis/audit_profitable_leagues_gaps.py
Auditoría de cobertura y partidos faltantes en ligas rentables (Euroleague, ACB, BBL, Serie A, etc.)
"""
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path("matches.db")

TARGET_LEAGUES = [
    "Euroleague",
    "Liga ACB",
    "BBL",
    "Serie A",
    "LNB Élite",
    "Pro A",
    "Eurocup",
    "Champions League",
    "NBA",
    "CBA",
    "China CBA",
    "Adriatic League",
    "TBSL",
    "Super Ligi",
    "KBL",
    "NBL"
]

def main():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    
    # Obtener nombres exactos en la base de datos
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT league FROM matches WHERE league IS NOT NULL")
    all_leagues = [r[0] for r in cur.fetchall()]
    
    matched_leagues = {}
    for target in TARGET_LEAGUES:
        matches = [lg for lg in all_leagues if target.lower() in lg.lower()]
        if matches:
            matched_leagues[target] = matches
            
    print(f"Ligas rentables identificadas en DB: {len(matched_leagues)}")
    
    query = """
        SELECT m.league, strftime('%Y', m.date) as yr,
               COUNT(m.match_id) as total_matches,
               SUM(CASE WHEN m.status_type = 'finished' THEN 1 ELSE 0 END) as finished_matches,
               SUM(CASE WHEN q.cnt >= 4 THEN 1 ELSE 0 END) as matches_with_4_quarters,
               SUM(CASE WHEN p.cnt > 0 THEN 1 ELSE 0 END) as matches_with_pbp
        FROM matches m
        LEFT JOIN (
            SELECT match_id, COUNT(*) as cnt 
            FROM quarter_scores 
            WHERE quarter IN ('Q1','Q2','Q3','Q4') 
            GROUP BY match_id
        ) q ON m.match_id = q.match_id
        LEFT JOIN (
            SELECT match_id, COUNT(*) as cnt 
            FROM play_by_play 
            GROUP BY match_id
        ) p ON m.match_id = p.match_id
        WHERE m.date >= '2021-01-01' AND m.league IN ({seq})
        GROUP BY m.league, yr
        ORDER BY m.league, yr
    """
    
    # Aplanamos la lista de ligas
    flat_lgs = [lg for lgs in matched_leagues.values() for lg in lgs]
    placeholders = ','.join('?' for _ in flat_lgs)
    sql = query.format(seq=placeholders)
    
    df = pd.read_sql_query(sql, conn, params=flat_lgs)
    conn.close()
    
    print("\n" + "="*95)
    print("COBERTURA DE LIGAS RENTABLES (2021-2026)")
    print("="*95)
    
    for category, lgs in matched_leagues.items():
        sub = df[df['league'].isin(lgs)]
        if sub.empty:
            continue
        print(f"\n>>> CATEGORÍA: {category} (Variantes: {len(lgs)})")
        summary_yr = sub.groupby('yr').agg(
            total=('total_matches', 'sum'),
            finished=('finished_matches', 'sum'),
            with_qtrs=('matches_with_4_quarters', 'sum'),
            with_pbp=('matches_with_pbp', 'sum')
        ).reset_index()
        
        for _, r in summary_yr.iterrows():
            q_pct = (r['with_qtrs'] / r['finished'] * 100) if r['finished'] > 0 else 0
            pbp_pct = (r['with_pbp'] / r['finished'] * 100) if r['finished'] > 0 else 0
            print(f"  Año {r['yr']}: {r['total']} partidos ({r['finished']} finished) | Cuartos Q1-Q4: {r['with_qtrs']} ({q_pct:.1f}%) | PBP: {r['with_pbp']} ({pbp_pct:.1f}%)")

if __name__ == "__main__":
    main()
