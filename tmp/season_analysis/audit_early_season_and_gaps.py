"""
tmp/season_analysis/audit_early_season_and_gaps.py

Auditoría para analizar:
1. Desempeño en inicio de temporada (cold-start / primeros partidos de equipo vs resto de la temporada).
2. Brechas de datos y partidos faltantes en las ligas rentables (Euroleague, ACB, BBL, Serie A, Pro A, etc.).
"""

import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DB_PATH = Path("matches.db")

def run_audit():
    conn = sqlite3.connect(DB_PATH)
    
    print("=== 1. AUDITORÍA DE INICIO DE TEMPORADA (COLD-START) ===")
    # Buscaremos el orden de partido por equipo dentro de cada temporada/año
    # Cargamos partidos terminados con fecha y marcador
    query = """
        SELECT m.match_id, m.date, m.league, m.home_team, m.away_team, 
               m.home_score, m.away_score,
               q.q1_home, q.q1_away, q.q2_home, q.q2_away, 
               q.q3_home, q.q3_away, q.q4_home, q.q4_away
        FROM matches m
        JOIN quarter_scores_v2 q ON m.match_id = q.match_id
        WHERE m.status_type = 'finished' AND m.date IS NOT NULL
        ORDER BY m.date ASC
    """
    df = pd.read_sql_query(query, conn)
    print(f"Total partidos terminados con cuartos: {len(df):,}")
    
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).sort_values('date')
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Marcador Q4
    df['q4_margin'] = df['q4_home'] - df['q4_away']
    df['q4_home_win'] = (df['q4_margin'] > 0).astype(int)
    # Margen al min 27 (aprox Q1+Q2+Q3)
    df['margin_m27'] = (df['q1_home'] + df['q2_home'] + df['q3_home']) - (df['q1_away'] + df['q2_away'] + df['q3_away'])
    
    # Calcular número de partido en la temporada para cada equipo
    # Para ligas de invierno (Sep a Jun), la temporada es año/año+1.
    def get_season(row):
        year = row['year']
        month = row['month']
        if month >= 8:
            return f"{year}-{year+1}"
        else:
            return f"{year-1}-{year}"
            
    df['season'] = df.apply(get_season, axis=1)
    
    # Conteo acumulado de partidos jugados en la temporada por home y away
    team_game_counts = defaultdict(lambda: defaultdict(int)) # (league, season) -> team -> count
    
    home_game_nums = []
    away_game_nums = []
    
    for idx, row in df.iterrows():
        key = (row['league'], row['season'])
        h_team = row['home_team']
        a_team = row['away_team']
        
        team_game_counts[key][h_team] += 1
        team_game_counts[key][a_team] += 1
        
        home_game_nums.append(team_game_counts[key][h_team])
        away_game_nums.append(team_game_counts[key][a_team])
        
    df['h_game_num'] = home_game_nums
    df['a_game_num'] = away_game_nums
    df['min_game_num'] = np.minimum(df['h_game_num'], df['a_game_num'])
    
    # Agrupar por fase de temporada:
    # 1. Inicio temprano: partidos 1 a 4 del equipo
    # 2. Asentamiento: partidos 5 a 10
    # 3. Mitad de temporada: partidos 11 a 24
    # 4. Recta final / Playoffs: partidos 25+
    
    def stage_label(g):
        if g <= 4:
            return "1. Inicio (Jornada 1-4)"
        elif g <= 10:
            return "2. Asentamiento (Jornada 5-10)"
        elif g <= 24:
            return "3. Mitad Temporada (Jornada 11-24)"
        else:
            return "4. Madurez / Cierre (Jornada 25+)"
            
    df['season_stage'] = df['min_game_num'].apply(stage_label)
    
    print("\n--- Desempeño por Etapa de Temporada (Todas las ligas) ---")
    summary = df.groupby('season_stage').agg(
        total_matches=('match_id', 'count'),
        avg_q4_margin=('q4_margin', 'mean'),
        q4_home_win_pct=('q4_home_win', 'mean'),
        q4_abs_margin=('q4_margin', lambda x: np.mean(np.abs(x))),
        q3_abs_lead=('margin_m27', lambda x: np.mean(np.abs(x)))
    )
    print(summary)
    
    # Ahora evaluamos con el modelo real m27_v5 si es posible
    print("\n=== 2. AUDITORÍA DE BRECHAS Y PARTIDOS FALTANTES POR LIGA RENTABLE ===")
    target_leagues = [
        "Euroleague", "Liga ACB", "BBL", "Serie A", "France Pro A", 
        "NBA", "China CBA", "Eurocup", "Champions League", "VTB United League",
        "Turkey Super Ligi", "Greece A1", "Adriatic League", "Australia NBL"
    ]
    
    # Comprobar partidos por liga y año, cuántos tienen PBP
    pbp_counts = pd.read_sql_query("""
        SELECT m.league, strftime('%Y', m.date) as yr,
               COUNT(m.match_id) as total_matches,
               SUM(CASE WHEN p.cnt > 0 THEN 1 ELSE 0 END) as matches_with_pbp,
               SUM(CASE WHEN q.match_id IS NOT NULL THEN 1 ELSE 0 END) as matches_with_quarters
        FROM matches m
        LEFT JOIN (SELECT match_id, COUNT(*) as cnt FROM play_by_play GROUP BY match_id) p ON m.match_id = p.match_id
        LEFT JOIN quarter_scores_v2 q ON m.match_id = q.match_id
        WHERE m.status_type = 'finished'
        GROUP BY m.league, yr
    """, conn)
    
    print("\n--- Cobertura en Ligas Clave (Últimos 5 años) ---")
    pbp_counts['yr'] = pd.to_numeric(pbp_counts['yr'], errors='coerce')
    filtered = pbp_counts[pbp_counts['yr'] >= 2021]
    
    for league in target_leagues:
        sub = filtered[filtered['league'].str.contains(league, case=False, na=False)]
        if not sub.empty:
            print(f"\nLiga: {league}")
            for _, r in sub.iterrows():
                pbp_pct = (r['matches_with_pbp'] / r['total_matches'] * 100) if r['total_matches'] > 0 else 0
                qtr_pct = (r['matches_with_quarters'] / r['total_matches'] * 100) if r['total_matches'] > 0 else 0
                print(f"  Año {int(r['yr'])}: {r['total_matches']} partidos | Con Cuartos: {r['matches_with_quarters']} ({qtr_pct:.1f}%) | Con PBP: {r['matches_with_pbp']} ({pbp_pct:.1f}%)")
        else:
            print(f"\nLiga: {league} -> NO ENCONTRADA CON COBERTURA RECIENTE O NOMBRE DISTINTO")

    conn.close()

if __name__ == "__main__":
    run_audit()
