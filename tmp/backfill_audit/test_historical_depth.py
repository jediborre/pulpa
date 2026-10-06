"""
Verifica la calidad y completitud de datos históricos en SofaScore (2015, 2020, 2024).
Comprueba la existencia de:
- quarter_scores (Q1-Q4)
- play_by_play
- graph_points
- lineups / player_stats
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path('.').resolve()))

from match.scraper import fetch_finished_match_ids_for_date, fetch_match_by_id

dates_to_test = ['2024-01-15', '2020-01-15', '2015-01-15']

for dt in dates_to_test:
    print(f"\n=================== FECHA: {dt} ===================")
    matches = fetch_finished_match_ids_for_date(dt, backend='mobile')
    print(f"Total partidos en {dt}: {len(matches)}")
    if not matches:
        continue
    
    # Probar 3 partidos
    for m in matches[:3]:
        mid = str(m.get('match_id'))
        slug = m.get('slug', 'unknown')
        print(f"\n--- Analizando Match ID: {mid} ({slug}) ---")
        try:
            data = fetch_match_by_id(mid, backend='mobile')
            if not data:
                print("  [ERROR] No se pudo obtener payload")
                continue
            
            quarters = data.get('score', {}).get('quarters', {})
            has_q4 = all(k in quarters for k in ['Q1', 'Q2', 'Q3', 'Q4'])
            pbp = data.get('play_by_play', {})
            pbp_q_count = len(pbp) if isinstance(pbp, dict) else 0
            pbp_total_events = sum(len(v) for v in pbp.values()) if isinstance(pbp, dict) else 0
            gp = data.get('graph_points', [])
            gp_count = len(gp) if isinstance(gp, list) else 0
            lineups = data.get('lineups', [])
            has_lineups = bool(lineups)
            league = data.get('match', {}).get('league', 'unknown')
            print(f"  - Liga: {league}")
            
            print(f"  - Marcadores Cuartos (Q1-Q4): {'SI' if has_q4 else 'NO'} ({list(quarters.keys())})")
            print(f"  - Play-by-Play: {'SI' if pbp_total_events > 0 else 'NO'} ({pbp_total_events} eventos en {pbp_q_count} cuartos)")
            print(f"  - Graph Points: {'SI' if gp_count > 0 else 'NO'} ({gp_count} puntos)")
            print(f"  - Lineups confirmadas: {'SI' if has_lineups else 'NO'}")
        except Exception as e:
            print(f"  [ERROR] Excepción: {e}")
