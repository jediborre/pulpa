"""
tmp/season_analysis/audit_early_season_v5.py
Auditoría rigurosa del rendimiento de M27_V5 en Inicios de Temporada (Cold-Start)
utilizando el dataset real de 198,423 partidos y los modelos calibrados en producción.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

CACHE_PATH = Path("tmp/m27_v5/dataset_v5_clean_cache.pkl")
MODELS_DIR = Path("models/m27_v5/model_outputs")

def main():
    print("1. Cargando cache de dataset v5...")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
        
    X = data["X"]
    y_class = data["y_class"]
    meta = data["meta"]
    
    print("2. Cargando modelos en producción...")
    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(MODELS_DIR / "hgb_model.pkl", "rb") as f:
        hgb = pickle.load(f)
    with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
        calibrator = pickle.load(f)
        
    print("3. Generando predicciones de ensamble calibrado...")
    p_xgb = xgb.predict_proba(X)[:, 1]
    p_hgb = hgb.predict_proba(X)[:, 1]
    p_raw = 0.5 * p_xgb + 0.5 * p_hgb
    p_cal = calibrator.predict(p_raw)
    
    df = pd.DataFrame(meta)
    df['y_true'] = y_class
    df['p_cal'] = p_cal
    df['dt'] = pd.to_datetime(df['dt'], errors='coerce')
    df = df.dropna(subset=['dt']).sort_values('dt').reset_index(drop=True)
    
    df['year'] = df['dt'].dt.year
    df['month'] = df['dt'].dt.month
    
    # Definir temporada deportiva (Sep a Jun)
    def get_season(row):
        yr, mo = row['year'], row['month']
        if mo >= 8:
            return f"{yr}-{yr+1}"
        else:
            return f"{yr-1}-{yr}"
            
    df['season'] = df.apply(get_season, axis=1)
    
    # Calcular orden de partido por equipo dentro de su liga y temporada
    team_counts = defaultdict(lambda: defaultdict(int))
    h_nums, a_nums = [], []
    
    for idx, row in df.iterrows():
        key = (row['lg'], row['season'])
        ht, at = row['ht'], row['at']
        team_counts[key][ht] += 1
        team_counts[key][at] += 1
        h_nums.append(team_counts[key][ht])
        a_nums.append(team_counts[key][at])
        
    df['h_game'] = h_nums
    df['a_game'] = a_nums
    df['min_game'] = np.minimum(df['h_game'], df['a_game'])
    
    def stage_group(g):
        if g <= 3:
            return "1. Inicio Crítico (Jornadas 1-3)"
        elif g <= 6:
            return "2. Asentamiento Temprano (Jornadas 4-6)"
        elif g <= 15:
            return "3. Plena Temporada (Jornadas 7-15)"
        else:
            return "4. Madurez / Cierre (Jornadas 16+)"
            
    df['stage'] = df['min_game'].apply(stage_group)
    
    # Filtro de señales operables: Verde P >= 0.68
    # En test set (por ejemplo últimos 3 años: 2023-2026)
    df_test = df[df['year'] >= 2023].copy()
    
    print("\n" + "="*80)
    print("ANÁLISIS DE RENDIMIENTO POR ETAPA DE TEMPORADA (TEST 2023-2026, P >= 0.68)")
    print("="*80)
    
    for stage, grp in df_test.groupby('stage'):
        tot_matches = len(grp)
        # Apuestas Home (P >= 0.68)
        home_bets = grp[grp['p_cal'] >= 0.68]
        # Apuestas Away (P <= 0.32)
        away_bets = grp[grp['p_cal'] <= 0.32]
        
        n_bets = len(home_bets) + len(away_bets)
        wins = home_bets['y_true'].sum() + (len(away_bets) - away_bets['y_true'].sum())
        wr = (wins / n_bets * 100) if n_bets > 0 else 0
        
        # ROI @ 1.40 pesimista
        roi_140 = ((wins * 1.40 - n_bets) / n_bets * 100) if n_bets > 0 else 0
        # ROI @ 1.83 flat
        roi_183 = ((wins * 1.83 - n_bets) / n_bets * 100) if n_bets > 0 else 0
        
        pct_bets = (n_bets / tot_matches * 100) if tot_matches > 0 else 0
        
        print(f"\nEtapa: {stage}")
        print(f"  Partidos totales evaluados: {tot_matches:,}")
        print(f"  Apuestas disparadas (P >= 0.68 o P <= 0.32): {n_bets:,} ({pct_bets:.1f}% de cobertura)")
        print(f"  Win Rate: {wr:.2f}% | ROI @ 1.40: {roi_140:+.2f}% | ROI @ 1.83: {roi_183:+.2f}%")
        
    print("\n" + "="*80)
    print("ANÁLISIS POR MES CALENDARIO (TEST 2023-2026, P >= 0.68)")
    print("="*80)
    
    month_names = {
        1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic"
    }
    
    for mo in sorted(df_test['month'].unique()):
        grp = df_test[df_test['month'] == mo]
        home_bets = grp[grp['p_cal'] >= 0.68]
        away_bets = grp[grp['p_cal'] <= 0.32]
        n_bets = len(home_bets) + len(away_bets)
        if n_bets < 50:
            continue
        wins = home_bets['y_true'].sum() + (len(away_bets) - away_bets['y_true'].sum())
        wr = (wins / n_bets * 100) if n_bets > 0 else 0
        roi_140 = ((wins * 1.40 - n_bets) / n_bets * 100) if n_bets > 0 else 0
        print(f"Mes {mo:02d} ({month_names.get(mo, '')}): Bets: {n_bets:4d} | WR: {wr:.2f}% | ROI @ 1.40: {roi_140:+.2f}%")

if __name__ == "__main__":
    main()
