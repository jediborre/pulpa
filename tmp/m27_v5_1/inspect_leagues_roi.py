"""
tmp/m27_v5_1/inspect_leagues_roi.py
Inspección de ligas con rendimiento pobre o sobreestimadas en m27_v5 para aplicar la poda adicional.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

CACHE_PATH = Path("tmp/m27_v5/dataset_v5_clean_cache.pkl")
MODELS_DIR = Path("models/m27_v5/model_outputs")

def main():
    print("Cargando dataset...")
    with open(CACHE_PATH, "rb") as f:
        d = pickle.load(f)
    X = d["X"]
    y = d["y_class"]
    meta = d["meta"]
    
    with open(MODELS_DIR / "xgb_model.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(MODELS_DIR / "hgb_model.pkl", "rb") as f:
        hgb = pickle.load(f)
    with open(MODELS_DIR / "calibrator.pkl", "rb") as f:
        cal = pickle.load(f)
        
    p_raw = 0.5 * xgb.predict_proba(X)[:, 1] + 0.5 * hgb.predict_proba(X)[:, 1]
    p_cal = cal.predict(p_raw)
    
    df = pd.DataFrame(meta)
    df['y'] = y
    df['p'] = p_cal
    
    # Evaluar solo en test set (años >= 2023)
    df_test = df[df['yr'] >= 2023].copy()
    
    print("\n--- ANÁLISIS DE LIGAS CON N >= 100 EN TEST ---")
    results = []
    for lg, grp in df_test.groupby('lg'):
        # Filtro de disparo verde: P >= 0.68 o P <= 0.32
        home_b = grp[grp['p'] >= 0.68]
        away_b = grp[grp['p'] <= 0.32]
        n_b = len(home_b) + len(away_b)
        if n_b < 40:
            continue
        wins = home_b['y'].sum() + (len(away_b) - away_b['y'].sum())
        wr = wins / n_b * 100
        roi_140 = (wins * 1.40 - n_b) / n_b * 100
        roi_183 = (wins * 1.83 - n_b) / n_b * 100
        
        # Sobreestimación media: diferencia entre confianza predicha y winrate real
        # Para home: p - y; para away: (1-p) - (1-y) = y - p
        overest = (home_b['p'] - home_b['y']).mean() if len(home_b) else 0
        
        results.append({
            'league': lg,
            'total_matches': len(grp),
            'bets': n_b,
            'wr': wr,
            'roi_140': roi_140,
            'roi_183': roi_183,
            'overest': overest
        })
        
    res_df = pd.DataFrame(results).sort_values('roi_140', ascending=True)
    print("\n>>> TOP 15 PEORES LIGAS (ROI @ 1.40 NEGATIVO / SOBREESTIMADAS):")
    print(res_df.head(15).to_string(index=False))
    
    print("\n>>> TOP 15 MEJORES LIGAS (ROI @ 1.40 POSITIVO):")
    print(res_df.tail(15).to_string(index=False))

if __name__ == "__main__":
    main()
