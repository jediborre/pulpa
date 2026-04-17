"""
analyze_pace_distribution.py — Análisis de distribución de puntos para validar pace buckets de V13.

Calcula percentiles reales de puntos por cuarto (Q1+Q2 para Q3, Q1+Q2+Q3 para Q4)
para definir los umbrales de buckets de ritmo (low/medium/high).
"""

import sqlite3
import json
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm

DB_PATH = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa\match\matches.db")

def get_connection():
    """Crear conexión con contexto de fila."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def pivot_quarter_scores(conn):
    """Convertir quarter_scores (formato largo) a formato ancho por partido."""
    cursor = conn.cursor()
    
    print("\n" + "="*80)
    print("🔄 PIVOTEANDO QUARTER_SCORES A FORMATO ANCHO")
    print("="*80)
    
    # Obtener todos los quarter_scores
    cursor.execute("""
        SELECT match_id, quarter, home, away
        FROM quarter_scores
        WHERE quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    """)
    
    rows = cursor.fetchall()
    print(f"📊 Total de quarter_scores: {len(rows)}")
    
    # Pivotear
    pivot = {}
    for row in tqdm(rows, desc="Pivoteando"):
        mid = row['match_id']
        q = row['quarter']  # 'Q1', 'Q2', etc.
        home = row['home']
        away = row['away']
        
        if mid not in pivot:
            pivot[mid] = {}
        
        # Guardar con key en minúscula: 'q1_home', 'q1_away', etc.
        q_lower = q.lower()  # 'q1', 'q2', etc.
        pivot[mid][f'{q_lower}_home'] = home
        pivot[mid][f'{q_lower}_away'] = away
    
    print(f"✅ Partidos con quarter_scores: {len(pivot)}")
    
    # Debug: ver un ejemplo
    if pivot:
        sample_mid = list(pivot.keys())[0]
        print(f"   Debug - Sample match {sample_mid}: {pivot[sample_mid]}")
    
    # Filtrar solo los que tienen Q1+Q2+Q3 completos
    complete_q3 = {
        mid: scores for mid, scores in pivot.items()
        if all(f'q{i}_{side}' in scores for i in [1,2,3] for side in ['home', 'away'])
    }
    
    complete_q4 = {
        mid: scores for mid, scores in pivot.items()
        if all(f'q{i}_{side}' in scores for i in [1,2,3,4] for side in ['home', 'away'])
    }
    
    print(f"✅ Partidos con Q1+Q2+Q3 completos: {len(complete_q3)}")
    print(f"✅ Partidos con Q1+Q2+Q3+Q4 completos: {len(complete_q4)}")
    
    return pivot, complete_q3, complete_q4

def get_league_gender_map(conn):
    """Obtener mapeo de match_id -> league, gender."""
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT match_id, league,
               CASE 
                   WHEN LOWER(league) LIKE '%women%' OR LOWER(league) LIKE '%femenino%' THEN 'women'
                   ELSE 'men'
               END as gender
        FROM matches
    """)
    
    return {row['match_id']: {'league': row['league'], 'gender': row['gender']} 
            for row in cursor.fetchall()}

def analyze_distribution(pivot, complete_q3, complete_q4, metadata):
    """Analizar distribución de puntos."""
    print("\n" + "="*80)
    print("📈 ANÁLISIS DE DISTRIBUCIÓN DE PUNTOS")
    print("="*80)
    
    # Calcular totales
    q3_totals = []  # Q1+Q2 total (para bucket de Q3)
    q4_totals = []  # Q1+Q2+Q3 total (para bucket de Q4)
    
    q3_by_gender = {'men': [], 'women': []}
    q4_by_gender = {'men': [], 'women': []}
    
    # Para Q3: usar partidos con al menos Q1+Q2
    for mid, scores in tqdm(complete_q3.items(), desc="Procesando Q3"):
        q1_home = scores.get('q1_home', 0)
        q1_away = scores.get('q1_away', 0)
        q2_home = scores.get('q2_home', 0)
        q2_away = scores.get('q2_away', 0)
        
        q3_total = q1_home + q1_away + q2_home + q2_away
        
        meta = metadata.get(mid, {})
        gender = meta.get('gender', 'men')
        
        q3_totals.append(q3_total)
        q3_by_gender[gender].append(q3_total)
    
    # Para Q4: usar partidos con Q1+Q2+Q3
    for mid, scores in tqdm(complete_q4.items(), desc="Procesando Q4"):
        q1_home = scores.get('q1_home', 0)
        q1_away = scores.get('q1_away', 0)
        q2_home = scores.get('q2_home', 0)
        q2_away = scores.get('q2_away', 0)
        q3_home = scores.get('q3_home', 0)
        q3_away = scores.get('q3_away', 0)
        
        q4_total = q1_home + q1_away + q2_home + q2_away + q3_home + q3_away
        
        meta = metadata.get(mid, {})
        gender = meta.get('gender', 'men')
        
        q4_totals.append(q4_total)
        q4_by_gender[gender].append(q4_total)
    
    # Estadísticas generales
    print("\n" + "-"*60)
    print("📦 DISTRIBUCIÓN Q3 (Q1+Q2 total de ambos equipos)")
    print("-"*60)
    print(f"   Muestras: {len(q3_totals)}")
    print(f"   Media: {np.mean(q3_totals):.1f}")
    print(f"   Mediana: {np.median(q3_totals):.1f}")
    print(f"   Min: {np.min(q3_totals)}, Max: {np.max(q3_totals)}")
    print(f"   Percentil 33: {np.percentile(q3_totals, 33):.1f}")
    print(f"   Percentil 66: {np.percentile(q3_totals, 66):.1f}")
    print(f"   Propuestos V13: low ≤42, medium 42-54, high ≥54")
    
    print("\n" + "-"*60)
    print("📦 DISTRIBUCIÓN Q4 (Q1+Q2+Q3 total de ambos equipos)")
    print("-"*60)
    print(f"   Muestras: {len(q4_totals)}")
    print(f"   Media: {np.mean(q4_totals):.1f}")
    print(f"   Mediana: {np.median(q4_totals):.1f}")
    print(f"   Min: {np.min(q4_totals)}, Max: {np.max(q4_totals)}")
    print(f"   Percentil 33: {np.percentile(q4_totals, 33):.1f}")
    print(f"   Percentil 66: {np.percentile(q4_totals, 66):.1f}")
    print(f"   Propuestos V13: low ≤63, medium 63-78, high ≥78")
    
    # Distribución por género
    print("\n" + "-"*60)
    print("👥 DISTRIBUCIÓN POR GÉNERO")
    print("-"*60)
    
    for gender in ['men', 'women']:
        q3_vals = q3_by_gender[gender]
        q4_vals = q4_by_gender[gender]
        
        if q3_vals:
            print(f"\n   {gender.upper()} Q3 ({len(q3_vals)} partidos):")
            print(f"     Media={np.mean(q3_vals):.1f}, p33={np.percentile(q3_vals, 33):.1f}, p66={np.percentile(q3_vals, 66):.1f}")
        
        if q4_vals:
            print(f"   {gender.upper()} Q4 ({len(q4_vals)} partidos):")
            print(f"     Media={np.mean(q4_vals):.1f}, p33={np.percentile(q4_vals, 33):.1f}, p66={np.percentile(q4_vals, 66):.1f}")
    
    # Guardar resultados
    results = {
        "q3_total": {
            "n": len(q3_totals),
            "mean": float(np.mean(q3_totals)),
            "median": float(np.median(q3_totals)),
            "min": int(np.min(q3_totals)),
            "max": int(np.max(q3_totals)),
            "p33": float(np.percentile(q3_totals, 33)),
            "p66": float(np.percentile(q3_totals, 66)),
        },
        "q4_total": {
            "n": len(q4_totals),
            "mean": float(np.mean(q4_totals)),
            "median": float(np.median(q4_totals)),
            "min": int(np.min(q4_totals)),
            "max": int(np.max(q4_totals)),
            "p33": float(np.percentile(q4_totals, 33)),
            "p66": float(np.percentile(q4_totals, 66)),
        },
        "by_gender": {
            gender: {
                "q3": {
                    "n": len(q3_by_gender[gender]),
                    "mean": float(np.mean(q3_by_gender[gender])) if q3_by_gender[gender] else 0,
                    "p33": float(np.percentile(q3_by_gender[gender], 33)) if q3_by_gender[gender] else 0,
                    "p66": float(np.percentile(q3_by_gender[gender], 66)) if q3_by_gender[gender] else 0,
                },
                "q4": {
                    "n": len(q4_by_gender[gender]),
                    "mean": float(np.mean(q4_by_gender[gender])) if q4_by_gender[gender] else 0,
                    "p33": float(np.percentile(q4_by_gender[gender], 33)) if q4_by_gender[gender] else 0,
                    "p66": float(np.percentile(q4_by_gender[gender], 66)) if q4_by_gender[gender] else 0,
                }
            }
            for gender in ['men', 'women']
        }
    }
    
    output_path = Path(__file__).parent / "pace_distribution.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Resultados guardados en: {output_path}")
    
    return results

def main():
    print("\n" + "🏀"*40)
    print("ANÁLISIS DE DISTRIBUCIÓN DE PUNTOS PARA V13")
    print("🏀"*40)
    
    conn = get_connection()
    
    try:
        # Paso 1: Pivotear quarter_scores
        pivot, complete_q3, complete_q4 = pivot_quarter_scores(conn)
        
        # Paso 2: Obtener metadata de ligas
        metadata = get_league_gender_map(conn)
        
        # Paso 3: Analizar distribución
        results = analyze_distribution(pivot, complete_q3, complete_q4, metadata)
        
        if results:
            print("\n" + "="*80)
            print("✅ ANÁLISIS COMPLETADO")
            print("="*80)
            print("\n📊 Umbrales recomendados para V13:")
            
            q3_p33 = results["q3_total"]["p33"]
            q3_p66 = results["q3_total"]["p66"]
            q4_p33 = results["q4_total"]["p33"]
            q4_p66 = results["q4_total"]["p66"]
            
            print(f"\n   Q3 (halftime total):")
            print(f"     low    ≤ {q3_p33:.0f} pts")
            print(f"     medium  {q3_p33:.0f}-{q3_p66:.0f} pts")
            print(f"     high   ≥ {q3_p66:.0f} pts")
            
            print(f"\n   Q4 (3 cuartos total):")
            print(f"     low    ≤ {q4_p33:.0f} pts")
            print(f"     medium  {q4_p33:.0f}-{q4_p66:.0f} pts")
            print(f"     high   ≥ {q4_p66:.0f} pts")
            
            print("\n💡 Estos valores deben reemplazar los defaults en config.py")
            
            # Estimar muestras por bucket
            print("\n" + "="*80)
            print("📊 ESTIMACIÓN DE MUESTRAS POR BUCKET")
            print("="*80)
            
            for gender in ['men', 'women']:
                n_q3 = results["by_gender"][gender]["q3"]["n"]
                n_q4 = results["by_gender"][gender]["q4"]["n"]
                
                print(f"\n   {gender.upper()}:")
                print(f"     Q3: ~{n_q3} partidos → ~{n_q3//3} por bucket")
                print(f"     Q4: ~{n_q4} partidos → ~{n_q4//3} por bucket")
                
                if n_q3//3 < 200:
                    print(f"     ⚠️ Q3 buckets pueden tener <200 muestras!")
                if n_q4//3 < 200:
                    print(f"     ⚠️ Q4 buckets pueden tener <200 muestras!")
        else:
            print("\n⚠️ No se pudo completar el análisis de distribución")
            
    finally:
        conn.close()
    
    print("\n" + "🏀"*40 + "\n")

if __name__ == "__main__":
    main()
