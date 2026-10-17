"""
Script: prove_m27_clock_bug.py
Ubicación: tmp/eda_nuevo_modelo/prove_m27_clock_bug.py
Propósito: Demostrar empíricamente qué eventos selecciona pbp_events_upto(data, 27) y recent_window_features(data, 27, 2.0)
           en los modelos m27_v1/v2/v3 sobre un partido FIBA de 10 minutos cuando se entrena con el partido terminado en DB
           vs cuando corre en vivo en el minuto 27 real (time_seconds <= 1620).
Hipótesis/Problema: Confirmar que la combinación de _period_from_game_seconds (720s múltiplos) en match/scraper.py
                    y q_start=(q_idx-1)*12 en pbp_utils.py provocaba que en entrenamiento a snapshot 27 se filtraran
                    los puntos del primer minuto de Q4 (1800..1860s) dentro de recent_2m_points_diff, mientras que los
                    minutos 25..27 reales de Q3 (1500..1620s) recibían event_min = 31..33 y quedaban excluidos.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "match"))
sys.stdout.reconfigure(encoding="utf-8")

import db as db_mod
from models.common.pbp_utils import pbp_events_upto, recent_window_features, score_upto

def main():
    conn = db_mod.get_conn(str(ROOT / "matches.db"))
    data = db_mod.get_match(conn, "14210656") # Bayern vs Maccabi (Euroleague 10m)
    conn.close()

    print("=== Partido 14210656 (Euroleague 10m): Q1=31-23, Q2=19-18 (HT 50-41), Q3=26-13 (Fin Q3 76-54), Q4=19-17 (FT 95-71) ===")
    evs_27 = pbp_events_upto(data, 27)
    print(f"\nEventos devueltos por pbp_events_upto(data, 27): {len(evs_27)}")
    for e in evs_27[-8:]:
        print(f"  _global_min={e['_global_min']:.2f} | time={e['time']} | pts={e['points']} | score={e['home_score']}-{e['away_score']}")

    r2 = recent_window_features(data, 27, 2.0, "recent_2m")
    r3 = recent_window_features(data, 27, 3.0, "recent_3m")
    sc = score_upto(data, 27)
    print("\nEn ENTRENAMIENTO (partido terminado en DB):")
    print("  score_upto(data, 27):", sc, "  <-- ¡Nota que Fin de Q3 era 76-54 y aquí ya va 78-54 (en Q4)!")
    print("  recent_2m:", r2)
    print("  recent_3m:", r3)

if __name__ == "__main__":
    main()
