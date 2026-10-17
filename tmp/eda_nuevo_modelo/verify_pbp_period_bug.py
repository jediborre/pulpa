"""
Script: verify_pbp_period_bug.py
Ubicación: tmp/eda_nuevo_modelo/verify_pbp_period_bug.py
Propósito: Verificar cómo match/scraper.py etiquetó los cuartos en play_by_play y match_events
           usando _period_from_game_seconds (720s/1440s/2160s/2880s de 12 min NBA) sobre partidos FIBA de 10 min,
           y comprobar que se puede reconstruir el cuarto y minuto REAL exacto de cada jugada usando
           time_seconds / marcadores de cuarto / orden secuencial.
Hipótesis/Problema: Confirmar el hallazgo de que quarter='Q3' en play_by_play incluye los primeros 6 minutos
                    de Q4 (1800s..2160s) en ligas de 10 minutos, lo que causaba contaminación de Q4 en m27/m30.
"""
import sys
import sqlite3
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    mid = "14210656" # Bayern vs Maccabi (Euroleague 10m: Q1 31-23, Q2 19-18 [HT 50-41], Q3 26-13 [End Q3 76-54], Q4 19-17 [FT 95-71])
    print(f"=== Eventos completos de {mid} en match_events ===")
    cur.execute("""
        SELECT id, quarter, seq, time, time_seconds, incident_type, subtype, home_score, away_score
        FROM match_events
        WHERE match_id = ?
        ORDER BY id ASC
    """, (mid,))
    rows = cur.fetchall()
    for r in rows[:35]:
        print(" ", r)

    conn.close()

if __name__ == "__main__":
    main()
