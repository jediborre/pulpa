"""
Script para calcular métricas de comportamiento comparativas por arquetipo de liga en matches.db.
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    queries = {
        "NBA / 12 Minutos": "quarter_duration_minutes = 12",
        "FIBA Top Pro (ACB, Euroleague, etc.)": "tier_level = 'top_pro' AND quarter_duration_minutes = 10 AND is_women = 0",
        "FIBA Second Pro (Serie A2, FEB, etc.)": "tier_level = 'second_pro' AND quarter_duration_minutes = 10 AND is_women = 0",
        "College NCAA (Men + Women)": "is_college = 1",
        "Youth / Canteras (U16-U23)": "is_youth = 1",
        "Baloncesto Femenino (Adultas)": "is_women = 1 AND is_youth = 0 AND is_college = 0",
        "Playoffs / Eliminatorias": "is_playoffs = 1",
        "Gran Final / Partidos de Título": "is_final = 1",
        "Fase Regular Senior Pro": "stage_detail = 'Regular Season' AND is_women = 0 AND is_youth = 0 AND is_college = 0"
    }

    print(f"{'ARQUETIPO':<38} | {'PARTIDOS':<8} | {'PTS':<5} | {'PACE':<5} | {'STD':<5} | {'BLOW%':<6} | {'CLUTCH%':<7} | {'Q4 PTS':<6} | {'LOC%':<5}")
    print("-" * 105)
    for label, where in queries.items():
        r = cur.execute(f"""
            SELECT 
                SUM(finished_match_count) as matches,
                AVG(avg_total_points) as pts,
                AVG(scoring_pace_per_minute) as pace,
                AVG(points_std_dev) as std,
                AVG(blowout_rate) as blow,
                AVG(clutch_rate) as clutch,
                AVG(avg_q4_total_points) as q4_pts,
                AVG(home_win_pct) as loc
            FROM leagues_classification
            WHERE {where}
        """).fetchone()
        
        matches = r["matches"] if r["matches"] is not None else 0
        pts = r["pts"] if r["pts"] is not None else 0
        pace = r["pace"] if r["pace"] is not None else 0
        std = r["std"] if r["std"] is not None else 0
        blow = r["blow"] if r["blow"] is not None else 0
        clutch = r["clutch"] if r["clutch"] is not None else 0
        q4_pts = r["q4_pts"] if r["q4_pts"] is not None else 0
        loc = r["loc"] if r["loc"] is not None else 0
        
        print(f"{label:<38} | {matches:<8} | {pts:<5.1f} | {pace:<5.2f} | {std:<5.1f} | {blow:<6.1f} | {clutch:<7.1f} | {q4_pts:<6.1f} | {loc:<5.1f}")

if __name__ == "__main__":
    main()
