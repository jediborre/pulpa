"""
Ubicación original: temp_scripts/validate_h2h_features.py
Propósito / Qué hacía:
Validación del cálculo de las features F8-F15 de H2H en un partido específico.
"""

"""
Validación de features H2H para un partido específico.
Muestra el cálculo paso a paso para verificar que sea correcto.
"""

import sys
import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "match" / "training"))

DB_PATH = ROOT / "match" / "matches.db"


def get_match_h2h_data(match_id):
    """Get raw H2H data from match_h2h table."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT date, home_team, away_team,
               q1_home, q1_away, q2_home, q2_away,
               q3_home, q3_away, q4_home, q4_away
        FROM match_h2h
        WHERE match_id = ? AND q1_home IS NOT NULL
        ORDER BY date
    """, (match_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_match_info(match_id):
    """Get match info from bet_monitor_log_v2."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute("""
        SELECT l.match_id, l.result, l.picked_side, l.confidence,
               s.home_team, s.away_team, s.event_date
        FROM bet_monitor_log_v2 l
        JOIN bet_monitor_schedule_v2 s ON l.match_id = s.match_id
        WHERE l.match_id = ? AND l.model_version = 'm27_v3'
    """, (match_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def compute_h2h_features_manual(h2h_rows, home_team, away_team, match_date):
    """Compute H2H features manually with step-by-step output."""
    print("\n" + "=" * 70)
    print("  VALIDACIÓN DE FEATURES H2H")
    print("=" * 70)
    print(f"\nPartido: {home_team} vs {away_team}")
    print(f"Fecha: {match_date}")
    print(f"Total partidos H2H encontrados: {len(h2h_rows)}")
    
    if not h2h_rows:
        print("\n[ERROR] No hay datos H2H para este partido")
        return None
    
    # Show raw H2H data
    print("\n" + "-" * 70)
    print("  DATOS H2H CRUDOS (de SofaScore)")
    print("-" * 70)
    for i, r in enumerate(h2h_rows):
        total_h = (r.get("q1_home") or 0) + (r.get("q2_home") or 0) + \
                  (r.get("q3_home") or 0) + (r.get("q4_home") or 0)
        total_a = (r.get("q1_away") or 0) + (r.get("q2_away") or 0) + \
                  (r.get("q3_away") or 0) + (r.get("q4_away") or 0)
        winner = "HOME" if total_h > total_a else "AWAY"
        print(f"  [{i+1}] {r['date']} | {r['home_team'][:20]} {total_h} - {total_a} {r['away_team'][:20]} | Q1: {r['q1_home']}-{r['q1_away']} | Ganador: {winner}")
    
    # Filter to games before match_date
    past_games = [r for r in h2h_rows if str(r["date"]) < str(match_date)]
    print(f"\nPartidos ANTERIORES a {match_date}: {len(past_games)}")
    
    if not past_games:
        print("\n[ERROR] No hay partidos anteriores para calcular features")
        return None
    
    # Compute features step by step
    print("\n" + "-" * 70)
    print("  CÁLCULO DE FEATURES")
    print("-" * 70)
    
    q1_diffs = []
    home_won_last3 = 0
    last_home_won = 0
    n = len(past_games)
    
    for i, g in enumerate(past_games):
        # Determine if home_team was home or away in this past game
        is_home = g["home_team"] == home_team
        sign = 1 if is_home else -1
        
        # Q1 diff
        q1_h = g.get("q1_home") or 0
        q1_a = g.get("q1_away") or 0
        q1_diff = (q1_h - q1_a) * sign
        q1_diffs.append(q1_diff)
        
        # Total score
        total_h = (g.get("q1_home") or 0) + (g.get("q2_home") or 0) + \
                  (g.get("q3_home") or 0) + (g.get("q4_home") or 0)
        total_a = (g.get("q1_away") or 0) + (g.get("q2_away") or 0) + \
                  (g.get("q3_away") or 0) + (g.get("q4_away") or 0)
        
        # Who won?
        home_won = (total_h > total_a) == is_home
        
        # Last 3 games
        if n - i <= 3:
            if home_won:
                home_won_last3 += 1
        
        # Last game
        if i == n - 1:
            last_home_won = 1 if home_won else 0
        
        role = "HOME" if is_home else "AWAY"
        winner = "SI" if home_won else "NO"
        print(f"  [{i+1}] {g['date']} | {g['home_team'][:15]} vs {g['away_team'][:15]} | {home_team[:15]} jugo como {role}")
        print(f"       Q1: {q1_h}-{q1_a} (diff para {home_team[:15]}: {q1_diff:+d}) | Total: {total_h}-{total_a} | {home_team[:15]} gano? {winner}")
    
    # Final features
    print("\n" + "-" * 70)
    print("  FEATURES FINALES")
    print("-" * 70)
    
    h2h_avg_q1_diff = round(sum(q1_diffs) / len(q1_diffs), 3) if q1_diffs else 0.0
    h2h_recent3_home_won = round(home_won_last3 / min(n, 3), 3)
    h2h_last_home_won = last_home_won
    
    print(f"\n  h2h_avg_q1_diff = {h2h_avg_q1_diff}")
    print(f"    -> Promedio de diffs Q1: {q1_diffs} -> {sum(q1_diffs)}/{len(q1_diffs)} = {h2h_avg_q1_diff}")
    
    print(f"\n  h2h_recent3_home_won = {h2h_recent3_home_won}")
    print(f"    -> Victorias en ultimos 3: {home_won_last3}/{min(n, 3)} = {h2h_recent3_home_won}")
    
    print(f"\n  h2h_last_home_won = {h2h_last_home_won}")
    print(f"    -> Ultimo partido: {'gano' if last_home_won else 'perdio'}")
    
    print("\n" + "=" * 70)
    
    return {
        "h2h_avg_q1_diff": h2h_avg_q1_diff,
        "h2h_recent3_home_won": h2h_recent3_home_won,
        "h2h_last_home_won": h2h_last_home_won,
    }


def main():
    # Get a match with H2H data
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Find matches with H2H data
    matches = conn.execute("""
        SELECT DISTINCT l.match_id
        FROM bet_monitor_log_v2 l
        JOIN match_h2h h ON h.match_id = l.match_id
        WHERE l.model_version = 'm27_v3'
          AND h.q1_home IS NOT NULL
        LIMIT 10
    """).fetchall()
    conn.close()
    
    if not matches:
        print("No hay partidos con datos H2H")
        return
    
    print("Partidos con H2H disponible:")
    for i, m in enumerate(matches):
        print(f"  [{i+1}] {m['match_id']}")
    
    # Pick first match
    match_id = str(matches[0]["match_id"])
    print(f"\nValidando partido: {match_id}")
    
    # Get match info
    match_info = get_match_info(match_id)
    if not match_info:
        print(f"❌ No se encontró info del partido {match_id}")
        return
    
    # Get H2H data
    h2h_rows = get_match_h2h_data(match_id)
    
    # Compute features
    features = compute_h2h_features_manual(
        h2h_rows,
        match_info["home_team"],
        match_info["away_team"],
        match_info["event_date"]
    )
    
    if features:
        print("\n[OK] Features calculadas correctamente")
        print(f"\nResultado del partido: {match_info['result']}")
        print(f"Pick original: {match_info['picked_side']}")
        print(f"Confianza original: {match_info['confidence']}")


if __name__ == "__main__":
    main()
