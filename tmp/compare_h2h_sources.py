"""
Ubicación original: temp_scripts/compare_h2h_sources.py
Propósito / Qué hacía:
Compara features H2H extraídas de dos fuentes distintas para validar consistencia.
"""

"""
Comparar features H2H desde DOS fuentes para el mismo partido:
1. quarter_scores (DB original - como en entrenamiento)
2. match_h2h (SofaScore - datos nuevos)
"""

import sys
import sqlite3
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "match" / "training"))

DB_PATH = ROOT / "match" / "matches.db"


def get_match_info(match_id):
    """Get match info."""
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


def compute_h2h_from_quarter_scores(match_id, home_team, away_team, match_date):
    """Compute H2H features from quarter_scores (original method)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    pair = tuple(sorted([home_team, away_team]))
    qs_rows = conn.execute(
        """SELECT m.date, qs.quarter, qs.home, qs.away
           FROM quarter_scores qs
           JOIN matches m ON m.match_id = qs.match_id
           WHERE (m.home_team = ? AND m.away_team = ?)
              OR (m.home_team = ? AND m.away_team = ?)
           ORDER BY m.date""",
        (*pair, *pair),
    ).fetchall()
    conn.close()
    
    past_games = []
    for dt, qtr, hs, aw in qs_rows:
        if hs is None or aw is None:
            continue
        g = past_games[-1] if past_games and past_games[-1].get("_date") == dt else None
        if g is None:
            g = {"_date": dt, "home_team": None, "away_team": None, "quarters": {}}
            past_games.append(g)
        g["quarters"][qtr] = (int(hs), int(aw))
    
    # Populate teams
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    for g in past_games:
        row = conn.execute(
            "SELECT home_team, away_team FROM matches WHERE date = ? AND "
            "((home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?))",
            (g["_date"], *pair, *pair),
        ).fetchone()
        if row:
            g["home_team"], g["away_team"] = row
    conn.close()
    
    # Filter to games before match_date
    past = [g for g in past_games if str(g["_date"]) < str(match_date)]
    
    if not past:
        return None, []
    
    # Compute features
    q1_diffs = []
    home_won_last3 = 0
    last_home_won = 0
    n = len(past)
    
    details = []
    for i, g in enumerate(past):
        qs = g.get("quarters", {})
        is_home = g["home_team"] == home_team
        sign = 1 if is_home else -1
        q1 = qs.get("Q1")
        if q1:
            q1_diffs.append((q1[0] - q1[1]) * sign)
        
        total_h = sum(v[0] for v in qs.values())
        total_a = sum(v[1] for v in qs.values())
        home_won = (total_h > total_a) == is_home
        
        if n - i <= 3:
            if home_won:
                home_won_last3 += 1
        
        if i == n - 1:
            last_home_won = 1 if home_won else 0
        
        details.append({
            "date": g["_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "is_home": is_home,
            "q1_diff": (q1[0] - q1[1]) * sign if q1 else 0,
            "total_h": total_h,
            "total_a": total_a,
            "home_won": home_won,
        })
    
    feats = {
        "h2h_avg_q1_diff": round(sum(q1_diffs) / len(q1_diffs), 3) if q1_diffs else 0.0,
        "h2h_recent3_home_won": round(home_won_last3 / min(n, 3), 3),
        "h2h_last_home_won": last_home_won,
    }
    
    return feats, details


def compute_h2h_from_sofascore(match_id, home_team, away_team, match_date):
    """Compute H2H features from match_h2h (SofaScore data)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    h2h_rows = conn.execute(
        """SELECT date, home_team, away_team,
                  q1_home, q1_away, q2_home, q2_away,
                  q3_home, q3_away, q4_home, q4_away
           FROM match_h2h
           WHERE match_id = ? AND q1_home IS NOT NULL
           ORDER BY date""",
        (match_id,),
    ).fetchall()
    conn.close()
    
    if not h2h_rows:
        return None, []
    
    # Filter to games before match_date
    past_games = []
    for r in h2h_rows:
        if str(r["date"]) >= str(match_date):
            continue
        quarters = {}
        for q in ("q1", "q2", "q3", "q4"):
            h, a = r[f"{q}_home"], r[f"{q}_away"]
            if h is not None and a is not None:
                quarters[q.upper()] = (int(h), int(a))
        if quarters:
            past_games.append({
                "_date": r["date"],
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "quarters": quarters,
            })
    
    if not past_games:
        return None, []
    
    # Compute features
    q1_diffs = []
    home_won_last3 = 0
    last_home_won = 0
    n = len(past_games)
    
    details = []
    for i, g in enumerate(past_games):
        qs = g.get("quarters", {})
        is_home = g["home_team"] == home_team
        sign = 1 if is_home else -1
        q1 = qs.get("Q1")
        if q1:
            q1_diffs.append((q1[0] - q1[1]) * sign)
        
        total_h = sum(v[0] for v in qs.values())
        total_a = sum(v[1] for v in qs.values())
        home_won = (total_h > total_a) == is_home
        
        if n - i <= 3:
            if home_won:
                home_won_last3 += 1
        
        if i == n - 1:
            last_home_won = 1 if home_won else 0
        
        details.append({
            "date": g["_date"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "is_home": is_home,
            "q1_diff": (q1[0] - q1[1]) * sign if q1 else 0,
            "total_h": total_h,
            "total_a": total_a,
            "home_won": home_won,
        })
    
    feats = {
        "h2h_avg_q1_diff": round(sum(q1_diffs) / len(q1_diffs), 3) if q1_diffs else 0.0,
        "h2h_recent3_home_won": round(home_won_last3 / min(n, 3), 3),
        "h2h_last_home_won": last_home_won,
    }
    
    return feats, details


def main():
    # Find matches with both sources - optimized
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Get first 20 matches with SofaScore data
    matches = conn.execute("""
        SELECT DISTINCT 
            l.match_id, 
            s.home_team, 
            s.away_team, 
            s.event_date
        FROM bet_monitor_log_v2 l
        JOIN bet_monitor_schedule_v2 s ON l.match_id = s.match_id
        JOIN match_h2h h ON h.match_id = l.match_id
        WHERE l.model_version = 'm27_v3'
          AND h.q1_home IS NOT NULL
        LIMIT 20
    """).fetchall()
    conn.close()
    
    if not matches:
        print("No hay partidos con datos H2H de SofaScore")
        return
    
    print(f"Buscando partidos con datos en AMBAS fuentes...")
    print(f"Revisando {len(matches)} partidos con SofaScore...")
    
    # Check which ones have data in quarter_scores
    matches_with_both = []
    for i, m in enumerate(matches):
        feats_qs, _ = compute_h2h_from_quarter_scores(
            str(m['match_id']), m['home_team'], m['away_team'], m['event_date']
        )
        if feats_qs:
            matches_with_both.append(m)
            print(f"  [{i+1}] {m['match_id']} - ENCONTRADO en ambas fuentes")
        else:
            print(f"  [{i+1}] {m['match_id']} - Solo en SofaScore")
        
        if len(matches_with_both) >= 3:
            break
    
    if not matches_with_both:
        print("\n[!] No se encontraron partidos con datos en AMBAS fuentes")
        print("\nEsto significa que los partidos de SofaScore son de ligas/equipos")
        print("que NO estaban en la DB original de 30k partidos.")
        return
    
    print(f"\nPartidos con datos en AMBAS fuentes: {len(matches_with_both)}")
    
    # Compare first match
    match_id = str(matches_with_both[0]["match_id"])
    home_team = matches_with_both[0]["home_team"]
    away_team = matches_with_both[0]["away_team"]
    match_date = matches_with_both[0]["event_date"]
    
    print(f"\n{'='*70}")
    print(f"  COMPARACION: {match_id}")
    print(f"  {home_team} vs {away_team} ({match_date})")
    print(f"{'='*70}")
    
    # Get features from both sources
    feats_qs, details_qs = compute_h2h_from_quarter_scores(match_id, home_team, away_team, match_date)
    feats_ss, details_ss = compute_h2h_from_sofascore(match_id, home_team, away_team, match_date)
    
    print(f"\n{'='*70}")
    print(f"  FUENTE 1: quarter_scores (DB original - 30k partidos)")
    print(f"{'='*70}")
    
    if feats_qs:
        print(f"\n  Partidos encontrados: {len(details_qs)}")
        print(f"\n  Detalle:")
        for i, d in enumerate(details_qs):
            role = "HOME" if d["is_home"] else "AWAY"
            won = "SI" if d["home_won"] else "NO"
            print(f"    [{i+1}] {d['date']} | {d['home_team'][:18]} vs {d['away_team'][:18]} | {home_team[:15]}={role} | Q1diff={d['q1_diff']:+d} | Total={d['total_h']}-{d['total_a']} | Gano={won}")
        
        print(f"\n  Features:")
        print(f"    h2h_avg_q1_diff = {feats_qs['h2h_avg_q1_diff']}")
        print(f"    h2h_recent3_home_won = {feats_qs['h2h_recent3_home_won']}")
        print(f"    h2h_last_home_won = {feats_qs['h2h_last_home_won']}")
    else:
        print("\n  [SIN DATOS] No se encontraron partidos en quarter_scores")
    
    print(f"\n{'='*70}")
    print(f"  FUENTE 2: match_h2h (SofaScore - datos nuevos)")
    print(f"{'='*70}")
    
    if feats_ss:
        print(f"\n  Partidos encontrados: {len(details_ss)}")
        print(f"\n  Detalle:")
        for i, d in enumerate(details_ss):
            role = "HOME" if d["is_home"] else "AWAY"
            won = "SI" if d["home_won"] else "NO"
            print(f"    [{i+1}] {d['date']} | {d['home_team'][:18]} vs {d['away_team'][:18]} | {home_team[:15]}={role} | Q1diff={d['q1_diff']:+d} | Total={d['total_h']}-{d['total_a']} | Gano={won}")
        
        print(f"\n  Features:")
        print(f"    h2h_avg_q1_diff = {feats_ss['h2h_avg_q1_diff']}")
        print(f"    h2h_recent3_home_won = {feats_ss['h2h_recent3_home_won']}")
        print(f"    h2h_last_home_won = {feats_ss['h2h_last_home_won']}")
    else:
        print("\n  [SIN DATOS] No se encontraron partidos en match_h2h")
    
    # Comparison
    print(f"\n{'='*70}")
    print(f"  COMPARACION DE FEATURES")
    print(f"{'='*70}")
    
    if feats_qs and feats_ss:
        print(f"\n  {'Feature':<25} {'quarter_scores':>15} {'SofaScore':>15} {'Diferencia':>15}")
        print(f"  {'-'*70}")
        
        for key in ["h2h_avg_q1_diff", "h2h_recent3_home_won", "h2h_last_home_won"]:
            qs_val = feats_qs[key]
            ss_val = feats_ss[key]
            diff = ss_val - qs_val
            print(f"  {key:<25} {qs_val:>15.3f} {ss_val:>15.3f} {diff:>+15.3f}")
        
        print(f"\n  Partidos: {len(details_qs)} vs {len(details_ss)}")
        
        if len(details_ss) > len(details_qs):
            print(f"\n  [!] SofaScore tiene {len(details_ss) - len(details_qs)} partidos MAS")
        elif len(details_qs) > len(details_ss):
            print(f"\n  [!] quarter_scores tiene {len(details_qs) - len(details_ss)} partidos MAS")
    else:
        print("\n  No se puede comparar - una o ambas fuentes no tienen datos")


if __name__ == "__main__":
    main()
