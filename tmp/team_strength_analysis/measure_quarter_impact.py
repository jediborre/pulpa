"""
Script para medir empíricamente el impacto de la fuerza relativa de los equipos
(fuerza rodante, calidad de titulares y profundidad de banquillo) en cada cuarto (Q1, Q2, Q3, Q4).
Aborda la hipótesis de fuerza dinámica punto-en-el-tiempo vs estática.
"""

import sqlite3
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("1. Extrayendo muestra de partidos con quarter_scores, team_strength y player_stats...", flush=True)
    
    # Extraemos partidos con parciales completos Q1-Q4
    query = """
    SELECT 
        m.match_id,
        m.date,
        m.league,
        m.home_team,
        m.away_team,
        q1.home as q1_h, q1.away as q1_a,
        q2.home as q2_h, q2.away as q2_a,
        q3.home as q3_h, q3.away as q3_a,
        q4.home as q4_h, q4.away as q4_a,
        m.home_score, m.away_score
    FROM matches m
    JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
    JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
    JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
    JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
    WHERE m.status_type = 'finished'
      AND q1.home IS NOT NULL AND q2.home IS NOT NULL 
      AND q3.home IS NOT NULL AND q4.home IS NOT NULL
      AND m.home_score > 0
    ORDER BY m.date DESC
    LIMIT 10000;
    """
    matches = cur.execute(query).fetchall()
    print(f"Partidos extraídos: {len(matches)}", flush=True)

    # 2. Cargar datos de team_strength punto-en-el-tiempo (wins, losses, form)
    print("Cargando métricas de team_strength previas al partido...", flush=True)
    ts_rows = cur.execute("""
        SELECT match_id, team_name, wins, losses, position
        FROM team_strength
    """).fetchall()
    ts_map = {}
    for r in ts_rows:
        key = (r["match_id"], r["team_name"])
        w = r["wins"] or 0
        l = r["losses"] or 0
        tot = w + l
        win_pct = (w / tot) if tot > 0 else None
        ts_map[key] = {
            "win_pct": win_pct,
            "position": r["position"]
        }

    # 3. Cargar agregaciones de titulares vs banquillo por partido
    print("Cargando métricas de quintetos iniciales y banquillo...", flush=True)
    roster_query = """
    SELECT 
        ps.match_id,
        ps.team,
        AVG(CASE WHEN l.is_starter = 1 THEN ps.sofascore_rating END) as starter_rating,
        AVG(CASE WHEN l.is_starter = 0 THEN ps.sofascore_rating END) as bench_rating,
        SUM(CASE WHEN l.is_starter = 1 THEN ps.points ELSE 0 END) as starter_pts,
        SUM(CASE WHEN l.is_starter = 0 THEN ps.points ELSE 0 END) as bench_pts,
        MAX(ps.points) as star_pts,
        SUM(ps.rebounds) as team_reb,
        SUM(ps.assists) as team_ast,
        SUM(ps.fouls) as team_fouls
    FROM player_stats ps
    JOIN lineups l ON ps.id = l.id
    GROUP BY ps.match_id, ps.team;
    """
    roster_rows = cur.execute(roster_query).fetchall()
    roster_map = {}
    for r in roster_rows:
        roster_map[(r["match_id"], r["team"])] = dict(r)

    print("Procesando métricas de impacto por cuarto...", flush=True)
    
    samples = []
    for m in matches:
        mid = m["match_id"]
        h_ts = ts_map.get((mid, m["home_team"]))
        a_ts = ts_map.get((mid, m["away_team"]))

        h_ros = roster_map.get((mid, "home"))
        a_ros = roster_map.get((mid, "away"))

        # Si tenemos datos de roster para ambos equipos
        if not h_ros or not a_ros:
            continue

        h_starter = h_ros["starter_rating"]
        a_starter = a_ros["starter_rating"]
        h_bench = h_ros["bench_rating"]
        a_bench = a_ros["bench_rating"]

        delta_starters = (h_starter - a_starter) if (h_starter and a_starter) else None
        delta_bench = (h_bench - a_bench) if (h_bench and a_bench) else None
        
        # Profundidad interna: ¿qué tanta caída hay entre titulares y suplentes?
        h_depth = (h_starter - h_bench) if (h_starter and h_bench) else None
        a_depth = (a_starter - a_bench) if (a_starter and a_bench) else None
        delta_depth_advantage = (a_depth - h_depth) if (h_depth and a_depth) else None # menor caída = mayor profundidad

        # Dependencia de la estrella (% puntos del máximo anotador)
        h_pts = (h_ros["starter_pts"] or 0) + (h_ros["bench_pts"] or 0)
        a_pts = (a_ros["starter_pts"] or 0) + (a_ros["bench_pts"] or 0)
        h_star = h_ros["star_pts"] or 0
        a_star = a_ros["star_pts"] or 0
        h_star_share = (h_star / h_pts) if h_pts > 0 else 0
        a_star_share = (a_star / a_pts) if a_pts > 0 else 0
        delta_star_reliance = h_star_share - a_star_share

        # Fuerza rodante de clasificación previa (si existe en team_strength)
        delta_standings_win_pct = None
        if h_ts and a_ts and h_ts["win_pct"] is not None and a_ts["win_pct"] is not None:
            delta_standings_win_pct = h_ts["win_pct"] - a_ts["win_pct"]

        # Margen en cada cuarto
        margin_q1 = m["q1_h"] - m["q1_a"]
        margin_q2 = m["q2_h"] - m["q2_a"]
        margin_q3 = m["q3_h"] - m["q3_a"]
        margin_q4 = m["q4_h"] - m["q4_a"]
        margin_ft = m["home_score"] - m["away_score"]

        samples.append({
            "delta_starters": delta_starters,
            "delta_bench": delta_bench,
            "delta_depth_advantage": delta_depth_advantage,
            "delta_star_reliance": delta_star_reliance,
            "delta_standings_win_pct": delta_standings_win_pct,
            "win_q1": 1 if margin_q1 > 0 else (0 if margin_q1 < 0 else 0.5),
            "win_q2": 1 if margin_q2 > 0 else (0 if margin_q2 < 0 else 0.5),
            "win_q3": 1 if margin_q3 > 0 else (0 if margin_q3 < 0 else 0.5),
            "win_q4": 1 if margin_q4 > 0 else (0 if margin_q4 < 0 else 0.5),
            "win_ft": 1 if margin_ft > 0 else 0,
            "margin_q1": margin_q1,
            "margin_q2": margin_q2,
            "margin_q3": margin_q3,
            "margin_q4": margin_q4,
            "margin_ft": margin_ft
        })

    print(f"Total muestras procesadas con métricas de quintetos: {len(samples)}", flush=True)

    # 4. Calcular correlaciones de Pearson entre cada dimensión de fuerza y cada cuarto
    def pearson(x_list, y_list):
        pairs = [(x, y) for x, y in zip(x_list, y_list) if x is not None and y is not None]
        n = len(pairs)
        if n < 100: return 0.0
        sum_x = sum(p[0] for p in pairs)
        sum_y = sum(p[1] for p in pairs)
        sum_x2 = sum(p[0]**2 for p in pairs)
        sum_y2 = sum(p[1]**2 for p in pairs)
        sum_xy = sum(p[0]*p[1] for p in pairs)
        den = math.sqrt((n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2))
        return (n * sum_xy - sum_x * sum_y) / den if den != 0 else 0.0

    features = [
        ("Diferencial Titulares (Rating)", "delta_starters"),
        ("Diferencial Banquillo (Rating)", "delta_bench"),
        ("Ventaja Profundidad Banquillo", "delta_depth_advantage"),
        ("Dependencia de 1 Estrella", "delta_star_reliance"),
        ("Diferencial % Victorias Previas", "delta_standings_win_pct")
    ]

    targets = [
        ("Cuarto 1 (Q1)", "margin_q1"),
        ("Cuarto 2 (Q2)", "margin_q2"),
        ("Cuarto 3 (Q3)", "margin_q3"),
        ("Cuarto 4 (Q4)", "margin_q4"),
        ("Partido Final (FT)", "margin_ft")
    ]

    results = {}
    for feat_label, feat_key in features:
        results[feat_label] = {}
        feat_vals = [s[feat_key] for s in samples]
        for tgt_label, tgt_key in targets:
            tgt_vals = [s[tgt_key] for s in samples]
            r_corr = pearson(feat_vals, tgt_vals)
            results[feat_label][tgt_label] = round(r_corr, 4)

    # 5. Precisión simple de la regla (si delta > 0, gana el local el cuarto?)
    accuracy_rules = {}
    for feat_label, feat_key in [("Diferencial Titulares", "delta_starters"), ("Diferencial Banquillo", "delta_bench"), ("Victorias Previas", "delta_standings_win_pct")]:
        accuracy_rules[feat_label] = {}
        for q_label, q_key in [("Q1", "win_q1"), ("Q2", "win_q2"), ("Q3", "win_q3"), ("Q4", "win_q4"), ("FT", "win_ft")]:
            valid = [s for s in samples if s[feat_key] is not None]
            hits = sum(1 for s in valid if (s[feat_key] > 0 and s[q_key] == 1) or (s[feat_key] < 0 and s[q_key] == 0))
            acc = (hits / len(valid)) * 100.0 if valid else 0.0
            accuracy_rules[feat_label][q_label] = round(acc, 2)

    output = {
        "sample_size": len(samples),
        "correlations": results,
        "directional_accuracy": accuracy_rules
    }

    Path("tmp/team_strength_analysis/quarter_impact_results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("Resultados guardados exitosamente en quarter_impact_results.json", flush=True)

if __name__ == "__main__":
    main()
