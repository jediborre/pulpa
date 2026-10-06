"""
Script de análisis de clusters de ligas para arquitectura multi-modelo.
Clasifica las ligas en los 3 modelos propuestos:
1. FIBA Senior Masculino (10 min)
2. NBA / 12 Minutos (12 min)
3. FIBA Femenino (10 min)
4. Descartadas / Blacklist (NCAA, juveniles, amistosos)
Identifica ligas con bajo soporte (< 50 o 50-100 matches) para priorizar backfill histórico.
"""
import sqlite3
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Consultar ligas
    rows = cur.execute("""
        SELECT league, country_or_region, confederation, competition_type, stage_detail,
               quarter_duration_minutes, is_women, is_youth, is_college,
               match_count, avg_total_points, scoring_pace_per_minute, blowout_rate
        FROM leagues_classification
        ORDER BY match_count DESC
    """).fetchall()

    clusters = {
        "m27_fiba_men": [],
        "m34_nba_12m": [],
        "m27_fiba_women": [],
        "blacklist": []
    }

    for r in rows:
        d = dict(r)
        q_dur = d["quarter_duration_minutes"]
        is_w = d["is_women"]
        is_y = d["is_youth"]
        is_col = d["is_college"]
        c_type = d["competition_type"]

        # Blacklist
        if is_y == 1 or is_col == 1 or c_type == "friendly":
            clusters["blacklist"].append(d)
        elif q_dur == 12:
            clusters["m34_nba_12m"].append(d)
        elif is_w == 1 and q_dur == 10:
            clusters["m27_fiba_women"].append(d)
        elif is_w == 0 and q_dur == 10:
            clusters["m27_fiba_men"].append(d)
        else:
            clusters["blacklist"].append(d)

    print("=== RESUMEN DE CLUSTERS DE MODELOS ===")
    for name, list_leagues in clusters.items():
        total_m = sum(x["match_count"] for x in list_leagues)
        num_leagues = len(list_leagues)
        leagues_ge_100 = len([x for x in list_leagues if x["match_count"] >= 100])
        leagues_50_99 = len([x for x in list_leagues if 50 <= x["match_count"] < 100])
        leagues_lt_50 = len([x for x in list_leagues if x["match_count"] < 50])
        print(f"\nCluster: {name.upper()}")
        print(f"  - Ligas totales: {num_leagues}")
        print(f"  - Partidos totales: {total_m:,}")
        print(f"  - Ligas con >= 100 partidos: {leagues_ge_100}")
        print(f"  - Ligas con 50-99 partidos: {leagues_50_99}")
        print(f"  - Ligas con < 50 partidos (Candidatas a Backfill): {leagues_lt_50}")

    # Guardar dump JSON para análisis detallado
    output_dir = Path("tmp/archetype_modeling")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "cluster_breakdown.json", "w", encoding="utf-8") as f:
        json.dump(clusters, f, ensure_ascii=False, indent=2)

    # Detalle de ligas clave con pocos partidos en cada cluster
    print("\n=== TOP LIGAS DESTACADAS CON POCOS PARTIDOS (< 50) EN FIBA MEN ===")
    fiba_men_low = [x for x in clusters["m27_fiba_men"] if x["match_count"] < 50]
    for x in sorted(fiba_men_low, key=lambda k: k["match_count"], reverse=True)[:15]:
        print(f"  - {x['league']} ({x['country_or_region']}): {x['match_count']} partidos | Pace: {x['scoring_pace_per_minute']:.2f}")

    print("\n=== TOP LIGAS DESTACADAS CON POCOS PARTIDOS (< 50) EN FIBA WOMEN ===")
    fiba_women_low = [x for x in clusters["m27_fiba_women"] if x["match_count"] < 50]
    for x in sorted(fiba_women_low, key=lambda k: k["match_count"], reverse=True)[:15]:
        print(f"  - {x['league']} ({x['country_or_region']}): {x['match_count']} partidos | Pace: {x['scoring_pace_per_minute']:.2f}")

    print("\n=== LIGAS EN 12M (NBA / OTROS) ===")
    for x in clusters["m34_nba_12m"]:
        print(f"  - {x['league']} ({x['country_or_region']}): {x['match_count']} partidos | Pace: {x['scoring_pace_per_minute']:.2f}")

if __name__ == "__main__":
    main()
