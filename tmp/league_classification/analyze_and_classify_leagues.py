"""
Script de auditoría y prototipo de clasificación de todas las ligas de matches.db.
Analiza patrones de texto, palabras clave, duraciones y genera estadísticas agregadas.
"""

import sqlite3
import re
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

WOMEN_KEYWORDS = [
    "women", "woman", "femen", "femin", "mujeres", "dames", "damen", "kobiet",
    "kobiety", "ženy", "zeny", "frauen", "donne", "naiste", "nainen", "moterys",
    "damer", "mulheres", "wnba", "lnbf", "lbf", "lfb", "wbbl", "wnbl", "waba",
    "ewbl", "w-league", "w league", "liga femenina", "1. zls", "zls"
]

YOUTH_KEYWORDS = [
    "u14", "u15", "u16", "u17", "u18", "u19", "u20", "u21", "u22", "u23",
    "youth", "junior", "juniores", "juniors", "cadet", "cadete", "jaunimo",
    "young", "primy", "next gen", "angt", "development", "developmental"
]

COLLEGE_KEYWORDS = [
    "ncaa", "naia", "njcaa", "u sports", "usports", "university", "college",
    "march madness", "nit", "national invitation tournament", "cbi"
]

PLAYOFF_KEYWORDS = [
    "playoff", "play-off", "play off", "knockout", "final", "finals", "semifinal",
    "quarterfinal", "cuartos", "semis", "postseason", "play-in", "play in",
    "relegation", "placement", "bronze", "super final"
]

CUP_KEYWORDS = [
    "cup", "copa", "coppa", "pokal", "taça", "taca", "coupe", "puchar", "kupa",
    "trophy", "supercup", "supercopa", "supercoppa", "super coupe", "tournament",
    "torneo", "championship", "all star", "all-star"
]

FRIENDLY_KEYWORDS = [
    "friendly", "club friendly", "amistoso", "preparation", "pre-season", "preseason"
]

TWELVE_MIN_KEYWORDS = [
    "nba", "cba", "pba", "nba g league"
]

def classify_single_league(league_name: str) -> dict:
    name_lower = league_name.lower().strip()
    
    # 1. Separar fase/etapa si existe (delimitado por coma o guión)
    clean_name = league_name
    stage = "Regular season"
    
    if "," in league_name:
        parts = [p.strip() for p in league_name.split(",", 1)]
        clean_name = parts[0]
        stage = parts[1]
    elif " - " in league_name:
        parts = [p.strip() for p in league_name.split(" - ", 1)]
        clean_name = parts[0]
        stage = parts[1]

    # 2. Género
    is_women = 0
    # Comprobación de palabras completas o subcadenas seguras
    if any(k in name_lower for k in WOMEN_KEYWORDS):
        is_women = 1
    # Casos especiales de letras sueltas como 'Ž' en lituano/checo/eslovaco (Ženy / Žiurkės)
    elif " ž" in name_lower or "(w)" in name_lower or " w " in name_lower or name_lower.endswith(" w") or name_lower.endswith(" ž"):
        is_women = 1
    
    gender = "women" if is_women else "men"

    # 3. Categoría de edad / desarrollo
    is_youth = 0
    age_category = "Senior"
    
    for yk in ["u16", "u17", "u18", "u19", "u20", "u21", "u22", "u23"]:
        if re.search(rf"\b{yk}\b", name_lower):
            is_youth = 1
            age_category = yk.upper()
            break
            
    if not is_youth:
        if any(k in name_lower for k in ["youth", "junior", "juniores", "juniors", "cadet", "cadete", "jaunimo"]):
            is_youth = 1
            age_category = "Youth"

    # 4. Universidad / College
    is_college = 1 if any(k in name_lower for k in COLLEGE_KEYWORDS) else 0

    # 5. Playoffs
    is_playoffs = 1 if any(k in name_lower for k in PLAYOFF_KEYWORDS) else 0

    # 6. Tipo de competición
    if any(k in name_lower for k in FRIENDLY_KEYWORDS):
        comp_type = "friendly"
        is_tournament = 0
    elif "all star" in name_lower or "all-star" in name_lower:
        comp_type = "all_star"
        is_tournament = 1
    elif is_playoffs:
        comp_type = "playoffs"
        is_tournament = 1
    elif any(k in name_lower for k in CUP_KEYWORDS):
        comp_type = "cup"
        is_tournament = 1
    else:
        comp_type = "league"
        is_tournament = 0

    # 7. Internacional vs Nacional
    intl_keywords = [
        "euroleague", "eurocup", "champions league", "fiba", "olympic", "world cup",
        "americup", "asiacup", "afrobasket", "vtb united", "bnxt", "adriatic", "aba",
        "alpe adria", "baltic", "bibl", "enbl", "super 8", "wasl", "intercontinental"
    ]
    is_international = 1 if any(k in name_lower for k in intl_keywords) else 0

    # 8. País o Región (heurística basada en nombres)
    country = "Other"
    if is_college or "nba" in name_lower or "wnba" in name_lower or "usa" in name_lower:
        country = "USA"
    elif "spain" in name_lower or "acb" in name_lower or "feb" in name_lower or "españa" in name_lower:
        country = "Spain"
    elif "italy" in name_lower or "serie a" in name_lower or "serie b" in name_lower or "lega" in name_lower:
        country = "Italy"
    elif "germany" in name_lower or "bbl" in name_lower or "pro a" in name_lower and "germany" in name_lower:
        country = "Germany"
    elif "france" in name_lower or "pro a" in name_lower or "pro b" in name_lower or "lnb" in name_lower:
        country = "France"
    elif "poland" in name_lower or "pbl" in name_lower or "plk" in name_lower:
        country = "Poland"
    elif "lithuania" in name_lower or "lkl" in name_lower or "nkl" in name_lower or "rkl" in name_lower:
        country = "Lithuania"
    elif "japan" in name_lower or "b league" in name_lower or "b.league" in name_lower:
        country = "Japan"
    elif "argentina" in name_lower:
        country = "Argentina"
    elif "brazil" in name_lower or "nbb" in name_lower:
        country = "Brazil"
    elif "china" in name_lower or "cba" in name_lower:
        country = "China"
    elif "australia" in name_lower or "nbl" in name_lower:
        country = "Australia"
    elif "philippines" in name_lower or "pba" in name_lower or "mpbl" in name_lower:
        country = "Philippines"
    elif "turkey" in name_lower or "bsl" in name_lower:
        country = "Turkey"
    elif "greece" in name_lower or "gbl" in name_lower:
        country = "Greece"
    elif "israel" in name_lower:
        country = "Israel"
    elif "serbia" in name_lower or "kbl" in name_lower and "korea" in name_lower:
        country = "Korea" if "korea" in name_lower else "Serbia"
    elif is_international:
        country = "International"

    # 9. Regulación de duración de cuartos
    # 12 min: NBA, NBA G League, CBA China, PBA Filipinas
    if any(k in name_lower for k in ["nba", "nba g league", "cba", "pba"]) and "wnba" not in name_lower and "u1" not in name_lower and "u2" not in name_lower:
        quarter_duration = 12
        total_game_minutes = 48
    elif is_college:
        quarter_duration = 10
        total_game_minutes = 40
    else:
        quarter_duration = 10
        total_game_minutes = 40

    # 10. Nivel o Tier competitivo
    if is_college:
        tier_level = "college"
    elif is_youth:
        tier_level = "youth"
    elif any(k in name_lower for k in ["nba", "euroleague", "liga acb", "germany bbl", "france pro a", "italy serie a", "china cba", "australia nbl", "brazil nbb", "argentina liga nacional", "wnba"]):
        tier_level = "top_pro"
    elif any(k in name_lower for k in ["2nd", "g league", "serie a2", "pro b", "division 2", "division b", "primera feb", "leb oro", "1st division", "b league one"]):
        tier_level = "second_pro"
    elif any(k in name_lower for k in ["3rd", "serie b", "leb plata", "division 3"]):
        tier_level = "lower_pro_amateur"
    elif is_international:
        tier_level = "top_pro"
    else:
        tier_level = "standard_pro"

    return {
        "league": league_name,
        "clean_name": clean_name,
        "stage": stage,
        "gender": gender,
        "is_women": is_women,
        "is_youth": is_youth,
        "age_category": age_category,
        "is_college": is_college,
        "competition_type": comp_type,
        "is_tournament": is_tournament,
        "is_playoffs": is_playoffs,
        "is_international": is_international,
        "country_or_region": country,
        "tier_level": tier_level,
        "quarter_duration_minutes": quarter_duration,
        "total_game_minutes": total_game_minutes
    }

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # Extraer ligas y estadísticas
    query = """
    SELECT 
        m.league,
        COUNT(*) as match_count,
        SUM(CASE WHEN m.status_type = 'finished' THEN 1 ELSE 0 END) as finished_matches,
        AVG(CASE WHEN m.status_type = 'finished' THEN m.home_score END) as avg_home,
        AVG(CASE WHEN m.status_type = 'finished' THEN m.away_score END) as avg_away,
        AVG(CASE WHEN m.status_type = 'finished' THEN m.home_score + m.away_score END) as avg_total,
        SUM(CASE WHEN m.status_type = 'finished' AND m.home_score > m.away_score THEN 1 ELSE 0 END) * 100.0 / 
            NULLIF(SUM(CASE WHEN m.status_type = 'finished' THEN 1 ELSE 0 END), 0) as home_win_pct,
        SUM(CASE WHEN m.status_description = 'AET' THEN 1 ELSE 0 END) * 100.0 / 
            NULLIF(SUM(CASE WHEN m.status_type = 'finished' THEN 1 ELSE 0 END), 0) as ot_rate
    FROM matches m
    WHERE m.league IS NOT NULL AND m.league != ''
    GROUP BY m.league
    ORDER BY match_count DESC;
    """
    
    rows = cur.execute(query).fetchall()
    print(f"Total ligas extraidas: {len(rows)}")

    classified = []
    for r in rows:
        data = classify_single_league(r["league"])
        data["match_count"] = r["match_count"]
        data["finished_match_count"] = r["finished_matches"]
        data["avg_home_score"] = round(r["avg_home"], 2) if r["avg_home"] is not None else None
        data["avg_away_score"] = round(r["avg_away"], 2) if r["avg_away"] is not None else None
        data["avg_total_points"] = round(r["avg_total"], 2) if r["avg_total"] is not None else None
        data["home_win_pct"] = round(r["home_win_pct"], 2) if r["home_win_pct"] is not None else None
        data["ot_rate"] = round(r["ot_rate"], 2) if r["ot_rate"] is not None else None
        classified.append(data)

    # Resumen rápido
    women_count = sum(1 for c in classified if c["is_women"] == 1)
    youth_count = sum(1 for c in classified if c["is_youth"] == 1)
    college_count = sum(1 for c in classified if c["is_college"] == 1)
    playoff_count = sum(1 for c in classified if c["is_playoffs"] == 1)
    twelve_min_count = sum(1 for c in classified if c["quarter_duration_minutes"] == 12)

    print(f"Ligas Femeninas: {women_count}")
    print(f"Ligas Juveniles: {youth_count}")
    print(f"Ligas College: {college_count}")
    print(f"Ligas/Fases Playoffs: {playoff_count}")
    print(f"Ligas 12 Minutos: {twelve_min_count}")

    # Guardar en tmp/league_classification/classified_sample.json
    out_path = Path("tmp/league_classification/classified_sample.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(classified[:50], indent=2, ensure_ascii=False), encoding="utf-8")
    print("Muestra guardada en", out_path)

if __name__ == "__main__":
    main()
