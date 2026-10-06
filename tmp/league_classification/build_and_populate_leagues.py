"""
Script para generar la tabla completa leagues_classification con métricas profundas
(Q4 scores, pbp coverage, graph coverage) y crear la tabla en matches.db.
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
    "ewbl", "w-league", "w league", "liga femenina", "1. zls", "zls", "dmvl",
    "sbl women", "nbl1 women", "superliga women", "dameligaligaen"
]

YOUTH_KEYWORDS = [
    "u14", "u15", "u16", "u17", "u18", "u19", "u20", "u21", "u22", "u23",
    "youth", "junior", "juniores", "juniors", "cadet", "cadete", "jaunimo",
    "young", "primy", "next gen", "angt", "development", "developmental",
    "esperanzas", "liga de desarrollo", "ldd"
]

COLLEGE_KEYWORDS = [
    "ncaa", "naia", "njcaa", "u sports", "usports", "university", "college",
    "march madness", "nit", "national invitation tournament", "cbi"
]

PLAYOFF_KEYWORDS = [
    "playoff", "play-off", "play off", "knockout", "final", "finals", "semifinal",
    "quarterfinal", "cuartos", "semis", "postseason", "play-in", "play in",
    "relegation", "placement", "bronze", "super final", "championship game",
    "consolation", "3rd place", "third place"
]

CUP_KEYWORDS = [
    "cup", "copa", "coppa", "pokal", "taça", "taca", "coupe", "puchar", "kupa",
    "trophy", "supercup", "supercopa", "supercoppa", "super coupe", "tournament",
    "torneo", "championship", "all star", "all-star"
]

FRIENDLY_KEYWORDS = [
    "friendly", "club friendly", "amistoso", "preparation", "pre-season", "preseason"
]

def classify_league(league_name: str) -> dict:
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
    if any(k in name_lower for k in WOMEN_KEYWORDS):
        is_women = 1
    elif re.search(r"\b(w|ž)\b", name_lower):
        is_women = 1
    elif "(w)" in name_lower or name_lower.endswith(" w") or name_lower.endswith(" ž"):
        is_women = 1
    
    gender = "women" if is_women else "men"

    # 3. Categoría de edad / desarrollo
    is_youth = 0
    age_category = "Senior"
    
    for yk in ["u14", "u15", "u16", "u17", "u18", "u19", "u20", "u21", "u22", "u23"]:
        if re.search(rf"\b{yk}\b", name_lower):
            is_youth = 1
            age_category = yk.upper()
            break
            
    if not is_youth:
        if any(k in name_lower for k in ["youth", "junior", "juniores", "juniors", "cadet", "cadete", "jaunimo", "desarrollo", "ldd", "angt"]):
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

    # 8. País o Región
    country = "Other"
    if is_college or "nba" in name_lower or "wnba" in name_lower or "usa" in name_lower:
        country = "USA"
    elif any(k in name_lower for k in ["spain", "acb", "feb", "españa"]):
        country = "Spain"
    elif any(k in name_lower for k in ["italy", "serie a", "serie b", "lega a", "lega basket"]):
        country = "Italy"
    elif any(k in name_lower for k in ["germany", "bbl"]):
        country = "Germany"
    elif any(k in name_lower for k in ["france", "pro a", "pro b", "lnb"]):
        country = "France"
    elif any(k in name_lower for k in ["poland", "pbl", "plk", "polska"]):
        country = "Poland"
    elif any(k in name_lower for k in ["lithuania", "lkl", "nkl", "rkl"]):
        country = "Lithuania"
    elif any(k in name_lower for k in ["japan", "b league", "b.league"]):
        country = "Japan"
    elif "argentina" in name_lower:
        country = "Argentina"
    elif any(k in name_lower for k in ["brazil", "nbb", "brasil"]):
        country = "Brazil"
    elif any(k in name_lower for k in ["china", "cba"]):
        country = "China"
    elif any(k in name_lower for k in ["australia", "nbl"]):
        country = "Australia"
    elif any(k in name_lower for k in ["philippines", "pba", "mpbl"]):
        country = "Philippines"
    elif any(k in name_lower for k in ["turkey", "bsl", "türkiye"]):
        country = "Turkey"
    elif any(k in name_lower for k in ["greece", "gbl", "hellas"]):
        country = "Greece"
    elif "israel" in name_lower:
        country = "Israel"
    elif "serbia" in name_lower:
        country = "Serbia"
    elif "korea" in name_lower or "kbl" in name_lower:
        country = "Korea"
    elif is_international:
        country = "International"

    # 9. Regulación de duración de cuartos
    # 12 min: NBA, NBA G League, CBA China, PBA Filipinas
    if any(k in name_lower for k in ["nba", "nba g league", "cba", "pba"]) and "wnba" not in name_lower and not is_youth:
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
    elif any(k in name_lower for k in ["nba", "euroleague", "liga acb", "germany bbl", "france pro a", "italy serie a", "china cba", "australia nbl", "brazil nbb", "argentina liga nacional", "wnba", "korean basketball league"]):
        tier_level = "top_pro"
    elif any(k in name_lower for k in ["2nd", "g league", "serie a2", "pro b", "division 2", "division b", "primera feb", "leb oro", "1st division", "b league one", "b league premier", "super league", "mpbl", "adriatic league"]):
        tier_level = "second_pro"
    elif any(k in name_lower for k in ["3rd", "serie b", "leb plata", "division 3", "rkl division b"]):
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

def populate_database_table(con: sqlite3.Connection):
    cur = con.cursor()

    # 1. Crear tabla si no existe
    cur.execute("DROP TABLE IF EXISTS leagues_classification;")
    cur.execute("""
    CREATE TABLE leagues_classification (
        league                    TEXT PRIMARY KEY,
        clean_name                TEXT NOT NULL,
        stage                     TEXT NOT NULL,
        gender                    TEXT NOT NULL,
        is_women                  INTEGER NOT NULL,
        is_youth                  INTEGER NOT NULL,
        age_category              TEXT NOT NULL,
        is_college                INTEGER NOT NULL,
        competition_type          TEXT NOT NULL,
        is_tournament             INTEGER NOT NULL,
        is_playoffs               INTEGER NOT NULL,
        is_international          INTEGER NOT NULL,
        country_or_region         TEXT NOT NULL,
        tier_level                TEXT NOT NULL,
        quarter_duration_minutes  INTEGER NOT NULL,
        total_game_minutes        INTEGER NOT NULL,
        match_count               INTEGER NOT NULL,
        finished_match_count      INTEGER NOT NULL,
        avg_home_score            REAL,
        avg_away_score            REAL,
        avg_total_points          REAL,
        home_win_pct              REAL,
        ot_rate                   REAL,
        avg_q4_total_points       REAL,
        avg_q4_margin             REAL,
        pbp_coverage_pct          REAL,
        graph_coverage_pct        REAL,
        updated_at                TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """)

    # 2. Consultar estadísticas de matches agrupadas por liga
    print("Calculando estadísticas base por liga...")
    base_stats = cur.execute("""
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
        GROUP BY m.league;
    """).fetchall()

    print("Calculando estadísticas de Q4 por liga...")
    # Agrupar Q4 scores por liga
    q4_stats = cur.execute("""
        SELECT 
            m.league,
            AVG(qs.home + qs.away) as avg_q4_total,
            AVG(ABS(qs.home - qs.away)) as avg_q4_margin
        FROM matches m
        JOIN quarter_scores qs ON m.match_id = qs.match_id AND qs.quarter = 'Q4'
        WHERE m.status_type = 'finished' AND qs.home IS NOT NULL AND qs.away IS NOT NULL
        GROUP BY m.league;
    """).fetchall()
    q4_map = {r["league"]: (r["avg_q4_total"], r["avg_q4_margin"]) for r in q4_stats}

    print("Calculando cobertura de PBP y Graph por liga...")
    pbp_stats = cur.execute("""
        SELECT 
            m.league,
            COUNT(DISTINCT pbp.match_id) * 100.0 / COUNT(DISTINCT m.match_id) as pbp_cov
        FROM matches m
        LEFT JOIN play_by_play pbp ON m.match_id = pbp.match_id
        GROUP BY m.league;
    """).fetchall()
    pbp_map = {r["league"]: r["pbp_cov"] for r in pbp_stats}

    graph_stats = cur.execute("""
        SELECT 
            m.league,
            COUNT(DISTINCT gp.match_id) * 100.0 / COUNT(DISTINCT m.match_id) as graph_cov
        FROM matches m
        LEFT JOIN graph_points gp ON m.match_id = gp.match_id
        GROUP BY m.league;
    """).fetchall()
    graph_map = {r["league"]: r["graph_cov"] for r in graph_stats}

    # 3. Insertar registros clasificados
    print("Insertando registros en leagues_classification...")
    insert_records = []
    for r in base_stats:
        league_name = r["league"]
        c = classify_league(league_name)
        
        q4_tot, q4_mar = q4_map.get(league_name, (None, None))
        pbp_cov = pbp_map.get(league_name, 0.0)
        graph_cov = graph_map.get(league_name, 0.0)

        record = (
            c["league"],
            c["clean_name"],
            c["stage"],
            c["gender"],
            c["is_women"],
            c["is_youth"],
            c["age_category"],
            c["is_college"],
            c["competition_type"],
            c["is_tournament"],
            c["is_playoffs"],
            c["is_international"],
            c["country_or_region"],
            c["tier_level"],
            c["quarter_duration_minutes"],
            c["total_game_minutes"],
            r["match_count"],
            r["finished_matches"],
            round(r["avg_home"], 2) if r["avg_home"] is not None else None,
            round(r["avg_away"], 2) if r["avg_away"] is not None else None,
            round(r["avg_total"], 2) if r["avg_total"] is not None else None,
            round(r["home_win_pct"], 2) if r["home_win_pct"] is not None else None,
            round(r["ot_rate"], 2) if r["ot_rate"] is not None else None,
            round(q4_tot, 2) if q4_tot is not None else None,
            round(q4_mar, 2) if q4_mar is not None else None,
            round(pbp_cov, 2) if pbp_cov is not None else 0.0,
            round(graph_cov, 2) if graph_cov is not None else 0.0
        )
        insert_records.append(record)

    cur.executemany("""
    INSERT INTO leagues_classification (
        league, clean_name, stage, gender, is_women, is_youth, age_category,
        is_college, competition_type, is_tournament, is_playoffs, is_international,
        country_or_region, tier_level, quarter_duration_minutes, total_game_minutes,
        match_count, finished_match_count, avg_home_score, avg_away_score,
        avg_total_points, home_win_pct, ot_rate, avg_q4_total_points, avg_q4_margin,
        pbp_coverage_pct, graph_coverage_pct
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, insert_records)

    # Crear índices
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_clean_name ON leagues_classification(clean_name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_gender ON leagues_classification(gender);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_is_women ON leagues_classification(is_women);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_is_youth ON leagues_classification(is_youth);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_is_college ON leagues_classification(is_college);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_is_playoffs ON leagues_classification(is_playoffs);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_tier_level ON leagues_classification(tier_level);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_country ON leagues_classification(country_or_region);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_q_duration ON leagues_classification(quarter_duration_minutes);")

    con.commit()
    print(f"Tabla leagues_classification creada e indexada con {len(insert_records)} filas exitosamente.")

if __name__ == "__main__":
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    populate_database_table(con)
