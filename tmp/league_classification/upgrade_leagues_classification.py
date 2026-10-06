"""
Script mejorado y ultra-optimizado para ampliar la tabla leagues_classification con:
1. Clasificación 100% precisa de ligas femeninas (incorporando vocablos internacionales y acrónimos).
2. Clasificación detallada de fases eliminatorias: is_final, is_semifinal, is_quarterfinal, is_relegation, stage_detail.
3. Métricas cuantitativas avanzadas para ML: scoring_pace_per_minute, points_std_dev, blowout_rate (>=15 pts), clutch_rate (<=5 pts), q4_home_win_pct, q4_points_ratio, confederation.
"""

import sqlite3
import re
import math
import sys
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

# Palabras clave y abreviaturas expandidas para ligas femeninas
WOMEN_PATTERNS = [
    r"\bwomen\b", r"\bwoman\b", r"\bfemen\w*", r"\bfemin\w*", r"\bfem\b", r"\bmujeres\b",
    r"\bdamen\b", r"\bdames\b", r"\bkobiet\w*", r"\bženy\b", r"\bzeny\b", r"\bfrauen\b",
    r"\bdonne\b", r"\bnaiste\b", r"\bnaiset\b", r"\bmoter\w*", r"\bdamer\b", r"\bmulheres\b",
    r"\bwnba\b", r"\blnbf\b", r"\blbf\b", r"\blfb\b", r"\bwbbl\b", r"\bwnbl\b", r"\bwaba\b",
    r"\bewbl\b", r"\bw-league\b", r"\bw league\b", r"\bliga femenina\b", r"\b1\. zls\b",
    r"\b2\. zls\b", r"\bzls\b", r"\bžls\b", r"\bdmvl\b", r"\bsbl women\b", r"\bnbl1 women\b",
    r"\bsuperliga women\b", r"\bdameligaen\b", r"\bžbl\b", r"\bzbl\b", r"\bdbbl\b",
    r"\btkbl\b", r"\bkbsl\b", r"\bmlkl\b", r"\bkadın\w*", r"\bkadin\w*", r"\bkvinn\w*",
    r"\bsievie\w*", r"\bnői\b", r"\bnoi\b", r"\bženska\b", r"\bzenska\b", r"\bženske\b",
    r"\bzenske\b", r"\blf challenge\b", r"\blf endesa\b", r"\blf2\b", r"\bnf1\b", r"\bnf2\b",
    r"\bnf3\b", r"\bserie a1 femminile\b", r"\bserie a2 femminile\b", r"\bkorisliiga naiset\b"
]
REGEX_WOMEN = re.compile("|".join(WOMEN_PATTERNS), re.IGNORECASE)

FINAL_PATTERNS = [
    r"\bfinal\b", r"\bfinals\b", r"\bgran final\b", r"\bchampionship game\b", r"\bsuper final\b",
    r"\bgold medal\b", r"\bchampionship final\b", r"\bfinal match\b"
]
REGEX_FINAL = re.compile("|".join(FINAL_PATTERNS), re.IGNORECASE)

SEMIFINAL_PATTERNS = [
    r"\bsemifinal\b", r"\bsemifinals\b", r"\bsemis\b", r"\bfinal-four\b", r"\bfinal four\b",
    r"\bsemi-final\b", r"\bsemi-finals\b"
]
REGEX_SEMI = re.compile("|".join(SEMIFINAL_PATTERNS), re.IGNORECASE)

QUARTERFINAL_PATTERNS = [
    r"\bquarterfinal\b", r"\bquarterfinals\b", r"\bcuartos\b", r"\bquarter-final\b", r"\bquarter-finals\b"
]
REGEX_QUARTER = re.compile("|".join(QUARTERFINAL_PATTERNS), re.IGNORECASE)

PLAYIN_PATTERNS = [
    r"\bplay-in\b", r"\bplay in\b", r"\bplayin\b", r"\bwild card\b"
]
REGEX_PLAYIN = re.compile("|".join(PLAYIN_PATTERNS), re.IGNORECASE)

RELEGATION_PATTERNS = [
    r"\brelegation\b", r"\bplayout\b", r"\bplay-out\b", r"\bplay out\b", r"\bdescenso\b",
    r"\bpermanencia\b", r"\bizlučne\b", r"\bplayouts\b"
]
REGEX_RELEGATION = re.compile("|".join(RELEGATION_PATTERNS), re.IGNORECASE)

BRONZE_PATTERNS = [
    r"\bbronze\b", r"\b3rd place\b", r"\bthird place\b", r"\btercer puesto\b", r"\bconsolation\b"
]
REGEX_BRONZE = re.compile("|".join(BRONZE_PATTERNS), re.IGNORECASE)

YOUTH_PATTERNS = [
    r"\bu14\b", r"\bu15\b", r"\bu16\b", r"\bu17\b", r"\bu18\b", r"\bu19\b", r"\bu20\b", r"\bu21\b",
    r"\bu22\b", r"\bu23\b", r"\byouth\b", r"\bjunior\b", r"\bjuniores\b", r"\bjuniors\b",
    r"\bcadet\b", r"\bcadete\b", r"\bjaunimo\b", r"\byoung\b", r"\bprimy\b", r"\bnext gen\b",
    r"\bangt\b", r"\bdevelopment\b", r"\bdevelopmental\b", r"\besperanzas\b", r"\bldd\b"
]
REGEX_YOUTH = re.compile("|".join(YOUTH_PATTERNS), re.IGNORECASE)

COLLEGE_PATTERNS = [
    r"\bncaa\b", r"\bnaia\b", r"\bnjcaa\b", r"\bu sports\b", r"\busports\b", r"\buniversity\b",
    r"\bcollege\b", r"\bmarch madness\b", r"\bnit\b", r"\bnational invitation tournament\b", r"\bcbi\b"
]
REGEX_COLLEGE = re.compile("|".join(COLLEGE_PATTERNS), re.IGNORECASE)

PLAYOFF_PATTERNS = [
    r"\bplayoff\b", r"\bplay-off\b", r"\bplay off\b", r"\bknockout\b", r"\bpostseason\b",
    r"\bchampionship\b", r"\bcuartos\b", r"\bsemifinal\b", r"\bfinal\b", r"\bplay-in\b", r"\bplay in\b"
]
REGEX_PLAYOFF = re.compile("|".join(PLAYOFF_PATTERNS), re.IGNORECASE)

CUP_PATTERNS = [
    r"\bcup\b", r"\bcopa\b", r"\bcoppa\b", r"\bpokal\b", r"\btaça\b", r"\btaca\b", r"\bcoupe\b",
    r"\bpuchar\b", r"\bkupa\b", r"\btrophy\b", r"\bsupercup\b", r"\bsupercopa\b", r"\bsupercoppa\b",
    r"\bsuper coupe\b", r"\btournament\b", r"\btorneo\b"
]
REGEX_CUP = re.compile("|".join(CUP_PATTERNS), re.IGNORECASE)

FRIENDLY_PATTERNS = [
    r"\bfriendly\b", r"\bclub friendly\b", r"\bamistoso\b", r"\bpreparation\b", r"\bpre-season\b", r"\bpreseason\b"
]
REGEX_FRIENDLY = re.compile("|".join(FRIENDLY_PATTERNS), re.IGNORECASE)

def classify_league_enhanced(league_name: str) -> dict:
    name_lower = league_name.lower().strip()
    
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

    # 1. Género
    is_women = 0
    if REGEX_WOMEN.search(name_lower):
        is_women = 1
    elif re.search(r"\b(w|ž)\b", name_lower) or "(w)" in name_lower or name_lower.endswith(" w") or name_lower.endswith(" ž"):
        is_women = 1
    elif "seleções fem" in name_lower or "cbi u15 fem" in name_lower or "cbi u17 fem" in name_lower:
        is_women = 1
    
    gender = "women" if is_women else "men"

    # 2. Categoría de edad
    is_youth = 0
    age_category = "Senior"
    for yk in ["u14", "u15", "u16", "u17", "u18", "u19", "u20", "u21", "u22", "u23"]:
        if re.search(rf"\b{yk}\b", name_lower):
            is_youth = 1
            age_category = yk.upper()
            break
            
    if not is_youth and REGEX_YOUTH.search(name_lower):
        is_youth = 1
        age_category = "Youth"

    # 3. College
    is_college = 1 if REGEX_COLLEGE.search(name_lower) else 0

    # 4. Fases granulares con prioridad jerárquica
    is_quarterfinal = 1 if REGEX_QUARTER.search(name_lower) else 0
    is_semifinal = 1 if (REGEX_SEMI.search(name_lower) and not is_quarterfinal) else 0
    is_playin = 1 if REGEX_PLAYIN.search(name_lower) else 0
    is_relegation = 1 if REGEX_RELEGATION.search(name_lower) else 0
    is_bronze = 1 if REGEX_BRONZE.search(name_lower) else 0
    
    # is_final solo si es la Gran Final por el título (no cuartos, no semis, no bronce)
    is_final = 1 if (
        REGEX_FINAL.search(name_lower) and 
        not is_semifinal and 
        not is_quarterfinal and 
        not is_bronze and 
        not is_playin and 
        not is_relegation
    ) else 0
    
    is_playoffs = 1 if (REGEX_PLAYOFF.search(name_lower) or is_final or is_semifinal or is_quarterfinal or is_playin) else 0

    if is_final:
        stage_detail = "Final / Championship"
    elif is_bronze:
        stage_detail = "Bronze / 3rd Place"
    elif is_semifinal:
        stage_detail = "Semifinal / Final Four"
    elif is_quarterfinal:
        stage_detail = "Quarterfinal"
    elif is_playin:
        stage_detail = "Play-in"
    elif is_relegation:
        stage_detail = "Relegation / Playout"
    elif is_playoffs:
        stage_detail = "Playoffs"
    elif "group" in name_lower or "grupo" in name_lower:
        stage_detail = "Group Stage"
    elif "cup" in name_lower or "copa" in name_lower:
        stage_detail = "Cup Stage"
    else:
        stage_detail = "Regular Season"

    # 5. Tipo de competición
    if REGEX_FRIENDLY.search(name_lower):
        comp_type = "friendly"
        is_tournament = 0
    elif "all star" in name_lower or "all-star" in name_lower:
        comp_type = "all_star"
        is_tournament = 1
    elif is_playoffs:
        comp_type = "playoffs"
        is_tournament = 1
    elif REGEX_CUP.search(name_lower):
        comp_type = "cup"
        is_tournament = 1
    else:
        comp_type = "league"
        is_tournament = 0

    # 6. Internacional
    intl_keywords = [
        "euroleague", "eurocup", "champions league", "fiba", "olympic", "world cup",
        "americup", "asiacup", "afrobasket", "vtb united", "bnxt", "adriatic", "aba",
        "alpe adria", "baltic", "bibl", "enbl", "super 8", "wasl", "intercontinental", "waba"
    ]
    is_international = 1 if any(k in name_lower for k in intl_keywords) else 0

    # 7. País y Confederación
    country = "Other"
    confederation = "FIBA_EUROPE"

    if is_college or "nba" in name_lower or "wnba" in name_lower or "usa" in name_lower:
        country = "USA"
        confederation = "NBA" if ("nba" in name_lower or "wnba" in name_lower) else "NCAA"
    elif any(k in name_lower for k in ["spain", "acb", "feb", "españa"]):
        country = "Spain"
        confederation = "FIBA_EUROPE"
    elif any(k in name_lower for k in ["italy", "serie a", "serie b", "lega a", "lega basket"]):
        country = "Italy"
        confederation = "FIBA_EUROPE"
    elif any(k in name_lower for k in ["germany", "bbl"]):
        country = "Germany"
        confederation = "FIBA_EUROPE"
    elif any(k in name_lower for k in ["france", "pro a", "pro b", "lnb"]):
        country = "France"
        confederation = "FIBA_EUROPE"
    elif any(k in name_lower for k in ["poland", "pbl", "plk", "polska"]):
        country = "Poland"
        confederation = "FIBA_EUROPE"
    elif any(k in name_lower for k in ["lithuania", "lkl", "nkl", "rkl"]):
        country = "Lithuania"
        confederation = "FIBA_EUROPE"
    elif any(k in name_lower for k in ["japan", "b league", "b.league"]):
        country = "Japan"
        confederation = "FIBA_ASIA"
    elif "argentina" in name_lower:
        country = "Argentina"
        confederation = "FIBA_AMERICAS"
    elif any(k in name_lower for k in ["brazil", "nbb", "brasil", "cbi", "lnbf", "lbf"]):
        country = "Brazil"
        confederation = "FIBA_AMERICAS"
    elif any(k in name_lower for k in ["china", "cba"]):
        country = "China"
        confederation = "FIBA_ASIA"
    elif any(k in name_lower for k in ["australia", "nbl", "wnbl"]):
        country = "Australia"
        confederation = "FIBA_OCEANIA"
    elif any(k in name_lower for k in ["philippines", "pba", "mpbl"]):
        country = "Philippines"
        confederation = "FIBA_ASIA"
    elif any(k in name_lower for k in ["turkey", "bsl", "türkiye", "kbsl", "tkbl"]):
        country = "Turkey"
        confederation = "FIBA_EUROPE"
    elif any(k in name_lower for k in ["greece", "gbl", "hellas"]):
        country = "Greece"
        confederation = "FIBA_EUROPE"
    elif "israel" in name_lower:
        country = "Israel"
        confederation = "FIBA_EUROPE"
    elif "korea" in name_lower or "kbl" in name_lower:
        country = "Korea"
        confederation = "FIBA_ASIA"
    elif is_international:
        country = "International"
        confederation = "INTERNATIONAL"

    # 8. Duraciones
    if any(k in name_lower for k in ["nba", "nba g league", "cba", "pba"]) and "wnba" not in name_lower and not is_youth:
        quarter_duration = 12
        total_game_minutes = 48
    else:
        quarter_duration = 10
        total_game_minutes = 40

    # 9. Tier
    if is_college:
        tier_level = "college"
    elif is_youth:
        tier_level = "youth"
    elif any(k in name_lower for k in ["nba", "euroleague", "liga acb", "germany bbl", "france pro a", "italy serie a", "china cba", "australia nbl", "brazil nbb", "argentina liga nacional", "wnba"]):
        tier_level = "top_pro"
    elif any(k in name_lower for k in ["2nd", "g league", "serie a2", "pro b", "division 2", "division b", "primera feb", "leb oro", "1st division", "b league one", "b league premier", "super league", "mpbl", "adriatic league", "lf challenge"]):
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
        "stage_detail": stage_detail,
        "gender": gender,
        "is_women": is_women,
        "is_youth": is_youth,
        "age_category": age_category,
        "is_college": is_college,
        "competition_type": comp_type,
        "is_tournament": is_tournament,
        "is_playoffs": is_playoffs,
        "is_final": is_final,
        "is_semifinal": is_semifinal,
        "is_quarterfinal": is_quarterfinal,
        "is_relegation": is_relegation,
        "is_international": is_international,
        "country_or_region": country,
        "confederation": confederation,
        "tier_level": tier_level,
        "quarter_duration_minutes": quarter_duration,
        "total_game_minutes": total_game_minutes
    }

def upgrade_database():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    print("1. Re-creando tabla leagues_classification...", flush=True)
    cur.execute("DROP TABLE IF EXISTS leagues_classification;")
    cur.execute("""
    CREATE TABLE leagues_classification (
        league                    TEXT PRIMARY KEY,
        clean_name                TEXT NOT NULL,
        stage                     TEXT NOT NULL,
        stage_detail              TEXT NOT NULL,
        gender                    TEXT NOT NULL,
        is_women                  INTEGER NOT NULL,
        is_youth                  INTEGER NOT NULL,
        age_category              TEXT NOT NULL,
        is_college                INTEGER NOT NULL,
        competition_type          TEXT NOT NULL,
        is_tournament             INTEGER NOT NULL,
        is_playoffs               INTEGER NOT NULL,
        is_final                  INTEGER NOT NULL,
        is_semifinal              INTEGER NOT NULL,
        is_quarterfinal           INTEGER NOT NULL,
        is_relegation             INTEGER NOT NULL,
        is_international          INTEGER NOT NULL,
        country_or_region         TEXT NOT NULL,
        confederation             TEXT NOT NULL,
        tier_level                TEXT NOT NULL,
        quarter_duration_minutes  INTEGER NOT NULL,
        total_game_minutes        INTEGER NOT NULL,
        match_count               INTEGER NOT NULL,
        finished_match_count      INTEGER NOT NULL,
        avg_home_score            REAL,
        avg_away_score            REAL,
        avg_total_points          REAL,
        points_std_dev            REAL,
        scoring_pace_per_minute   REAL,
        home_win_pct              REAL,
        ot_rate                   REAL,
        blowout_rate              REAL,
        clutch_rate               REAL,
        avg_q4_total_points       REAL,
        avg_q4_margin             REAL,
        q4_home_win_pct           REAL,
        q4_points_ratio           REAL,
        pbp_coverage_pct          REAL,
        graph_coverage_pct        REAL,
        updated_at                TEXT NOT NULL DEFAULT (datetime('now'))
    );
    """)

    print("2. Calculando métricas de matches agrupadas...", flush=True)
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
                NULLIF(SUM(CASE WHEN m.status_type = 'finished' THEN 1 ELSE 0 END), 0) as ot_rate,
            SUM(CASE WHEN m.status_type = 'finished' AND ABS(m.home_score - m.away_score) >= 15 THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(SUM(CASE WHEN m.status_type = 'finished' THEN 1 ELSE 0 END), 0) as blowout_rate,
            SUM(CASE WHEN m.status_type = 'finished' AND ABS(m.home_score - m.away_score) <= 5 THEN 1 ELSE 0 END) * 100.0 /
                NULLIF(SUM(CASE WHEN m.status_type = 'finished' THEN 1 ELSE 0 END), 0) as clutch_rate
        FROM matches m
        WHERE m.league IS NOT NULL AND m.league != ''
        GROUP BY m.league;
    """).fetchall()

    print("3. Calculando desviación estándar...", flush=True)
    std_stats = cur.execute("""
        SELECT 
            league,
            AVG((home_score + away_score) * (home_score + away_score)) - 
            (AVG(home_score + away_score) * AVG(home_score + away_score)) as variance
        FROM matches
        WHERE status_type = 'finished' AND home_score IS NOT NULL AND away_score IS NOT NULL
        GROUP BY league;
    """).fetchall()
    std_map = {}
    for r in std_stats:
        var = r["variance"]
        std_map[r["league"]] = round(math.sqrt(var), 2) if var is not None and var > 0 else 0.0

    print("4. Calculando métricas de Q4...", flush=True)
    q4_stats = cur.execute("""
        SELECT 
            m.league,
            AVG(qs.home + qs.away) as avg_q4_total,
            AVG(ABS(qs.home - qs.away)) as avg_q4_margin,
            SUM(CASE WHEN qs.home > qs.away THEN 1 ELSE 0 END) * 100.0 / 
                NULLIF(COUNT(*), 0) as q4_home_win_pct
        FROM matches m
        JOIN quarter_scores qs ON m.match_id = qs.match_id AND qs.quarter = 'Q4'
        WHERE m.status_type = 'finished' AND qs.home IS NOT NULL AND qs.away IS NOT NULL
        GROUP BY m.league;
    """).fetchall()
    q4_map = {r["league"]: (r["avg_q4_total"], r["avg_q4_margin"], r["q4_home_win_pct"]) for r in q4_stats}

    print("5. Calculando cobertura PBP y Graph en memoria ultrarrápida...", flush=True)
    # Cargar sets de IDs únicos (100% libre de joins cartesianos)
    pbp_matches = set(r[0] for r in cur.execute("SELECT DISTINCT match_id FROM play_by_play").fetchall())
    graph_matches = set(r[0] for r in cur.execute("SELECT DISTINCT match_id FROM graph_points").fetchall())
    
    # Contar por liga
    league_counts = defaultdict(lambda: {"total": 0, "pbp": 0, "graph": 0})
    for r in cur.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL").fetchall():
        mid, l = r["match_id"], r["league"]
        league_counts[l]["total"] += 1
        if mid in pbp_matches:
            league_counts[l]["pbp"] += 1
        if mid in graph_matches:
            league_counts[l]["graph"] += 1

    cov_map = {}
    for l, d in league_counts.items():
        tot = d["total"]
        cov_map[l] = (
            round((d["pbp"] * 100.0 / tot), 2) if tot > 0 else 0.0,
            round((d["graph"] * 100.0 / tot), 2) if tot > 0 else 0.0
        )

    print("6. Insertando registros enriquecidos...", flush=True)
    insert_records = []
    women_count = 0
    final_count = 0

    for r in base_stats:
        lname = r["league"]
        c = classify_league_enhanced(lname)
        if c["is_women"]: women_count += 1
        if c["is_final"]: final_count += 1

        pts_std = std_map.get(lname, 0.0)
        q4_tot, q4_mar, q4_hwin = q4_map.get(lname, (None, None, None))
        pbp_cov, graph_cov = cov_map.get(lname, (0.0, 0.0))

        avg_tot = r["avg_total"]
        game_mins = c["total_game_minutes"]
        pace = round(avg_tot / game_mins, 3) if avg_tot is not None and game_mins > 0 else None
        q4_ratio = round((q4_tot / avg_tot) * 100.0, 2) if q4_tot is not None and avg_tot is not None and avg_tot > 0 else None

        insert_records.append((
            c["league"],
            c["clean_name"],
            c["stage"],
            c["stage_detail"],
            c["gender"],
            c["is_women"],
            c["is_youth"],
            c["age_category"],
            c["is_college"],
            c["competition_type"],
            c["is_tournament"],
            c["is_playoffs"],
            c["is_final"],
            c["is_semifinal"],
            c["is_quarterfinal"],
            c["is_relegation"],
            c["is_international"],
            c["country_or_region"],
            c["confederation"],
            c["tier_level"],
            c["quarter_duration_minutes"],
            c["total_game_minutes"],
            r["match_count"],
            r["finished_matches"],
            round(r["avg_home"], 2) if r["avg_home"] is not None else None,
            round(r["avg_away"], 2) if r["avg_away"] is not None else None,
            round(avg_tot, 2) if avg_tot is not None else None,
            pts_std,
            pace,
            round(r["home_win_pct"], 2) if r["home_win_pct"] is not None else None,
            round(r["ot_rate"], 2) if r["ot_rate"] is not None else None,
            round(r["blowout_rate"], 2) if r["blowout_rate"] is not None else None,
            round(r["clutch_rate"], 2) if r["clutch_rate"] is not None else None,
            round(q4_tot, 2) if q4_tot is not None else None,
            round(q4_mar, 2) if q4_mar is not None else None,
            round(q4_hwin, 2) if q4_hwin is not None else None,
            q4_ratio,
            pbp_cov,
            graph_cov
        ))

    cur.executemany("""
    INSERT INTO leagues_classification (
        league, clean_name, stage, stage_detail, gender, is_women, is_youth,
        age_category, is_college, competition_type, is_tournament, is_playoffs,
        is_final, is_semifinal, is_quarterfinal, is_relegation, is_international,
        country_or_region, confederation, tier_level, quarter_duration_minutes,
        total_game_minutes, match_count, finished_match_count, avg_home_score,
        avg_away_score, avg_total_points, points_std_dev, scoring_pace_per_minute,
        home_win_pct, ot_rate, blowout_rate, clutch_rate, avg_q4_total_points,
        avg_q4_margin, q4_home_win_pct, q4_points_ratio, pbp_coverage_pct,
        graph_coverage_pct
    ) VALUES (
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
    );
    """, insert_records)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_clean_name ON leagues_classification(clean_name);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_gender ON leagues_classification(gender);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_is_women ON leagues_classification(is_women);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_is_final ON leagues_classification(is_final);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_is_playoffs ON leagues_classification(is_playoffs);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_stage_detail ON leagues_classification(stage_detail);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_confederation ON leagues_classification(confederation);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_lc_q_duration ON leagues_classification(quarter_duration_minutes);")

    con.commit()
    print(f"Upgrade exitoso: {len(insert_records)} ligas actualizadas.", flush=True)
    print(f"Total ligas femeninas identificadas: {women_count}", flush=True)
    print(f"Total ligas/fases de FINALES identificadas: {final_count}", flush=True)

if __name__ == "__main__":
    upgrade_database()
