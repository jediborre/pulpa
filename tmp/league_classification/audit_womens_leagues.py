"""
Script de auditoría exhaustiva para ligas femeninas en matches.db.
Revisa nombres de ligas, acrónimos internacionales y nombres de equipos participantes
para detectar ligas femeninas con nombres inusuales o no estándar.
"""

import sqlite3
import re
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

    # 1. Obtener todas las ligas que actualmente están marcadas como 'men'
    men_leagues = cur.execute("""
        SELECT league, clean_name, match_count, avg_total_points
        FROM leagues_classification
        WHERE is_women = 0
    """).fetchall()

    print(f"Total ligas actualmente marcadas como 'men': {len(men_leagues)}")

    # 2. Palabras clave sospechosas en el nombre de los EQUIPOS
    # Si los equipos de una liga tienen 'women', 'fem', 'damen', '(w)', etc., la liga es femenina.
    team_women_patterns = [
        r"\bwomen\b", r"\bwoman\b", r"\bfem\b", r"\bfemenin\w*", r"\bfeminin\w*",
        r"\bmujeres\b", r"\bdamen\b", r"\bdames\b", r"\(w\)", r"\b\(f\)", r"\bženy\b",
        r"\bzeny\b", r"\bmoter\w*", r"\bkadın\w*", r"\bkadin\w*", r"\bnaiste\b",
        r"\bnaiset\b", r"\bkobiet\w*", r"\bkvinner\b", r"\bdamer\b", r"\bnői\b",
        r"\bnoi\b", r"\blbf\b", r"\blfb\b", r"\bwnba\b"
    ]
    regex_team_women = re.compile("|".join(team_women_patterns), re.IGNORECASE)

    # 3. Nombres de liga sospechosos por acrónimos o vocablos extranjeros
    league_women_patterns = [
        r"\bžbl\b", r"\bzbl\b", r"\blnbf\b", r"\blbf\b", r"\blfb\b", r"\bdbbl\b",
        r"\btkbl\b", r"\bkbsl\b", r"\bmlkl\b", r"\bwbbl\b", r"\bwnbl\b", r"\bwaba\b",
        r"\bewbl\b", r"\bžls\b", r"\bzls\b", r"\bkadın\w*", r"\bkadin\w*", r"\bkvinn\w*",
        r"\bdamelig\w*", r"\bnaiste\b", r"\bnaiset\b", r"\bsievie\w*", r"\bmoter\w*",
        r"\bnői\b", r"\bnoi\b", r"\bdam\b", r"\bdamen\b", r"\bdames\b", r"\bkobiet\w*",
        r"\bženy\b", r"\bzeny\b", r"\bdivisão feminina\b", r"\bliga femenina\b",
        r"\blf endesa\b", r"\blf challenge\b", r"\blf2\b", r"\bnf1\b", r"\bnf2\b",
        r"\bnf3\b", r"\bserie a1 femminile\b", r"\ba1 femminile\b", r"\ba2 femminile\b"
    ]
    regex_league_women = re.compile("|".join(league_women_patterns), re.IGNORECASE)

    suspicious_leagues = []

    for l in men_leagues:
        lname = l["league"]
        
        # Check por patrón de liga
        match_league_kw = regex_league_women.search(lname)
        
        # Check por nombres de equipos en esa liga
        teams_sample = cur.execute("""
            SELECT home_team, away_team
            FROM matches
            WHERE league = ?
            LIMIT 25
        """, (lname,)).fetchall()

        women_team_hits = 0
        total_teams_checked = len(teams_sample) * 2
        
        sample_detected = []
        for row in teams_sample:
            for t in (row["home_team"], row["away_team"]):
                if t and regex_team_women.search(t):
                    women_team_hits += 1
                    if len(sample_detected) < 3:
                        sample_detected.append(t)

        team_ratio = (women_team_hits / total_teams_checked) if total_teams_checked > 0 else 0

        # Si el ratio de equipos con indicación femenina es > 20% o el nombre de liga hace match
        if match_league_kw or team_ratio >= 0.20:
            suspicious_leagues.append({
                "league": lname,
                "clean_name": l["clean_name"],
                "match_count": l["match_count"],
                "avg_total_points": l["avg_total_points"],
                "reason": "league_kw" if match_league_kw else f"team_ratio_{team_ratio:.2f}",
                "detected_teams": sample_detected
            })

    print(f"\nLigas detectadas que ERAN FEMENINAS pero estaban marcadas como 'men': {len(suspicious_leagues)}")
    for sl in suspicious_leagues[:30]:
        print(f"  - [{sl['match_count']} m] {sl['league']} (Razón: {sl['reason']}, Equipos: {sl['detected_teams']})")

    Path("tmp/league_classification/missed_women_leagues.json").write_text(
        json.dumps(suspicious_leagues, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
