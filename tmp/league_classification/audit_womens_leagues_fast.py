"""
Script de auditoría ultrarrápido para ligas femeninas en matches.db.
Hace una única pasada agrupada sobre matches para extraer equipos y patrones.
"""

import sqlite3
import re
import json
import sys
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    # 1. Obtener todas las ligas que actualmente están marcadas como 'men'
    men_leagues = {
        r["league"]: dict(r)
        for r in cur.execute("""
            SELECT league, clean_name, match_count, avg_total_points
            FROM leagues_classification
            WHERE is_women = 0
        """).fetchall()
    }
    print(f"Total ligas actualmente marcadas como 'men': {len(men_leagues)}")

    # 2. Palabras clave sospechosas en el nombre de los EQUIPOS
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

    # 4. Una única consulta para obtener los equipos por liga
    print("Extrayendo combinaciones (league, teams) en una sola consulta...")
    league_teams = defaultdict(set)
    for r in cur.execute("SELECT league, home_team, away_team FROM matches WHERE league IS NOT NULL"):
        l = r["league"]
        if l in men_leagues:
            if r["home_team"]: league_teams[l].add(r["home_team"])
            if r["away_team"]: league_teams[l].add(r["away_team"])

    print(f"Ligas procesadas con equipos: {len(league_teams)}")

    suspicious_leagues = []

    for lname, linfo in men_leagues.items():
        match_league_kw = regex_league_women.search(lname)
        
        teams = list(league_teams.get(lname, []))
        total_teams = len(teams)
        
        detected_teams = [t for t in teams if regex_team_women.search(t)]
        team_ratio = (len(detected_teams) / total_teams) if total_teams > 0 else 0

        # Si el nombre tiene keywords de liga femenina O al menos 20% de los equipos tienen indicadores femeninos
        if match_league_kw or (total_teams >= 2 and team_ratio >= 0.20) or (len(detected_teams) >= 2):
            suspicious_leagues.append({
                "league": lname,
                "clean_name": linfo["clean_name"],
                "match_count": linfo["match_count"],
                "avg_total_points": linfo["avg_total_points"],
                "reason": "league_kw" if match_league_kw else f"team_ratio_{team_ratio:.2f}",
                "total_teams": total_teams,
                "women_teams_count": len(detected_teams),
                "detected_teams_sample": detected_teams[:4]
            })

    print(f"\nLigas FEMENINAS adicionales detectadas con nombres no estándar: {len(suspicious_leagues)}")
    # Ordenar por match_count descendente
    suspicious_leagues.sort(key=lambda x: x["match_count"], reverse=True)
    
    total_missed_matches = sum(sl["match_count"] for sl in suspicious_leagues)
    print(f"Total partidos afectados: {total_missed_matches}")

    for sl in suspicious_leagues[:25]:
        print(f"  - [{sl['match_count']} m] '{sl['league']}' | Pts: {sl['avg_total_points']} | Teams: {sl['detected_teams_sample']}")

    Path("tmp/league_classification/missed_women_leagues.json").write_text(
        json.dumps(suspicious_leagues, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("\nResultados guardados en tmp/league_classification/missed_women_leagues.json")

if __name__ == "__main__":
    main()
