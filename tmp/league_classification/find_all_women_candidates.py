"""
Script para analizar todas las palabras clave y acrónimos en las ligas marcadas como men
para descubrir términos femeninos en cualquier idioma.
"""

import sqlite3
import re
import json
import sys
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path("matches.db")

def main():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    men_leagues = [
        r["league"] for r in cur.execute("""
            SELECT league FROM leagues_classification WHERE is_women = 0
        """).fetchall()
    ]

    # Contar palabras
    words = Counter()
    for l in men_leagues:
        for w in re.findall(r"[A-Za-zÀ-ÿ0-9]+", l):
            words[w.lower()] += 1

    # Ver palabras con baja o alta frecuencia que puedan sonar a femenino
    # Let's inspect potential female terms
    female_indicators = [
        "dam", "damen", "damer", "dames", "fem", "femenina", "femenino", "feminina", "feminino",
        "femme", "femmes", "frauen", "kadin", "kadinlar", "kadın", "kadınlar", "kobiet", "kobiety",
        "kvinn", "kvinner", "moterys", "moteru", "moterų", "naised", "naiset", "naiste", "noi",
        "női", "sieviesu", "sieviešu", "vrouwen", "zeny", "ženy", "zenske", "ženske", "zenska",
        "ženska", "zls", "žls", "zbl", "žbl", "dbbl", "tkbl", "kbsl", "lnbf", "lbf", "lfb",
        "lf", "lf1", "lf2", "wnba", "wnbl", "wbbl", "waba", "ewbl"
    ]

    found = {}
    for l in men_leagues:
        low = l.lower()
        matched = []
        for fi in female_indicators:
            # Word boundary search
            if re.search(rf"\b{fi}\b", low):
                matched.append(fi)
        if matched:
            found[l] = matched

    print(f"Total ligas men que coinciden con algún indicador femenino: {len(found)}")
    for l, matches in found.items():
        cnt = cur.execute("SELECT match_count, avg_total_points FROM leagues_classification WHERE league = ?", (l,)).fetchone()
        print(f"  - '{l}' [{cnt['match_count']} m, {cnt['avg_total_points']} pts] -> {matches}")

if __name__ == "__main__":
    main()
