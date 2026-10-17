"""
Script: inspect_db.py
Ubicación: tmp/eda_nuevo_modelo/inspect_db.py
Propósito: Auditar rápidamente el estado actual de matches.db tras las descargas históricas recientes,
           comparar el volumen y cobertura por año/tabla/arquetipo frente al corte previo (48,623 partidos),
           usando consultas indexadas en memoria para ejecución en <3 segundos.
Hipótesis/Problema: Determinar qué nuevos datos históricos existen en matches.db, qué cobertura tienen
                    (PBP, graph_points, lineups, player_stats, team_statistics, match_odds, match_h2h, leagues_classification)
                    y cómo cambia la cobertura de H2H y ELO multi-temporada.
"""
import sys
import sqlite3
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("=== 1. TABLAS EN MATCHES.DB ===", flush=True)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        cur.execute(f"SELECT COUNT(*) FROM [{t}]")
        cnt = cur.fetchone()[0]
        cur.execute(f"PRAGMA table_info([{t}])")
        cols = [f"{c[1]}({c[2]})" for c in cur.fetchall()]
        print(f" - {t}: {cnt:,} filas | cols: {', '.join(cols[:10])}{' ...' if len(cols)>10 else ''}", flush=True)

    print("\n=== 2. DISTRIBUCIÓN ANUAL DE PARTIDOS ===", flush=True)
    cur.execute("SELECT match_id, substr(date, 1, 4), date, status_type, league FROM matches")
    matches_rows = cur.fetchall()
    print(f"Total filas en matches: {len(matches_rows):,}", flush=True)

    match_yr = {}
    match_league = {}
    yr_stats = defaultdict(lambda: {"total": 0, "fin": 0, "min_d": "9999", "max_d": "0000"})
    for mid, yr, dt, st, lg in matches_rows:
        yr = yr or "NULL"
        s = yr_stats[yr]
        s["total"] += 1
        if st == "finished":
            s["fin"] += 1
            match_yr[mid] = yr
            match_league[mid] = lg
        if dt and dt < s["min_d"]:
            s["min_d"] = dt
        if dt and dt > s["max_d"]:
            s["max_d"] = dt

    for yr in sorted(yr_stats.keys()):
        s = yr_stats[yr]
        print(f" Año {yr}: {s['total']:,} total ({s['fin']:,} fin) | [{s['min_d']} -> {s['max_d']}]", flush=True)

    print("\n=== 3. COBERTURA DE TABLAS RELACIONALES POR AÑO (PARTIDOS FINISHED) ===", flush=True)
    rel_tables = ["quarter_scores", "play_by_play", "graph_points", "player_stats", "team_statistics", "match_h2h", "match_odds", "lineups", "team_strength"]
    rel_sets = {}
    for t in rel_tables:
        if t in tables:
            cur.execute(f"SELECT DISTINCT match_id FROM [{t}]")
            rel_sets[t] = {r[0] for r in cur.fetchall()}
            print(f" -> {t}: {len(rel_sets[t]):,} partidos únicos", flush=True)

    for yr in sorted(yr_stats.keys()):
        mids = [m for m, y in match_yr.items() if y == yr]
        n = len(mids)
        if n == 0:
            continue
        mset = set(mids)
        parts = [f"N={n:,}"]
        for t, short in [("quarter_scores","QS"), ("play_by_play","PBP"), ("graph_points","GP"), ("player_stats","PS"), ("team_statistics","TS"), ("match_h2h","H2H"), ("match_odds","Odds"), ("lineups","LU"), ("team_strength","TSt")]:
            if t in rel_sets:
                c = len(mset & rel_sets[t])
                parts.append(f"{short}={c:,}({c*100/n:.0f}%)")
        print(f" {yr}: " + " | ".join(parts), flush=True)

    print("\n=== 4. ARQUETIPOS SEGÚN LEAGUES_CLASSIFICATION ===", flush=True)
    if "leagues_classification" in tables:
        cur.execute("SELECT league, quarter_duration_minutes, is_youth, is_college, is_women, competition_type, match_count, clean_name FROM leagues_classification")
        lc_map = {}
        for r in cur.fetchall():
            lc_map[r[0]] = {
                "qmin": r[1], "youth": r[2], "college": r[3], "women": r[4],
                "comp": r[5], "mcnt": r[6], "clean": r[7]
            }
        arch_counts = defaultdict(lambda: {"total": 0, "full_ml": 0, "has_lu": 0, "has_ts": 0, "has_odds": 0})
        pbp_set = rel_sets.get("play_by_play", set())
        gp_set = rel_sets.get("graph_points", set())
        lu_set = rel_sets.get("lineups", set())
        ts_set = rel_sets.get("team_statistics", set())
        odds_set = rel_sets.get("match_odds", set())

        for mid, lg in match_league.items():
            lc = lc_map.get(lg)
            if not lc:
                arch = "UNCLASSIFIED"
            elif lc["youth"] == 1 or lc["college"] == 1 or lc["comp"] == "friendly":
                arch = "BLACKLIST (Youth/NCAA/Friendly)"
            elif lc["qmin"] == 12:
                arch = "NBA_12M"
            elif lc["women"] == 1 and lc["qmin"] == 10:
                arch = "FIBA_WOMEN_10M"
            elif lc["women"] == 0 and lc["qmin"] == 10:
                arch = "FIBA_MEN_10M"
            else:
                arch = "OTHER"
            
            a = arch_counts[arch]
            a["total"] += 1
            if mid in pbp_set and mid in gp_set:
                a["full_ml"] += 1
            if mid in lu_set:
                a["has_lu"] += 1
            if mid in ts_set:
                a["has_ts"] += 1
            if mid in odds_set:
                a["has_odds"] += 1

        for arch, d in sorted(arch_counts.items(), key=lambda x: -x[1]["total"]):
            print(f" - {arch}: {d['total']:,} total | FULL_ML(PBP+GP)={d['full_ml']:,} | Lineups={d['has_lu']:,} | TeamStats={d['has_ts']:,} | Odds={d['has_odds']:,}", flush=True)

    conn.close()

if __name__ == "__main__":
    main()
