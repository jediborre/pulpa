"""
tmp/eda_nuevo_modelo/run_part3_quarters_players_shots.py
========================================================
Propósito:
Ejecutar la Parte 3 del análisis profundo sobre `matches.db` (modo solo lectura):
1. Teoría y anatomía de cada cuarto (Q1, Q2, Q3, Q4) por Arquetipo y por Liga:
   - Correlación de la fuerza pre-partido (ELO) con Q1, Q2, Q3 y Q4.
   - Autocorrelación entre cuartos (r(Q1,Q4), r(Q2,Q4), r(Q3,Q4), r(Q13,Q4)).
   - Volatilidad de cada cuarto (std), Tasa de Descanso de Titulares en Blowout (+15 pts al Q3),
     y Tasa de Remontada del Favorito (Rubber-Banding).
2. Estadísticas de ELO Individual de Jugadores (`player_stats`, 1.88M registros):
   - `d_starter_elo` (Top 5 jugadores por ELO/puntos pre-partido)
   - `d_bench_elo` (Jugadores #6 a #10 de rotación/banquillo)
   - `d_star_dep` (Dependencia de la estrella #1 frente al resto del quinteto)
   - `d_player_vol` y `sum_player_vol` (Volatilidad histórica del plus_minus individual)
   - Descanso empírico de las Estrellas en Q4 medido directamente desde `match_events`
     (`incident_type='score'` / `points > 0` en Q1-Q3 vs Q4 cuando hay partido cerrado <=8p vs paliza >=18p).
3. Extracción y Patrones de Tiros (2PT, 3PT, FT) desde `match_events`, `player_stats` y `team_statistics`:
   - Sostenibilidad del dominio en Pintura + Tiros Libres (2PT + FT) frente a la regresión
     a la media de las rachas de Triples (3PT) hacia el Q4.
"""
import sqlite3
import sys
import time
from collections import defaultdict
from itertools import groupby
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    t0 = time.time()
    uri = f"file:{ROOT / 'matches.db'}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    cur = conn.cursor()

    print("=" * 100)
    print("1. CARGANDO PARTIDOS, CUARTOS (Q1..Q4) Y TAXONOMÍA DE LIGAS")
    print("=" * 100)

    tax_map = {}
    for lg, qd, iw in cur.execute("SELECT league, quarter_duration_minutes, is_women FROM leagues_classification").fetchall():
        arch = "nba_12m" if int(qd or 10) == 12 else ("fiba_women_10m" if int(iw or 0) == 1 else "fiba_men_10m")
        tax_map[lg] = arch

    q_rows = cur.execute("""
        SELECT m.match_id, m.date, m.league, m.home_team, m.away_team,
               q1.home, q1.away,
               q2.home, q2.away,
               q3.home, q3.away,
               q4.home, q4.away
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        WHERE q4.home != q4.away
        ORDER BY m.date ASC, m.match_id ASC
    """).fetchall()

    # Calcular ELO FT y ELO por Cuarto en barrido cronológico rápido
    elo_ft = defaultdict(lambda: 1500.0)
    elo_q4 = defaultdict(lambda: 1500.0)
    last_dt = {}

    matches_list = []
    rec_by_mid = {}

    for row in q_rows:
        mid, dt_s, lg, ht, at, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a = row
        arch = tax_map.get(lg, "fiba_men_10m")
        q1_d = float((q1h or 0) - (q1a or 0))
        q2_d = float((q2h or 0) - (q2a or 0))
        q3_d = float((q3h or 0) - (q3a or 0))
        q4_d = float((q4h or 0) - (q4a or 0))
        ht_d = q1_d + q2_d
        q13_d = ht_d + q3_d
        ft_d = q13_d + q4_d
        if ft_d == 0:
            continue

        d_elo = (elo_ft[ht] - elo_ft[at]) + 35.0
        d_elo4 = elo_q4[ht] - elo_q4[at]

        rec = {
            "mid": mid,
            "date": str(dt_s or "")[:10],
            "league": lg or "Unknown",
            "arch": arch,
            "ht": ht,
            "at": at,
            "q1_d": q1_d,
            "q2_d": q2_d,
            "q3_d": q3_d,
            "ht_d": ht_d,
            "q13_d": q13_d,
            "q4_m": q4_d,
            "q4_win": 1 if q4_d > 0 else 0,
            "q4_tot": float((q4h or 0) + (q4a or 0)),
            "ft_win": 1 if ft_d > 0 else 0,
            "d_elo_ft": d_elo,
            "d_elo_q4": d_elo4,
        }
        matches_list.append(rec)
        rec_by_mid[mid] = rec

        # Update ELO
        exp_ft = 1.0 / (1.0 + 10.0 ** (-d_elo / 400.0))
        s_ft = 1.0 if ft_d > 0 else 0.0
        elo_ft[ht] += 20.0 * (s_ft - exp_ft)
        elo_ft[at] -= 20.0 * (s_ft - exp_ft)

        exp_q4 = 1.0 / (1.0 + 10.0 ** (-d_elo4 / 400.0))
        s_q4 = 1.0 if q4_d > 0 else 0.0
        elo_q4[ht] += 16.0 * (s_q4 - exp_q4)
        elo_q4[at] -= 16.0 * (s_q4 - exp_q4)

    print(f"Cargados {len(matches_list):,} partidos con Q1..Q4 en {time.time()-t0:.1f}s")

    # =========================================================================
    # SECCIÓN A: TEORÍA DE CADA CUARTO (Q1, Q2, Q3, Q4) POR ARQUETIPO Y POR LIGA
    # =========================================================================
    print("\n" + "=" * 100)
    print("2. TEORÍA DE CADA CUARTO (Q1, Q2, Q3, Q4) POR ARQUETIPO Y POR LIGA")
    print("=" * 100)
    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        sub = [r for r in matches_list if r["arch"] == arch]
        q1 = np.array([r["q1_d"] for r in sub])
        q2 = np.array([r["q2_d"] for r in sub])
        q3 = np.array([r["q3_d"] for r in sub])
        ht = np.array([r["ht_d"] for r in sub])
        q13 = np.array([r["q13_d"] for r in sub])
        q4 = np.array([r["q4_m"] for r in sub])
        elo = np.array([r["d_elo_ft"] for r in sub])

        m15 = np.abs(q13) >= 15
        lose15 = float(np.mean(np.sign(q13[m15]) != np.sign(q4[m15])) * 100.0)
        m22 = np.abs(q13) >= 22
        lose22 = float(np.mean(np.sign(q13[m22]) != np.sign(q4[m22])) * 100.0)
        m_ft = ((elo > 75) & (q13 < 0)) | ((elo < -75) & (q13 > 0))
        fav_cb = float(np.mean(np.sign(elo[m_ft]) == np.sign(q4[m_ft])) * 100.0)

        print(
            f"\n  >>> ARQUETIPO: {arch} (N = {len(sub):,})\n"
            f"      - Correlación Fuerza Pre-Partido (ELO) con cada cuarto:\n"
            f"          r(ELO, Q1) = {np.corrcoef(elo,q1)[0,1]:+.4f} | r(ELO, Q2) = {np.corrcoef(elo,q2)[0,1]:+.4f} | "
            f"r(ELO, Q3) = {np.corrcoef(elo,q3)[0,1]:+.4f} | r(ELO, Q4) = {np.corrcoef(elo,q4)[0,1]:+.4f}\n"
            f"      - Autocorrelación de cada cuarto hacia el Q4:\n"
            f"          r(Q1, Q4)  = {np.corrcoef(q1,q4)[0,1]:+.4f} | r(Q2, Q4)  = {np.corrcoef(q2,q4)[0,1]:+.4f} | "
            f"r(Q3, Q4)  = {np.corrcoef(q3,q4)[0,1]:+.4f} | r(Acum_Q3, Q4) = {np.corrcoef(q13,q4)[0,1]:+.4f}\n"
            f"      - Volatilidad (Desviación Estándar del diferencial de puntos por cuarto):\n"
            f"          std(Q1) = {np.std(q1):.2f} pts | std(Q2) = {np.std(q2):.2f} pts | std(Q3) = {np.std(q3):.2f} pts | std(Q4) = {np.std(q4):.2f} pts\n"
            f"      - Efecto Descanso Titulares: Líder por >=15p al Q3 PIERDE el Q4 el {lose15:.1f}% de las veces (>=22p: {lose22:.1f}%)\n"
            f"      - Efecto Rubber-Band: Favorito ELO (>75) que va perdiendo al Q3 GANA el Q4 el {fav_cb:.1f}% de las veces"
        )

    print("\n  >>> TABLA COMPARATIVA POR LIGA: ANATOMÍA DE CUARTOS, DESCANSO DE TITULARES Y VOLATILIDAD")
    print(f"  {'Liga':<42} | {'Arq':<10} | {'N':>5} | {'r(ELO,Q1)':>9} | {'r(ELO,Q3)':>9} | {'r(ELO,Q4)':>9} | {'r(Q13,Q4)':>9} | {'Líder+15 PierdeQ4':>17} | {'FavPerd GanaQ4':>14} | {'std(Q4)':>7}")
    print("  " + "-" * 150)

    by_lg = defaultdict(list)
    for r in matches_list:
        by_lg[r["league"]].append(r)

    selected_lgs = []
    # Top ligas por volumen + ligas élite + ligas tóxicas
    priority_patterns = [
        "NBA", "NBA G League", "China CBA", "PBA Philippine Cup", "Euroleague",
        "Liga ACB", "Lega A", "BBL", "Pro A", "BSL", "VTB", "Adriatic", "ABA League",
        "NBL1, South", "NBL1, East", "NBL1, Central", "NBL1 East, Women",
        "PGE Basket Liga", "Austrian Basketball Superliga", "Enovos League",
        "Israeli Basketball Super League", "Premijer ženska liga", "Liga Femenina, Women",
        "Mexico LNBP", "Argentina Liga Nacional", "Brazil - NBB", "NCAA Men's Division I",
        "NCAA Women's Division I", "1. Liga", "Poland 1st Division",
    ]
    for pat in priority_patterns:
        for k in sorted(by_lg.keys(), key=lambda x: -len(by_lg[x])):
            if pat.lower() in k.lower() and k not in selected_lgs and len(by_lg[k]) >= 45:
                selected_lgs.append(k)
                break

    for lg in selected_lgs[:26]:
        sub = by_lg[lg]
        arch_short = sub[0]["arch"].replace("_10m", "").replace("_12m", "")
        q1 = np.array([r["q1_d"] for r in sub])
        q3 = np.array([r["q3_d"] for r in sub])
        q13 = np.array([r["q13_d"] for r in sub])
        q4 = np.array([r["q4_m"] for r in sub])
        elo = np.array([r["d_elo_ft"] for r in sub])
        m15 = np.abs(q13) >= 15
        lose15 = float(np.mean(np.sign(q13[m15]) != np.sign(q4[m15])) * 100.0) if m15.sum() >= 5 else np.nan
        m_ft = ((elo > 50) & (q13 < 0)) | ((elo < -50) & (q13 > 0))
        fav_cb = float(np.mean(np.sign(elo[m_ft]) == np.sign(q4[m_ft])) * 100.0) if m_ft.sum() >= 5 else np.nan
        print(
            f"  {lg[:42]:<42} | {arch_short:<10} | {len(sub):5d} | {np.corrcoef(elo,q1)[0,1]:+9.3f} | {np.corrcoef(elo,q3)[0,1]:+9.3f} | "
            f"{np.corrcoef(elo,q4)[0,1]:+9.3f} | {np.corrcoef(q13,q4)[0,1]:+9.3f} | "
            f"{lose15:16.1f}% | {fav_cb:13.1f}% | {np.std(q4):7.2f}"
        )

    # =========================================================================
    # SECCIÓN B: ELO DE JUGADORES, VOLATILIDAD Y DESCANSO REAL DE ESTRELLAS EN Q4
    # =========================================================================
    print("\n" + "=" * 100)
    print("3. ELO INDIVIDUAL DE JUGADORES, VOLATILIDAD Y DESCANSO REAL DE ESTRELLAS EN Q4")
    print("=" * 100)

    t1 = time.time()
    ps_rows = cur.execute("""
        SELECT ps.match_id, ps.team, ps.player_name, ps.minutes_played, ps.points,
               ps.plus_minus, ps.sofascore_rating,
               ps.field_goals_made, ps.field_goals_attempted,
               ps.three_made, ps.three_attempted,
               ps.free_throws_made, ps.free_throws_attempted
        FROM player_stats ps
        JOIN matches m ON ps.match_id = m.match_id
        ORDER BY m.date ASC, ps.match_id ASC
    """).fetchall()
    print(f"Leídas {len(ps_rows):,} filas de player_stats en {time.time()-t1:.1f}s")

    player_elo = defaultdict(lambda: 1500.0)
    player_q4_elo = defaultdict(lambda: 1500.0)
    player_pm_hist = defaultdict(list)
    player_pts_hist = defaultdict(list)

    match_p_feats = {}
    match_top_stars = {}

    for mid, grp in groupby(ps_rows, key=lambda x: x[0]):
        if mid not in rec_by_mid:
            continue
        r_rec = rec_by_mid[mid]
        ht_name = r_rec["ht"]
        rows_m = list(grp)

        h_players, a_players = [], []
        for row in rows_m:
            _, team_s, pname, mins_s, pts, pm, rat, fgm, fga, fg3m, fg3a, ftm, fta = row
            if not pname:
                continue
            try:
                if isinstance(mins_s, str) and ":" in mins_s:
                    mins = float(mins_s.split(":")[0]) + float(mins_s.split(":")[1]) / 60.0
                else:
                    mins = float(mins_s or 0.0)
            except Exception:
                mins = 0.0
            pm_v = float(pm or 0.0)
            rat_v = float(rat or 6.5)
            pts_v = float(pts or 0.0)

            p_key = f"{team_s}::{pname}"
            pre_elo = player_elo[p_key]
            pre_q4_elo = player_q4_elo[p_key]
            pm_h = player_pm_hist[p_key]
            pts_h = player_pts_hist[p_key]
            pre_pm_std = float(np.std(pm_h[-10:])) if len(pm_h) >= 3 else 5.5
            pre_pm_mean = float(np.mean(pm_h[-10:])) if len(pm_h) >= 1 else 0.0
            pre_pts_avg = float(np.mean(pts_h[-10:])) if len(pts_h) >= 1 else 0.0

            entry = {
                "p_key": p_key,
                "pname": pname,
                "mins": mins,
                "pts": pts_v,
                "pm": pm_v,
                "rat": rat_v,
                "pre_elo": pre_elo,
                "pre_q4_elo": pre_q4_elo,
                "pre_pm_std": pre_pm_std,
                "pre_pm_mean": pre_pm_mean,
                "pre_pts_avg": pre_pts_avg,
            }
            if team_s == ht_name or str(team_s).lower() == "home":
                h_players.append(entry)
            else:
                a_players.append(entry)

        if len(h_players) >= 5 and len(a_players) >= 5:
            h_pre = sorted(h_players, key=lambda x: (x["pre_elo"], x["pre_pts_avg"]), reverse=True)
            a_pre = sorted(a_players, key=lambda x: (x["pre_elo"], x["pre_pts_avg"]), reverse=True)

            h_top5_elo = float(np.mean([p["pre_elo"] for p in h_pre[:5]]))
            a_top5_elo = float(np.mean([p["pre_elo"] for p in a_pre[:5]]))
            h_bench_elo = float(np.mean([p["pre_elo"] for p in h_pre[5:10]])) if len(h_pre) > 5 else 1500.0
            a_bench_elo = float(np.mean([p["pre_elo"] for p in a_pre[5:10]])) if len(a_pre) > 5 else 1500.0
            h_bench_q4_elo = float(np.mean([p["pre_q4_elo"] for p in h_pre[5:10]])) if len(h_pre) > 5 else 1500.0
            a_bench_q4_elo = float(np.mean([p["pre_q4_elo"] for p in a_pre[5:10]])) if len(a_pre) > 5 else 1500.0

            h_star_dep = h_pre[0]["pre_elo"] - h_top5_elo
            a_star_dep = a_pre[0]["pre_elo"] - a_top5_elo

            h_vol = float(np.mean([p["pre_pm_std"] for p in h_pre[:5]]))
            a_vol = float(np.mean([p["pre_pm_std"] for p in a_pre[:5]]))

            match_p_feats[mid] = {
                "d_starter_elo": h_top5_elo - a_top5_elo,
                "d_bench_elo": h_bench_elo - a_bench_elo,
                "d_bench_q4_elo": h_bench_q4_elo - a_bench_q4_elo,
                "d_star_dep": h_star_dep - a_star_dep,
                "d_player_vol": h_vol - a_vol,
                "sum_player_vol": h_vol + a_vol,
            }
            match_top_stars[mid] = (
                {p["pname"] for p in h_pre[:2]},
                {p["pname"] for p in a_pre[:2]},
            )

        # Actualizar ELO de cada jugador post-partido
        ft_w = r_rec["ft_win"]
        q4_w = r_rec["q4_win"]
        for p in h_players:
            if p["mins"] > 0:
                w_m = min(1.0, p["mins"] / 26.0)
                perf = 1.0 if p["pm"] > 0 else (0.0 if p["pm"] < 0 else float(ft_w))
                exp_p = 1.0 / (1.0 + 10.0 ** ((1500.0 - p["pre_elo"]) / 400.0))
                player_elo[p["p_key"]] += 14.0 * w_m * (perf - exp_p)
                exp_q4 = 1.0 / (1.0 + 10.0 ** ((1500.0 - p["pre_q4_elo"]) / 400.0))
                player_q4_elo[p["p_key"]] += 12.0 * w_m * (float(q4_w) - exp_q4)
                player_pm_hist[p["p_key"]].append(p["pm"])
                player_pts_hist[p["p_key"]].append(p["pts"])
        for p in a_players:
            if p["mins"] > 0:
                w_m = min(1.0, p["mins"] / 26.0)
                perf = 1.0 if p["pm"] > 0 else (0.0 if p["pm"] < 0 else float(1 - ft_w))
                exp_p = 1.0 / (1.0 + 10.0 ** ((1500.0 - p["pre_elo"]) / 400.0))
                player_elo[p["p_key"]] += 14.0 * w_m * (perf - exp_p)
                exp_q4 = 1.0 / (1.0 + 10.0 ** ((1500.0 - p["pre_q4_elo"]) / 400.0))
                player_q4_elo[p["p_key"]] += 12.0 * w_m * (float(1 - q4_w) - exp_q4)
                player_pm_hist[p["p_key"]].append(p["pm"])
                player_pts_hist[p["p_key"]].append(p["pts"])

    print("\n  >>> CORRELACIONES DEL ELO DE JUGADORES (TITULARES VS BANCA) Y VOLATILIDAD CON Q4 Y FT:")
    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        pairs = [(rec_by_mid[m], f) for m, f in match_p_feats.items() if rec_by_mid[m]["arch"] == arch]
        yw = np.array([r["q4_win"] for r, _ in pairs])
        ym = np.array([r["q4_m"] for r, _ in pairs])
        yft = np.array([r["ft_win"] for r, _ in pairs])
        q13 = np.array([abs(r["q13_d"]) for r, _ in pairs])
        blow_mask = q13 >= 16

        print(f"\n    Arquetipo: {arch} (N con lineups = {len(pairs):,}):")
        for k in ["d_starter_elo", "d_bench_elo", "d_bench_q4_elo", "d_star_dep", "d_player_vol", "sum_player_vol"]:
            arr = np.array([f[k] for _, f in pairs])
            rw_all = float(np.corrcoef(arr, yw)[0, 1])
            rm_all = float(np.corrcoef(arr, ym)[0, 1])
            rw_blow = float(np.corrcoef(arr[blow_mask], yw[blow_mask])[0, 1]) if blow_mask.sum() > 50 else 0.0
            rft = float(np.corrcoef(arr, yft)[0, 1])
            print(
                f"      * {k:<18} -> r(Q4_Win)={rw_all:+.4f} | r(Q4_Margin)={rm_all:+.4f} | "
                f"r(Q4_Win en Blowout >=16p)={rw_blow:+.4f} | r(FT_Win)={rft:+.4f}"
            )

    # Medir en match_events la participación real de las Top-2 estrellas en Q1-Q3 vs Q4
    print("\n  >>> EVIDENCIA EMPÍRICA EN `match_events`: ¿DESCANSAN LOS MEJORES JUGADORES EN EL Q4?")
    ev_rows = cur.execute("""
        SELECT match_id, quarter, team, player, points
        FROM match_events
        WHERE points > 0 AND player IS NOT NULL AND player != ''
    """).fetchall()

    star_q13 = defaultdict(lambda: [0.0, 0.0])
    star_q4 = defaultdict(lambda: [0.0, 0.0])

    for mid, qtr, team_s, pname, pts in ev_rows:
        if mid not in match_top_stars or mid not in rec_by_mid:
            continue
        r_rec = rec_by_mid[mid]
        leader_home = r_rec["q13_d"] >= 0
        is_leader_ev = (leader_home and (team_s == "home" or team_s == r_rec["ht"])) or (
            not leader_home and (team_s == "away" or team_s == r_rec["at"])
        )
        if not is_leader_ev:
            continue
        stars = match_top_stars[mid][0] if leader_home else match_top_stars[mid][1]
        p_val = float(pts or 0.0)
        in_stars = pname in stars
        if qtr in ("Q1", "Q2", "Q3"):
            star_q13[mid][1] += p_val
            if in_stars:
                star_q13[mid][0] += p_val
        elif qtr == "Q4":
            star_q4[mid][1] += p_val
            if in_stars:
                star_q4[mid][0] += p_val

    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        c_q13, c_q4, b_q13, b_q4 = [], [], [], []
        c_zero, b_zero = [], []
        for mid, (s13, t13) in star_q13.items():
            r_rec = rec_by_mid[mid]
            if r_rec["arch"] != arch or t13 < 20:
                continue
            s4, t4 = star_q4.get(mid, [0.0, 0.0])
            if t4 < 6:
                continue
            lead = abs(r_rec["q13_d"])
            if lead <= 8:
                c_q13.append(s13 / t13 * 100.0)
                c_q4.append(s4 / t4 * 100.0)
                c_zero.append(1.0 if s4 == 0 else 0.0)
            elif lead >= 18:
                b_q13.append(s13 / t13 * 100.0)
                b_q4.append(s4 / t4 * 100.0)
                b_zero.append(1.0 if s4 == 0 else 0.0)
        if c_q13 and b_q13:
            print(
                f"    * {arch:<15}:\n"
                f"        - Partido Cerrado (<=8p al Q3): Top-2 Estrellas aportan {np.mean(c_q13):.1f}% de los pts en Q1-Q3 y {np.mean(c_q4):.1f}% en Q4 (Descanso total 0 pts en Q4: {np.mean(c_zero)*100:.1f}%)\n"
                f"        - Paliza / Blowout (>=18p Q3) : Top-2 Estrellas aportan {np.mean(b_q13):.1f}% de los pts en Q1-Q3 y CAEN a {np.mean(b_q4):.1f}% en Q4 (Descanso total 0 pts en Q4: {np.mean(b_zero)*100:.1f}%)"
            )

    # =========================================================================
    # SECCIÓN C: EXTRACCIÓN DE TIROS (2PT, 3PT, FT) Y PATRONES DE CORRELACIÓN
    # =========================================================================
    print("\n" + "=" * 100)
    print("4. EXTRACCIÓN DE TIROS (2PT, 3PT, FT) Y PATRONES DE SOSTENIBILIDAD VS REGRESIÓN A LA MEDIA")
    print("=" * 100)

    shot_q13 = defaultdict(lambda: [0.0, 0.0, 0.0])  # mid -> [d_2pt, d_3pt, d_ft] in Q1-Q3
    shot_q4 = defaultdict(lambda: [0.0, 0.0, 0.0])   # mid -> [d_2pt, d_3pt, d_ft] in Q4

    all_sc_rows = cur.execute("""
        SELECT match_id, quarter, team, points
        FROM match_events
        WHERE points IN (1, 2, 3)
    """).fetchall()

    for mid, qtr, team_s, pts in all_sc_rows:
        if mid not in rec_by_mid:
            continue
        r_rec = rec_by_mid[mid]
        sgn = 1.0 if (team_s == "home" or team_s == r_rec["ht"]) else -1.0
        target_dict = shot_q13 if qtr in ("Q1", "Q2", "Q3") else (shot_q4 if qtr == "Q4" else None)
        if target_dict is None:
            continue
        if pts == 2:
            target_dict[mid][0] += 2.0 * sgn
        elif pts == 3:
            target_dict[mid][1] += 3.0 * sgn
        elif pts == 1:
            target_dict[mid][2] += 1.0 * sgn

    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        sub = [r for r in matches_list if r["arch"] == arch and r["mid"] in shot_q13]
        d2_13 = np.array([shot_q13[r["mid"]][0] for r in sub])
        d3_13 = np.array([shot_q13[r["mid"]][1] for r in sub])
        dft_13 = np.array([shot_q13[r["mid"]][2] for r in sub])
        d2_4 = np.array([shot_q4[r["mid"]][0] for r in sub])
        d3_4 = np.array([shot_q4[r["mid"]][1] for r in sub])
        dft_4 = np.array([shot_q4[r["mid"]][2] for r in sub])
        q13 = np.array([r["q13_d"] for r in sub])
        q4 = np.array([r["q4_m"] for r in sub])
        q4w = np.array([r["q4_win"] for r in sub])

        # Persistencia de cada tipo de tiro de Q1-Q3 hacia Q4
        r_2pt_persist = float(np.corrcoef(d2_13, d2_4)[0, 1])
        r_ft_persist = float(np.corrcoef(dft_13, dft_4)[0, 1])
        r_3pt_persist = float(np.corrcoef(d3_13, d3_4)[0, 1])

        # Comparación en ventaja moderada (4..12 pts al Q3):
        # Caso 1: Ventaja construida con Pintura + Tiros Libres (d2+dft >= 80% de la ventaja)
        # Caso 2: Ventaja construida con Lluvia de Triples (d3 >= 80% de la ventaja y d2 <= 0)
        paint_m = ((q13 >= 4) & (q13 <= 12) & (d2_13 + dft_13 >= q13 * 0.8)) | (
            (q13 <= -4) & (q13 >= -12) & (d2_13 + dft_13 <= q13 * 0.8)
        )
        three_m = ((q13 >= 4) & (q13 <= 12) & (d3_13 >= q13 * 0.8) & (d2_13 <= 0)) | (
            (q13 <= -4) & (q13 >= -12) & (d3_13 <= q13 * 0.8) & (d2_13 >= 0)
        )
        wr_paint = float(np.mean(np.sign(q13[paint_m]) == np.sign(q4[paint_m])) * 100.0)
        wr_three = float(np.mean(np.sign(q13[three_m]) == np.sign(q4[three_m])) * 100.0)

        print(
            f"\n  >>> ARQUETIPO: {arch} (N = {len(sub):,})\n"
            f"      - Correlación de Tiros en Q1-Q3 con Ganar el Q4:\n"
            f"          r(2PT_Q13, Q4_Win) = {np.corrcoef(d2_13,q4w)[0,1]:+.4f} | r(FT_Q13, Q4_Win) = {np.corrcoef(dft_13,q4w)[0,1]:+.4f} | r(3PT_Q13, Q4_Win) = {np.corrcoef(d3_13,q4w)[0,1]:+.4f}\n"
            f"      - Persistencia del tipo de tiro (Correlación entre Q1-Q3 y Q4 del mismo tiro):\n"
            f"          Persistencia 2PT: r = {r_2pt_persist:+.4f} (Alta) | Persistencia FT: r = {r_ft_persist:+.4f} | Persistencia 3PT: r = {r_3pt_persist:+.4f} (Baja / Regresión)\n"
            f"      - Patrón Operativo (Líder por 4 a 12 pts al cierre del Q3):\n"
            f"          * Si lidera dominando PINTURA + TIROS LIBRES (2PT+FT) -> Gana el Q4 el {wr_paint:5.2f}% de las veces (N={int(paint_m.sum()):,})\n"
            f"          * Si lidera solo por RACHA DE TRIPLES (3PT sin pintura) -> Gana el Q4 solo el {wr_three:5.2f}% de las veces (N={int(three_m.sum()):,})"
        )

    conn.close()


if __name__ == "__main__":
    main()
