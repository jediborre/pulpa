"""
Script: full_leakfree_eda.py
Ubicación: tmp/eda_nuevo_modelo/full_leakfree_eda.py
Propósito: Ejecutar el EDA cuantitativo completo y 100% libre de data leakage sobre matches.db (227,028 partidos),
           evaluando todas las familias de variables (Pre-partido, Rolling Roster, ELO Dinámico, Cuotas Bookmaker,
           Box Score 1ª Mitad Q1+Q2, Momentum Graph Points al min 27, PBP recorte exacto al min 27, e Interacciones
           No Lineales / Rubber-Band / Garbage Time) frente a múltiples targets (Q4 Winner/Margin, Q4 Total, FT Rem).
Hipótesis/Problema: Determinar con rigor matemático qué features tienen verdadera señal predictiva sin fugas del futuro,
                    comparar el impacto real de las nuevas tablas descargadas, y fundamentar la arquitectura del nuevo modelo.
"""
import sys
import sqlite3
import math
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def calc_stats(x_arr, y_bin, y_cont):
    x = np.asarray(x_arr, dtype=float)
    yb = np.asarray(y_bin, dtype=float)
    yc = np.asarray(y_cont, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(yb) & ~np.isnan(yc) & ~np.isinf(x)
    x = x[mask]
    yb = yb[mask]
    yc = yc[mask]
    n = len(x)
    if n < 300 or np.std(x) < 1e-12:
        return {"n": int(n), "r_margin": 0.0, "r_win": 0.0, "auc": 0.5, "auc_dev": 0.0, "dir_acc": 50.0}
    r_m = float(np.corrcoef(x, yc)[0, 1])
    r_w = float(np.corrcoef(x, yb)[0, 1])
    try:
        auc = float(roc_auc_score(yb.astype(int), x))
    except Exception:
        auc = 0.5
    nz = x != 0
    if np.sum(nz) > 100:
        if r_w >= 0:
            hits = np.sum((x[nz] > 0) == (yb[nz] == 1))
        else:
            hits = np.sum((x[nz] < 0) == (yb[nz] == 1))
        acc = float(hits * 100.0 / np.sum(nz))
    else:
        acc = 50.0
    return {
        "n": int(n),
        "r_margin": round(r_m, 4),
        "r_win": round(r_w, 4),
        "auc": round(auc, 4),
        "auc_dev": round(abs(auc - 0.5), 4),
        "dir_acc": round(acc, 2),
    }

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("1. Cargando matches y quarter_scores...", flush=True)
    cur.execute("""
        SELECT match_id, date, time, league, home_team, away_team, home_score, away_score
        FROM matches
        WHERE status_type = 'finished' AND home_score > 0 AND away_score > 0
        ORDER BY date ASC, time ASC, match_id ASC
    """)
    matches = cur.fetchall()

    cur.execute("SELECT match_id, quarter, home, away FROM quarter_scores")
    qs_map = defaultdict(dict)
    for mid, q, h, a in cur.fetchall():
        if h is not None and a is not None:
            qs_map[mid][q] = (int(h), int(a))

    print("2. Cargando match_odds (Full time Home/Away)...", flush=True)
    cur.execute("""
        SELECT match_id, home_value, away_value
        FROM match_odds
        WHERE odds_type = 'Home/Away' AND market_period = 'Match'
          AND home_value > 1.0 AND away_value > 1.0
    """)
    odds_map = {}
    for mid, hv, av in cur.fetchall():
        p_h = (1.0 / hv) / ((1.0 / hv) + (1.0 / av))
        log_ratio = math.log(av / hv)
        odds_map[mid] = (p_h, log_ratio)

    print("3. Cargando team_strength (wins, losses, position, form)...", flush=True)
    cur.execute("SELECT match_id, team_name, wins, losses, position, form FROM team_strength")
    ts_map = {}
    for mid, tname, w, l, pos, form_str in cur.fetchall():
        w = w or 0
        l = l or 0
        tot = w + l
        wp = (w / tot) if tot >= 3 else np.nan
        f_score = np.nan
        if form_str:
            # e.g. '["W","L","W"]' or 'WLW'
            w_cnt = form_str.count("W")
            l_cnt = form_str.count("L")
            if w_cnt + l_cnt > 0:
                f_score = w_cnt / (w_cnt + l_cnt)
        ts_map[(mid, tname)] = (wp, pos, f_score)

    print("4.Calculando Rolling Player Ratings (Sin Leakage vs Con Leakage) sobre muestra reciente 2024-2026...", flush=True)
    # Para velocidad y máxima cobertura de lineups, cargamos lineups+player_stats ordenados por fecha
    cur.execute("""
        SELECT m.match_id, m.date, ps.team, ps.player_id, ps.player_name, l.is_starter,
               ps.sofascore_rating, ps.points, ps.plus_minus, ps.minutes_played, ps.fouls
        FROM player_stats ps
        JOIN lineups l ON ps.id = l.id
        JOIN matches m ON ps.match_id = m.match_id
        WHERE m.status_type = 'finished' AND m.date >= '2024-01-01'
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)
    # Procesamos por match_id en orden cronológico para mantener el historial rolling de cada jugador ANTES del partido
    player_hist = defaultdict(list) # player_key -> list of (rating, pts, pm, mins)
    match_roster_Rolling = {}       # mid -> dict of leak-free pre-match roster metrics
    match_roster_Leaked = {}        # mid -> dict of post-match leaked roster metrics

    current_mid = None
    current_date = None
    current_rows = []

    def process_match_roster(mid, rows):
        h_st_leak, a_st_leak, h_bn_leak, a_bn_leak = [], [], [], []
        h_st_roll, a_st_roll, h_bn_roll, a_bn_roll = [], [], [], []
        h_pm_roll, a_pm_roll = [], []
        h_pts_roll, a_pts_roll = [], []

        for _, dt, team, pid, pname, is_st, rat, pts, pm, mins, fouls in rows:
            pkey = pid if pid else pname
            # 1. Leer historial PREVIO (Strict pre-match rolling, últimos 10 partidos)
            past = player_hist.get(pkey)
            if past:
                recent = past[-10:]
                r_avg = sum(x[0] for x in recent if x[0] is not None) / max(1, sum(1 for x in recent if x[0] is not None))
                p_avg = sum(x[1] for x in recent) / len(recent)
                pm_avg = sum(x[2] for x in recent if x[2] is not None) / max(1, sum(1 for x in recent if x[2] is not None))
                if team == "home":
                    if is_st == 1:
                        if r_avg > 0: h_st_roll.append(r_avg)
                        h_pm_roll.append(pm_avg)
                        h_pts_roll.append(p_avg)
                    else:
                        if r_avg > 0: h_bn_roll.append(r_avg)
                        h_pts_roll.append(p_avg)
                else:
                    if is_st == 1:
                        if r_avg > 0: a_st_roll.append(r_avg)
                        a_pm_roll.append(pm_avg)
                        a_pts_roll.append(p_avg)
                    else:
                        if r_avg > 0: a_bn_roll.append(r_avg)
                        a_pts_roll.append(p_avg)

            # 2. Guardar métrica leaked del partido actual
            if rat is not None and rat > 0:
                if team == "home":
                    (h_st_leak if is_st == 1 else h_bn_leak).append(rat)
                else:
                    (a_st_leak if is_st == 1 else a_bn_leak).append(rat)

        # Guardar agregados pre-partido
        if len(h_st_roll) >= 3 and len(a_st_roll) >= 3:
            hs_r = sum(h_st_roll) / len(h_st_roll)
            as_r = sum(a_st_roll) / len(a_st_roll)
            hb_r = (sum(h_bn_roll) / len(h_bn_roll)) if len(h_bn_roll) >= 2 else hs_r - 0.4
            ab_r = (sum(a_bn_roll) / len(a_bn_roll)) if len(a_bn_roll) >= 2 else as_r - 0.4
            h_top_share = (max(h_pts_roll) / max(1.0, sum(h_pts_roll))) if h_pts_roll else 0.25
            a_top_share = (max(a_pts_roll) / max(1.0, sum(a_pts_roll))) if a_pts_roll else 0.25
            match_roster_Rolling[mid] = {
                "delta_starter_roll": hs_r - as_r,
                "delta_bench_roll": hb_r - ab_r,
                "delta_depth_roll": (as_r - ab_r) - (hs_r - hb_r),
                "delta_pm_roll": (sum(h_pm_roll)/len(h_pm_roll)) - (sum(a_pm_roll)/len(a_pm_roll)),
                "delta_star_rel_roll": h_top_share - a_top_share,
            }

        if len(h_st_leak) >= 3 and len(a_st_leak) >= 3:
            match_roster_Leaked[mid] = {
                "delta_starter_leaked": (sum(h_st_leak)/len(h_st_leak)) - (sum(a_st_leak)/len(a_st_leak)),
                "delta_bench_leaked": ((sum(h_bn_leak)/len(h_bn_leak)) - (sum(a_bn_leak)/len(a_bn_leak))) if (h_bn_leak and a_bn_leak) else np.nan,
            }

        # 3. Actualizar historial del jugador DESPUÉS de extraer features pre-partido
        for _, dt, team, pid, pname, is_st, rat, pts, pm, mins, fouls in rows:
            pkey = pid if pid else pname
            if (mins or 0) > 0 or (pts or 0) > 0 or (rat or 0) > 0:
                player_hist[pkey].append((rat, pts or 0, pm, mins or 0))

    for r in cur.fetchall():
        mid = r[0]
        if mid != current_mid:
            if current_mid is not None:
                process_match_roster(current_mid, current_rows)
            current_mid = mid
            current_rows = [r]
        else:
            current_rows.append(r)
    if current_mid is not None:
        process_match_roster(current_mid, current_rows)

    print(f"   Partidos con Rolling Roster pre-partido (2024-2026): {len(match_roster_Rolling):,}", flush=True)

    print("5. Cargando métricas en vivo al minuto 27 (graph_points y play_by_play) para partidos 2024-2026...", flush=True)
    # Graph points al minuto 27, 25, 24, 22 y Q3 (21..27)
    cur.execute("""
        SELECT gp.match_id, gp.minute, gp.value
        FROM graph_points gp
        JOIN matches m ON gp.match_id = m.match_id
        WHERE m.date >= '2024-01-01' AND gp.minute <= 27
        ORDER BY gp.match_id, gp.minute
    """)
    gp_m27_map = {}
    cur_mid = None
    pts_list = []
    def proc_gp(mid, plist):
        if len(plist) < 15:
            return
        vals = [v for _, v in plist]
        m_dict = {m: v for m, v in plist}
        v27 = m_dict.get(27, vals[-1])
        v25 = m_dict.get(25, vals[-3] if len(vals) >= 3 else vals[0])
        v24 = m_dict.get(24, vals[-4] if len(vals) >= 4 else vals[0])
        v22 = m_dict.get(22, vals[-6] if len(vals) >= 6 else vals[0])
        q3_vals = [v for m, v in plist if m >= 21]
        gp_m27_map[mid] = {
            "gp_val_m27": v27,
            "gp_slope_2m": v27 - v25,
            "gp_slope_3m": v27 - v24,
            "gp_slope_5m": v27 - v22,
            "gp_accel": (v27 - v25) - (v25 - v22),
            "gp_mean_q3": (sum(q3_vals) / len(q3_vals)) if q3_vals else v27,
            "gp_area_m27": sum(vals),
            "gp_std_q3": float(np.std(q3_vals)) if len(q3_vals) >= 3 else 0.0,
        }
    for mid, m_min, val in cur.fetchall():
        if mid != cur_mid:
            if cur_mid is not None:
                proc_gp(cur_mid, pts_list)
            cur_mid = mid
            pts_list = [(m_min, val)]
        else:
            pts_list.append((m_min, val))
    if cur_mid is not None:
        proc_gp(cur_mid, pts_list)
    print(f"   Partidos con Graph Points <=27 (2024-2026): {len(gp_m27_map):,}", flush=True)

    # Play-by-play en Q3 hasta minuto 27 (en cuartos de 10m, min 27 = 03:00 restantes en Q3)
    print("6. Cargando PBP en Q3 hasta min 27 (03:00 en reloj de Q3)...", flush=True)
    cur.execute("""
        SELECT p.match_id, p.time, p.team, p.points, p.home_score, p.away_score
        FROM play_by_play p
        JOIN matches m ON p.match_id = m.match_id
        WHERE m.date >= '2024-01-01' AND p.quarter = 'Q3'
        ORDER BY p.match_id, p.seq ASC
    """)
    pbp_m27_map = {}
    cur_mid = None
    ev_list = []
    def proc_pbp_q3(mid, evs):
        # Filtrar eventos donde clock >= 3:00 (180 segs restantes en Q3 de 10m)
        valid = []
        for t_str, tm, pts, hs, aws in evs:
            if not t_str or ":" not in t_str:
                continue
            parts = t_str.split(":")
            try:
                rem_s = int(parts[0]) * 60 + int(parts[1])
            except ValueError:
                continue
            if rem_s >= 180 and rem_s <= 600:
                valid.append((rem_s, tm, pts or 0, hs, aws))
        if not valid:
            return
        # Últimos 2 minutos antes del min 27: rem_s entre 180 y 300 (min 25 a 27)
        # Últimos 3 minutos antes del min 27: rem_s entre 180 y 360 (min 24 a 27)
        r2_h = sum(pts for rem_s, tm, pts, _, _ in valid if rem_s <= 300 and tm == "home")
        r2_a = sum(pts for rem_s, tm, pts, _, _ in valid if rem_s <= 300 and tm == "away")
        r3_h = sum(pts for rem_s, tm, pts, _, _ in valid if rem_s <= 360 and tm == "home")
        r3_a = sum(pts for rem_s, tm, pts, _, _ in valid if rem_s <= 360 and tm == "away")
        # Desglose pintura/TL (1 o 2 pts) vs triple (3 pts) en últimos 3m
        r3_paint_diff = sum((pts if tm == "home" else -pts) for rem_s, tm, pts, _, _ in valid if rem_s <= 360 and pts in (1, 2))
        r3_3pt_diff = sum((pts if tm == "home" else -pts) for rem_s, tm, pts, _, _ in valid if rem_s <= 360 and pts == 3)
        # Racha abierta al min 27
        run_h, run_a = 0, 0
        for _, tm, pts, _, _ in reversed(valid):
            if pts <= 0:
                continue
            if tm == "home" and run_a == 0:
                run_h += pts
            elif tm == "away" and run_h == 0:
                run_a += pts
            else:
                break
        last_hs, last_aws = valid[-1][3], valid[-1][4]
        pbp_m27_map[mid] = {
            "r2_pts_diff": r2_h - r2_a,
            "r3_pts_diff": r3_h - r3_a,
            "r3_paint_diff": r3_paint_diff,
            "r3_3pt_diff": r3_3pt_diff,
            "open_run_diff": run_h - run_a,
            "est_home_m27": last_hs,
            "est_away_m27": last_aws,
        }
    for mid, t_str, tm, pts, hs, aws in cur.fetchall():
        if mid != cur_mid:
            if cur_mid is not None:
                proc_pbp_q3(cur_mid, ev_list)
            cur_mid = mid
            ev_list = [(t_str, tm, pts, hs, aws)]
        else:
            ev_list.append((t_str, tm, pts, hs, aws))
    if cur_mid is not None:
        proc_pbp_q3(cur_mid, ev_list)
    print(f"   Partidos con PBP Q3 al min 27 (2024-2026): {len(pbp_m27_map):,}", flush=True)

    # 7. Box Score de 1ª Mitad (Q1 + Q2 cerrados) desde team_statistics
    print("7. Cargando Box Score de 1ª mitad (Q1+Q2) desde team_statistics...", flush=True)
    cur.execute("""
        SELECT ts.match_id, ts.stat_key, SUM(ts.home_value - ts.away_value)
        FROM team_statistics ts
        JOIN matches m ON ts.match_id = m.match_id
        WHERE m.date >= '2024-01-01'
          AND ts.period IN ('Q1', 'Q2', '1Q', '2Q', '1ST')
          AND ts.stat_key IN ('rebounds', 'turnovers', 'assists', 'fouls', 'freeThrows', 'threePointFieldGoals', 'pointsInPaint')
        GROUP BY ts.match_id, ts.stat_key
    """)
    ht_box_map = defaultdict(dict)
    for mid, sk, diff_val in cur.fetchall():
        if diff_val is not None:
            ht_box_map[mid][sk] = float(diff_val)
    print(f"   Partidos con Box Score 1ª mitad: {len(ht_box_map):,}", flush=True)

    # 8. Recorrido cronológico completo (2015-2026) calculando ELO Dinámico por Cuartos, Descanso, H2H y ensamblando vectores
    print("8. Ejecutando motor ELO cronológico (2015-2026) y calculando correlaciones...", flush=True)
    elo_ft = defaultdict(lambda: 1500.0)
    elo_q4 = defaultdict(lambda: 1500.0)
    elo_q1 = defaultdict(lambda: 1500.0)
    elo_q2 = defaultdict(lambda: 1500.0)
    elo_q3 = defaultdict(lambda: 1500.0)
    team_last_date = {}
    team_q4_margins = defaultdict(list)
    team_ft_wins = defaultdict(list)
    pair_hist = defaultdict(list)

    # Cargamos clasificación de ligas
    cur.execute("SELECT league, quarter_duration_minutes, is_youth, is_college, is_women, is_playoffs, competition_type, avg_q4_total_points, home_win_pct FROM leagues_classification")
    lc_map = {}
    for r in cur.fetchall():
        lc_map[r[0]] = r

    # Acumuladores de features para evaluar correlaciones sobre FIBA 10m (y global)
    feat_cols = defaultdict(list)
    y_q4_win = []
    y_q4_margin = []
    y_q4_total = []
    y_ft_rem_margin = []

    last_year = None
    for mid, dt, tm, lg, ht, at, hs, aws in matches:
        qs = qs_map.get(mid)
        if not qs or "Q1" not in qs or "Q2" not in qs or "Q3" not in qs or "Q4" not in qs:
            continue
        q1h, q1a = qs["Q1"]
        q2h, q2a = qs["Q2"]
        q3h, q3a = qs["Q3"]
        q4h, q4a = qs["Q4"]
        if q4h == q4a:
            continue

        d_str = dt.split(" ")[0] if dt else "2020-01-01"
        yr = d_str[:4]
        # Regresión a la media inter-anual suave en agosto/cambio de año
        if last_year is not None and yr != last_year:
            for d_elo in (elo_ft, elo_q4, elo_q1, elo_q2, elo_q3):
                for k in list(d_elo.keys()):
                    d_elo[k] = 0.80 * d_elo[k] + 0.20 * 1500.0
        last_year = yr

        # 1. Leer ELO ANTES del partido (100% leak-free)
        d_elo_ft = elo_ft[ht] - elo_ft[at]
        d_elo_q4 = elo_q4[ht] - elo_q4[at]
        d_elo_q1 = elo_q1[ht] - elo_q1[at]
        d_elo_q2 = elo_q2[ht] - elo_q2[at]
        d_elo_q3 = elo_q3[ht] - elo_q3[at]

        # Rolling Q4 margin y FT win rate de los últimos 12 partidos de cada equipo
        h_q4_r = sum(team_q4_margins[ht][-12:]) / len(team_q4_margins[ht][-12:]) if len(team_q4_margins[ht]) >= 3 else np.nan
        a_q4_r = sum(team_q4_margins[at][-12:]) / len(team_q4_margins[at][-12:]) if len(team_q4_margins[at]) >= 3 else np.nan
        d_roll_q4_margin = (h_q4_r - a_q4_r) if (not np.isnan(h_q4_r) and not np.isnan(a_q4_r)) else np.nan

        h_wr_12 = sum(team_ft_wins[ht][-12:]) / len(team_ft_wins[ht][-12:]) if len(team_ft_wins[ht]) >= 3 else np.nan
        a_wr_12 = sum(team_ft_wins[at][-12:]) / len(team_ft_wins[at][-12:]) if len(team_ft_wins[at]) >= 3 else np.nan
        d_prior_wr_12 = (h_wr_12 - a_wr_12) if (not np.isnan(h_wr_12) and not np.isnan(a_wr_12)) else np.nan

        # Días de descanso / Back-to-Back
        rest_diff = np.nan
        b2b_diff = np.nan
        try:
            cur_dt_obj = datetime.strptime(d_str, "%Y-%m-%d")
            if ht in team_last_date and at in team_last_date:
                rh = min(14, max(0, (cur_dt_obj - team_last_date[ht]).days))
                ra = min(14, max(0, (cur_dt_obj - team_last_date[at]).days))
                rest_diff = float(rh - ra)
                b2b_diff = float((1 if ra <= 1 else 0) - (1 if rh <= 1 else 0))
            team_last_date[ht] = cur_dt_obj
            team_last_date[at] = cur_dt_obj
        except Exception:
            pass

        # H2H limpio (strictly < d_str)
        pair = tuple(sorted([ht, at]))
        past_h2h = [g for g in pair_hist[pair] if g[0] < d_str]
        if past_h2h:
            h2h_ft_margins = [(g[1] if g[2] == ht else -g[1]) for g in past_h2h]
            h2h_q4_margins = [(g[3] if g[2] == ht else -g[3]) for g in past_h2h]
            h2h_avg_ft_margin = sum(h2h_ft_margins) / len(h2h_ft_margins)
            h2h_avg_q4_margin = sum(h2h_q4_margins) / len(h2h_q4_margins)
        else:
            h2h_avg_ft_margin = np.nan
            h2h_avg_q4_margin = np.nan

        # Solo agregamos al estudio de correlaciones de Q4 los partidos FIBA 10m (o sin clasificar 10m) de 2024-2026 donde tenemos telemetría rica
        lc = lc_map.get(lg)
        is_10m_valid = True
        if lc is not None:
            if lc[1] != 10 or lc[2] == 1 or lc[3] == 1 or lc[6] == "friendly":
                is_10m_valid = False

        if d_str >= "2024-01-01" and is_10m_valid:
            ht_diff = (q1h + q2h) - (q1a + q2a)
            q3_end_diff = ht_diff + (q3h - q3a)
            pbp_m = pbp_m27_map.get(mid)
            gp_m = gp_m27_map.get(mid)

            if pbp_m and pbm_has_score(pbp_m):
                m27_diff = float(pbp_m["est_home_m27"] - pbp_m["est_away_m27"])
                q3_part_diff = m27_diff - ht_diff
            else:
                m27_diff = np.nan
                q3_part_diff = np.nan

            odds_info = odds_map.get(mid)
            odds_prob = odds_info[0] if odds_info else np.nan
            odds_logr = odds_info[1] if odds_info else np.nan

            # Expected margin from odds (~6.5 pts per unit of log_ratio) vs current m27_diff (Rubber-band!)
            odds_rubber_band = (odds_logr * 6.5 - m27_diff * 0.65) if (not np.isnan(odds_logr) and not np.isnan(m27_diff)) else np.nan
            elo_rubber_band = ((d_elo_ft / 25.0) - m27_diff * 0.65) if not np.isnan(m27_diff) else np.nan

            ts_h = ts_map.get((mid, ht))
            ts_a = ts_map.get((mid, at))
            d_ts_wp = (ts_h[0] - ts_a[0]) if (ts_h and ts_a and not np.isnan(ts_h[0]) and not np.isnan(ts_a[0])) else np.nan
            d_ts_form = (ts_h[2] - ts_a[2]) if (ts_h and ts_a and not np.isnan(ts_h[2]) and not np.isnan(ts_a[2])) else np.nan

            ros_roll = match_roster_Rolling.get(mid, {})
            ros_leak = match_roster_Leaked.get(mid, {})
            ht_box = ht_box_map.get(mid, {})

            # Target values
            y_q4_win.append(1 if q4h > q4a else 0)
            y_q4_margin.append(float(q4h - q4a))
            y_q4_total.append(float(q4h + q4a))
            y_ft_rem_margin.append(float((hs - aws) - m27_diff) if not np.isnan(m27_diff) else np.nan)

            # Features
            feat_cols["[LEAKED_DEMO] delta_starter_rating_POST_MATCH"].append(ros_leak.get("delta_starter_leaked", np.nan))
            feat_cols["[LEAKED_DEMO] delta_bench_rating_POST_MATCH"].append(ros_leak.get("delta_bench_leaked", np.nan))

            feat_cols["[ROSTER_CLEAN] rolling_delta_starter_rating"].append(ros_roll.get("delta_starter_roll", np.nan))
            feat_cols["[ROSTER_CLEAN] rolling_delta_bench_rating"].append(ros_roll.get("delta_bench_roll", np.nan))
            feat_cols["[ROSTER_CLEAN] rolling_delta_depth"].append(ros_roll.get("delta_depth_roll", np.nan))
            feat_cols["[ROSTER_CLEAN] rolling_delta_plus_minus"].append(ros_roll.get("delta_pm_roll", np.nan))
            feat_cols["[ROSTER_CLEAN] rolling_delta_star_reliance"].append(ros_roll.get("delta_star_rel_roll", np.nan))

            feat_cols["[ODDS_PRE] odds_implied_home_prob"].append(odds_prob)
            feat_cols["[ODDS_PRE] odds_log_ratio (Spread Proxy)"].append(odds_logr)
            feat_cols["[ODDS_LIVE_INTERACT] odds_rubber_band_m27"].append(odds_rubber_band)

            feat_cols["[ELO_CLEAN] delta_elo_ft"].append(d_elo_ft)
            feat_cols["[ELO_CLEAN] delta_elo_q4"].append(d_elo_q4)
            feat_cols["[ELO_CLEAN] delta_elo_q1"].append(d_elo_q1)
            feat_cols["[ELO_CLEAN] delta_rolling12_q4_margin"].append(d_roll_q4_margin)
            feat_cols["[ELO_CLEAN] delta_prior_wr_12"].append(d_prior_wr_12)
            feat_cols["[ELO_LIVE_INTERACT] elo_rubber_band_m27"].append(elo_rubber_band)

            feat_cols["[STRENGTH_REST] ts_delta_standings_win_pct"].append(d_ts_wp)
            feat_cols["[STRENGTH_REST] ts_delta_form_last5"].append(d_ts_form)
            feat_cols["[STRENGTH_REST] rest_days_diff"].append(rest_diff)
            feat_cols["[STRENGTH_REST] opponent_b2b_advantage"].append(b2b_diff)

            feat_cols["[H2H_CLEAN] h2h_avg_ft_margin"].append(h2h_avg_ft_margin)
            feat_cols["[H2H_CLEAN] h2h_avg_q4_margin"].append(h2h_avg_q4_margin)

            feat_cols["[SCORE_LIVE] q1_diff"].append(float(q1h - q1a))
            feat_cols["[SCORE_LIVE] q2_diff"].append(float(q2h - q2a))
            feat_cols["[SCORE_LIVE] halftime_diff"].append(float(ht_diff))
            feat_cols["[SCORE_LIVE] score_diff_m27"].append(m27_diff)
            feat_cols["[SCORE_LIVE] q3_partial_diff_m27"].append(q3_part_diff)
            feat_cols["[SCORE_LIVE] q3_end_diff (Min 30)"].append(float(q3_end_diff))
            feat_cols["[SCORE_LIVE] q3_vs_ht_momentum"].append(q3_part_diff - ht_diff if not np.isnan(q3_part_diff) else np.nan)

            feat_cols["[GRAPH_M27] gp_val_m27"].append(gp_m["gp_val_m27"] if gp_m else np.nan)
            feat_cols["[GRAPH_M27] gp_slope_2m"].append(gp_m["gp_slope_2m"] if gp_m else np.nan)
            feat_cols["[GRAPH_M27] gp_slope_3m"].append(gp_m["gp_slope_3m"] if gp_m else np.nan)
            feat_cols["[GRAPH_M27] gp_slope_5m"].append(gp_m["gp_slope_5m"] if gp_m else np.nan)
            feat_cols["[GRAPH_M27] gp_mean_q3"].append(gp_m["gp_mean_q3"] if gp_m else np.nan)
            feat_cols["[GRAPH_M27] gp_area_m27"].append(gp_m["gp_area_m27"] if gp_m else np.nan)

            feat_cols["[PBP_M27] recent_2m_pts_diff"].append(pbp_m["r2_pts_diff"] if pbp_m else np.nan)
            feat_cols["[PBP_M27] recent_3m_pts_diff"].append(pbp_m["r3_pts_diff"] if pbp_m else np.nan)
            feat_cols["[PBP_M27] recent_3m_paint_ft_diff"].append(pbp_m["r3_paint_diff"] if pbp_m else np.nan)
            feat_cols["[PBP_M27] recent_3m_3pt_diff"].append(pbp_m["r3_3pt_diff"] if pbp_m else np.nan)
            feat_cols["[PBP_M27] open_unanswered_run_diff"].append(pbp_m["open_run_diff"] if pbp_m else np.nan)

            feat_cols["[HT_BOX_Q1Q2] ht_rebounds_diff"].append(ht_box.get("rebounds", np.nan))
            feat_cols["[HT_BOX_Q1Q2] ht_turnovers_diff"].append(ht_box.get("turnovers", np.nan))
            feat_cols["[HT_BOX_Q1Q2] ht_assists_diff"].append(ht_box.get("assists", np.nan))
            feat_cols["[HT_BOX_Q1Q2] ht_fouls_diff"].append(ht_box.get("fouls", np.nan))
            feat_cols["[HT_BOX_Q1Q2] ht_freeThrows_diff"].append(ht_box.get("freeThrows", np.nan))

        # 2. Actualizar ELOs y acumulados DESPUÉS del partido
        update_elo(elo_ft, ht, at, hs, aws, k=20.0, home_adv=35.0)
        update_elo(elo_q4, ht, at, q4h, q4a, k=16.0, home_adv=10.0)
        update_elo(elo_q1, ht, at, q1h, q1a, k=16.0, home_adv=15.0)
        update_elo(elo_q2, ht, at, q2h, q2a, k=16.0, home_adv=10.0)
        update_elo(elo_q3, ht, at, q3h, q3a, k=16.0, home_adv=10.0)

        team_q4_margins[ht].append(q4h - q4a)
        team_q4_margins[at].append(q4a - q4h)
        team_ft_wins[ht].append(1 if hs > aws else 0)
        team_ft_wins[at].append(1 if aws > hs else 0)
        pair_hist[pair].append((d_str, hs - aws, ht, q4h - q4a))

    print("\n=== TABLA MAESTRA DE CORRELACIONES CON Q4 WINNER & Q4 MARGIN (FIBA 10M, 2024-2026) ===", flush=True)
    rows_out = []
    for fname, vals in feat_cols.items():
        st = calc_stats(vals, y_q4_win, y_q4_margin)
        st["feature"] = fname
        rows_out.append(st)

    rows_out.sort(key=lambda r: abs(r["r_margin"]), reverse=True)
    print(f"{'Feature':<48} | {'N':>7} | {'r(Q4_Mgn)':>9} | {'r(Q4_Win)':>9} | {'ROC_AUC':>7} | {'DirAcc%':>7}", flush=True)
    print("-" * 100, flush=True)
    for r in rows_out:
        print(f"{r['feature']:<48} | {r['n']:>7,} | {r['r_margin']:>+9.4f} | {r['r_win']:>+9.4f} | {r['auc']:>7.4f} | {r['dir_acc']:>6.2f}%", flush=True)

    # También analizar el Efecto Rubber-Band / Garbage Time por tramos de marcador al min 27
    print("\n=== ANÁLISIS NO LINEAL: RUBBER-BAND & GARBAGE TIME SEGÚN VENTAJA AL MIN 27 ===", flush=True)
    m27_arr = np.asarray(feat_cols["[SCORE_LIVE] score_diff_m27"], dtype=float)
    elo_arr = np.asarray(feat_cols["[ELO_CLEAN] delta_elo_ft"], dtype=float)
    odds_arr = np.asarray(feat_cols["[ODDS_PRE] odds_log_ratio (Spread Proxy)"], dtype=float)
    q4w_arr = np.asarray(y_q4_win, dtype=float)
    q4m_arr = np.asarray(y_q4_margin, dtype=float)

    bins = [(-999, -15, "Local pierde por 15+ (Blowout Visitante)"),
            (-14, -8,  "Local pierde por 8..14"),
            (-7, -4,   "Local pierde por 4..7"),
            (-3, 3,    "Partido empatado / cerrado (-3..+3)"),
            (4, 7,     "Local gana por 4..7"),
            (8, 14,    "Local gana por 8..14"),
            (15, 999,  "Local gana por 15+ (Blowout Local)")]
    for lo, hi, label in bins:
        msk = (~np.isnan(m27_arr)) & (m27_arr >= lo) & (m27_arr <= hi)
        if np.sum(msk) > 50:
            w_rate = float(np.mean(q4w_arr[msk]) * 100.0)
            m_avg = float(np.mean(q4m_arr[msk]))
            # Dentro de este tramo, correlación de odds_log_ratio y delta_elo_ft con Q4
            r_elo = float(np.corrcoef(elo_arr[msk], q4m_arr[msk])[0, 1]) if np.sum(msk) > 100 else 0.0
            msk_o = msk & (~np.isnan(odds_arr))
            r_odds = float(np.corrcoef(odds_arr[msk_o], q4m_arr[msk_o])[0, 1]) if np.sum(msk_o) > 100 else 0.0
            print(f" {label:<42}: N={int(np.sum(msk)):>6,} | Home Q4 Win%={w_rate:5.1f}% | Avg Q4 Margin={m_avg:+5.2f} | r(ELO)={r_elo:+.3f} | r(Odds)={r_odds:+.3f}", flush=True)

    # Guardar JSON para referencia
    out_json = Path(__file__).resolve().parent / "eda_correlations_results.json"
    out_json.write_text(json.dumps(rows_out, indent=2, ensure_ascii=False), encoding="utf-8")
    conn.close()

def pbm_has_score(pbp_m):
    return pbp_m.get("est_home_m27") is not None and pbp_m.get("est_away_m27") is not None

def update_elo(elo_dict, ht, at, sc_h, sc_a, k=20.0, home_adv=25.0):
    rh = elo_dict[ht]
    ra = elo_dict[at]
    exp_h = 1.0 / (1.0 + 10.0 ** ((ra - (rh + home_adv)) / 400.0))
    act_h = 1.0 if sc_h > sc_a else (0.0 if sc_a > sc_h else 0.5)
    margin = abs(sc_h - sc_a)
    mult = math.log(max(margin, 1) + 1.0) * (2.2 / (((rh - ra) if act_h == 1.0 else (ra - rh)) * 0.001 + 2.2))
    delta = k * mult * (act_h - exp_h)
    elo_dict[ht] = rh + delta
    elo_dict[at] = ra - delta

if __name__ == "__main__":
    main()
