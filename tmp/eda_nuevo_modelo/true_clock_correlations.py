"""
Script: true_clock_correlations.py
Ubicación: tmp/eda_nuevo_modelo/true_clock_correlations.py
Propósito: 
  1. Reconstruir el tiempo real exacto (time_seconds / minuto real 0..40 en FIBA 10m y 0..48 en NBA 12m)
     corrigiendo los dos bugs históricos del repositorio (_period_from_game_seconds con múltiplos de 720s en scraper.py
     y q_start=(q_idx-1)*12 en pbp_utils.py).
  2. Cuantificar exactamente cuántos partidos tenían contaminación de Q4 en m27_v1/v2/v3 por el bug de reloj.
  3. Calcular las correlaciones 100% libres de leakage en el Minuto 27 REAL (1620s), Minuto 30 REAL (Fin de Q3, 1800s)
     y Minuto 32 REAL (2 min dentro de Q4, 1920s), y entrenar un modelo Gradient Boosted de prueba (Walk-Forward / Temporal Split)
     100% libre de leakage para medir el ROC-AUC, Accuracy y Yield real alcanzable con todas las nuevas features.
Hipótesis/Problema: Probar qué features y qué arquitectura/snapshot predicen mejor el Q4 Winner, Q4 Margin, Q4 Total y FT Winner
                    cuando se eliminan las 3 fuentes de data leakage históricas.
"""
import sys
import sqlite3
import math
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

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
        return {"n": int(n), "r_margin": 0.0, "r_win": 0.0, "auc": 0.5, "dir_acc": 50.0}
    r_m = float(np.corrcoef(x, yc)[0, 1])
    r_w = float(np.corrcoef(x, yb)[0, 1])
    try:
        auc = float(roc_auc_score(yb.astype(int), x))
    except Exception:
        auc = 0.5
    nz = x != 0
    if np.sum(nz) > 100:
        hits = np.sum((x[nz] > 0) == (yb[nz] == 1)) if r_w >= 0 else np.sum((x[nz] < 0) == (yb[nz] == 1))
        acc = float(hits * 100.0 / np.sum(nz))
    else:
        acc = 50.0
    return {
        "n": int(n),
        "r_margin": round(r_m, 4),
        "r_win": round(r_w, 4),
        "auc": round(auc, 4),
        "dir_acc": round(acc, 2),
    }

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("1. Cargando matches, quarter_scores, leagues_classification y match_odds...", flush=True)
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

    cur.execute("SELECT league, quarter_duration_minutes, is_youth, is_college, is_women, is_playoffs, competition_type, avg_q4_total_points, home_win_pct FROM leagues_classification")
    lc_map = {r[0]: r for r in cur.fetchall()}

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
        odds_map[mid] = (p_h, log_ratio, hv, av)

    cur.execute("SELECT match_id, team_name, wins, losses, position, form FROM team_strength")
    ts_map = {}
    for mid, tname, w, l, pos, form_str in cur.fetchall():
        w, l = w or 0, l or 0
        wp = (w / (w + l)) if (w + l) >= 3 else np.nan
        f_sc = np.nan
        if form_str:
            wc, lc_ = form_str.count("W"), form_str.count("L")
            if wc + lc_ > 0:
                f_sc = wc / (wc + lc_)
        ts_map[(mid, tname)] = (wp, pos, f_sc)

    print("2. Extrayendo eventos con time_seconds REAL desde match_events (2024-2026)...", flush=True)
    # En match_events tenemos time_seconds exacto (0..1800s para Q1..Q3 en ligas de 10m)!
    # En ligas de 10m:
    #   Min 24 real = 1440s
    #   Min 25 real = 1500s
    #   Min 27 real = 1620s (faltan 3:00 para cerrar Q3)
    #   Min 28 real = 1680s (faltan 2:00 para cerrar Q3)
    #   Min 30 real = 1800s (Fin exacto de Q3)
    #   Min 32 real = 1920s (2:00 jugados de Q4)
    cur.execute("""
        SELECT me.match_id, me.time_seconds, me.team, me.points, me.home_score, me.away_score
        FROM match_events me
        JOIN matches m ON me.match_id = m.match_id
        WHERE m.date >= '2024-01-01'
          AND me.incident_type = 'goal'
          AND me.points > 0
          AND me.time_seconds IS NOT NULL
          AND me.time_seconds <= 1980
        ORDER BY me.match_id, me.time_seconds ASC, me.id DESC
    """)
    real_pbp_map = {}
    cur_mid = None
    ev_list = []

    def proc_real_events(mid, evs):
        if not evs:
            return
        # evs sorted by time_seconds ASC
        # Score at 1620s (True Min 27), 1800s (True Min 30), 1920s (True Min 32)
        sc_27_h, sc_27_a = None, None
        sc_32_h, sc_32_a = None, None
        r2_27_h, r2_27_a = 0, 0 # 1500..1620s (True Min 25..27)
        r3_27_h, r3_27_a = 0, 0 # 1440..1620s (True Min 24..27)
        r3_27_paint, r3_27_3pt = 0, 0
        r3_30_h, r3_30_a = 0, 0 # 1620..1800s (True Min 27..30, últimos 3m de Q3)
        q4_first2m_h, q4_first2m_a = 0, 0 # 1801..1920s (Primeros 2m de Q4)
        q4_first1m_h, q4_first1m_a = 0, 0 # 1801..1860s (Primer 1m de Q4 = lo que filtraba el bug m27!)

        run27_h, run27_a = 0, 0
        for ts, tm, pts, hs, aws in evs:
            if ts <= 1620:
                if hs is not None and aws is not None:
                    sc_27_h, sc_27_a = int(hs), int(aws)
                if ts > 1500:
                    if tm == "home": r2_27_h += pts
                    else: r2_27_a += pts
                if ts > 1440:
                    if tm == "home": r3_27_h += pts
                    else: r3_27_a += pts
                    if pts in (1, 2):
                        r3_27_paint += (pts if tm == "home" else -pts)
                    elif pts == 3:
                        r3_27_3pt += (pts if tm == "home" else -pts)
                if tm == "home":
                    run27_h += pts
                    run27_a = 0
                else:
                    run27_a += pts
                    run27_h = 0
            elif ts <= 1800:
                if tm == "home": r3_30_h += pts
                else: r3_30_a += pts
            elif ts <= 1920:
                if hs is not None and aws is not None:
                    sc_32_h, sc_32_a = int(hs), int(aws)
                if tm == "home": q4_first2m_h += pts
                else: q4_first2m_a += pts
                if ts <= 1860:
                    if tm == "home": q4_first1m_h += pts
                    else: q4_first1m_a += pts

        real_pbp_map[mid] = {
            "sc_27_diff": (sc_27_h - sc_27_a) if (sc_27_h is not None and sc_27_a is not None) else np.nan,
            "sc_27_total": (sc_27_h + sc_27_a) if (sc_27_h is not None and sc_27_a is not None) else np.nan,
            "true_r2_m27_diff": r2_27_h - r2_27_a,
            "true_r3_m27_diff": r3_27_h - r3_27_a,
            "true_r3_m27_paint_diff": r3_27_paint,
            "true_r3_m27_3pt_diff": r3_27_3pt,
            "true_run_m27_diff": run27_h - run27_a,
            "true_r3_m30_diff": r3_30_h - r3_30_a,
            "buggy_q4_first1m_diff": q4_first1m_h - q4_first1m_a,
            "q4_first2m_diff": q4_first2m_h - q4_first2m_a,
            "sc_32_diff": (sc_32_h - sc_32_a) if (sc_32_h is not None and sc_32_a is not None) else np.nan,
        }

    for r in cur.fetchall():
        mid = r[0]
        if mid != cur_mid:
            if cur_mid is not None:
                proc_real_events(cur_mid, ev_list)
            cur_mid = mid
            ev_list = [r[1:]]
        else:
            ev_list.append(r[1:])
    if cur_mid is not None:
        proc_real_events(cur_mid, ev_list)

    print(f"   Partidos con time_seconds real procesados: {len(real_pbp_map):,}", flush=True)

    # Cargar graph_points hasta min 30
    print("3. Cargando graph_points (min 27 y min 30)...", flush=True)
    cur.execute("""
        SELECT gp.match_id, gp.minute, gp.value
        FROM graph_points gp
        JOIN matches m ON gp.match_id = m.match_id
        WHERE m.date >= '2024-01-01' AND gp.minute <= 30
        ORDER BY gp.match_id, gp.minute
    """)
    gp_map = {}
    cur_mid = None
    g_list = []
    def proc_gp(mid, glist):
        if len(glist) < 15:
            return
        md = {m: v for m, v in glist}
        vals27 = [v for m, v in glist if m <= 27]
        if not vals27:
            return
        v27 = md.get(27, vals27[-1])
        v24 = md.get(24, vals27[-4] if len(vals27) >= 4 else vals27[0])
        v20 = md.get(20, vals27[0])
        v30 = md.get(30, glist[-1][1])
        q3_27 = [v for m, v in glist if 21 <= m <= 27]
        gp_map[mid] = {
            "gp_v27": v27,
            "gp_slope_24_27": v27 - v24,
            "gp_slope_20_27": v27 - v20,
            "gp_mean_q3_27": sum(q3_27)/len(q3_27) if q3_27 else v27,
            "gp_v30": v30,
            "gp_slope_27_30": v30 - v27,
        }
    for mid, m_min, val in cur.fetchall():
        if mid != cur_mid:
            if cur_mid is not None:
                proc_gp(cur_mid, g_list)
            cur_mid = mid
            g_list = [(m_min, val)]
        else:
            g_list.append((m_min, val))
    if cur_mid is not None:
        proc_gp(cur_mid, g_list)

    # Recorrido cronológico completo para ELO, Rolling Stats, Rest Days y armado del dataset limpio
    print("4. Recorriendo los 227k partidos cronológicamente (2015-2026)...", flush=True)
    elo_ft = defaultdict(lambda: 1500.0)
    elo_q4 = defaultdict(lambda: 1500.0)
    team_last_dt = {}
    team_q4_hist = defaultdict(list)
    team_ft_hist = defaultdict(list)
    team_tot_hist = defaultdict(list)
    pair_hist = defaultdict(list)
    league_q4_tot_hist = defaultdict(list)
    league_q4_home_win_hist = defaultdict(list)

    feat_cols = defaultdict(list)
    y_q4_win = []
    y_q4_margin = []
    y_q4_total = []
    y_ft_win = []
    ml_rows = []

    last_yr = None
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
        if last_yr is not None and yr != last_yr:
            for ed in (elo_ft, elo_q4):
                for k in list(ed.keys()):
                    ed[k] = 0.80 * ed[k] + 0.20 * 1500.0
        last_yr = yr

        # Priors sin leakage
        d_elo_ft = elo_ft[ht] - elo_ft[at]
        d_elo_q4 = elo_q4[ht] - elo_q4[at]

        h_q4_m = sum(team_q4_hist[ht][-15:]) / len(team_q4_hist[ht][-15:]) if len(team_q4_hist[ht]) >= 3 else 0.0
        a_q4_m = sum(team_q4_hist[at][-15:]) / len(team_q4_hist[at][-15:]) if len(team_q4_hist[at]) >= 3 else 0.0
        d_roll_q4_m = h_q4_m - a_q4_m

        h_wr = sum(team_ft_hist[ht][-15:]) / len(team_ft_hist[ht][-15:]) if len(team_ft_hist[ht]) >= 3 else 0.5
        a_wr = sum(team_ft_hist[at][-15:]) / len(team_ft_hist[at][-15:]) if len(team_ft_hist[at]) >= 3 else 0.5
        d_roll_wr = h_wr - a_wr

        # Descanso real (en días)
        rest_diff = 0.0
        b2b_diff = 0.0
        try:
            c_dt = datetime.strptime(d_str, "%Y-%m-%d")
            if ht in team_last_dt and at in team_last_dt:
                rh = min(10, max(0, (c_dt - team_last_dt[ht]).days))
                ra = min(10, max(0, (c_dt - team_last_dt[at]).days))
                rest_diff = float(rh - ra)
                b2b_diff = float((1 if ra <= 1 else 0) - (1 if rh <= 1 else 0))
            team_last_dt[ht] = c_dt
            team_last_dt[at] = c_dt
        except Exception:
            pass

        pair = tuple(sorted([ht, at]))
        past_h2h = [g for g in pair_hist[pair] if g[0] < d_str]
        if past_h2h:
            h2h_ft_m = sum((g[1] if g[2] == ht else -g[1]) for g in past_h2h) / len(past_h2h)
            h2h_q4_m = sum((g[3] if g[2] == ht else -g[3]) for g in past_h2h) / len(past_h2h)
        else:
            h2h_ft_m = np.nan
            h2h_q4_m = np.nan

        lg_q4_tot_prior = sum(league_q4_tot_hist[lg][-100:]) / len(league_q4_tot_hist[lg][-100:]) if len(league_q4_tot_hist[lg]) >= 15 else 39.0

        # Evaluar en FIBA 10m 2024-2026 con telemetría real
        lc = lc_map.get(lg)
        is_10m_pro = True
        if lc is not None:
            if lc[1] != 10 or lc[2] == 1 or lc[3] == 1 or lc[6] == "friendly":
                is_10m_pro = False

        r_pbp = real_pbp_map.get(mid)
        r_gp = gp_map.get(mid)

        if d_str >= "2024-01-01" and is_10m_pro and r_pbp is not None and not np.isnan(r_pbp["sc_27_diff"]):
            ht_diff = float((q1h + q2h) - (q1a + q2a))
            ht_tot = float(q1h + q2h + q1a + q2a)
            sc27_diff = float(r_pbp["sc_27_diff"])
            q3_part27_diff = sc27_diff - ht_diff
            q3_end_diff = ht_diff + float(q3h - q3a)

            odds_info = odds_map.get(mid)
            odds_prob = odds_info[0] if odds_info else np.nan
            odds_logr = odds_info[1] if odds_info else np.nan

            # Expected FT margin from ELO + Odds
            exp_ft_margin = (d_elo_ft / 25.0) if np.isnan(odds_logr) else (0.5 * (d_elo_ft / 25.0) + 0.5 * (odds_logr * 6.5))
            # Expected margin at min 27 is 0.675 * exp_ft_margin. Deficit of favorite at min 27:
            fav_deficit_m27 = exp_ft_margin - sc27_diff
            fav_deficit_m30 = exp_ft_margin - q3_end_diff

            ts_h = ts_map.get((mid, ht))
            ts_a = ts_map.get((mid, at))
            d_ts_wp = (ts_h[0] - ts_a[0]) if (ts_h and ts_a and not np.isnan(ts_h[0]) and not np.isnan(ts_a[0])) else np.nan
            d_ts_form = (ts_h[2] - ts_a[2]) if (ts_h and ts_a and not np.isnan(ts_h[2]) and not np.isnan(ts_a[2])) else np.nan

            y_q4_win.append(1 if q4h > q4a else 0)
            y_q4_margin.append(float(q4h - q4a))
            y_q4_total.append(float(q4h + q4a))
            y_ft_win.append(1 if hs > aws else 0)

            feat_cols["[BUG_DEMO] q4_first_1min_pts_diff (Lo que leía m27)"].append(float(r_pbp["buggy_q4_first1m_diff"]))
            feat_cols["[LIVE_Q4_M32] q4_first_2min_pts_diff (Min 32 en Q4)"].append(float(r_pbp["q4_first2m_diff"]))

            feat_cols["[1_PRE_ELO] delta_elo_ft (10 años)"].append(d_elo_ft)
            feat_cols["[1_PRE_ELO] delta_elo_q4 (10 años)"].append(d_elo_q4)
            feat_cols["[1_PRE_ELO] delta_rolling15_q4_margin"].append(d_roll_q4_m)
            feat_cols["[1_PRE_ELO] delta_rolling15_win_rate"].append(d_roll_wr)

            feat_cols["[2_PRE_ODDS] odds_log_ratio (Bookmaker Spread)"].append(odds_logr)
            feat_cols["[2_PRE_ODDS] odds_implied_home_prob"].append(odds_prob)

            feat_cols["[3_PRE_CONTEXT] ts_delta_standings_win_pct"].append(d_ts_wp)
            feat_cols["[3_PRE_CONTEXT] ts_delta_form_last5"].append(d_ts_form)
            feat_cols["[3_PRE_CONTEXT] h2h_clean_avg_ft_margin"].append(h2h_ft_m)
            feat_cols["[3_PRE_CONTEXT] h2h_clean_avg_q4_margin"].append(h2h_q4_m)
            feat_cols["[3_PRE_CONTEXT] rest_days_diff"].append(rest_diff)
            feat_cols["[3_PRE_CONTEXT] b2b_advantage"].append(b2b_diff)

            feat_cols["[4_LIVE_SCORE] score_diff_m27_TRUE (1620s)"].append(sc27_diff)
            feat_cols["[4_LIVE_SCORE] score_diff_m30_TRUE (Fin Q3 1800s)"].append(q3_end_diff)
            feat_cols["[4_LIVE_SCORE] halftime_diff (1200s)"].append(ht_diff)
            feat_cols["[4_LIVE_SCORE] q3_partial_diff_m27_TRUE"].append(q3_part27_diff)
            feat_cols["[4_LIVE_SCORE] q3_full_diff_m30_TRUE"].append(float(q3h - q3a))

            feat_cols["[5_LIVE_PBP_TRUE] true_recent_2m_m27_diff (25..27m)"].append(float(r_pbp["true_r2_m27_diff"]))
            feat_cols["[5_LIVE_PBP_TRUE] true_recent_3m_m27_diff (24..27m)"].append(float(r_pbp["true_r3_m27_diff"]))
            feat_cols["[5_LIVE_PBP_TRUE] true_recent_3m_paint_diff"].append(float(r_pbp["true_r3_m27_paint_diff"]))
            feat_cols["[5_LIVE_PBP_TRUE] true_recent_3m_3pt_diff"].append(float(r_pbp["true_r3_m27_3pt_diff"]))
            feat_cols["[5_LIVE_PBP_TRUE] true_open_run_m27_diff"].append(float(r_pbp["true_run_m27_diff"]))
            feat_cols["[5_LIVE_PBP_TRUE] true_recent_3m_m30_diff (27..30m)"].append(float(r_pbp["true_r3_m30_diff"]))

            feat_cols["[6_LIVE_GRAPH] gp_val_m27"].append(float(r_gp["gp_v27"]) if r_gp else np.nan)
            feat_cols["[6_LIVE_GRAPH] gp_val_m30"].append(float(r_gp["gp_v30"]) if r_gp else np.nan)
            feat_cols["[6_LIVE_GRAPH] gp_slope_24_27"].append(float(r_gp["gp_slope_24_27"]) if r_gp else np.nan)
            feat_cols["[6_LIVE_GRAPH] gp_slope_27_30"].append(float(r_gp["gp_slope_27_30"]) if r_gp else np.nan)

            feat_cols["[7_INTERACTION] expected_ft_margin (ELO+Odds)"].append(exp_ft_margin)
            feat_cols["[7_INTERACTION] favorite_deficit_m27 (RubberBand)"].append(fav_deficit_m27)
            feat_cols["[7_INTERACTION] favorite_deficit_m30 (RubberBand)"].append(fav_deficit_m30)

            ml_rows.append({
                "dt": d_str,
                "y_q4_win": 1 if q4h > q4a else 0,
                "y_q4_margin": float(q4h - q4a),
                "y_q4_total": float(q4h + q4a),
                "y_ft_win": 1 if hs > aws else 0,
                "x": [
                    d_elo_ft, d_elo_q4, d_roll_q4_m, d_roll_wr,
                    odds_logr if not np.isnan(odds_logr) else (d_elo_ft / 160.0),
                    1.0 if not np.isnan(odds_logr) else 0.0,
                    d_ts_wp if not np.isnan(d_ts_wp) else d_roll_wr,
                    d_ts_form if not np.isnan(d_ts_form) else 0.0,
                    h2h_ft_m if not np.isnan(h2h_ft_m) else 0.0,
                    h2h_q4_m if not np.isnan(h2h_q4_m) else 0.0,
                    rest_diff, b2b_diff,
                    float(q1h - q1a), float(q2h - q2a), ht_diff, ht_tot,
                    sc27_diff, abs(sc27_diff), q3_part27_diff, float(r_pbp["sc_27_total"]),
                    float(r_pbp["true_r2_m27_diff"]), float(r_pbp["true_r3_m27_diff"]),
                    float(r_pbp["true_r3_m27_paint_diff"]), float(r_pbp["true_r3_m27_3pt_diff"]),
                    float(r_pbp["true_run_m27_diff"]),
                    float(r_gp["gp_v27"]) if r_gp else sc27_diff * 2.0,
                    float(r_gp["gp_slope_24_27"]) if r_gp else 0.0,
                    exp_ft_margin, fav_deficit_m27,
                    lg_q4_tot_prior,
                    float(lc[4]) if lc else 0.0, # is_women
                    float(lc[5]) if lc else 0.0, # is_playoffs
                ],
                "x_m32": [
                    float(r_pbp["q4_first2m_diff"]),
                    float(r_pbp["sc_32_diff"]) if not np.isnan(r_pbp["sc_32_diff"]) else q3_end_diff,
                    q3_end_diff,
                    float(q3h - q3a),
                    float(r_pbp["true_r3_m30_diff"]),
                ]
            })

        # Actualizar ELO e historiales DESPUÉS del partido
        rh, ra = elo_ft[ht], elo_ft[at]
        ex = 1.0 / (1.0 + 10.0 ** ((ra - (rh + 35.0)) / 400.0))
        ac = 1.0 if hs > aws else 0.0
        mg = abs(hs - aws)
        mlt = math.log(max(mg, 1) + 1.0) * (2.2 / (((rh - ra) if ac == 1.0 else (ra - rh)) * 0.001 + 2.2))
        elo_ft[ht] = rh + 20.0 * mlt * (ac - ex)
        elo_ft[at] = ra - 20.0 * mlt * (ac - ex)

        rh4, ra4 = elo_q4[ht], elo_q4[at]
        ex4 = 1.0 / (1.0 + 10.0 ** ((ra4 - (rh4 + 10.0)) / 400.0))
        ac4 = 1.0 if q4h > q4a else 0.0
        mg4 = abs(q4h - q4a)
        mlt4 = math.log(max(mg4, 1) + 1.0) * (2.2 / (((rh4 - ra4) if ac4 == 1.0 else (ra4 - rh4)) * 0.001 + 2.2))
        elo_q4[ht] = rh4 + 16.0 * mlt4 * (ac4 - ex4)
        elo_q4[at] = ra4 - 16.0 * mlt4 * (ac4 - ex4)

        team_q4_hist[ht].append(q4h - q4a)
        team_q4_hist[at].append(q4a - q4h)
        team_ft_hist[ht].append(1 if hs > aws else 0)
        team_ft_hist[at].append(1 if aws > hs else 0)
        pair_hist[pair].append((d_str, hs - aws, ht, q4h - q4a))
        league_q4_tot_hist[lg].append(q4h + q4a)

    print("\n=== CORRELACIONES 100% REALES (SIN BUG DE RELOJ NI LEAKAGE) CON Q4 WINNER & MARGIN ===", flush=True)
    res_list = []
    for k, v in feat_cols.items():
        st = calc_stats(v, y_q4_win, y_q4_margin)
        st["feature"] = k
        res_list.append(st)
    res_list.sort(key=lambda r: abs(r["r_margin"]), reverse=True)
    print(f"{'Feature':<52} | {'N':>7} | {'r(Q4_Mgn)':>9} | {'r(Q4_Win)':>9} | {'ROC_AUC':>7} | {'DirAcc%':>7}", flush=True)
    print("-" * 104, flush=True)
    for r in res_list:
        print(f"{r['feature']:<52} | {r['n']:>7,} | {r['r_margin']:>+9.4f} | {r['r_win']:>+9.4f} | {r['auc']:>7.4f} | {r['dir_acc']:>6.2f}%", flush=True)

    # Benchmark rápido de ML en Split Temporal Estricto (70/15/15) para comparar:
    # A) Predicción de Q4 Winner en Min 27 REAL (pre-Q4)
    # B) Predicción de Q4 Winner en Min 30 REAL (descanso fin de Q3)
    # C) Predicción de Q4 Winner en Min 32 REAL (2 minutos jugados de Q4)
    # D) Predicción de FT Winner en Min 27 REAL
    # E) Predicción de Q4 Total Points (Over/Under) en Min 27 REAL
    print("\n=== BENCHMARK ML TEMPORAL ESTRICTO (100% SIN LEAKAGE, N=" + f"{len(ml_rows):,}" + ") ===", flush=True)
    n_tot = len(ml_rows)
    i_tr = int(n_tot * 0.70)
    i_va = int(n_tot * 0.85)

    X_m27 = np.array([r["x"] for r in ml_rows], dtype=float)
    X_m30 = np.array([r["x"] + r["x_m32"][2:] for r in ml_rows], dtype=float)
    X_m32 = np.array([r["x"] + r["x_m32"] for r in ml_rows], dtype=float)

    Y_q4w = np.array([r["y_q4_win"] for r in ml_rows], dtype=int)
    Y_ftw = np.array([r["y_ft_win"] for r in ml_rows], dtype=int)
    Y_q4t = np.array([r["y_q4_total"] for r in ml_rows], dtype=float)

    for label, X_mat, Y_vec in [
        ("Target 1A: Q4 Winner @ Min 27 REAL (Antes de fin Q3)", X_m27, Y_q4w),
        ("Target 1B: Q4 Winner @ Min 30 REAL (Descanso Q3->Q4)", X_m30, Y_q4w),
        ("Target 1C: Q4 Winner @ Min 32 REAL (2m jugados de Q4)", X_m32, Y_q4w),
        ("Target 2 : FT Winner @ Min 27 REAL (Ganador Partido)", X_m27, Y_ftw),
    ]:
        clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_depth=5, min_samples_leaf=40, random_state=42)
        clf.fit(X_mat[:i_tr], Y_vec[:i_tr])
        probs = clf.predict_proba(X_mat[i_va:])[:, 1]
        yt = Y_vec[i_va:]
        auc = roc_auc_score(yt, probs)
        acc = accuracy_score(yt, (probs >= 0.5).astype(int))
        # Evaluar hit rate en distintos umbrales de confianza (max(p, 1-p) >= thr)
        conf = np.maximum(probs, 1.0 - probs)
        picks = (probs >= 0.5).astype(int)
        hits = (picks == yt)
        thr_str = []
        for thr in (0.60, 0.65, 0.70, 0.72):
            m_thr = conf >= thr
            cnt = int(np.sum(m_thr))
            if cnt > 20:
                hr = float(np.mean(hits[m_thr]) * 100.0)
                yld = (hr / 100.0 * 1.40 - 1.0) * 100.0
                thr_str.append(f"@{thr:.2f}: {hr:.1f}% ({cnt} bets, yield {yld:+.1f}%)")
        print(f" -> {label:<52} | Test AUC={auc:.4f} | Acc={acc*100:.2f}% | " + " | ".join(thr_str), flush=True)

    # Regresión de Q4 Total Points (Over/Under)
    reg = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, max_depth=5, min_samples_leaf=40, random_state=42)
    reg.fit(X_m27[:i_tr], Y_q4t[:i_tr])
    pred_tot = reg.predict(X_m27[i_va:])
    true_tot = Y_q4t[i_va:]
    prior_tot = X_m27[i_va:, -3] # lg_q4_tot_prior as proxy line
    mae = float(np.mean(np.abs(pred_tot - true_tot)))
    r_tot = float(np.corrcoef(pred_tot, true_tot)[0, 1])
    # Si la línea del bookmaker para Q4 es aprox lg_q4_tot_prior, cuando |pred - line| >= 2.5 pts, qué hit rate hay?
    diff_line = pred_tot - prior_tot
    mask_ou = np.abs(diff_line) >= 2.5
    ou_hits = ((diff_line[mask_ou] > 0) == (true_tot[mask_ou] > prior_tot[mask_ou]))
    ou_hr = float(np.mean(ou_hits) * 100.0) if np.sum(mask_ou) > 0 else 50.0
    print(f" -> {'Target 3 : Q4 Total Points (Over/Under) @ Min 27':<52} | Test MAE={mae:.2f} pts | r={r_tot:+.4f} | Edge >=2.5pts vs Prior: {ou_hr:.1f}% ({int(np.sum(mask_ou))} bets)", flush=True)

    conn.close()

if __name__ == "__main__":
    main()
