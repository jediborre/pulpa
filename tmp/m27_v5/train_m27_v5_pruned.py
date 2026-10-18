"""
Pipeline de Entrenamiento M27 V5 con Poda de Ruido (Clean Data Pruning).
Excluye de raíz:
1. Amistosos y Pretemporadas (Friendly, Preseason) - WR real 50%
2. Descenso y Playouts (Relegation, Playout, Rel/Prom) - WR real 45%
3. Juveniles y Canteras Formativas (U14 a U20, Youth, Espoirs)
4. College NCAA (mitades de 20 min, sin 4 cuartos naturales)
5. Ligas de 12 minutos (NBA, CBA, PBA) -> Se canalizan a su propio modelo dedicado m34_nba.

Resultado: Un modelo Campeón FIBA 10m con máxima pureza de gradientes.
"""

import math
import os
import pickle
import sqlite3
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DB_PATH = Path("matches.db")
PROD_MODEL_DIR = Path("models/m27_v5/model_outputs")
PROD_MODEL_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp/m27_v5")
TMP_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_NAMES = [
    # Grupo A: Marcadores y Momentum (0..12)
    "cur_score_diff",
    "abs_cur_score_diff",
    "ht_diff",
    "q1_diff",
    "q2_diff",
    "q3_eff_diff",
    "q3_vs_ht_momentum",
    "recent_2m_run_diff",
    "pace_pts_per_min",
    "pace_vs_league_prior",
    "gp_val",
    "gp_slope_3m",
    "fraction_q3_completed",
    # Grupo B: Roster ELO y Estrategia Táctica (13..23)
    "d_lineup_starters_elo_ft",
    "d_lineup_starters_elo_q4",
    "d_lineup_bench_elo_q4",
    "d_lineup_top3_elo_q4",
    "d_roster_drift_adv",
    "d_strat_q4_net",
    "q3_tactical_surprise",
    "d_elo_q1",
    "elo_reliability",
    "d_q4_bayes",
    "d_wr_bayes",
    # Grupo C: Mercado y Momios (24..29)
    "has_odds",
    "odds_log_ratio",
    "expected_ft_margin",
    "h2h_bayes_ft_margin",
    "h2h_bayes_q4_margin",
    "ts_delta_wp_bayes",
    # Grupo D: Tiros, Fatiga, Situacional y Priors (30..36)
    "pbp_2pt_diff_q1_q3",
    "pbp_3pt_diff_q1_q3",
    "pbp_ft_diff_q1_q3",
    "rubber_band_gap",
    "white_flag_risk_zone",
    "blowout_garbage_time",
    "lg_avg_q4_prior",
]

def fast_mean(dq, default=0.0):
    n = len(dq)
    return (sum(dq) / n) if n else default

def parse_mins(m_val):
    if m_val is None:
        return 0.0
    s = str(m_val).strip()
    if not s:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0

def build_pruned_matrices():
    clean_cache_file = TMP_DIR / "dataset_v5_clean_cache.pkl"
    if clean_cache_file.exists():
        print(f"Cargando dataset podado pre-calculado desde {clean_cache_file}...", flush=True)
        with open(clean_cache_file, "rb") as f:
            data = pickle.load(f)
        return (
            data["X"], data["y_class"], data["y_reg"], data["meta"],
            data["elo_ft"], data["elo_q4"], data["elo_q1"],
            data["player_state"], data["team_q_profile"], data["lc_map"]
        )

    t0 = time.time()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-256000;")
    cur = conn.cursor()

    print("[1/6] Cargando taxonomía de ligas con FILTRO DE PODA...", flush=True)
    cur.execute("""
        SELECT league, quarter_duration_minutes, is_women, is_youth, is_college, is_playoffs,
               avg_total_points, avg_q4_total_points, home_win_pct, q4_home_win_pct, is_relegation
        FROM leagues_classification
    """)
    lc_map = {}
    pruned_leagues = set()

    for row in cur.fetchall():
        lg = row[0]
        q_dur = int(row[1] or 10)
        is_w = int(row[2] or 0)
        is_y = int(row[3] or 0)
        is_c = int(row[4] or 0)
        is_r = int(row[10] or 0)
        lg_up = lg.upper()

        # Criterios de Poda de Ruido:
        # 1. No juveniles / formativas
        # 2. No college NCAA (mitades 20m)
        # 3. No ligas de 12 minutos (NBA, CBA, PBA tienen su propio modelo)
        # 4. No amistosos / pretemporadas
        # 5. No descenso / playout
        # 6. No esports / cyber
        is_noise = (
            q_dur != 10 or
            is_y == 1 or
            is_c == 1 or
            is_r == 1 or
            any(w in lg_up for w in [
                "FRIENDLY", "PRESEASON", "RELEGATION", "PLAYOUT", "REL/PROM",
                "QUALIFYING ROUND", "U13", "U14", "U15", "U16", "U17", "U18",
                "U19", "U20", "YOUTH", "ESPOIRS", "NCAA", "COLLEGE", "CYBER",
                "ESPORTS", "2K", "SUMMER LEAGUE", "TPBL", "SPARKASSE PRVA LIGA"
            ])
        )

        if is_noise:
            pruned_leagues.add(lg)
            continue

        arch = "fiba_women_10m" if is_w == 1 else "fiba_men_10m"
        lc_map[lg] = {
            "arch": arch,
            "q_dur": 10,
            "is_w": is_w,
            "lg_avg_ft": float(row[6] or 158.0),
            "lg_avg_q4": float(row[7] or 39.5),
        }

    print(f"   -> Ligas válidas FIBA 10m conservadas: {len(lc_map):,} | Ligas podadas: {len(pruned_leagues):,}", flush=True)

    cur.execute("""
        SELECT match_id, home_value, away_value
        FROM match_odds
        WHERE odds_type = 'Home/Away' AND market_period = 'Match'
          AND home_value > 1.0 AND away_value > 1.0
    """)
    odds_map = {}
    for mid, ho, ao in cur.fetchall():
        odds_map[mid] = (1.0, float(np.clip(math.log(ao / ho), -2.5, 2.5)))

    cur.execute("SELECT match_id, team_name, wins, losses, position, form FROM team_strength")
    ts_map = {}
    for mid, tname, w, l, pos, form_s in cur.fetchall():
        w, l = int(w or 0), int(l or 0)
        tot = w + l
        wp_bayes = (w + 3.0) / (tot + 6.0)
        f_val = 0.5
        if form_s:
            wc, lc_ = form_s.count("W"), form_s.count("L")
            if wc + lc_ > 0:
                f_val = (wc + 1.5) / (wc + lc_ + 3.0)
        ts_map[(mid, tname)] = (wp_bayes, f_val)

    print("[2/6] Cargando player_stats y lineups...", flush=True)
    ps_by_match = defaultdict(list)
    cur.execute("""
        SELECT match_id, team, player_id, player_name, sofascore_rating, minutes_played, points, fouls, plus_minus
        FROM player_stats
    """)
    for row in cur.fetchall():
        ps_by_match[row[0]].append(row[1:])

    starter_map = {}
    cur.execute("SELECT match_id, team, player_id, player_name, is_starter FROM lineups")
    for mid, team, pid, pname, is_st in cur.fetchall():
        key = pid if pid else pname
        starter_map[(mid, team, key)] = int(is_st or 0)

    print("[3/6] Agregando tiros 2PT, 3PT y FT desde match_events...", flush=True)
    cur.execute("""
        SELECT match_id, team, points, COUNT(*)
        FROM match_events
        WHERE time_seconds <= 1800 AND points IN (1, 2, 3) AND team IS NOT NULL
        GROUP BY match_id, team, points
    """)
    shots_map = defaultdict(lambda: {"2pt": 0, "3pt": 0, "ft": 0})
    for mid, tm, pts, cnt in cur.fetchall():
        k = "2pt" if pts == 2 else ("3pt" if pts == 3 else "ft")
        shots_map[(mid, tm)][k] = cnt

    print("[4/6] Cargando graph_points de momentum...", flush=True)
    cur.execute("SELECT match_id, minute, value FROM graph_points WHERE minute IN (24, 27, 30)")
    gp_map = defaultdict(dict)
    for mid, mn, val in cur.fetchall():
        gp_map[mid][int(mn)] = float(val or 0.0)

    print("[5/6] Cargando partidos limpios y marcadores Q1..Q4...", flush=True)
    cur.execute("""
        SELECT match_id, quarter, home, away
        FROM quarter_scores INDEXED BY idx_qs_covering
        WHERE quarter IN ('Q1', 'Q2', 'Q3', 'Q4')
    """)
    q_map = defaultdict(dict)
    for mid, q, h, a in cur.fetchall():
        if h is not None and a is not None:
            q_map[mid][q] = (int(h), int(a))

    cur.execute("""
        SELECT match_id, date, time, league, home_team, away_team, home_team_id, away_team_id
        FROM matches INDEXED BY idx_matches_date_id
        WHERE date IS NOT NULL
        ORDER BY date ASC, match_id ASC
    """)
    raw_matches = cur.fetchall()
    all_matches = []
    for mid, dt, tm, lg, ht, at, htid, atid in raw_matches:
        if lg in pruned_leagues or lg not in lc_map:
            continue
        qs = q_map.get(mid)
        if not qs or len(qs) < 4:
            continue
        all_matches.append((
            mid, dt, tm, lg, ht, at, htid, atid,
            qs["Q1"][0], qs["Q1"][1],
            qs["Q2"][0], qs["Q2"][1],
            qs["Q3"][0], qs["Q3"][1],
            qs["Q4"][0], qs["Q4"][1],
        ))
    conn.close()

    print(f"   -> Partidos profesionales FIBA 10m limpios tras poda: {len(all_matches):,} ({time.time()-t0:.1f}s)", flush=True)

    # State tracking
    elo_ft = defaultdict(lambda: 1500.0)
    elo_q4 = defaultdict(lambda: 1500.0)
    elo_q1 = defaultdict(lambda: 1500.0)
    last_team_date = {}
    team_q4_hist = defaultdict(list)
    team_ft_hist = defaultdict(list)
    team_q_profile = defaultdict(
        lambda: {
            "q1_net": deque(maxlen=12),
            "q2_net": deque(maxlen=12),
            "q3_net": deque(maxlen=12),
            "q4_net": deque(maxlen=12),
        }
    )
    h2h_hist = defaultdict(list)

    player_state = defaultdict(
        lambda: {
            "elo_ft": 1500.0,
            "elo_q4": 1500.0,
            "n": 0,
            "ratings": deque(maxlen=10),
            "pm": deque(maxlen=10),
            "mins": deque(maxlen=10),
        }
    )

    X_list = []
    y_class_list = []
    y_reg_list = []
    meta_list = []

    print("[6/6] Extrayendo 37 features sobre dataset limpio...", flush=True)
    t_feat = time.time()

    for idx_m, m in enumerate(all_matches):
        mid, dt, tm, lg, ht, at, htid, atid, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a = m
        lg_info = lc_map[lg]
        arch = lg_info["arch"]
        lg_avg_ft = lg_info["lg_avg_ft"]
        lg_avg_q4 = lg_info["lg_avg_q4"]

        # Parse year
        yr = int(dt.split("-")[0]) if dt and "-" in dt else 2020

        # Snapshot a min 27 (cierre Q3 a 30m en FIBA)
        cur_h = q1h + q2h + int(round(q3h * 0.70))
        cur_a = q1a + q2a + int(round(q3a * 0.70))
        cur_diff = float(cur_h - cur_a)
        abs_cur_score_diff = abs(cur_diff)

        q1_diff = float(q1h - q1a)
        q2_diff = float(q2h - q2a)
        ht_diff = q1_diff + q2_diff

        eff_q3h = int(round(q3h * 0.70))
        eff_q3a = int(round(q3a * 0.70))
        q3_diff = float(eff_q3h - eff_q3a)
        q3_tot = float(eff_q3h + eff_q3a)
        q3_eff_diff = (q3_diff / (q3_tot + 1.0)) * 10.0
        q3_vs_ht_momentum = q3_diff - (ht_diff * 0.5)

        recent_2m_run_diff = float(np.clip(q3_diff * 0.40, -12.0, 12.0))
        pace_pts_per_min = float(cur_h + cur_a) / 27.0
        lg_expected_ppm = (lg_avg_ft * 0.75) / 30.0
        pace_vs_league_prior = pace_pts_per_min - lg_expected_ppm

        gp_m = gp_map.get(mid, {})
        gp_val = float(gp_m.get(27, cur_diff * 0.5))
        gp_prev = float(gp_m.get(24, gp_val))
        gp_slope_3m = (gp_val - gp_prev) / 3.0
        fraction_q3_completed = 0.70

        # Decay seasonal
        if ht in last_team_date and dt:
            days = (datetime.strptime(dt, "%Y-%m-%d") - datetime.strptime(last_team_date[ht], "%Y-%m-%d")).days
            if days > 75:
                elo_ft[ht] = 1500.0 + (elo_ft[ht] - 1500.0) * 0.75
                elo_q4[ht] = 1500.0 + (elo_q4[ht] - 1500.0) * 0.75
                elo_q1[ht] = 1500.0 + (elo_q1[ht] - 1500.0) * 0.75
        if at in last_team_date and dt:
            days = (datetime.strptime(dt, "%Y-%m-%d") - datetime.strptime(last_team_date[at], "%Y-%m-%d")).days
            if days > 75:
                elo_ft[at] = 1500.0 + (elo_ft[at] - 1500.0) * 0.75
                elo_q4[at] = 1500.0 + (elo_q4[at] - 1500.0) * 0.75
                elo_q1[at] = 1500.0 + (elo_q1[at] - 1500.0) * 0.75

        last_team_date[ht] = dt
        last_team_date[at] = dt

        h_elo_ft, a_elo_ft = elo_ft[ht], elo_ft[at]
        h_elo_q4, a_elo_q4 = elo_q4[ht], elo_q4[at]
        h_elo_q1, a_elo_q1 = elo_q1[ht], elo_q1[at]

        d_elo_ft = float(h_elo_ft - a_elo_ft)
        d_elo_q4 = float(h_elo_q4 - a_elo_q4)
        d_elo_q1 = float(h_elo_q1 - a_elo_q1)

        n_ht = len(team_ft_hist[ht])
        n_at = len(team_ft_hist[at])
        elo_reliability = float(min(1.0, (n_ht + n_at) / 20.0))
        d_q4_bayes = d_elo_q4 / 35.0
        d_wr_bayes = d_elo_ft / 180.0

        # Roster ELO
        p_list = ps_by_match.get(mid, [])
        h_st_ft, a_st_ft, h_st_q4, a_st_q4 = [], [], [], []
        h_bn_q4, a_bn_q4 = [], []

        for p_row in p_list:
            team_s, pid, pname, s_rat, m_ply, pts_p, fouls_p, pm_p = p_row
            is_home_p = (team_s == ht) or ("home" in str(team_s).lower())
            pkey = pid if pid else pname
            is_starter = starter_map.get((mid, team_s, pkey), 0)
            p_prior = player_state[pkey]
            p_ft = p_prior["elo_ft"]
            p_q4 = p_prior["elo_q4"]

            if is_starter == 1:
                if is_home_p:
                    h_st_ft.append(p_ft)
                    h_st_q4.append(p_q4)
                else:
                    a_st_ft.append(p_ft)
                    a_st_q4.append(p_q4)
            else:
                if is_home_p:
                    h_bn_q4.append(p_q4)
                else:
                    a_bn_q4.append(p_q4)

        if len(h_st_ft) >= 3 and len(a_st_ft) >= 3:
            d_lineup_starters_elo_ft = float(np.mean(h_st_ft) - np.mean(a_st_ft))
            d_lineup_starters_elo_q4 = float(np.mean(h_st_q4) - np.mean(a_st_q4))
            d_roster_drift_adv = float(d_lineup_starters_elo_q4 - d_elo_q4)
            h_top3 = sorted(h_st_q4, reverse=True)[:3]
            a_top3 = sorted(a_st_q4, reverse=True)[:3]
            d_lineup_top3_elo_q4 = float(np.mean(h_top3) - np.mean(a_top3))
            d_lineup_bench_elo_q4 = float(np.mean(h_bn_q4) - np.mean(a_bn_q4)) if (h_bn_q4 and a_bn_q4) else 0.0
        else:
            d_lineup_starters_elo_ft = d_elo_ft
            d_lineup_starters_elo_q4 = d_elo_q4
            d_roster_drift_adv = 0.0
            d_lineup_top3_elo_q4 = d_elo_q4
            d_lineup_bench_elo_q4 = 0.0

        h_q4_net = fast_mean(team_q_profile[ht]["q4_net"], 0.0)
        a_q4_net = fast_mean(team_q_profile[at]["q4_net"], 0.0)
        d_strat_q4_net = float(h_q4_net - a_q4_net)

        h_q3_net = fast_mean(team_q_profile[ht]["q3_net"], 0.0)
        a_q3_net = fast_mean(team_q_profile[at]["q3_net"], 0.0)
        q3_tactical_surprise = float(q3_diff - (h_q3_net - a_q3_net))

        # Mercado
        has_odds, odds_log_ratio = odds_map.get(mid, (0.0, 0.0))
        if has_odds:
            expected_ft_margin = 6.5 * odds_log_ratio + 0.50 * (d_elo_ft / 28.0)
        else:
            expected_ft_margin = d_elo_ft / 28.0

        h2h_key = tuple(sorted([ht, at]))
        prior_h2h = h2h_hist[h2h_key]
        if prior_h2h:
            h2h_ft = [x[0] if x[2] == ht else -x[0] for x in prior_h2h[-8:]]
            h2h_q4 = [x[1] if x[2] == ht else -x[1] for x in prior_h2h[-8:]]
            shrink = len(h2h_ft) / (len(h2h_ft) + 4.0)
            h2h_bayes_ft_margin = float(np.mean(h2h_ft)) * shrink
            h2h_bayes_q4_margin = float(np.mean(h2h_q4)) * shrink
        else:
            h2h_bayes_ft_margin, h2h_bayes_q4_margin = 0.0, 0.0

        ts_h = ts_map.get((mid, ht), (0.5, 0.5))
        ts_a = ts_map.get((mid, at), (0.5, 0.5))
        ts_delta_wp_bayes = float(ts_h[0] - ts_a[0])

        sh_h = shots_map.get((mid, ht), {"2pt": 0, "3pt": 0, "ft": 0})
        sh_a = shots_map.get((mid, at), {"2pt": 0, "3pt": 0, "ft": 0})
        pbp_2pt_diff_q1_q3 = float(sh_h["2pt"] - sh_a["2pt"])
        pbp_3pt_diff_q1_q3 = float(sh_h["3pt"] - sh_a["3pt"])
        pbp_ft_diff_q1_q3 = float(sh_h["ft"] - sh_a["ft"])

        exp_cur_lead = expected_ft_margin * 0.75
        rubber_band_gap = float(cur_diff - exp_cur_lead)
        white_flag_risk_zone = 1.0 if abs_cur_score_diff >= 18.0 else 0.0
        blowout_garbage_time = 1.0 if abs_cur_score_diff >= 25.0 else 0.0

        feat_row = [
            cur_diff, abs_cur_score_diff, ht_diff, q1_diff, q2_diff, q3_eff_diff,
            q3_vs_ht_momentum, recent_2m_run_diff, pace_pts_per_min, pace_vs_league_prior,
            gp_val, gp_slope_3m, fraction_q3_completed,
            d_lineup_starters_elo_ft, d_lineup_starters_elo_q4, d_lineup_bench_elo_q4,
            d_lineup_top3_elo_q4, d_roster_drift_adv, d_strat_q4_net, q3_tactical_surprise,
            d_elo_q1, elo_reliability, d_q4_bayes, d_wr_bayes,
            has_odds, odds_log_ratio, expected_ft_margin, h2h_bayes_ft_margin,
            h2h_bayes_q4_margin, ts_delta_wp_bayes,
            pbp_2pt_diff_q1_q3, pbp_3pt_diff_q1_q3, pbp_ft_diff_q1_q3,
            rubber_band_gap, white_flag_risk_zone, blowout_garbage_time,
            lg_avg_q4
        ]

        q4_diff = float(q4h - q4a)
        y_class = 1 if q4_diff > 0 else 0
        y_reg = q4_diff

        X_list.append(feat_row)
        y_class_list.append(y_class)
        y_reg_list.append(y_reg)
        meta_list.append({
            "mid": mid, "dt": dt, "yr": yr, "arch": arch, "lg": lg,
            "ht": ht, "at": at, "cur_diff": cur_diff, "q4_diff": q4_diff
        })

        # State updates
        k_ft = 22.0
        ft_diff = float((q1h+q2h+q3h+q4h) - (q1a+q2a+q3a+q4a))
        s_ft = 1.0 if ft_diff > 0 else (0.5 if ft_diff == 0 else 0.0)
        e_ft = 1.0 / (1.0 + 10.0 ** (-d_elo_ft / 400.0))
        elo_ft[ht] += k_ft * (s_ft - e_ft)
        elo_ft[at] -= k_ft * (s_ft - e_ft)

        k_q4 = 18.0
        s_q4 = 1.0 if q4_diff > 0 else (0.5 if q4_diff == 0 else 0.0)
        e_q4 = 1.0 / (1.0 + 10.0 ** (-d_elo_q4 / 280.0))
        elo_q4[ht] += k_q4 * (s_q4 - e_q4)
        elo_q4[at] -= k_q4 * (s_q4 - e_q4)

        s_q1 = 1.0 if q1_diff > 0 else (0.5 if q1_diff == 0 else 0.0)
        e_q1 = 1.0 / (1.0 + 10.0 ** (-d_elo_q1 / 280.0))
        elo_q1[ht] += 12.0 * (s_q1 - e_q1)
        elo_q1[at] -= 12.0 * (s_q1 - e_q1)

        team_q4_hist[ht].append(q4_diff)
        team_q4_hist[at].append(-q4_diff)
        team_ft_hist[ht].append(ft_diff)
        team_ft_hist[at].append(-ft_diff)

        team_q_profile[ht]["q1_net"].append(q1_diff)
        team_q_profile[at]["q1_net"].append(-q1_diff)
        team_q_profile[ht]["q2_net"].append(q2_diff)
        team_q_profile[at]["q2_net"].append(-q2_diff)
        team_q_profile[ht]["q3_net"].append(float(q3h - q3a))
        team_q_profile[at]["q3_net"].append(-float(q3h - q3a))
        team_q_profile[ht]["q4_net"].append(q4_diff)
        team_q_profile[at]["q4_net"].append(-q4_diff)

        h2h_hist[h2h_key].append((ft_diff, q4_diff, ht))

        # Update player ELOs
        for p_row in p_list:
            team_s, pid, pname, s_rat, m_ply, pts_p, fouls_p, pm_p = p_row
            is_home_p = (team_s == ht) or ("home" in str(team_s).lower())
            pkey = pid if pid else pname
            p_st = player_state[pkey]
            p_st["n"] += 1
            if pm_p is not None:
                p_st["pm"].append(float(pm_p))
            if m_ply:
                p_st["mins"].append(parse_mins(m_ply))
            p_actual = s_ft if is_home_p else (1.0 - s_ft)
            p_st["elo_ft"] += 10.0 * (p_actual - e_ft)
            p_actual_q4 = s_q4 if is_home_p else (1.0 - s_q4)
            p_st["elo_q4"] += 10.0 * (p_actual_q4 - e_q4)

        if (idx_m + 1) % 50000 == 0:
            print(f"   -> Procesados {idx_m+1:,} partidos ({time.time()-t_feat:.1f}s)...", flush=True)

    X = np.array(X_list, dtype=np.float32)
    y_class = np.array(y_class_list, dtype=np.int32)
    y_reg = np.array(y_reg_list, dtype=np.float32)

    print(f"Guardando caché limpio en {clean_cache_file}...", flush=True)
    with open(clean_cache_file, "wb") as f:
        pickle.dump({
            "X": X, "y_class": y_class, "y_reg": y_reg, "meta": meta_list,
            "elo_ft": dict(elo_ft), "elo_q4": dict(elo_q4), "elo_q1": dict(elo_q1),
            "player_state": dict(player_state), "team_q_profile": dict(team_q_profile),
            "lc_map": lc_map, "feature_names": FEATURE_NAMES
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    return X, y_class, y_reg, meta_list, elo_ft, elo_q4, elo_q1, player_state, team_q_profile, lc_map

def train_and_evaluate():
    print("=" * 95)
    print("ENTRENAMIENTO M27 V5 CON DATASET PODADO LIMPIO (FIBA 10M PROFESIONAL)")
    print("=" * 95)

    X, y_class, y_reg, meta, elo_ft, elo_q4, elo_q1, player_state, team_q_profile, lc_map = build_pruned_matrices()

    years = np.array([m["yr"] for m in meta])
    print(f"\nDataset Total Podado: {len(X):,} partidos limpios")
    print(f"Distribución por año: {pd.Series(years).value_counts().sort_index().to_dict()}")

    # 1. Validación Walk-Forward Temporal
    cuts = [
        ("Corte 1: Train <= 2022 -> Test 2023", years <= 2022, years == 2023),
        ("Corte 2: Train <= 2023 -> Test 2024", years <= 2023, years == 2024),
        ("Corte 3: Train <= 2024 -> Test 2025-2026", years <= 2024, years >= 2025),
    ]

    print("\n--- VALIDACIÓN WALK-FORWARD MULTI-CORTE EN EL TIEMPO ---")
    wf_results = []
    for cut_name, tr_mask, te_mask in cuts:
        X_tr, y_tr = X[tr_mask], y_class[tr_mask]
        X_te, y_te = X[te_mask], y_class[te_mask]
        yr_tr = y_reg[tr_mask]

        clf_xgb = xgb.XGBClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.055,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=50.0,
            random_state=42, n_jobs=-1, eval_metric="logloss"
        )
        clf_xgb.fit(X_tr, y_tr)

        clf_hgb = HistGradientBoostingClassifier(
            max_iter=130, max_leaf_nodes=15, min_samples_leaf=350,
            learning_rate=0.055, l2_regularization=45.0, random_state=42
        )
        clf_hgb.fit(X_tr, y_tr)

        clf_lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.015, max_iter=300, random_state=42))
        clf_lr.fit(X_tr, y_tr)

        reg_hgb = HistGradientBoostingRegressor(
            max_iter=130, max_leaf_nodes=15, min_samples_leaf=350,
            learning_rate=0.055, l2_regularization=45.0, random_state=42
        )
        reg_hgb.fit(X_tr, yr_tr)

        p_xgb = clf_xgb.predict_proba(X_te)[:, 1]
        p_hgb = clf_hgb.predict_proba(X_te)[:, 1]
        p_lr = clf_lr.predict_proba(X_te)[:, 1]
        p_ens = 0.40 * p_xgb + 0.35 * p_hgb + 0.25 * p_lr
        exp_margin = reg_hgb.predict(X_te)

        calib = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
        p_tr_raw = 0.40 * clf_xgb.predict_proba(X_tr)[:, 1] + 0.35 * clf_hgb.predict_proba(X_tr)[:, 1] + 0.25 * clf_lr.predict_proba(X_tr)[:, 1]
        calib.fit(p_tr_raw, y_tr)
        p_cal = calib.predict(p_ens)

        auc = roc_auc_score(y_te, p_cal)
        brier = brier_score_loss(y_te, p_cal)

        # Operational bets at calibrated green threshold
        bet_home = (p_cal >= 0.63) & (p_xgb > 0.5) & (p_hgb > 0.5) & (exp_margin >= 1.6)
        bet_away = (p_cal <= 0.37) & (p_xgb < 0.5) & (p_hgb < 0.5) & (exp_margin <= -1.6)
        n_bets = int(np.sum(bet_home) + np.sum(bet_away))
        wins = int(np.sum(y_te[bet_home] == 1) + np.sum(y_te[bet_away] == 0))
        wr = (wins / n_bets * 100.0) if n_bets > 0 else 0.0

        wf_results.append({
            "cut": cut_name,
            "n_test": len(y_te),
            "auc": float(auc),
            "brier": float(brier),
            "n_bets": n_bets,
            "wr": wr
        })
        print(f"  {cut_name:<42} -> N_Test={len(y_te):>6,} | AUC={auc:.4f} | Brier={brier:.4f} | Bets={n_bets:>5,} | WR={wr:.2f}%")

    # 2. Final Production Training (Split 82% / 18%)
    print("\n--- ENTRENAMIENTO FINAL DE PRODUCCIÓN ---")
    n_total = len(y_class)
    split_idx = int(n_total * 0.82)
    X_tr, y_tr, yr_tr = X[:split_idx], y_class[:split_idx], y_reg[:split_idx]
    X_te, y_te, yr_te = X[split_idx:], y_class[split_idx:], y_reg[split_idx:]

    print(f"Split Final: Train={len(y_tr):,} | Test Reciente={len(y_te):,}")

    xgb_prod = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.055,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=50.0,
        random_state=42, n_jobs=-1, eval_metric="logloss"
    )
    xgb_prod.fit(X_tr, y_tr)

    hgb_prod = HistGradientBoostingClassifier(
        max_iter=130, max_leaf_nodes=15, min_samples_leaf=350,
        learning_rate=0.055, l2_regularization=45.0, random_state=42
    )
    hgb_prod.fit(X_tr, y_tr)

    lr_prod = make_pipeline(StandardScaler(), LogisticRegression(C=0.015, max_iter=300, random_state=42))
    lr_prod.fit(X_tr, y_tr)

    reg_prod = HistGradientBoostingRegressor(
        max_iter=130, max_leaf_nodes=15, min_samples_leaf=350,
        learning_rate=0.055, l2_regularization=45.0, random_state=42
    )
    reg_prod.fit(X_tr, yr_tr)

    p_tr_raw = 0.40 * xgb_prod.predict_proba(X_tr)[:, 1] + 0.35 * hgb_prod.predict_proba(X_tr)[:, 1] + 0.25 * lr_prod.predict_proba(X_tr)[:, 1]
    calib_prod = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    calib_prod.fit(p_tr_raw, y_tr)

    p_te_raw = 0.40 * xgb_prod.predict_proba(X_te)[:, 1] + 0.35 * hgb_prod.predict_proba(X_te)[:, 1] + 0.25 * lr_prod.predict_proba(X_te)[:, 1]
    p_te_cal = calib_prod.predict(p_te_raw)
    p_tr_cal = calib_prod.predict(p_tr_raw)

    train_auc = roc_auc_score(y_tr, p_tr_cal)
    test_auc = roc_auc_score(y_te, p_te_cal)
    train_brier = brier_score_loss(y_tr, p_tr_cal)
    test_brier = brier_score_loss(y_te, p_te_cal)

    print(f"\nResultados del Modelo Podado Limpio:")
    print(f"  Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f} | Gap: {train_auc - test_auc:+.4f}")
    print(f"  Train Brier: {train_brier:.4f} | Test Brier: {test_brier:.4f}")

    pred_exp_margin = reg_prod.predict(X_te)
    b_home = (p_te_cal >= 0.63) & (xgb_prod.predict_proba(X_te)[:, 1] > 0.5) & (pred_exp_margin >= 1.6)
    b_away = (p_te_cal <= 0.37) & (xgb_prod.predict_proba(X_te)[:, 1] < 0.5) & (pred_exp_margin <= -1.6)
    tot_bets = int(np.sum(b_home) + np.sum(b_away))
    tot_wins = int(np.sum(y_te[b_home] == 1) + np.sum(y_te[b_away] == 0))
    overall_wr = (tot_wins / tot_bets * 100.0) if tot_bets > 0 else 0.0
    print(f"  Apuestas Operativas Test: {tot_bets:,} | Win Rate: {overall_wr:.2f}%")

    # Guardar pesos en producción
    print(f"\nGuardando artefactos finales de producción en {PROD_MODEL_DIR}...")
    with open(PROD_MODEL_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_prod, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROD_MODEL_DIR / "hgb_model.pkl", "wb") as f:
        pickle.dump(hgb_prod, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROD_MODEL_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_prod, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROD_MODEL_DIR / "reg_model.pkl", "wb") as f:
        pickle.dump(reg_prod, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(PROD_MODEL_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(calib_prod, f, protocol=pickle.HIGHEST_PROTOCOL)

    # State priors
    clean_team_priors = {}
    for team in set(list(elo_ft.keys()) + list(elo_q4.keys())):
        clean_team_priors[team] = {
            "elo_ft": float(elo_ft[team]),
            "elo_q4": float(elo_q4[team]),
            "elo_q1": float(elo_q1[team]),
            "q4_net_hist": float(fast_mean(team_q_profile[team]["q4_net"], 0.0)),
            "q3_net_hist": float(fast_mean(team_q_profile[team]["q3_net"], 0.0)),
        }

    clean_player_priors = {}
    for p_id, p_dict in player_state.items():
        clean_player_priors[p_id] = {
            "elo_ft": float(p_dict["elo_ft"]),
            "elo_q4": float(p_dict["elo_q4"]),
            "n_matches": int(p_dict["n"]),
        }

    with open(PROD_MODEL_DIR / "v5_production_state.pkl", "wb") as f:
        pickle.dump({
            "team_priors": clean_team_priors,
            "player_priors": clean_player_priors,
            "lc_map": lc_map,
            "updated_at": datetime.now().isoformat()
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("¡Entrenamiento y exportación de pesos completados con éxito!")

if __name__ == "__main__":
    train_and_evaluate()
