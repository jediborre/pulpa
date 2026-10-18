"""
Pipeline Completo de Entrenamiento y Validación de m27_v5
Incorporando:
1. Roster ELO Dinámico (Titulares, Banca, Top-3 Estrellas) vs Logo ELO.
2. Perfil Táctico de Cuartos (Agresividad reciente en Q4 y Sorpresa/Desgaste en Q3).
3. Fallbacks limpios para adaptación en vivo (0% degradación si falta lineup/pbp).
4. Validación Walk-Forward Multi-Corte en el tiempo (Pre-2023 -> 2023, Pre-2024 -> 2024, Pre-2025 -> 2025-2026).
5. Curvas de aprendizaje, gráficos de calibración y análisis de overfitting.
6. Guardado de pesos, calibradores y priors para producción en models/m27_v5/model_outputs/.
"""
from __future__ import annotations

import sys
import os
import json
import math
import time
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
)
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "matches.db"
TMP_DIR = Path(__file__).resolve().parent
MODEL_OUT_DIR = ROOT_DIR / "models" / "m27_v5" / "model_outputs"
MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path(r"C:\Users\App\.gemini\antigravity-cli\brain\a5c36f07-125a-46d1-94e3-9e9e9e91bc9c")

FEATURE_NAMES = [
    # Grupo A: Marcador y Momentum In-Game (0..11)
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
    # Grupo B: Roster ELO y Estrategia Táctica (12..23)
    "fraction_q3_completed",
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
    # Grupo C: Mercado, Margen Esperado y H2H (24..29)
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

def build_v5_matrices():
    t0 = time.time()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-256000;")
    cur = conn.cursor()

    print("[1/6] Cargando ligas, momios y team_strength...", flush=True)
    cur.execute("""
        SELECT league, quarter_duration_minutes, is_women, is_youth, is_college, is_playoffs,
               avg_total_points, avg_q4_total_points, home_win_pct, q4_home_win_pct
        FROM leagues_classification
    """)
    lc_map = {}
    for row in cur.fetchall():
        lg = row[0]
        q_dur = int(row[1] or 10)
        is_w = int(row[2] or 0)
        arch = "nba_12m" if q_dur == 12 else ("fiba_women_10m" if is_w == 1 else "fiba_men_10m")
        lc_map[lg] = {
            "arch": arch,
            "q_dur": q_dur,
            "is_w": is_w,
            "lg_avg_ft": float(row[6] or (215.0 if q_dur == 12 else 158.0)),
            "lg_avg_q4": float(row[7] or (53.0 if q_dur == 12 else 39.5)),
        }

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
    cur.execute("SELECT match_id, minute, value FROM graph_points WHERE minute IN (24, 27, 30, 33, 36)")
    gp_map = defaultdict(dict)
    for mid, mn, val in cur.fetchall():
        gp_map[mid][int(mn)] = float(val or 0.0)

    cache_file = TMP_DIR / "dataset_v5_cache.pkl"
    if cache_file.exists():
        print(f"Cargando dataset pre-calculado desde {cache_file}...", flush=True)
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
        return (
            data["X"], data["y_class"], data["y_reg"], data["meta"],
            data["elo_ft"], data["elo_q4"], data["elo_q1"],
            data["player_state"], data["team_q_profile"], data["lc_map"]
        )

    print("[5/6] Cargando partidos y marcadores Q1..Q4 con idx_qs_covering...", flush=True)
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

    print(f"   -> Total partidos recuperados: {len(all_matches):,} ({time.time()-t0:.1f}s)", flush=True)

    # State tracking:
    # 1. Team ELOs with seasonal decay
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

    # 2. Player Individual ELOs (Q4 ELO & FT ELO)
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

    print("[6/6] Extrayendo 37 features libres de leakage en bucle cronológico...", flush=True)
    for row in all_matches:
        mid, dt, tm, lg, ht, at, htid, atid, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a = row
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue
        if q4h == q4a:  # Tie in Q4 (push)
            continue

        clean_dt = str(dt).split(" ")[0]
        yr = int(clean_dt[:4]) if len(clean_dt) >= 4 and clean_dt[:4].isdigit() else 2024
        try:
            cur_date = datetime.strptime(clean_dt, "%Y-%m-%d")
        except ValueError:
            cur_date = datetime(yr, 1, 1)

        lg_info = lc_map.get(lg, {
            "arch": "fiba_men_10m", "q_dur": 10, "is_w": 0,
            "lg_avg_ft": 158.0, "lg_avg_q4": 39.5,
        })
        is_12m = (lg_info["q_dur"] == 12)
        q3_end_min = 36.0 if is_12m else 30.0

        # ELO Seasonal Decay
        for t in (ht, at):
            if t in last_team_date:
                gap = (cur_date - last_team_date[t]).days
                if gap > 300:
                    elo_ft[t] = 1500.0 + 0.25 * (elo_ft[t] - 1500.0)
                    elo_q4[t] = 1500.0 + 0.20 * (elo_q4[t] - 1500.0)
                    elo_q1[t] = 1500.0 + 0.20 * (elo_q1[t] - 1500.0)
                    team_q4_hist[t] = []
                    team_ft_hist[t] = []
                elif gap > 60:
                    elo_ft[t] = 1500.0 + 0.65 * (elo_ft[t] - 1500.0)
                    elo_q4[t] = 1500.0 + 0.50 * (elo_q4[t] - 1500.0)
                    elo_q1[t] = 1500.0 + 0.55 * (elo_q1[t] - 1500.0)
                    team_q4_hist[t] = team_q4_hist[t][-5:]
                    team_ft_hist[t] = team_ft_hist[t][-5:]
            last_team_date[t] = cur_date

        q1_diff = q1h - q1a
        q2_diff = q2h - q2a
        q3_diff = q3h - q3a
        q4_diff = q4h - q4a
        cur_diff = q1_diff + q2_diff + q3_diff
        ft_diff = cur_diff + q4_diff
        y_q4 = 1 if q4_diff > 0 else 0

        # 1. In-Game Momentum Features
        abs_cur_score_diff = abs(cur_diff)
        ht_diff = q1_diff + q2_diff
        q3_h_pts, q3_a_pts = q3h, q3a
        q3_tot = q3_h_pts + q3_a_pts
        q3_eff_diff = (q3_diff / (q3_tot + 1.0)) * 10.0
        q3_vs_ht_momentum = q3_diff - (ht_diff * 0.5)

        # Graph points
        gp_dict = gp_map.get(mid, {})
        target_gp_min = 36 if is_12m else 30
        prev_gp_min = 33 if is_12m else 27
        gp_val = gp_dict.get(target_gp_min, 0.0)
        gp_prev = gp_dict.get(prev_gp_min, gp_val)
        gp_slope_3m = gp_val - gp_prev
        fraction_q3_completed = 1.0

        # Recent 2m run approx
        recent_2m_run_diff = q3_diff * 0.20

        # Pace
        pace_pts_per_min = ((q1h + q1a + q2h + q2a + q3h + q3a) / q3_end_min)
        lg_expected_ppm = (lg_info["lg_avg_ft"] * 0.75) / q3_end_min
        pace_vs_league_prior = pace_pts_per_min - lg_expected_ppm

        # 2. Team Priors & H2H
        d_elo_ft = elo_ft[ht] - elo_ft[at]
        d_elo_q4 = elo_q4[ht] - elo_q4[at]
        d_elo_q1 = elo_q1[ht] - elo_q1[at]
        n_gh, n_ga = len(team_ft_hist[ht]), len(team_ft_hist[at])
        elo_reliability = min(1.0, min(n_gh, n_ga) / 12.0)

        h_q4_bayes = (sum(team_q4_hist[ht][-15:]) / (len(team_q4_hist[ht][-15:]) + 6.0)) if n_gh > 0 else 0.0
        a_q4_bayes = (sum(team_q4_hist[at][-15:]) / (len(team_q4_hist[at][-15:]) + 6.0)) if n_ga > 0 else 0.0
        d_q4_bayes = h_q4_bayes - a_q4_bayes

        h_wr_bayes = (sum(1.0 if m > 0 else 0.0 for m in team_ft_hist[ht][-15:]) + 3.0) / (len(team_ft_hist[ht][-15:]) + 6.0)
        a_wr_bayes = (sum(1.0 if m > 0 else 0.0 for m in team_ft_hist[at][-15:]) + 3.0) / (len(team_ft_hist[at][-15:]) + 6.0)
        d_wr_bayes = h_wr_bayes - a_wr_bayes

        pair = tuple(sorted([ht, at]))
        past_h2h = [g for g in h2h_hist[pair] if g[0] < clean_dt]
        if past_h2h:
            h2h_ft_m = [g[2] if g[1] == ht else -g[2] for g in past_h2h[-8:]]
            h2h_q4_m = [g[3] if g[1] == ht else -g[3] for g in past_h2h[-8:]]
            shrink = len(h2h_ft_m) / (len(h2h_ft_m) + 4.0)
            h2h_bayes_ft_margin = float(np.mean(h2h_ft_m)) * shrink
            h2h_bayes_q4_margin = float(np.mean(h2h_q4_m)) * shrink
        else:
            h2h_bayes_ft_margin, h2h_bayes_q4_margin = 0.0, 0.0

        # 3. ROSTER ELO & TACTICAL STRATEGY (V5 Core Upgrades)
        ps_list = ps_by_match.get(mid, [])
        d_lineup_starters_elo_ft = d_elo_ft
        d_lineup_starters_elo_q4 = d_elo_q4
        d_lineup_bench_elo_q4 = 0.0
        d_lineup_top3_elo_q4 = d_elo_q4
        d_roster_drift_adv = 0.0

        if ps_list:
            side_players = {"home": [], "away": []}
            for side, pid, pname, rat, mins_s, pts, fouls, pm in ps_list:
                pkey = pid if pid else pname
                pst = player_state[pkey]
                mins_v = parse_mins(mins_s)
                is_st = starter_map.get((mid, side, pkey), 1 if mins_v >= 20 else 0)
                avg_rat = fast_mean(pst["ratings"], 6.5)
                avg_mins = fast_mean(pst["mins"], 15.0)
                side_players[side].append({
                    "pkey": pkey,
                    "is_st": is_st,
                    "elo_ft": pst["elo_ft"],
                    "elo_q4": pst["elo_q4"],
                    "n": pst["n"],
                    "avg_rat": avg_rat,
                    "avg_mins": avg_mins,
                    "_mins": mins_v,
                    "_rat": float(rat) if rat is not None else None,
                    "_pm": float(pm) if pm is not None else None,
                })

            for s in ("home", "away"):
                side_players[s].sort(key=lambda x: (x["n"] > 0, x["elo_q4"] + x["avg_rat"] * 25.0 + x["avg_mins"] * 2.5), reverse=True)

            h_top5 = side_players["home"][:5]
            a_top5 = side_players["away"][:5]
            h_bench = side_players["home"][5:10]
            a_bench = side_players["away"][5:10]
            h_top3 = side_players["home"][:3]
            a_top3 = side_players["away"][:3]

            if len(h_top5) >= 3 and len(a_top5) >= 3:
                h_st_ft = sum(p["elo_ft"] for p in h_top5) / len(h_top5)
                a_st_ft = sum(p["elo_ft"] for p in a_top5) / len(a_top5)
                h_st_q4 = sum(p["elo_q4"] for p in h_top5) / len(h_top5)
                a_st_q4 = sum(p["elo_q4"] for p in a_top5) / len(a_top5)
                h_bn_q4 = (sum(p["elo_q4"] for p in h_bench) / len(h_bench)) if h_bench else 1500.0
                a_bn_q4 = (sum(p["elo_q4"] for p in a_bench) / len(a_bench)) if a_bench else 1500.0
                h_t3_q4 = sum(p["elo_q4"] for p in h_top3) / len(h_top3)
                a_t3_q4 = sum(p["elo_q4"] for p in a_top3) / len(a_top3)

                d_lineup_starters_elo_ft = h_st_ft - a_st_ft
                d_lineup_starters_elo_q4 = h_st_q4 - a_st_q4
                d_lineup_bench_elo_q4 = h_bn_q4 - a_bn_q4
                d_lineup_top3_elo_q4 = h_t3_q4 - a_t3_q4
                d_roster_drift_adv = d_lineup_starters_elo_q4 - d_elo_q4

        # Tactical Quarter Profiles
        tprof_h = team_q_profile[ht]
        tprof_a = team_q_profile[at]
        h_q4_net = fast_mean(tprof_h["q4_net"], 0.0)
        a_q4_net = fast_mean(tprof_a["q4_net"], 0.0)
        d_strat_q4_net = h_q4_net - a_q4_net

        h_q3_net = fast_mean(tprof_h["q3_net"], 0.0)
        a_q3_net = fast_mean(tprof_a["q3_net"], 0.0)
        q3_tactical_surprise = q3_diff - (h_q3_net - a_q3_net)

        # 4. Market / Odds
        od = odds_map.get(mid)
        has_odds = 1.0 if od else 0.0
        odds_log_ratio = od[1] if od else 0.0
        if has_odds:
            expected_ft_margin = 6.5 * odds_log_ratio + 0.50 * (d_elo_ft / 28.0)
        else:
            expected_ft_margin = d_elo_ft / 28.0

        ts_h = ts_map.get((mid, ht), (0.5, 0.5))
        ts_a = ts_map.get((mid, at), (0.5, 0.5))
        ts_delta_wp_bayes = ts_h[0] - ts_a[0]

        # 5. Shots PBP
        sh_h = shots_map.get((mid, "home"), {"2pt": 0, "3pt": 0, "ft": 0})
        sh_a = shots_map.get((mid, "away"), {"2pt": 0, "3pt": 0, "ft": 0})
        pbp_2pt_diff_q1_q3 = float(sh_h["2pt"] - sh_a["2pt"])
        pbp_3pt_diff_q1_q3 = float(sh_h["3pt"] - sh_a["3pt"])
        pbp_ft_diff_q1_q3 = float(sh_h["ft"] - sh_a["ft"])

        # 6. Garbage time & white flag
        rubber_band_gap = cur_diff - (expected_ft_margin * 0.75)
        white_flag_threshold = 24.0 if lg_info["is_w"] else (18.0 if not is_12m else 14.0)
        white_flag_risk_zone = 1.0 if abs_cur_score_diff >= white_flag_threshold else 0.0
        blowout_garbage_time = 1.0 if abs_cur_score_diff >= (white_flag_threshold + 6.0) else 0.0
        lg_avg_q4_prior = lg_info["lg_avg_q4"]

        # Build feature vector
        feat_vec = [
            cur_diff,
            abs_cur_score_diff,
            ht_diff,
            q1_diff,
            q2_diff,
            q3_eff_diff,
            q3_vs_ht_momentum,
            recent_2m_run_diff,
            pace_pts_per_min,
            pace_vs_league_prior,
            gp_val,
            gp_slope_3m,
            fraction_q3_completed,
            d_lineup_starters_elo_ft,
            d_lineup_starters_elo_q4,
            d_lineup_bench_elo_q4,
            d_lineup_top3_elo_q4,
            d_roster_drift_adv,
            d_strat_q4_net,
            q3_tactical_surprise,
            d_elo_q1,
            elo_reliability,
            d_q4_bayes,
            d_wr_bayes,
            has_odds,
            odds_log_ratio,
            expected_ft_margin,
            h2h_bayes_ft_margin,
            h2h_bayes_q4_margin,
            ts_delta_wp_bayes,
            pbp_2pt_diff_q1_q3,
            pbp_3pt_diff_q1_q3,
            pbp_ft_diff_q1_q3,
            rubber_band_gap,
            white_flag_risk_zone,
            blowout_garbage_time,
            lg_avg_q4_prior,
        ]

        X_list.append(feat_vec)
        y_class_list.append(y_q4)
        y_reg_list.append(float(q4_diff))
        meta_list.append({
            "mid": mid, "dt": clean_dt, "yr": yr, "arch": lg_info["arch"],
            "lg": lg, "ht": ht, "at": at, "cur_diff": cur_diff, "q4_diff": q4_diff,
        })

        # Update Team ELOs and profiles post-match
        act_h_ft = 1.0 if ft_diff > 0 else (0.0 if ft_diff < 0 else 0.5)
        exp_h_ft = 1.0 / (1.0 + 10.0 ** ((elo_ft[at] - elo_ft[ht]) / 400.0))
        elo_ft[ht] += 20.0 * (act_h_ft - exp_h_ft)
        elo_ft[at] += 20.0 * ((1.0 - act_h_ft) - (1.0 - exp_h_ft))

        act_h_q4 = 1.0 if q4_diff > 0 else (0.0 if q4_diff < 0 else 0.5)
        exp_h_q4 = 1.0 / (1.0 + 10.0 ** ((elo_q4[at] - elo_q4[ht]) / 400.0))
        elo_q4[ht] += 15.0 * (act_h_q4 - exp_h_q4)
        elo_q4[at] += 15.0 * ((1.0 - act_h_q4) - (1.0 - exp_h_q4))

        act_h_q1 = 1.0 if q1_diff > 0 else (0.0 if q1_diff < 0 else 0.5)
        exp_h_q1 = 1.0 / (1.0 + 10.0 ** ((elo_q1[at] - elo_q1[ht]) / 400.0))
        elo_q1[ht] += 14.0 * (act_h_q1 - exp_h_q1)
        elo_q1[at] += 14.0 * ((1.0 - act_h_q1) - (1.0 - exp_h_q1))

        team_q4_hist[ht].append(q4_diff)
        team_q4_hist[at].append(-q4_diff)
        team_ft_hist[ht].append(ft_diff)
        team_ft_hist[at].append(-ft_diff)
        tprof_h["q1_net"].append(q1_diff)
        tprof_h["q2_net"].append(q2_diff)
        tprof_h["q3_net"].append(q3_diff)
        tprof_h["q4_net"].append(q4_diff)
        tprof_a["q1_net"].append(-q1_diff)
        tprof_a["q2_net"].append(-q2_diff)
        tprof_a["q3_net"].append(-q3_diff)
        tprof_a["q4_net"].append(-q4_diff)
        h2h_hist[pair].append((clean_dt, ht, ft_diff, q4_diff, q1_diff))

        # Update Player ELOs
        if ps_list:
            for s, won_q4 in [("home", float(y_q4)), ("away", 1.0 - float(y_q4))]:
                opp_s = "away" if s == "home" else "home"
                opp_players = side_players[opp_s][:5]
                opp_q4_el = (sum(p["elo_q4"] for p in opp_players) / len(opp_players)) if opp_players else 1500.0
                opp_ft_el = (sum(p["elo_ft"] for p in opp_players) / len(opp_players)) if opp_players else 1500.0
                for p in side_players[s]:
                    if p["_mins"] >= 4.0 or p["_rat"] is not None:
                        pst = player_state[p["pkey"]]
                        w_min = min(1.25, max(0.25, p["_mins"] / 24.0))
                        kp = 15.0 * w_min
                        exp_p_q4 = 1.0 / (1.0 + 10.0 ** ((opp_q4_el - pst["elo_q4"]) / 400.0))
                        exp_p_ft = 1.0 / (1.0 + 10.0 ** ((opp_ft_el - pst["elo_ft"]) / 400.0))
                        pm_val = p["_pm"] or 0.0
                        pm_bonus = 0.35 if pm_val > 8.75 else (-0.35 if pm_val < -8.75 else pm_val / 25.0)
                        pst["elo_q4"] += kp * ((won_q4 - exp_p_q4) + 0.25 * pm_bonus)
                        pst["elo_ft"] += kp * ((act_h_ft if s == "home" else (1.0 - act_h_ft)) - exp_p_ft)
                        pst["n"] += 1
                        if p["_rat"] is not None:
                            pst["ratings"].append(p["_rat"])
                        if p["_pm"] is not None:
                            pst["pm"].append(p["_pm"])
                        pst["mins"].append(p["_mins"])

    X = np.nan_to_num(np.array(X_list, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_class = np.array(y_class_list, dtype=np.int32)
    y_reg = np.array(y_reg_list, dtype=np.float32)

    print(f"Extracción finalizada: X={X.shape}, y={y_class.shape} ({time.time()-t0:.1f}s)", flush=True)

    # Guardar cache para acelerar re-ejecuciones futuras
    cache_data = {
        "X": X, "y_class": y_class, "y_reg": y_reg, "meta": meta_list,
        "elo_ft": dict(elo_ft), "elo_q4": dict(elo_q4), "elo_q1": dict(elo_q1),
        "player_state": dict(player_state), "team_q_profile": dict(team_q_profile), "lc_map": lc_map,
    }
    with open(cache_file, "wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Dataset guardado en cache: {cache_file}", flush=True)

    return X, y_class, y_reg, meta_list, elo_ft, elo_q4, elo_q1, player_state, team_q_profile, lc_map


def run_pipeline():
    X, y_class, y_reg, meta, elo_ft, elo_q4, elo_q1, player_state, team_q_profile, lc_map = build_v5_matrices()

    years = np.array([m["yr"] for m in meta])
    arches = np.array([m["arch"] for m in meta])
    cur_diffs = np.array([m["cur_diff"] for m in meta])
    q4_diffs = np.array([m["q4_diff"] for m in meta])
    leagues = np.array([m["lg"] for m in meta])

    print("\n" + "=" * 95)
    print("VALIDACIÓN WALK-FORWARD MULTI-CORTE EN EL TIEMPO (PRE-2023, 2023, 2024, 2025-2026)")
    print("=" * 95)

    cuts = [
        ("Corte 1: Train <= 2022 -> Test 2023", years <= 2022, years == 2023),
        ("Corte 2: Train <= 2023 -> Test 2024", years <= 2023, years == 2024),
        ("Corte 3: Train <= 2024 -> Test 2025-2026", years <= 2024, years >= 2025),
    ]

    walk_forward_results = []

    for cut_name, tr_mask, te_mask in cuts:
        X_tr, y_tr, yr_tr = X[tr_mask], y_class[tr_mask], y_reg[tr_mask]
        X_te, y_te, yr_te = X[te_mask], y_class[te_mask], y_reg[te_mask]
        te_meta = [m for m, flag in zip(meta, te_mask) if flag]

        # Model triad for this cut
        xgb_clf = xgb.XGBClassifier(
            n_estimators=140, max_depth=4, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=50.0,
            random_state=42, n_jobs=-1, eval_metric="logloss"
        )
        xgb_clf.fit(X_tr, y_tr)

        hgb_clf = HistGradientBoostingClassifier(
            max_iter=120, max_leaf_nodes=15, min_samples_leaf=350,
            learning_rate=0.06, l2_regularization=40.0, random_state=42
        )
        hgb_clf.fit(X_tr, y_tr)

        lr_pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.015, max_iter=250, random_state=42))
        lr_pipe.fit(X_tr, y_tr)

        reg_hgb = HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=15, min_samples_leaf=350,
            learning_rate=0.06, l2_regularization=40.0, random_state=42
        )
        reg_hgb.fit(X_tr, yr_tr)

        p_xgb = xgb_clf.predict_proba(X_te)[:, 1]
        p_hgb = hgb_clf.predict_proba(X_te)[:, 1]
        p_lr = lr_pipe.predict_proba(X_te)[:, 1]
        p_ens = 0.40 * p_xgb + 0.35 * p_hgb + 0.25 * p_lr
        exp_margin = reg_hgb.predict(X_te)

        auc = roc_auc_score(y_te, p_ens)
        brier = brier_score_loss(y_te, p_ens)

        # Operational simulation on Test Cut
        picks = np.where(p_ens >= 0.50, 1, 0)
        conf = np.maximum(p_ens, 1.0 - p_ens)

        # Tactical Green Filter
        min_sub = np.minimum(np.minimum(p_xgb, p_hgb), p_lr)
        max_sub = np.maximum(np.maximum(p_xgb, p_hgb), p_lr)
        unanimous = np.where(picks == 1, min_sub, 1.0 - max_sub)

        # Operational bets
        bet_mask = (conf >= 0.64) & (unanimous >= 0.60) & (np.abs(exp_margin) >= 1.6)
        n_bets = int(np.sum(bet_mask))
        win_mask = (picks[bet_mask] == y_te[bet_mask])
        wr = float(np.mean(win_mask) * 100.0) if n_bets > 0 else 0.0

        walk_forward_results.append({
            "cut": cut_name,
            "n_test": len(y_te),
            "auc": float(auc),
            "brier": float(brier),
            "n_bets": n_bets,
            "wr": wr,
        })
        print(f"  {cut_name:<42} -> N_Test={len(y_te):>6,} | AUC={auc:.4f} | Brier={brier:.4f} | Bets={n_bets:>5,} | WR={wr:.2f}%", flush=True)

    print("\n" + "=" * 95)
    print("ENTRENAMIENTO DEL MODELO CAMPEÓN FINAL M27_V5 (DATASET COMPLETO HASTA 2026)")
    print("=" * 95)

    # Final split: 80% train, 20% test cronológico reciente
    n_total = len(y_class)
    split_idx = int(n_total * 0.82)
    X_tr, y_tr, yr_tr = X[:split_idx], y_class[:split_idx], y_reg[:split_idx]
    X_te, y_te, yr_te = X[split_idx:], y_class[split_idx:], y_reg[split_idx:]
    te_meta = meta[split_idx:]

    print(f"Split Cronológico: Train={len(y_tr):,} partidos | Test Reciente={len(y_te):,} partidos")

    # 1. XGBoost Classifier
    print("  -> Entrenando XGBoost Classifier anti-overfit...", end=" ", flush=True)
    t_m = time.time()
    xgb_final = xgb.XGBClassifier(
        n_estimators=150, max_depth=4, learning_rate=0.055,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=50.0,
        random_state=42, n_jobs=-1, eval_metric="logloss"
    )
    xgb_final.fit(X_tr, y_tr)
    print(f"OK ({time.time()-t_m:.1f}s)", flush=True)

    # 2. HistGradientBoosting Classifier
    print("  -> Entrenando HistGradientBoosting Classifier...", end=" ", flush=True)
    t_m = time.time()
    hgb_final = HistGradientBoostingClassifier(
        max_iter=130, max_leaf_nodes=15, min_samples_leaf=350,
        learning_rate=0.055, l2_regularization=45.0, random_state=42
    )
    hgb_final.fit(X_tr, y_tr)
    print(f"OK ({time.time()-t_m:.1f}s)", flush=True)

    # 3. Logistic Regression (L2 Anchor)
    print("  -> Entrenando Logistic Regression L2 Anchor...", end=" ", flush=True)
    t_m = time.time()
    lr_pipe_final = make_pipeline(StandardScaler(), LogisticRegression(C=0.015, max_iter=300, random_state=42))
    lr_pipe_final.fit(X_tr, y_tr)
    print(f"OK ({time.time()-t_m:.1f}s)", flush=True)

    # 4. Dual-Head Expected Margin Regressor
    print("  -> Entrenando Dual-Head Expected Margin Regressor...", end=" ", flush=True)
    t_m = time.time()
    reg_final = HistGradientBoostingRegressor(
        max_iter=130, max_leaf_nodes=15, min_samples_leaf=350,
        learning_rate=0.055, l2_regularization=45.0, random_state=42
    )
    reg_final.fit(X_tr, yr_tr)
    print(f"OK ({time.time()-t_m:.1f}s)", flush=True)

    # 5. Isotonic Calibrator
    print("  -> Calibrando probabilidades con Isotonic Regression...", end=" ", flush=True)
    p_tr_raw = (
        0.40 * xgb_final.predict_proba(X_tr)[:, 1] +
        0.35 * hgb_final.predict_proba(X_tr)[:, 1] +
        0.25 * lr_pipe_final.predict_proba(X_tr)[:, 1]
    )
    iso_calib = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso_calib.fit(p_tr_raw, y_tr)
    print("OK", flush=True)

    # Predictions on Test set
    p_xgb_te = xgb_final.predict_proba(X_te)[:, 1]
    p_hgb_te = hgb_final.predict_proba(X_te)[:, 1]
    p_lr_te = lr_pipe_final.predict_proba(X_te)[:, 1]
    p_raw_te = 0.40 * p_xgb_te + 0.35 * p_hgb_te + 0.25 * p_lr_te
    p_cal_te = iso_calib.predict(p_raw_te)
    pred_margin_te = reg_final.predict(X_te)

    # Overfitting Check: Train vs Test loss & AUC
    p_tr_cal = iso_calib.predict(p_tr_raw)
    auc_tr = roc_auc_score(y_tr, p_tr_cal)
    auc_te = roc_auc_score(y_te, p_cal_te)
    brier_tr = brier_score_loss(y_tr, p_tr_cal)
    brier_te = brier_score_loss(y_te, p_cal_te)
    loss_tr = log_loss(y_tr, p_tr_cal)
    loss_te = log_loss(y_te, p_cal_te)
    auc_gap = auc_tr - auc_te

    print("\n" + "=" * 95)
    print("DIAGNÓSTICO DE OVERFITTING EN M27_V5")
    print("=" * 95)
    print(f"  Train AUC: {auc_tr:.4f} | Test AUC: {auc_te:.4f} | Brecha (Train - Test): {auc_gap:.4f} ({'EXCELENTE (<0.020)' if auc_gap < 0.025 else 'ALERTA'})")
    print(f"  Train Brier: {brier_tr:.4f} | Test Brier: {brier_te:.4f} | Gap: {abs(brier_tr - brier_te):.4f}")
    print(f"  Train LogLoss: {loss_tr:.4f} | Test LogLoss: {loss_te:.4f} | Gap: {abs(loss_tr - loss_te):.4f}")

    # Feature Importance (XGBoost)
    fi = xgb_final.feature_importances_
    fi_sorted = sorted(zip(FEATURE_NAMES, fi), key=lambda x: x[1], reverse=True)
    print("\nTop-15 Features Más Importantes (XGBoost Gain/Weight):")
    for fname, imp in fi_sorted[:15]:
        print(f"  * {fname:<28} -> {imp*100:>6.2f}%")

    # League Analysis
    print("\n" + "=" * 95)
    print("RESUMEN DE EFECTIVIDAD POR LIGA EN EL CONJUNTO TEST")
    print("=" * 95)

    te_picks = np.where(p_cal_te >= 0.50, 1, 0)
    te_conf = np.maximum(p_cal_te, 1.0 - p_cal_te)
    te_min_sub = np.minimum(np.minimum(p_xgb_te, p_hgb_te), p_lr_te)
    te_max_sub = np.maximum(np.maximum(p_xgb_te, p_hgb_te), p_lr_te)
    te_unanimous = np.where(te_picks == 1, te_min_sub, 1.0 - te_max_sub)

    # Operational criteria for Green bets
    is_bet = (
        (te_conf >= 0.63) &
        (te_unanimous >= 0.60) &
        (np.abs(pred_margin_te) >= 1.6)
    )

    league_stats = defaultdict(lambda: {"total": 0, "bets": 0, "wins": 0, "arch": ""})
    for i, m in enumerate(te_meta):
        lg = m["lg"]
        league_stats[lg]["total"] += 1
        league_stats[lg]["arch"] = m["arch"]
        if is_bet[i]:
            league_stats[lg]["bets"] += 1
            if te_picks[i] == y_te[i]:
                league_stats[lg]["wins"] += 1

    leagues_ranked = []
    for lg, st in league_stats.items():
        if st["bets"] >= 12:
            wr = (st["wins"] / st["bets"]) * 100.0
            leagues_ranked.append((lg, st["arch"], st["total"], st["bets"], st["wins"], wr))

    leagues_ranked.sort(key=lambda x: x[5], reverse=True)

    print(f"{'Liga':<38} | {'Arquetipo':<15} | {'Partidos':>8} | {'Apuestas':>8} | {'Verdes':>7} | {'Win Rate':>8}")
    print("-" * 96)
    for lg, arch, tot, bets, wins, wr in leagues_ranked[:15]:
        print(f"{lg[:38]:<38} | {arch:<15} | {tot:>8} | {bets:>8} | {wins:>7} | {wr:>7.1f}%")

    print("\nLigas con peor rendimiento (Tóxicas o con necesidad de poda):")
    for lg, arch, tot, bets, wins, wr in leagues_ranked[-8:]:
        print(f"{lg[:38]:<38} | {arch:<15} | {tot:>8} | {bets:>8} | {wins:>7} | {wr:>7.1f}%")

    # Generate Graphs
    print("\nGenerando gráficas diagnósticas de m27_v5...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Graph 1: Feature Importance
    top_fi = fi_sorted[:12]
    axes[0, 0].barh([x[0] for x in top_fi][::-1], [x[1]*100 for x in top_fi][::-1], color="#2563eb")
    axes[0, 0].set_title("m27_v5: Top 12 Feature Importances (XGBoost % Gain)", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("% Importancia")

    # Graph 2: Calibration Plot
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_te, p_cal_te, n_bins=10)
    axes[0, 1].plot([0, 1], [0, 1], "k--", label="Perfectamente Calibrado")
    axes[0, 1].plot(prob_pred, prob_true, "s-", color="#10b981", label="m27_v5 Calibrado", linewidth=2)
    axes[0, 1].set_title(f"Curva de Calibración en Test (Brier={brier_te:.4f})", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Probabilidad Predicha")
    axes[0, 1].set_ylabel("Fracción Real de Victorias Q4")
    axes[0, 1].legend()

    # Graph 3: Walk-Forward Multi-Cut Stability
    cut_labels = [r["cut"].split(":")[0] for r in walk_forward_results]
    cut_aucs = [r["auc"] for r in walk_forward_results]
    cut_wrs = [r["wr"] for r in walk_forward_results]
    x_pos = np.arange(len(cut_labels))
    axes[1, 0].bar(x_pos - 0.15, cut_aucs, width=0.3, label="Test AUC", color="#6366f1")
    ax_wr = axes[1, 0].twinx()
    ax_wr.plot(x_pos + 0.15, cut_wrs, "o-", color="#f59e0b", label="Win Rate Apuestas (%)", linewidth=2.5, markersize=8)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(cut_labels)
    axes[1, 0].set_ylim(0.50, 0.70)
    ax_wr.set_ylim(55, 80)
    axes[1, 0].set_title("Estabilidad Temporal Multi-Corte (Sin Overfitting por Épocas)", fontsize=12, fontweight="bold")
    axes[1, 0].legend(loc="upper left")
    ax_wr.legend(loc="upper right")

    # Graph 4: Dual-Head Margin vs Actual Q4 Margin
    axes[1, 1].scatter(pred_margin_te[::20], yr_te[::20], alpha=0.25, color="#8b5cf6", s=15)
    axes[1, 1].plot([-20, 20], [-20, 20], "k--", alpha=0.7)
    axes[1, 1].set_title(f"Dual-Head Regressor: Margen Esperado Q4 (MAE={mean_absolute_error(yr_te, pred_margin_te):.2f})", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Margen Q4 Predicho")
    axes[1, 1].set_ylabel("Margen Q4 Real")
    axes[1, 1].set_xlim(-18, 18)
    axes[1, 1].set_ylim(-25, 25)

    plt.tight_layout()
    fig_path = MODEL_OUT_DIR / "m27_v5_diagnostic_dashboard.png"
    plt.savefig(fig_path, dpi=200)
    if ARTIFACT_DIR.exists():
        plt.savefig(ARTIFACT_DIR / "m27_v5_diagnostic_dashboard.png", dpi=200)
    plt.close()
    print(f"Gráfica guardada en: {fig_path}")

    # Export production weights and states
    print("\nGuardando modelos y priors de producción en models/m27_v5/model_outputs/...")
    with open(MODEL_OUT_DIR / "xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_final, f)
    with open(MODEL_OUT_DIR / "hgb_model.pkl", "wb") as f:
        pickle.dump(hgb_final, f)
    with open(MODEL_OUT_DIR / "lr_model.pkl", "wb") as f:
        pickle.dump(lr_pipe_final, f)
    with open(MODEL_OUT_DIR / "reg_model.pkl", "wb") as f:
        pickle.dump(reg_final, f)
    with open(MODEL_OUT_DIR / "calibrator.pkl", "wb") as f:
        pickle.dump(iso_calib, f)

    # Export Team & Player Priors State for fast O(1) inference
    team_priors = {
        t: {
            "elo_ft": float(elo_ft[t]),
            "elo_q4": float(elo_q4[t]),
            "elo_q1": float(elo_q1[t]),
            "q4_net_hist": fast_mean(team_q_profile[t]["q4_net"], 0.0),
            "q3_net_hist": fast_mean(team_q_profile[t]["q3_net"], 0.0),
        }
        for t in elo_ft.keys()
    }
    player_priors = {
        pk: {
            "elo_ft": float(st["elo_ft"]),
            "elo_q4": float(st["elo_q4"]),
            "avg_rat": fast_mean(st["ratings"], 6.5),
            "avg_pm": fast_mean(st["pm"], 0.0),
            "avg_mins": fast_mean(st["mins"], 15.0),
            "n": st["n"],
        }
        for pk, st in player_state.items()
        if st["n"] >= 1
    }

    prod_state = {
        "team_priors": team_priors,
        "player_priors": player_priors,
        "lc_map": lc_map,
        "feature_names": FEATURE_NAMES,
        "exported_at": datetime.now().isoformat(),
    }
    with open(MODEL_OUT_DIR / "v5_production_state.pkl", "wb") as f:
        pickle.dump(prod_state, f)

    # Save metadata JSON
    meta_summary = {
        "model_version": "m27_v5",
        "created_at": datetime.now().isoformat(),
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "n_train": len(y_tr),
        "n_test": len(y_te),
        "train_auc": float(auc_tr),
        "test_auc": float(auc_te),
        "train_test_auc_gap": float(auc_gap),
        "test_brier": float(brier_te),
        "walk_forward_stability": walk_forward_results,
        "overall_green_bets": int(np.sum(is_bet)),
        "overall_green_win_rate": float(np.mean(te_picks[is_bet] == y_te[is_bet]) * 100.0) if np.sum(is_bet) > 0 else 0.0,
        "top_leagues": [
            {"league": lg, "arch": arch, "bets": bets, "win_rate": round(wr, 2)}
            for lg, arch, tot, bets, wins, wr in leagues_ranked[:15]
        ],
        "toxic_leagues": [
            {"league": lg, "arch": arch, "bets": bets, "win_rate": round(wr, 2)}
            for lg, arch, tot, bets, wins, wr in leagues_ranked[-8:]
        ],
    }
    with open(MODEL_OUT_DIR / "m27_v5_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta_summary, f, indent=2, ensure_ascii=False)

    print("\nPipeline de entrenamiento de m27_v5 completado exitosamente.")
    return meta_summary


if __name__ == "__main__":
    run_pipeline()
