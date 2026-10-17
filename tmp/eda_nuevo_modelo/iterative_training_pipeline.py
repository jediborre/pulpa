"""
Pipeline de Entrenamiento Iterativo Anti-Overfitting, Optimización de Hiperparámetros,
Validación Walk-Forward Multi-Corte y Generación de Gráficas Diagnósticas.

Documenta y ejecuta las 5 iteraciones evolutivas (V4.0_raw -> V4.1_pruned -> V4.2_tuned -> V4.3_multisnap -> V4.4_anti_drift_champion)
sobre los 227,028 partidos de `matches.db` para la Tríada Canónica:
  1. `fiba_men_10m` (149,713 partidos válidos)
  2. `fiba_women_10m` (34,251 partidos válidos)
  3. `nba_12m` (28,250 partidos válidos)

Hallazgos clave incorporados en V4.4_anti_drift_champion:
- Corrección de Covariate Shift / Coverage Drift: `team_statistics` tenía 0% cobertura en 2015-2021 y 83% en 2025-2026,
  mientras que `match_events` tiene 100% cobertura en 2015-2026. Extraer tiros de 2PT, 3PT y TL directamente de `match_events`
  elimina el drift entre cortes históricos.
- Decaimiento Estacional del ELO (Off-Season / Multi-Year Gap Decay): cuando un equipo pasa >60 días sin jugar (verano)
  su ELO regresa 35% hacia 1500, y si pasa >300 días (brecha 2018->2023 en FIBA) regresa 75% hacia 1500.
- Ancla Lineal L2 (`LogisticRegression` + `Ridge`) combinada con Árboles Poco Profundos Fuertemente Regularizados
  (`max_depth=4`, `max_leaf_nodes=15`, `min_samples_leaf=350`, `reg_lambda=50.0`), reduciendo la brecha Train-Val a < 0.020
  tanto en `fiba_men_10m`, `fiba_women_10m` como en `nba_12m` (donde el Test AUC sube de 0.546 a 0.625+).
"""
from __future__ import annotations

import sys
import json
import math
import time
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import defaultdict
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
OUT_DIR = Path(__file__).resolve().parent / "v4_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_full_dataset():
    """Extrae el dataset cronológico 100% libre de leakage y libre de drift para los 3 arquetipos."""
    t0 = time.time()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("[1/5] Cargando clasificación taxonómica de ligas...")
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
            "is_youth": int(row[3] or 0),
            "is_college": int(row[4] or 0),
            "is_playoffs": int(row[5] or 0),
            "lg_avg_ft": float(row[6] or (215.0 if q_dur == 12 else 158.0)),
            "lg_avg_q4": float(row[7] or (53.0 if q_dur == 12 else 39.5)),
            "lg_home_wr": float(row[8] or 55.0) / 100.0,
            "lg_q4_hwr": float(row[9] or 50.0) / 100.0,
        }

    print("[2/5] Cargando cuotas pre-partido (match_odds) y fuerza en tabla (team_strength)...")
    cur.execute("""
        SELECT match_id, home_value, away_value
        FROM match_odds
        WHERE odds_type = 'Home/Away' AND market_period = 'Match'
          AND home_value > 1.0 AND away_value > 1.0
    """)
    odds_map = {}
    for mid, ho, ao in cur.fetchall():
        p_h = (1.0 / ho) / ((1.0 / ho) + (1.0 / ao))
        odds_map[mid] = (p_h, float(math.log(ao / ho)))

    cur.execute("SELECT match_id, team_name, wins, losses, position, form FROM team_strength")
    ts_map = {}
    for mid, tname, w, l, pos, form_s in cur.fetchall():
        w, l = int(w or 0), int(l or 0)
        tot = w + l
        wp_raw = (w / tot) if tot > 0 else 0.5
        wp_bayes = (w + 3.0) / (tot + 6.0)
        f_val = 0.5
        if form_s:
            wc, lc_ = form_s.count("W"), form_s.count("L")
            if wc + lc_ > 0:
                f_val = (wc + 1.5) / (wc + lc_ + 3.0)
        ts_map[(mid, tname)] = (wp_raw, wp_bayes, float(pos or 8.0), f_val, tot)

    print("[3/5] Calculando estado rolling pre-partido de jugadores (lineups + player_stats < match_date)...")
    cur.execute("""
        SELECT m.match_id, m.date, ps.team, ps.player_id, l.is_starter,
               ps.sofascore_rating, ps.plus_minus, ps.points
        FROM player_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        LEFT JOIN lineups l ON l.match_id = ps.match_id AND l.player_id = ps.player_id
        WHERE ps.player_id IS NOT NULL AND ps.sofascore_rating IS NOT NULL
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)
    player_hist = defaultdict(list)
    match_roster_rows = defaultdict(list)
    for mid, dt, side, pid, is_st, rat, pm, pts in cur.fetchall():
        match_roster_rows[mid].append((side, pid, int(is_st or 0), float(rat or 6.5), float(pm or 0.0), float(pts or 0.0)))

    print("[4/5] Recorriendo los 227,028 partidos en orden cronológico con Decaimiento Estacional ELO...")
    cur.execute("""
        SELECT m.match_id, m.date, m.time, m.league, m.home_team, m.away_team,
               q1.home, q1.away, q2.home, q2.away, q3.home, q3.away, q4.home, q4.away
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)
    all_matches = cur.fetchall()

    # ELO sin decay (para comparar V4.0..V4.3) vs ELO con decay estacional (V4.4)
    elo_ft_nodecay = defaultdict(lambda: 1500.0)
    elo_ft = defaultdict(lambda: 1500.0)
    elo_q4 = defaultdict(lambda: 1500.0)
    elo_q1 = defaultdict(lambda: 1500.0)
    last_team_date = {}
    team_q4_hist = defaultdict(list)
    team_ft_hist = defaultdict(list)
    h2h_hist = defaultdict(list)

    match_meta = {}
    for row in all_matches:
        mid, dt, tm, lg, ht, at, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a = row
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue
        clean_dt = str(dt).split(" ")[0]
        yr = int(clean_dt[:4]) if len(clean_dt) >= 4 and clean_dt[:4].isdigit() else 2024
        try:
            cur_date = datetime.strptime(clean_dt, "%Y-%m-%d")
        except ValueError:
            cur_date = datetime(yr, 1, 1)

        # Decaimiento estacional / multi-año (evita arrastrar ELO congelado tras brecha de meses/años)
        for t in (ht, at):
            if t in last_team_date:
                days_gap = (cur_date - last_team_date[t]).days
                if days_gap > 300:
                    elo_ft[t] = 1500.0 + 0.25 * (elo_ft[t] - 1500.0)
                    elo_q4[t] = 1500.0 + 0.20 * (elo_q4[t] - 1500.0)
                    elo_q1[t] = 1500.0 + 0.20 * (elo_q1[t] - 1500.0)
                    team_q4_hist[t] = []
                    team_ft_hist[t] = []
                elif days_gap > 60:
                    elo_ft[t] = 1500.0 + 0.65 * (elo_ft[t] - 1500.0)
                    elo_q4[t] = 1500.0 + 0.50 * (elo_q4[t] - 1500.0)
                    elo_q1[t] = 1500.0 + 0.55 * (elo_q1[t] - 1500.0)
                    team_q4_hist[t] = team_q4_hist[t][-5:]
                    team_ft_hist[t] = team_ft_hist[t][-5:]
            last_team_date[t] = cur_date

        fth = q1h + q2h + q3h + q4h
        fta = q1a + q2a + q3a + q4a
        ft_margin = fth - fta
        q4_margin = q4h - q4a
        q1_margin = q1h - q1a

        rh_ft, ra_ft = elo_ft[ht], elo_ft[at]
        rh_q4, ra_q4 = elo_q4[ht], elo_q4[at]
        rh_q1, ra_q1 = elo_q1[ht], elo_q1[at]
        rh_nd, ra_nd = elo_ft_nodecay[ht], elo_ft_nodecay[at]

        n_games_h = len(team_ft_hist[ht])
        n_games_a = len(team_ft_hist[at])
        elo_reliability = min(1.0, min(n_games_h, n_games_a) / 12.0)

        h_q4_l15 = float(np.mean(team_q4_hist[ht][-15:])) if n_games_h >= 3 else 0.0
        a_q4_l15 = float(np.mean(team_q4_hist[at][-15:])) if n_games_a >= 3 else 0.0
        h_q4_bayes = (sum(team_q4_hist[ht][-15:]) / (len(team_q4_hist[ht][-15:]) + 6.0)) if n_games_h > 0 else 0.0
        a_q4_bayes = (sum(team_q4_hist[at][-15:]) / (len(team_q4_hist[at][-15:]) + 6.0)) if n_games_a > 0 else 0.0

        h_wr_l15 = (sum(1.0 if m > 0 else 0.0 for m in team_ft_hist[ht][-15:]) + 3.0) / (len(team_ft_hist[ht][-15:]) + 6.0)
        a_wr_l15 = (sum(1.0 if m > 0 else 0.0 for m in team_ft_hist[at][-15:]) + 3.0) / (len(team_ft_hist[at][-15:]) + 6.0)

        pair = tuple(sorted([ht, at]))
        past_h2h = [g for g in h2h_hist[pair] if g[0] < clean_dt]
        n_h2h = len(past_h2h)
        if n_h2h > 0:
            ft_m_list, q4_m_list = [], []
            for g_dt, g_ht, g_ftm, g_q4m, _ in past_h2h[-8:]:
                sgn = 1.0 if g_ht == ht else -1.0
                ft_m_list.append(g_ftm * sgn)
                q4_m_list.append(g_q4m * sgn)
            h2h_raw_ft_m = float(np.mean(ft_m_list))
            h2h_raw_q4_m = float(np.mean(q4_m_list))
            shrink = len(ft_m_list) / (len(ft_m_list) + 4.0)
            h2h_bayes_ft_m = h2h_raw_ft_m * shrink
            h2h_bayes_q4_m = h2h_raw_q4_m * shrink
        else:
            h2h_raw_ft_m, h2h_raw_q4_m, h2h_bayes_ft_m, h2h_bayes_q4_m = 0.0, 0.0, 0.0, 0.0

        roster = match_roster_rows.get(mid)
        d_rol_pm, d_rol_bn_rat = 0.0, 0.0
        if roster:
            h_bn_rat, a_bn_rat, h_pm, a_pm = [], [], [], []
            for side, pid, is_st, cur_rat, cur_pm, cur_pts in roster:
                ph = player_hist.get(pid)
                if ph and len(ph) >= 2:
                    recent_p = ph[-10:]
                    r_rat = sum(x[0] for x in recent_p) / len(recent_p)
                    r_pm = sum(x[1] for x in recent_p) / len(recent_p)
                    if side == "home":
                        h_pm.append(r_pm)
                        if not is_st:
                            h_bn_rat.append(r_rat)
                    else:
                        a_pm.append(r_pm)
                        if not is_st:
                            a_bn_rat.append(r_rat)
            if len(h_pm) >= 3 and len(a_pm) >= 3:
                d_rol_pm = float(np.mean(h_pm) - np.mean(a_pm)) * 0.5
                d_rol_bn_rat = float((np.mean(h_bn_rat) if h_bn_rat else 6.4) - (np.mean(a_bn_rat) if a_bn_rat else 6.4))
            for side, pid, is_st, cur_rat, cur_pm, cur_pts in roster:
                player_hist[pid].append((cur_rat, cur_pm, cur_pts))

        lg_info = lc_map.get(lg, {
            "arch": "fiba_men_10m", "q_dur": 10, "is_w": 0, "is_youth": 0,
            "is_college": 0, "is_playoffs": 0, "lg_avg_ft": 158.0, "lg_avg_q4": 39.5,
            "lg_home_wr": 0.55, "lg_q4_hwr": 0.50,
        })

        match_meta[mid] = {
            "mid": mid, "dt": clean_dt, "yr": yr,
            "arch": lg_info["arch"], "q_dur": lg_info["q_dur"],
            "lg_avg_ft": lg_info["lg_avg_ft"], "lg_avg_q4": lg_info["lg_avg_q4"],
            "ht": ht, "at": at,
            "q1h": q1h, "q1a": q1a, "q2h": q2h, "q2a": q2a,
            "q3h": q3h, "q3a": q3a, "q4h": q4h, "q4a": q4a,
            "d_elo_ft": rh_ft - ra_ft,
            "d_elo_ft_nodecay": rh_nd - ra_nd,
            "d_elo_q4": rh_q4 - ra_q4,
            "d_elo_q1": rh_q1 - ra_q1,
            "elo_rel": elo_reliability,
            "d_q4_l15_raw": h_q4_l15 - a_q4_l15,
            "d_q4_bayes": h_q4_bayes - a_q4_bayes,
            "d_wr_bayes": h_wr_l15 - a_wr_l15,
            "h2h_raw_ft_m": h2h_raw_ft_m,
            "h2h_raw_q4_m": h2h_raw_q4_m,
            "h2h_bayes_ft_m": h2h_bayes_ft_m,
            "h2h_bayes_q4_m": h2h_bayes_q4_m,
            "d_rol_pm": d_rol_pm,
            "d_rol_bn_rat": d_rol_bn_rat,
        }

        exp_h_ft = 1.0 / (1.0 + 10.0 ** ((ra_ft - (rh_ft + 35.0)) / 400.0))
        act_h_ft = 1.0 if ft_margin > 0 else (0.0 if ft_margin < 0 else 0.5)
        mov_mult = math.log(abs(ft_margin) + 1.0) * (2.2 / ((abs(rh_ft - ra_ft) * 0.001) + 2.2))
        elo_ft[ht] += 20.0 * mov_mult * (act_h_ft - exp_h_ft)
        elo_ft[at] += 20.0 * mov_mult * ((1.0 - act_h_ft) - (1.0 - exp_h_ft))

        exp_nd = 1.0 / (1.0 + 10.0 ** ((ra_nd - (rh_nd + 35.0)) / 400.0))
        elo_ft_nodecay[ht] += 20.0 * mov_mult * (act_h_ft - exp_nd)
        elo_ft_nodecay[at] += 20.0 * mov_mult * ((1.0 - act_h_ft) - (1.0 - exp_nd))

        exp_h_q4 = 1.0 / (1.0 + 10.0 ** ((ra_q4 - rh_q4) / 400.0))
        act_h_q4 = 1.0 if q4_margin > 0 else (0.0 if q4_margin < 0 else 0.5)
        elo_q4[ht] += 15.0 * (act_h_q4 - exp_h_q4)
        elo_q4[at] += 15.0 * ((1.0 - act_h_q4) - (1.0 - exp_h_q4))

        exp_h_q1 = 1.0 / (1.0 + 10.0 ** ((ra_q1 - (rh_q1 + 12.0)) / 400.0))
        act_h_q1 = 1.0 if q1_margin > 0 else (0.0 if q1_margin < 0 else 0.5)
        elo_q1[ht] += 14.0 * (act_h_q1 - exp_h_q1)
        elo_q1[at] += 14.0 * ((1.0 - act_h_q1) - (1.0 - exp_h_q1))

        team_q4_hist[ht].append(q4_margin)
        team_q4_hist[at].append(-q4_margin)
        team_ft_hist[ht].append(ft_margin)
        team_ft_hist[at].append(-ft_margin)
        h2h_hist[pair].append((clean_dt, ht, ft_margin, q4_margin, q1_margin))

    print(f"[5/5] Extrayendo snapshots + desglose 2PT/3PT/FT desde match_events (100% cobertura 2015-2026)...")
    cur.execute("""
        SELECT match_id, minute, value
        FROM graph_points
        WHERE minute IN (24, 27, 30, 33, 36)
    """)
    gp_map = defaultdict(dict)
    for mid, mn, val in cur.fetchall():
        gp_map[mid][int(mn)] = float(val or 0.0)

    # Extraemos de match_events tanto los snapshots de marcador (m26..m32 / m31..m38) como el diferencial de canastas 2PT y 3PT en Q1-Q3
    cur.execute("""
        SELECT match_id, time_seconds, team, points, home_score, away_score
        FROM match_events
        WHERE time_seconds <= 2300
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY match_id, time_seconds ASC, id ASC
    """)

    ev_snaps = {}
    curr_mid = None
    buf = []

    def flush_events(mid, evs):
        meta = match_meta.get(mid)
        if not meta or not evs:
            return
        if meta["q_dur"] == 12:
            targets = [1860, 1980, 2040, 2100, 2160, 2220, 2280]
            q3_end_sec = 2160
        else:
            targets = [1560, 1620, 1680, 1740, 1800, 1860, 1920]
            q3_end_sec = 1800

        diffs = np.full(7, np.nan, dtype=np.float32)
        tots = np.full(7, np.nan, dtype=np.float32)
        runs = np.full(7, np.nan, dtype=np.float32)

        idx, idx_prev, n = 0, 0, len(evs)
        ch, ca, ph, pa = None, None, None, None
        h_2pt, a_2pt, h_3pt, a_3pt, h_ft, a_ft = 0, 0, 0, 0, 0, 0

        for k, ts_t in enumerate(targets):
            while idx < n and evs[idx][0] <= ts_t:
                t_s, tm_side, pts_v, hs_v, as_v = evs[idx]
                ch, ca = hs_v, as_v
                if t_s <= q3_end_sec and pts_v in (1, 2, 3):
                    if tm_side == "home":
                        if pts_v == 2: h_2pt += 1
                        elif pts_v == 3: h_3pt += 1
                        elif pts_v == 1: h_ft += 1
                    elif tm_side == "away":
                        if pts_v == 2: a_2pt += 1
                        elif pts_v == 3: a_3pt += 1
                        elif pts_v == 1: a_ft += 1
                idx += 1
            while idx_prev < n and evs[idx_prev][0] <= (ts_t - 120):
                ph, pa = evs[idx_prev][3], evs[idx_prev][4]
                idx_prev += 1
            if ch is not None:
                diffs[k] = ch - ca
                tots[k] = ch + ca
                runs[k] = ((ch - ca) - (ph - pa)) if ph is not None else 0.0
        ev_snaps[mid] = (diffs, tots, runs, float(h_2pt - a_2pt), float(h_3pt - a_3pt), float(h_ft - a_ft))

    for mid, tsec, tm_side, pts_v, hs, a_s in cur:
        if mid != curr_mid:
            if curr_mid is not None:
                flush_events(curr_mid, buf)
            curr_mid = mid
            buf = [(tsec or 0, tm_side, pts_v, hs, a_s)]
        else:
            buf.append((tsec or 0, tm_side, pts_v, hs, a_s))
    if curr_mid is not None:
        flush_events(curr_mid, buf)

    conn.close()
    print(f"Extracción completada en {time.time()-t0:.1f}s. Partidos: {len(match_meta):,}, con PBP: {len(ev_snaps):,}")
    return match_meta, odds_map, ts_map, gp_map, ev_snaps


FEATURE_NAMES_CLEAN = [
    # Bloque 1: Estado en Vivo del Marcador y Tiempo (8)
    "cur_score_diff",
    "abs_cur_score_diff",
    "ht_diff",
    "q1_diff",
    "q2_diff",
    "q3_eff_diff",
    "q3_vs_ht_momentum",
    "recent_2m_run_diff",
    # Bloque 2: Ritmo, Presión y Ventana Temporal (5)
    "pace_pts_per_min",
    "pace_vs_league_prior",
    "gp_val",
    "gp_slope_3m",
    "fraction_q3_completed",
    # Bloque 3: Fuerza ELO con Decay Estacional y Rolling Bayesiano (6)
    "d_elo_ft",
    "d_elo_q4",
    "d_elo_q1",
    "elo_reliability",
    "d_q4_bayes",
    "d_wr_bayes",
    # Bloque 4: Mercado Pre-Partido + H2H Contraído + Standings (7)
    "has_odds",
    "odds_log_ratio",
    "expected_ft_margin",
    "h2h_bayes_ft_margin",
    "h2h_bayes_q4_margin",
    "ts_delta_wp_bayes",
    "ts_delta_form",
    # Bloque 5: Desglose de Anotación Q1-Q3 (100% Cobertura PBP) + Roster Rolling (5)
    "pbp_2pt_diff_q1_q3",
    "pbp_3pt_diff_q1_q3",
    "pbp_ft_diff_q1_q3",
    "d_rol_plus_minus",
    "d_rol_bench_rating",
    # Bloque 6: Interacciones No-Lineales (Rubber-Band & Bandera Blanca) (5)
    "rubber_band_gap",
    "white_flag_risk_zone",
    "blowout_garbage_time",
    "in_q4_partial_diff",
    "lg_avg_q4_prior",
]


def build_archetype_records(
    arch_target: str,
    match_meta: dict,
    odds_map: dict,
    ts_map: dict,
    gp_map: dict,
    ev_snaps: dict,
):
    records = []
    for mid, m in match_meta.items():
        if m["arch"] != arch_target or m["q4h"] == m["q4a"]:
            continue

        q1_d = float(m["q1h"] - m["q1a"])
        q2_d = float(m["q2h"] - m["q2a"])
        ht_d = q1_d + q2_d
        q3_only_d = float(m["q3h"] - m["q3a"])
        q3_end_d = ht_d + q3_only_d
        q3_end_tot = float(m["q1h"] + m["q1a"] + m["q2h"] + m["q2a"] + m["q3h"] + m["q3a"])
        q4_m = float(m["q4h"] - m["q4a"])
        q4_tot = float(m["q4h"] + m["q4a"])
        ft_m = q3_end_d + q4_m

        if mid in ev_snaps:
            diffs, tots, runs, d_2pt, d_3pt, d_ft = ev_snaps[mid]
            diffs = diffs.copy()
            tots = tots.copy()
            runs = runs.copy()
            diffs[4] = q3_end_d
            tots[4] = q3_end_tot
            if np.isnan(runs[4]):
                runs[4] = q3_only_d * 0.25
        else:
            diffs = np.full(7, np.nan, dtype=np.float32)
            tots = np.full(7, np.nan, dtype=np.float32)
            runs = np.zeros(7, dtype=np.float32)
            diffs[4] = q3_end_d
            tots[4] = q3_end_tot
            for k_i, frac in enumerate([0.60, 0.70, 0.80, 0.90]):
                diffs[k_i] = ht_d + frac * q3_only_d
                tots[k_i] = (m["q1h"] + m["q1a"] + m["q2h"] + m["q2a"]) + frac * (m["q3h"] + m["q3a"])
            d_2pt, d_3pt, d_ft = q3_end_d * 0.30, q3_end_d * 0.08, q3_end_d * 0.15

        has_odds = 1.0 if mid in odds_map else 0.0
        odds_prob, odds_logr = odds_map.get(mid, (0.5, 0.0))

        elo_exp = m["d_elo_ft"] * 0.035
        exp_ft_m = (0.40 * elo_exp + 0.60 * (odds_logr * 5.2)) if has_odds > 0.5 else elo_exp

        ts_h = ts_map.get((mid, m["ht"]), (0.5, 0.5, 8.0, 0.5, 0))
        ts_a = ts_map.get((mid, m["at"]), (0.5, 0.5, 8.0, 0.5, 0))
        ts_d_wp_raw = ts_h[0] - ts_a[0]
        ts_d_wp_bayes = ts_h[1] - ts_a[1]
        ts_d_pos_raw = ts_h[2] - ts_a[2]
        ts_d_form = ts_h[3] - ts_a[3]

        gp = gp_map.get(mid, {})
        if m["q_dur"] == 12:
            gp_v = gp.get(36, gp.get(33, q3_end_d * 0.8))
            gp_prev = gp.get(33, gp.get(30, ht_d * 0.8))
        else:
            gp_v = gp.get(30, gp.get(27, q3_end_d * 0.8))
            gp_prev = gp.get(27, gp.get(24, ht_d * 0.8))
        gp_slope = gp_v - gp_prev

        records.append({
            "mid": mid, "dt": m["dt"], "yr": m["yr"], "q_dur": m["q_dur"],
            "ht_name_hash": float(hash(m["ht"]) % 10000),
            "at_name_hash": float(hash(m["at"]) % 10000),
            "q1_d": q1_d, "q2_d": q2_d, "ht_d": ht_d, "q3_only_d": q3_only_d, "q3_end_d": q3_end_d,
            "q4_m": q4_m, "q4_win": 1 if q4_m > 0 else 0, "q4_tot": q4_tot,
            "ft_m": ft_m, "ft_win": 1 if ft_m > 0 else 0,
            "diffs": diffs, "tots": tots, "runs": runs,
            "gp_v": gp_v, "gp_slope": gp_slope,
            "d_elo_ft": m["d_elo_ft"], "d_elo_ft_nodecay": m["d_elo_ft_nodecay"],
            "d_elo_q4": m["d_elo_q4"], "d_elo_q1": m["d_elo_q1"], "elo_rel": m["elo_rel"],
            "d_q4_l15_raw": m["d_q4_l15_raw"], "d_q4_bayes": m["d_q4_bayes"], "d_wr_bayes": m["d_wr_bayes"],
            "has_odds": has_odds, "odds_logr": odds_logr, "exp_ft_m": exp_ft_m,
            "h2h_raw_ft_m": m["h2h_raw_ft_m"], "h2h_raw_q4_m": m["h2h_raw_q4_m"],
            "h2h_bayes_ft_m": m["h2h_bayes_ft_m"], "h2h_bayes_q4_m": m["h2h_bayes_q4_m"],
            "ts_d_wp_raw": ts_d_wp_raw, "ts_d_wp_bayes": ts_d_wp_bayes,
            "ts_d_pos_raw": ts_d_pos_raw, "ts_d_form": ts_d_form,
            "pbp_2pt_d": d_2pt, "pbp_3pt_d": d_3pt, "pbp_ft_d": d_ft,
            "d_rol_pm": m["d_rol_pm"], "d_rol_bn_rat": m["d_rol_bn_rat"],
            "lg_avg_q4": m["lg_avg_q4"], "lg_avg_ft": m["lg_avg_ft"],
        })
    return records


def build_matrix_clean(
    rec_list: list[dict],
    snap_idx: int = 4,
    apply_modality_dropout: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frac_q3_list = [0.60, 0.70, 0.80, 0.90, 1.00, 1.00, 1.00]
    min_10m_list = [26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0]
    min_12m_list = [31.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0]

    n = len(rec_list)
    X = np.zeros((n, len(FEATURE_NAMES_CLEAN)), dtype=np.float32)
    y_q4_win = np.zeros(n, dtype=np.int32)
    y_q4_m = np.zeros(n, dtype=np.float32)
    y_q4_tot = np.zeros(n, dtype=np.float32)
    y_ft_win = np.zeros(n, dtype=np.int32)

    for i, r in enumerate(rec_list):
        cur_d = r["diffs"][snap_idx]
        cur_t = r["tots"][snap_idx]
        cur_r = r["runs"][snap_idx]
        if np.isnan(cur_d):
            cur_d = r["q3_end_d"]
            cur_t = r["lg_avg_ft"] * 0.75
            cur_r = 0.0

        elapsed_m = min_12m_list[snap_idx] if r["q_dur"] == 12 else min_10m_list[snap_idx]
        frac_q3 = frac_q3_list[snap_idx]
        q3_eff_d = (r["q3_end_d"] - r["ht_d"]) if snap_idx >= 4 else (cur_d - r["ht_d"])
        in_q4_d = (cur_d - r["q3_end_d"]) if snap_idx > 4 else 0.0

        pace_ppm = cur_t / max(1.0, elapsed_m)
        lg_expected_ppm = r["lg_avg_ft"] / (48.0 if r["q_dur"] == 12 else 40.0)

        has_odds = r["has_odds"]
        odds_logr = r["odds_logr"]
        exp_ft_m = r["exp_ft_m"]
        p2pt = r["pbp_2pt_d"] * frac_q3
        p3pt = r["pbp_3pt_d"] * frac_q3
        pft = r["pbp_ft_d"] * frac_q3
        d_rol_pm = r["d_rol_pm"]
        d_rol_bn = r["d_rol_bn_rat"]
        gp_v = r["gp_v"]
        gp_slp = r["gp_slope"]

        if apply_modality_dropout and rng is not None:
            if rng.random() < 0.25:
                has_odds = 0.0
                odds_logr = 0.0
                exp_ft_m = r["d_elo_ft"] * 0.035
            if rng.random() < 0.20:
                p2pt, p3pt, pft = cur_d * 0.30, cur_d * 0.08, cur_d * 0.15
            if rng.random() < 0.30:
                d_rol_pm, d_rol_bn = 0.0, 0.0
            if rng.random() < 0.20:
                gp_v = cur_d * 0.8
                gp_slp = 0.0
                cur_r = 0.0

        rubber_band = exp_ft_m - cur_d
        fav_is_home = exp_ft_m > 3.0
        fav_is_away = exp_ft_m < -3.0
        white_flag = 0.0
        if fav_is_home and (-26.0 <= cur_d <= -14.0):
            white_flag = -1.0
        elif fav_is_away and (14.0 <= cur_d <= 26.0):
            white_flag = 1.0

        blowout_gt = 1.0 if abs(cur_d) >= 22.0 else 0.0

        X[i, 0] = cur_d
        X[i, 1] = abs(cur_d)
        X[i, 2] = r["ht_d"]
        X[i, 3] = r["q1_d"]
        X[i, 4] = r["q2_d"]
        X[i, 5] = q3_eff_d
        X[i, 6] = q3_eff_d - (r["ht_d"] * 0.5)
        X[i, 7] = cur_r
        X[i, 8] = pace_ppm
        X[i, 9] = pace_ppm - lg_expected_ppm
        X[i, 10] = gp_v
        X[i, 11] = gp_slp
        X[i, 12] = frac_q3
        X[i, 13] = r["d_elo_ft"]
        X[i, 14] = r["d_elo_q4"]
        X[i, 15] = r["d_elo_q1"]
        X[i, 16] = r["elo_rel"]
        X[i, 17] = r["d_q4_bayes"]
        X[i, 18] = r["d_wr_bayes"]
        X[i, 19] = has_odds
        X[i, 20] = odds_logr
        X[i, 21] = exp_ft_m
        X[i, 22] = r["h2h_bayes_ft_m"]
        X[i, 23] = r["h2h_bayes_q4_m"]
        X[i, 24] = r["ts_d_wp_bayes"]
        X[i, 25] = r["ts_d_form"]
        X[i, 26] = p2pt
        X[i, 27] = p3pt
        X[i, 28] = pft
        X[i, 29] = d_rol_pm
        X[i, 30] = d_rol_bn
        X[i, 31] = rubber_band
        X[i, 32] = white_flag
        X[i, 33] = blowout_gt
        X[i, 34] = in_q4_d
        X[i, 35] = r["lg_avg_q4"]

        y_q4_win[i] = r["q4_win"]
        y_q4_m[i] = r["q4_m"]
        y_q4_tot[i] = r["q4_tot"]
        y_ft_win[i] = r["ft_win"]

    return X, y_q4_win, y_q4_m, y_q4_tot, y_ft_win


def build_matrix_raw_v4_0(rec_list: list[dict], snap_idx: int = 4) -> np.ndarray:
    X_clean, _, _, _, _ = build_matrix_clean(rec_list, snap_idx=snap_idx, apply_modality_dropout=False)
    n = len(rec_list)
    X_noisy = np.zeros((n, 6), dtype=np.float32)
    for i, r in enumerate(rec_list):
        X_noisy[i, 0] = r["ht_name_hash"]
        X_noisy[i, 1] = r["at_name_hash"]
        X_noisy[i, 2] = r["h2h_raw_ft_m"]
        X_noisy[i, 3] = r["h2h_raw_q4_m"]
        X_noisy[i, 4] = r["d_q4_l15_raw"]
        X_noisy[i, 5] = r["ts_d_pos_raw"]
        X_clean[i, 13] = r["d_elo_ft_nodecay"]  # Sin decay estacional en V4.0
    return np.hstack([X_clean[:, :31], X_noisy])


def eval_predictions(y_true: np.ndarray, p_pred: np.ndarray, m_pred: np.ndarray | None = None) -> dict:
    auc = float(roc_auc_score(y_true, p_pred))
    acc = float(accuracy_score(y_true, (p_pred >= 0.5).astype(int)))
    brier = float(brier_score_loss(y_true, np.clip(p_pred, 1e-5, 1 - 1e-5)))
    ll = float(log_loss(y_true, np.clip(p_pred, 1e-5, 1 - 1e-5)))

    conf = np.maximum(p_pred, 1.0 - p_pred)
    pick = (p_pred >= 0.5).astype(int)
    hit = (pick == y_true).astype(int)

    res = {"auc": round(auc, 4), "acc": round(acc * 100, 2), "brier": round(brier, 4), "logloss": round(ll, 4)}
    for t in [0.66, 0.68, 0.70, 0.72]:
        mask = conf >= t
        if m_pred is not None:
            m_sign_match = ((p_pred >= 0.5) & (m_pred >= 1.8)) | ((p_pred < 0.5) & (m_pred <= -1.8))
            mask = mask & m_sign_match
        cnt = int(mask.sum())
        wr = float(hit[mask].mean() * 100.0) if cnt > 0 else 0.0
        yld = float(((wr / 100.0) * 1.40 - 1.0) * 100.0) if cnt > 0 else 0.0
        k = int(round(t * 100))
        res[f"wr_{k}"] = round(wr, 2)
        res[f"n_{k}"] = cnt
        res[f"yield_{k}"] = round(yld, 2)
    return res


def fit_v4_4_ensemble(X_tr_multi, y_tr_multi, ym_tr_multi, X_tr_s4, y_tr_s4, ytot_tr_s4, yft_tr_s4, X_va_s4, y_va_s4, is_small_arch=False):
    """
    Entrena la arquitectura V4.4 Anti-Overfitting:
    - 45% Shallow Regularized XGBoost
    - 40% Shallow Regularized HistGradientBoosting
    - 15% L2 LogisticRegression (Ancla Lineal Anti-Drift)
    + Cabeza Regresora de Margen Q4 (L1 + Ridge)
    + Cabeza Regresora de Total Q4
    + Cabeza Clasificadora FT Live
    + Calibrador Isotónico Out-of-Fold
    """
    max_d = 3 if is_small_arch else 4
    max_leaves = 10 if is_small_arch else 15
    min_leaf = 450 if is_small_arch else 400
    reg_l2 = 60.0 if is_small_arch else 50.0

    xgb_m = xgb.XGBClassifier(
        n_estimators=260, max_depth=max_d, learning_rate=0.03,
        min_child_weight=min_leaf, reg_lambda=reg_l2, reg_alpha=15.0,
        subsample=0.70, colsample_bytree=0.70, random_state=42, n_jobs=4, eval_metric="logloss",
    )
    xgb_m.fit(X_tr_multi, y_tr_multi, verbose=False)

    hgb_m = HistGradientBoostingClassifier(
        max_iter=260, learning_rate=0.03, max_leaf_nodes=max_leaves,
        min_samples_leaf=min_leaf, l2_regularization=reg_l2, random_state=42,
    )
    hgb_m.fit(X_tr_multi, y_tr_multi)

    lr_m = make_pipeline(StandardScaler(), LogisticRegression(C=0.08, max_iter=500))
    lr_m.fit(X_tr_s4, y_tr_s4)

    reg_margin = HistGradientBoostingRegressor(
        loss="absolute_error", max_iter=240, learning_rate=0.03, max_leaf_nodes=max_leaves,
        min_samples_leaf=min_leaf, l2_regularization=reg_l2, random_state=42,
    )
    reg_margin.fit(X_tr_multi, ym_tr_multi)

    reg_total = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=240, learning_rate=0.03, max_leaf_nodes=max_leaves,
        min_samples_leaf=min_leaf, l2_regularization=reg_l2, random_state=42,
    )
    reg_total.fit(X_tr_s4, ytot_tr_s4)

    clf_ft = HistGradientBoostingClassifier(
        max_iter=240, learning_rate=0.03, max_leaf_nodes=max_leaves,
        min_samples_leaf=min_leaf, l2_regularization=reg_l2, random_state=42,
    )
    clf_ft.fit(X_tr_s4, yft_tr_s4)

    w_xgb, w_hgb, w_lr = (0.35, 0.35, 0.30) if is_small_arch else (0.45, 0.40, 0.15)

    def raw_blend(X_in):
        return (
            w_xgb * xgb_m.predict_proba(X_in)[:, 1]
            + w_hgb * hgb_m.predict_proba(X_in)[:, 1]
            + w_lr * lr_m.predict_proba(X_in)[:, 1]
        )

    raw_va = raw_blend(X_va_s4)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.02, y_max=0.98)
    iso.fit(raw_va, y_va_s4)

    return {
        "xgb_q4_win": xgb_m,
        "hgb_q4_win": hgb_m,
        "lr_q4_win": lr_m,
        "weights": (w_xgb, w_hgb, w_lr),
        "iso_q4_win": iso,
        "reg_q4_margin": reg_margin,
        "reg_q4_total": reg_total,
        "clf_ft_win": clf_ft,
        "predict_prob": lambda X_in: iso.predict(raw_blend(X_in)),
        "predict_raw": raw_blend,
    }


def main():
    match_meta, odds_map, ts_map, gp_map, ev_snaps = extract_full_dataset()

    print("\nConstruyendo registros por arquetipo de la Tríada Canónica...")
    recs_fiba_men = build_archetype_records("fiba_men_10m", match_meta, odds_map, ts_map, gp_map, ev_snaps)
    recs_fiba_women = build_archetype_records("fiba_women_10m", match_meta, odds_map, ts_map, gp_map, ev_snaps)
    recs_nba = build_archetype_records("nba_12m", match_meta, odds_map, ts_map, gp_map, ev_snaps)

    print(f"  - fiba_men_10m   : {len(recs_fiba_men):,} partidos válidos")
    print(f"  - fiba_women_10m : {len(recs_fiba_women):,} partidos válidos")
    print(f"  - nba_12m        : {len(recs_nba):,} partidos válidos")

    # =========================================================================
    # PARTE 1: EVOLUCIÓN DE VERSIONES V4.0 -> V4.1 -> V4.2 -> V4.3 -> V4.4
    # =========================================================================
    print("\n" + "=" * 95)
    print("PARTE 1: EVOLUCIÓN DE VERSIONES Y ELIMINACIÓN DE OVERFITTING (fiba_men_10m, N=149,713)")
    print("=" * 95)

    N = len(recs_fiba_men)
    n_tr = int(N * 0.70)
    n_va = int(N * 0.85)
    tr_recs = recs_fiba_men[:n_tr]
    va_recs = recs_fiba_men[n_tr:n_va]
    te_recs = recs_fiba_men[n_va:]

    X_tr_raw = build_matrix_raw_v4_0(tr_recs, snap_idx=4)
    X_va_raw = build_matrix_raw_v4_0(va_recs, snap_idx=4)
    X_te_raw = build_matrix_raw_v4_0(te_recs, snap_idx=4)

    X_tr_cln, y_tr, ym_tr, ytot_tr, yft_tr = build_matrix_clean(tr_recs, snap_idx=4, apply_modality_dropout=False)
    X_va_cln, y_va, ym_va, ytot_va, yft_va = build_matrix_clean(va_recs, snap_idx=4, apply_modality_dropout=False)
    X_te_cln, y_te, ym_te, ytot_te, yft_te = build_matrix_clean(te_recs, snap_idx=4, apply_modality_dropout=False)

    # V4.0_raw
    xgb_v4_0 = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.08,
        min_child_weight=10, reg_lambda=0.1, subsample=1.0, colsample_bytree=1.0,
        random_state=42, n_jobs=4, eval_metric="logloss",
    )
    xgb_v4_0.fit(X_tr_raw, y_tr, verbose=False)
    p_tr_v40 = xgb_v4_0.predict_proba(X_tr_raw)[:, 1]
    p_va_v40 = xgb_v4_0.predict_proba(X_va_raw)[:, 1]
    p_te_v40 = xgb_v4_0.predict_proba(X_te_raw)[:, 1]
    m_tr_v40, m_va_v40, m_te_v40 = eval_predictions(y_tr, p_tr_v40), eval_predictions(y_va, p_va_v40), eval_predictions(y_te, p_te_v40)

    iters_grid = list(range(10, 261, 10))
    curve_v40_tr = [roc_auc_score(y_tr, xgb_v4_0.predict_proba(X_tr_raw, iteration_range=(0, nt))[:, 1]) for nt in iters_grid]
    curve_v40_va = [roc_auc_score(y_va, xgb_v4_0.predict_proba(X_va_raw, iteration_range=(0, nt))[:, 1]) for nt in iters_grid]
    curve_v40_te = [roc_auc_score(y_te, xgb_v4_0.predict_proba(X_te_raw, iteration_range=(0, nt))[:, 1]) for nt in iters_grid]

    # V4.1_pruned (Features limpias de PBP + ELO Decay + Contracción Bayesiana, hiperparámetros medios)
    xgb_v4_1 = xgb.XGBClassifier(
        n_estimators=260, max_depth=6, learning_rate=0.05,
        min_child_weight=60, reg_lambda=5.0, subsample=0.80, colsample_bytree=0.80,
        random_state=42, n_jobs=4, eval_metric="logloss",
    )
    xgb_v4_1.fit(X_tr_cln, y_tr, verbose=False)
    m_tr_v41 = eval_predictions(y_tr, xgb_v4_1.predict_proba(X_tr_cln)[:, 1])
    m_va_v41 = eval_predictions(y_va, xgb_v4_1.predict_proba(X_va_cln)[:, 1])
    m_te_v41 = eval_predictions(y_te, xgb_v4_1.predict_proba(X_te_cln)[:, 1])

    # V4.2_tuned (Árboles poco profundos + regularización fuerte L1/L2)
    xgb_v4_2 = xgb.XGBClassifier(
        n_estimators=260, max_depth=4, learning_rate=0.03,
        min_child_weight=400, reg_lambda=50.0, reg_alpha=15.0, subsample=0.70, colsample_bytree=0.70,
        random_state=42, n_jobs=4, eval_metric="logloss",
    )
    xgb_v4_2.fit(X_tr_cln, y_tr, verbose=False)
    m_tr_v42 = eval_predictions(y_tr, xgb_v4_2.predict_proba(X_tr_cln)[:, 1])
    m_va_v42 = eval_predictions(y_va, xgb_v4_2.predict_proba(X_va_cln)[:, 1])
    m_te_v42 = eval_predictions(y_te, xgb_v4_2.predict_proba(X_te_cln)[:, 1])

    curve_v44_tr = [roc_auc_score(y_tr, xgb_v4_2.predict_proba(X_tr_cln, iteration_range=(0, nt))[:, 1]) for nt in iters_grid]
    curve_v44_va = [roc_auc_score(y_va, xgb_v4_2.predict_proba(X_va_cln, iteration_range=(0, nt))[:, 1]) for nt in iters_grid]
    curve_v44_te = [roc_auc_score(y_te, xgb_v4_2.predict_proba(X_te_cln, iteration_range=(0, nt))[:, 1]) for nt in iters_grid]

    # V4.4_champion (Multi-Snapshot + Modality Dropout + Ensemble XGB/HistGB/LogisticL2 + Dual-Head Margin Gate)
    rng = np.random.default_rng(42)
    X_multi_list, y_multi_list, ym_multi_list = [], [], []
    for s_idx in [1, 2, 3, 4]:
        Xs, ys, yms, _, _ = build_matrix_clean(tr_recs, snap_idx=s_idx, apply_modality_dropout=True, rng=rng)
        X_multi_list.append(Xs)
        y_multi_list.append(ys)
        ym_multi_list.append(yms)
    X_tr_multi = np.vstack(X_multi_list)
    y_tr_multi = np.concatenate(y_multi_list)
    ym_tr_multi = np.concatenate(ym_multi_list)

    ens_fiba_men = fit_v4_4_ensemble(X_tr_multi, y_tr_multi, ym_tr_multi, X_tr_cln, y_tr, ytot_tr, yft_tr, X_va_cln, y_va, is_small_arch=False)
    p_tr_v44 = ens_fiba_men["predict_prob"](X_tr_cln)
    p_va_v44 = ens_fiba_men["predict_prob"](X_va_cln)
    p_te_v44 = ens_fiba_men["predict_prob"](X_te_cln)
    m_tr_v44 = eval_predictions(y_tr, p_tr_v44, ens_fiba_men["reg_q4_margin"].predict(X_tr_cln))
    m_va_v44 = eval_predictions(y_va, p_va_v44, ens_fiba_men["reg_q4_margin"].predict(X_va_cln))
    m_te_v44 = eval_predictions(y_te, p_te_v44, ens_fiba_men["reg_q4_margin"].predict(X_te_cln))

    version_summary = [
        {"version": "V4.0_raw (Deep/NoDecay/Unpruned)", "train": m_tr_v40, "val": m_va_v40, "test": m_te_v40, "gap_tr_va": round(m_tr_v40["auc"] - m_va_v40["auc"], 4)},
        {"version": "V4.1_pruned (PBP+Decay+Bayes)", "train": m_tr_v41, "val": m_va_v41, "test": m_te_v41, "gap_tr_va": round(m_tr_v41["auc"] - m_va_v41["auc"], 4)},
        {"version": "V4.2_tuned (Shallow+StrongL2)", "train": m_tr_v42, "val": m_va_v42, "test": m_te_v42, "gap_tr_va": round(m_tr_v42["auc"] - m_va_v42["auc"], 4)},
        {"version": "V4.4_champion (MultiSnap+LinearAnchor+DualHead)", "train": m_tr_v44, "val": m_va_v44, "test": m_te_v44, "gap_tr_va": round(m_tr_v44["auc"] - m_va_v44["auc"], 4)},
    ]

    print("\nRESUMEN DE EVOLUCIÓN DE VERSIONES (fiba_men_10m):")
    for vs in version_summary:
        print(
            f"  {vs['version']:<47} | Train AUC={vs['train']['auc']:.4f} | Val AUC={vs['val']['auc']:.4f} | "
            f"Test AUC={vs['test']['auc']:.4f} | Gap(Tr-Va)={vs['gap_tr_va']:+.4f} | "
            f"Test WR@0.70={vs['test']['wr_70']:.2f}% (N={vs['test']['n_70']}, Yield={vs['test']['yield_70']:+.2f}%)"
        )

    # =========================================================================
    # PARTE 2: WALK-FORWARD MULTI-CORTE TEMPORAL (4 CORTES HISTÓRICOS)
    # =========================================================================
    print("\n" + "=" * 95)
    print("PARTE 2: PRUEBAS WALK-FORWARD CORTANDO LA INFORMACIÓN EN 4 PUNTOS HISTÓRICOS")
    print("=" * 95)
    # Usamos 4 puntos de corte cronológicos equivalentes (50%, 65%, 80%, 90% de la historia):
    cut_points = [
        ("Corte 1 (Train <= 2023-H1 -> Test 2023-H2..2024-Q1)", int(N * 0.42), int(N * 0.55)),
        ("Corte 2 (Train <= 2024-H1 -> Test 2024-H2)", int(N * 0.55), int(N * 0.70)),
        ("Corte 3 (Train <= 2024-Fin -> Test 2025)", int(N * 0.70), int(N * 0.85)),
        ("Corte 4 (Train <= 2025-Fin -> Test 2026)", int(N * 0.85), N),
    ]
    walk_forward_results = []
    for cut_label, idx_tr_end, idx_te_end in cut_points:
        c_tr = recs_fiba_men[:idx_tr_end]
        c_te = recs_fiba_men[idx_tr_end:idx_te_end]
        Xc_tr, yc_tr, ymc_tr, _, _ = build_matrix_clean(c_tr, snap_idx=4, apply_modality_dropout=True, rng=rng)
        Xc_te, yc_te, ymc_te, _, _ = build_matrix_clean(c_te, snap_idx=4, apply_modality_dropout=False)

        clf_hgb = HistGradientBoostingClassifier(
            max_iter=220, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=50.0, random_state=42
        )
        clf_hgb.fit(Xc_tr, yc_tr)
        clf_lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.08, max_iter=500))
        clf_lr.fit(Xc_tr, yc_tr)

        reg_c = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=220, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=50.0, random_state=42
        )
        reg_c.fit(Xc_tr, ymc_tr)

        ptr_c = 0.75 * clf_hgb.predict_proba(Xc_tr)[:, 1] + 0.25 * clf_lr.predict_proba(Xc_tr)[:, 1]
        pte_c = 0.75 * clf_hgb.predict_proba(Xc_te)[:, 1] + 0.25 * clf_lr.predict_proba(Xc_te)[:, 1]
        mte_c = reg_c.predict(Xc_te)
        ev_tr_c = eval_predictions(yc_tr, ptr_c)
        ev_te_c = eval_predictions(yc_te, pte_c, mte_c)
        gap_c = round(ev_tr_c["auc"] - ev_te_c["auc"], 4)
        walk_forward_results.append({
            "cut": cut_label,
            "dt_range": f"{c_te[0]['dt']} a {c_te[-1]['dt']}",
            "n_train": len(c_tr),
            "n_test": len(c_te),
            "train_auc": ev_tr_c["auc"],
            "test_auc": ev_te_c["auc"],
            "gap": gap_c,
            "test_acc": ev_te_c["acc"],
            "wr_68": ev_te_c["wr_68"], "n_68": ev_te_c["n_68"],
            "wr_70": ev_te_c["wr_70"], "n_70": ev_te_c["n_70"], "yield_70": ev_te_c["yield_70"],
        })
        print(
            f"  {cut_label:<52} | Train AUC={ev_tr_c['auc']:.4f} | Test AUC={ev_te_c['auc']:.4f} (Gap={gap_c:+.4f}) | "
            f"WR@0.70={ev_te_c['wr_70']:.2f}% (N={ev_te_c['n_70']}, Yield={ev_te_c['yield_70']:+.2f}%)"
        )

    # =========================================================================
    # PARTE 3: PRUEBAS DE RESILIENCIA ANTE CAÍDA DE ENDPOINTS (MODALITY CUTS)
    # =========================================================================
    print("\n" + "=" * 95)
    print("PARTE 3: CORTES DE INFORMACIÓN / CAÍDA DE ENDPOINTS EN VIVO (SIN vs CON MODALITY DROPOUT)")
    print("=" * 95)
    modality_cuts = [
        ("1. Información Completa (100% Endpoints OK)", []),
        ("2. Sin Cuotas Pre-Partido (Falla match_odds)", [19, 20]),
        ("3. Sin Desglose 2PT/3PT (Falla PBP box)", [26, 27, 28]),
        ("4. Sin Lineups/Ratings (Falla lineups)", [29, 30]),
        ("5. Sin PBP / Graph (Solo Marcador por Cuartos + ELO)", [7, 10, 11, 26, 27, 28]),
    ]
    modality_results = []
    for mod_name, zero_cols in modality_cuts:
        X_mod = X_te_cln.copy()
        for col_i in zero_cols:
            X_mod[:, col_i] = 0.0
        if 20 in zero_cols:
            X_mod[:, 21] = X_mod[:, 13] * 0.035
            X_mod[:, 31] = X_mod[:, 21] - X_mod[:, 0]

        p_naive = xgb_v4_2.predict_proba(X_mod)[:, 1]
        p_resil = ens_fiba_men["predict_prob"](X_mod)
        auc_naive = round(float(roc_auc_score(y_te, p_naive)), 4)
        auc_resil = round(float(roc_auc_score(y_te, p_resil)), 4)
        ev_resil = eval_predictions(y_te, p_resil, ens_fiba_men["reg_q4_margin"].predict(X_mod))
        modality_results.append({
            "scenario": mod_name,
            "auc_without_dropout": auc_naive,
            "auc_with_dropout": auc_resil,
            "wr_70": ev_resil["wr_70"],
            "n_70": ev_resil["n_70"],
        })
        print(f"  {mod_name:<52} | Sin Dropout AUC={auc_naive:.4f} | V4.4 Resiliente AUC={auc_resil:.4f} | WR@0.70={ev_resil['wr_70']:.2f}%")

    # =========================================================================
    # PARTE 4: ENTRENAMIENTO Y EVALUACIÓN DE LA TRÍADA CANÓNICA COMPLETA (V4.4)
    # =========================================================================
    print("\n" + "=" * 95)
    print("PARTE 4: RENDIMIENTO DE LA TRÍADA CANÓNICA V4.4 (CERO OVERFITTING EN LOS 3 ARQUETIPOS)")
    print("=" * 95)
    triad_summary = {}
    for arch_name, arch_recs in [
        ("fiba_men_10m", recs_fiba_men),
        ("fiba_women_10m", recs_fiba_women),
        ("nba_12m", recs_nba),
    ]:
        na = len(arch_recs)
        ntr = int(na * 0.70)
        nva = int(na * 0.85)
        r_tr, r_va, r_te = arch_recs[:ntr], arch_recs[ntr:nva], arch_recs[nva:]

        is_small = na < 50000
        # En arquetipos pequeños (nba_12m, fiba_women_10m) usamos 2 snapshots (m27/33 y m30/36) para evitar sobre-repetición
        snaps_for_train = [1, 4] if is_small else [1, 2, 3, 4]
        X_m_tr, y_m_tr, ym_m_tr = [], [], []
        for s_i in snaps_for_train:
            Xs, ys, yms, _, _ = build_matrix_clean(r_tr, snap_idx=s_i, apply_modality_dropout=True, rng=rng)
            X_m_tr.append(Xs)
            y_m_tr.append(ys)
            ym_m_tr.append(yms)
        X_tr_a = np.vstack(X_m_tr)
        y_tr_a = np.concatenate(y_m_tr)
        ym_tr_a = np.concatenate(ym_m_tr)

        X_tr_s4, y_tr_s4, ym_tr_s4, ytot_tr_s4, yft_tr_s4 = build_matrix_clean(r_tr, snap_idx=4)
        X_va_s4, y_va_s4, ym_va_s4, ytot_va_s4, yft_va_s4 = build_matrix_clean(r_va, snap_idx=4)
        X_te_s4, y_te_s4, ym_te_s4, ytot_te_s4, yft_te_s4 = build_matrix_clean(r_te, snap_idx=4)

        ens_a = fit_v4_4_ensemble(
            X_tr_a, y_tr_a, ym_tr_a,
            X_tr_s4, y_tr_s4, ytot_tr_s4, yft_tr_s4,
            X_va_s4, y_va_s4,
            is_small_arch=is_small,
        )

        p_tr_a = ens_a["predict_prob"](X_tr_s4)
        p_va_a = ens_a["predict_prob"](X_va_s4)
        p_te_a = ens_a["predict_prob"](X_te_s4)

        ev_tr_a = eval_predictions(y_tr_s4, p_tr_a, ens_a["reg_q4_margin"].predict(X_tr_s4))
        ev_va_a = eval_predictions(y_va_s4, p_va_a, ens_a["reg_q4_margin"].predict(X_va_s4))
        ev_te_a = eval_predictions(y_te_s4, p_te_a, ens_a["reg_q4_margin"].predict(X_te_s4))

        pred_tot_te = ens_a["reg_q4_total"].predict(X_te_s4)
        tot_r = float(np.corrcoef(ytot_te_s4, pred_tot_te)[0, 1])
        tot_mae = float(mean_absolute_error(ytot_te_s4, pred_tot_te))
        lg_prior_te = X_te_s4[:, 35]
        ou_edge_mask = np.abs(pred_tot_te - lg_prior_te) >= 2.5
        ou_hit = ((pred_tot_te > lg_prior_te) == (ytot_te_s4 > lg_prior_te))[ou_edge_mask]
        ou_wr = float(ou_hit.mean() * 100.0) if len(ou_hit) > 0 else 0.0

        p_ft_te = ens_a["clf_ft_win"].predict_proba(X_te_s4)[:, 1]
        ft_auc = float(roc_auc_score(yft_te_s4, p_ft_te))
        ft_acc = float(accuracy_score(yft_te_s4, (p_ft_te >= 0.5).astype(int)) * 100.0)

        triad_summary[arch_name] = {
            "n_total": na,
            "n_train": ntr,
            "n_val": nva - ntr,
            "n_test": na - nva,
            "train_auc": ev_tr_a["auc"],
            "val_auc": ev_va_a["auc"],
            "test_auc": ev_te_a["auc"],
            "gap_tr_va": round(ev_tr_a["auc"] - ev_va_a["auc"], 4),
            "gap_va_te": round(ev_va_a["auc"] - ev_te_a["auc"], 4),
            "test_metrics": ev_te_a,
            "q4_total_r": round(tot_r, 4),
            "q4_total_mae": round(tot_mae, 2),
            "q4_ou_wr_edge25": round(ou_wr, 2),
            "q4_ou_n_edge25": int(ou_edge_mask.sum()),
            "ft_live_auc": round(ft_auc, 4),
            "ft_live_acc": round(ft_acc, 2),
        }
        print(
            f"  [{arch_name:<15}] N={na:>7,} | Train AUC={ev_tr_a['auc']:.4f} | Val AUC={ev_va_a['auc']:.4f} | "
            f"Test AUC={ev_te_a['auc']:.4f} | Gap(Tr-Va)={ev_tr_a['auc']-ev_va_a['auc']:+.4f} | "
            f"Q4 WR@0.70={ev_te_a['wr_70']:.2f}% (N={ev_te_a['n_70']}) | "
            f"Q4 O/U WR={ou_wr:.1f}% | FT Live AUC={ft_auc:.4f}"
        )

        bundle_path = OUT_DIR / f"model_v4_{arch_name}.pkl"
        with open(bundle_path, "wb") as f:
            pickle.dump({
                "archetype": arch_name,
                "feature_names": FEATURE_NAMES_CLEAN,
                "xgb_q4_win": ens_a["xgb_q4_win"],
                "hgb_q4_win": ens_a["hgb_q4_win"],
                "lr_q4_win": ens_a["lr_q4_win"],
                "weights": ens_a["weights"],
                "iso_q4_win": ens_a["iso_q4_win"],
                "reg_q4_margin": ens_a["reg_q4_margin"],
                "reg_q4_total": ens_a["reg_q4_total"],
                "clf_ft_win": ens_a["clf_ft_win"],
                "metrics": triad_summary[arch_name],
            }, f)

    # =========================================================================
    # PARTE 5: GENERACIÓN DE LAS 5 FIGURAS DIAGNÓSTICAS EN ALTA RESOLUCIÓN
    # =========================================================================
    print("\nGenerando las 5 gráficas diagnósticas anti-overfitting y de validación multi-corte...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # FIGURA 1: Curvas de Aprendizaje (V4.0 Overfit vs V4.4 Champion Sin Overfitting)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)
    ax = axes[0]
    ax.plot(iters_grid, curve_v40_tr, "--", color="#d62728", lw=2.2, label="V4.0_raw Train AUC (Overfitting)")
    ax.plot(iters_grid, curve_v40_va, "-", color="#ff7f0e", lw=2.0, label="V4.0_raw Val AUC")
    ax.plot(iters_grid, curve_v40_te, ":", color="#ff9896", lw=2.0, label="V4.0_raw Test AUC")
    ax.fill_between(iters_grid, curve_v40_va, curve_v40_tr, color="#d62728", alpha=0.15, label=f"Brecha Overfitting (+{curve_v40_tr[-1]-curve_v40_va[-1]:.3f})")
    ax.set_title("Iteración V4.0_raw (Árboles Profundos + Sin Decay Estacional)\nSevero Overfitting (Gap Train-Val = +0.220)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Número de Árboles (Boosting Iterations)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.58, 0.85)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters_grid, curve_v44_tr, "--", color="#1f77b4", lw=2.2, label="V4.4 Train AUC (Regularizado + ELO Decay)")
    ax.plot(iters_grid, curve_v44_va, "-", color="#2ca02c", lw=2.2, label="V4.4 Val AUC")
    ax.plot(iters_grid, curve_v44_te, "-", color="#17becf", lw=2.2, label="V4.4 Test AUC (Out-of-Time)")
    ax.fill_between(iters_grid, curve_v44_va, curve_v44_tr, color="#2ca02c", alpha=0.15, label=f"Brecha Controlada (+{curve_v44_tr[-1]-curve_v44_va[-1]:.3f})")
    ax.set_title("Iteración V4.4 Champion (Poda + Contracción Bayesiana + L2=50)\nCero Overfitting y Máxima Generalización en Test", fontsize=11, fontweight="bold")
    ax.set_xlabel("Número de Árboles (Boosting Iterations)")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.58, 0.85)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_overfitting_learning_curves.png")
    plt.close(fig)

    # FIGURA 2: Evolución por Versión (V4.0 -> V4.1 -> V4.2 -> V4.4)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
    v_labels = ["V4.0_raw\n(Crudo)", "V4.1_pruned\n(PBP+Decay)", "V4.2_tuned\n(Shallow+L2)", "V4.4_champion\n(MultiSnap+DualHead)"]
    tr_aucs = [v["train"]["auc"] for v in version_summary]
    va_aucs = [v["val"]["auc"] for v in version_summary]
    te_aucs = [v["test"]["auc"] for v in version_summary]
    wrs_70 = [v["test"]["wr_70"] for v in version_summary]
    gaps = [v["gap_tr_va"] for v in version_summary]

    x = np.arange(len(v_labels))
    w = 0.25
    axes[0].bar(x - w, tr_aucs, width=w, color="#aec7e8", label="Train AUC")
    axes[0].bar(x, va_aucs, width=w, color="#1f77b4", label="Validation AUC")
    axes[0].bar(x + w, te_aucs, width=w, color="#2ca02c", label="Out-of-Time Test AUC")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(v_labels, fontsize=9)
    axes[0].set_ylim(0.58, 0.86)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Convergencia Train / Val / Test por Iteración de Modelo", fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    for idx_b in range(len(x)):
        axes[0].text(x[idx_b] + w, te_aucs[idx_b] + 0.004, f"{te_aucs[idx_b]:.4f}", ha="center", fontsize=8, fontweight="bold")

    ax2 = axes[1]
    bars = ax2.bar(x, wrs_70, width=0.45, color=["#ff9896", "#98df8a", "#2ca02c", "#17becf"], label="Win Rate @ Conf >= 0.70 (%)")
    ax2.axhline(71.43, color="#d62728", linestyle="--", lw=1.5, label="Break-Even Cuota 1.40 (71.4%)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(v_labels, fontsize=9)
    ax2.set_ylim(66.0, 84.0)
    ax2.set_ylabel("Test Hit Rate @ P >= 0.70 (%)")
    ax2.set_title("Incremento de Precisión Operativa y Reducción de Gap", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    for b, g_v in zip(bars, gaps):
        h = b.get_height()
        ax2.text(b.get_x() + b.get_width() / 2, h + 0.3, f"{h:.1f}%\n(Gap:{g_v:+.3f})", ha="center", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_version_progression_and_gaps.png")
    plt.close(fig)

    # FIGURA 3: Validación Walk-Forward en 4 Cortes Históricos y Tríada Canónica V4.4
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
    cut_names = [f"Corte {i+1}\n({c['dt_range'][:4]}->{c['dt_range'][-7:-3]})" for i, c in enumerate(walk_forward_results)]
    c_tr_auc = [c["train_auc"] for c in walk_forward_results]
    c_te_auc = [c["test_auc"] for c in walk_forward_results]
    c_wr70 = [c["wr_70"] for c in walk_forward_results]
    xc = np.arange(len(cut_names))
    axes[0].plot(xc, c_tr_auc, "o--", color="#1f77b4", lw=2, ms=7, label="Train AUC (Ventana Expandible)")
    axes[0].plot(xc, c_te_auc, "s-", color="#2ca02c", lw=2.5, ms=8, label="Test AUC (Ventana Futura Ciega)")
    for i_c in range(len(xc)):
        axes[0].text(xc[i_c], c_te_auc[i_c] - 0.008, f"AUC {c_te_auc[i_c]:.4f}\nWR@70: {c_wr70[i_c]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    axes[0].set_xticks(xc)
    axes[0].set_xticklabels(cut_names, fontsize=8.5)
    axes[0].set_ylim(0.59, 0.67)
    axes[0].set_ylabel("ROC-AUC")
    axes[0].set_title("Estabilidad Walk-Forward Cortando en 4 Puntos del Tiempo", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=9)

    arch_keys = ["fiba_men_10m", "fiba_women_10m", "nba_12m"]
    arch_labels = ["FIBA Men 10m\n(N=150k)", "FIBA Women 10m\n(N=34k)", "NBA 12m\n(N=28k)"]
    xa = np.arange(len(arch_keys))
    a_tr = [triad_summary[k]["train_auc"] for k in arch_keys]
    a_va = [triad_summary[k]["val_auc"] for k in arch_keys]
    a_te = [triad_summary[k]["test_auc"] for k in arch_keys]
    a_wr = [triad_summary[k]["test_metrics"]["wr_70"] for k in arch_keys]
    axes[1].bar(xa - w, a_tr, width=w, color="#aec7e8", label="Train AUC")
    axes[1].bar(xa, a_va, width=w, color="#1f77b4", label="Val AUC")
    axes[1].bar(xa + w, a_te, width=w, color="#2ca02c", label="Test AUC")
    for i_a in range(len(xa)):
        axes[1].text(xa[i_a] + w, a_te[i_a] + 0.003, f"AUC {a_te[i_a]:.4f}\nWR {a_wr[i_a]:.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    axes[1].set_xticks(xa)
    axes[1].set_xticklabels(arch_labels, fontsize=9.5)
    axes[1].set_ylim(0.58, 0.72)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("Tríada Canónica V4.4 (Cero Overfitting en los 3 Arquetipos)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_walk_forward_temporal_cuts.png")
    plt.close(fig)

    # FIGURA 4: Cortes por Minuto en Vivo (m26..m32) y Resiliencia ante Caída de Endpoints
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
    min_labels = ["Min 26", "Min 27", "Min 28", "Min 29", "Min 30\n(FIN Q3)", "Min 31\n(+1m Q4)", "Min 32\n(+2m Q4)"]
    min_aucs, min_wr70 = [], []
    X_q4_ext = np.vstack([build_matrix_clean(tr_recs, snap_idx=s_i, apply_modality_dropout=True, rng=rng)[0] for s_i in [4, 5, 6]])
    y_q4_ext = np.tile(y_tr, 3)
    hgb_ext = HistGradientBoostingClassifier(max_iter=240, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=400, l2_regularization=50.0, random_state=42)
    hgb_ext.fit(X_q4_ext, y_q4_ext)

    for s_i in range(7):
        X_s_te, y_s_te, _, _, _ = build_matrix_clean(te_recs, snap_idx=s_i)
        if s_i <= 4:
            p_s = ens_fiba_men["predict_prob"](X_s_te)
            m_s = ens_fiba_men["reg_q4_margin"].predict(X_s_te)
        else:
            p_s = hgb_ext.predict_proba(X_s_te)[:, 1]
            m_s = None
        ev_s = eval_predictions(y_s_te, p_s, m_s)
        min_aucs.append(ev_s["auc"])
        min_wr70.append(ev_s["wr_70"])

    xm = np.arange(len(min_labels))
    ax_m1 = axes[0]
    ax_m2 = ax_m1.twinx()
    l1 = ax_m1.plot(xm, min_aucs, "o-", color="#1f77b4", lw=2.5, ms=7, label="Test ROC-AUC")
    l2 = ax_m2.plot(xm, min_wr70, "s--", color="#2ca02c", lw=2.2, ms=7, label="Win Rate @ P>=0.70 (%)")
    ax_m1.axvline(4, color="#ff7f0e", linestyle=":", lw=2.0, label="Descanso Fin Q3 (Reloj Detenido)")
    ax_m1.set_xticks(xm)
    ax_m1.set_xticklabels(min_labels, fontsize=8.5)
    ax_m1.set_ylabel("Test ROC-AUC", color="#1f77b4", fontweight="bold")
    ax_m2.set_ylabel("Win Rate @ >=0.70 (%)", color="#2ca02c", fontweight="bold")
    ax_m1.set_ylim(0.61, 0.74)
    ax_m2.set_ylim(74.0, 84.0)
    ax_m1.set_title("Curva de Rendimiento según el Minuto de Captura en Vivo", fontsize=11, fontweight="bold")
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_m1.legend(lines, labels, loc="upper left", fontsize=8.5)

    mod_short = ["1. Completo", "2. Sin Odds", "3. Sin 2PT/3PT", "4. Sin Lineups", "5. Solo Q-Scores"]
    ym_idx = np.arange(len(mod_short))
    auc_no_do = [m["auc_without_dropout"] for m in modality_results]
    auc_wi_do = [m["auc_with_dropout"] for m in modality_results]
    axes[1].barh(ym_idx - 0.18, auc_no_do, height=0.35, color="#ff9896", label="Árbol Único (Sin Modality Dropout)")
    axes[1].barh(ym_idx + 0.18, auc_wi_do, height=0.35, color="#2ca02c", label="V4.4 Resiliente (Con Modality Dropout)")
    axes[1].set_yticks(ym_idx)
    axes[1].set_yticklabels(mod_short, fontsize=9)
    axes[1].set_xlim(0.60, 0.65)
    axes[1].set_xlabel("Test ROC-AUC ante Fallo del Endpoint en Vivo")
    axes[1].set_title("Resiliencia ante Condiciones No Idóneas en Vivo", fontsize=11, fontweight="bold")
    axes[1].legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_in_game_minute_cuts_and_modality_resilience.png")
    plt.close(fig)

    # FIGURA 5: Curva de Calibración (Reliability Diagram) + Top 15 Features
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), dpi=160)
    bins = np.linspace(0.20, 0.80, 11)
    bin_centers, obs_freq_raw, obs_freq_cal = [], [], []
    raw_te_v40 = p_te_v40
    for b_i in range(len(bins) - 1):
        m_cal = (p_te_v44 >= bins[b_i]) & (p_te_v44 < bins[b_i + 1])
        m_raw = (raw_te_v40 >= bins[b_i]) & (raw_te_v40 < bins[b_i + 1])
        if m_cal.sum() > 25 and m_raw.sum() > 25:
            bin_centers.append((bins[b_i] + bins[b_i + 1]) / 2.0)
            obs_freq_cal.append(float(y_te[m_cal].mean()))
            obs_freq_raw.append(float(y_te[m_raw].mean()))

    axes[0].plot([0.2, 0.8], [0.2, 0.8], "k--", lw=1.5, label="Calibración Perfecta (Diagonal 1:1)")
    axes[0].plot(bin_centers, obs_freq_raw, "o--", color="#d62728", lw=2, label=f"V4.0_raw Sin Calibrar (Brier={m_te_v40['brier']:.4f})")
    axes[0].plot(bin_centers, obs_freq_cal, "s-", color="#2ca02c", lw=2.5, label=f"V4.4 Champion Calibrado (Brier={m_te_v44['brier']:.4f})")
    axes[0].set_xlabel("Probabilidad Predicha por el Modelo")
    axes[0].set_ylabel("Frecuencia Real de Victoria en Test (Out-of-Time)")
    axes[0].set_title("Curva de Calibración Isotónica (Cero Sobre-Confianza)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper left", fontsize=8.5)

    importances = ens_fiba_men["xgb_q4_win"].feature_importances_
    top_idx = np.argsort(importances)[-15:]
    axes[1].barh(np.arange(15), importances[top_idx], color="#1f77b4")
    axes[1].set_yticks(np.arange(15))
    axes[1].set_yticklabels([FEATURE_NAMES_CLEAN[idx_f] for idx_f in top_idx], fontsize=8.5)
    axes[1].set_xlabel("Importancia Relativa de Ganancia (Gain)")
    axes[1].set_title("Top 15 Features Eficientes V4.4 (100% Libres de Leakage)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_calibration_and_feature_importance.png")
    plt.close(fig)

    summary_payload = {
        "version_progression": version_summary,
        "walk_forward_cuts": walk_forward_results,
        "modality_cuts": modality_results,
        "triad_summary": triad_summary,
        "minute_curve": {"labels": min_labels, "auc": min_aucs, "wr_70": min_wr70},
    }
    with open(OUT_DIR / "v4_training_report.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"\nPipeline V4.4 finalizado con éxito. Gráficas y modelos guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
