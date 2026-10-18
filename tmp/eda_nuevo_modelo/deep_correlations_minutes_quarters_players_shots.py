"""
tmp/eda_nuevo_modelo/deep_correlations_minutes_quarters_players_shots.py
========================================================================
Propósito:
Calcular exhaustivamente sobre `matches.db` (en modo solo lectura):
1. Correlaciones exactas (Pearson r) de CADA UNA de las 36 features (`FEATURE_NAMES_CLEAN`)
   contra los 4 targets (`q4_win`, `q4_margin`, `q4_total`, `ft_win`) por arquetipo
   (`fiba_men_10m`, `fiba_women_10m`, `nba_12m`) + matriz de colinealidad entre bloques.
2. Efectividad minuto a minuto en vivo ("si fuera live calculado"):
   - FIBA 10m: minutos 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 (Fin Q3), 31 (+1m Q4), 32 (+2m Q4).
   - NBA 12m : minutos 25, 27, 29, 31, 32, 33, 34, 35, 36 (Fin Q3), 37 (+1m Q4), 38 (+2m Q4).
3. Teoría y anatomía de cada cuarto (Q1, Q2, Q3, Q4) por liga:
   - Correlación del favorito con Q1, Q2, Q3 y Q4.
   - Autocorrelación entre cuartos (Q1->Q4, Q2->Q4, Q3->Q4, HT->Q4).
   - Volatilidad de Q4 (std), Tasa de Descanso de Titulares en Blowout (+15 pts al Q3),
     y Tasa de Remontada del Favorito (Rubber-Banding).
4. ELO Individual de Jugadores (`Player ELO` titulares vs banca, dependencia de estrella y
   volatilidad individual) + Prueba empírica en `match_events` de cuánto descansan las
   estrellas en el Q4 según la ventaja al cierre del Q3 y según la liga.
5. Extracción de Tiros (2PT, 3PT, FT) desde `match_events` y `team_statistics`:
   - Cómo se extraen, qué cobertura tienen y qué patrones/correlaciones muestran (pintura + TL
     sostenible vs regresión a la media de rachas de triples en Q3).
"""
import json
import math
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.m27_v4.features import FEATURE_NAMES_CLEAN
from tmp.eda_nuevo_modelo.iterative_training_pipeline import (
    build_archetype_records,
    build_matrix_clean,
    extract_full_dataset,
)


def compute_all_feature_correlations_and_minute_curves(match_meta, odds_map, ts_map, gp_map, ev_snaps):
    print("=" * 95)
    print("1. CORRELACIÓN EXACTA DE CADA UNA DE LAS 36 FEATURES POR ARQUETIPO Y TARGET")
    print("=" * 95)

    corr_results = {}
    recs_by_arch = {}

    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        recs = build_archetype_records(arch, match_meta, odds_map, ts_map, gp_map, ev_snaps)
        recs_by_arch[arch] = recs
        # Evaluar correlaciones en snap_idx=1 (m27 FIBA / m33 NBA) y snap_idx=6 (m32/m38 para in_q4_partial_diff)
        X, y_win, y_m, y_tot, y_ft = build_matrix_clean(recs, snap_idx=1, apply_modality_dropout=False)
        X_s6, _, _, _, _ = build_matrix_clean(recs, snap_idx=6, apply_modality_dropout=False)

        arch_corrs = []
        for j, fname in enumerate(FEATURE_NAMES_CLEAN):
            col = X[:, j] if j != 34 else X_s6[:, j]
            if np.std(col) < 1e-8:
                r_win, r_m, r_tot, r_ft = 0.0, 0.0, 0.0, 0.0
            else:
                r_win = float(np.corrcoef(col, y_win)[0, 1])
                r_m = float(np.corrcoef(col, y_m)[0, 1])
                r_tot = float(np.corrcoef(col, y_tot)[0, 1])
                r_ft = float(np.corrcoef(col, y_ft)[0, 1])
            arch_corrs.append({
                "idx": j,
                "feature": fname,
                "r_q4_win": round(r_win, 4),
                "r_q4_margin": round(r_m, 4),
                "r_q4_total": round(r_tot, 4),
                "r_ft_win": round(r_ft, 4),
            })
        corr_results[arch] = arch_corrs

    print(f"\n{'#':>2} | {'Feature':<24} | {'FIBA_M r(Q4Win)':>15} | {'FIBA_M r(Q4Mrg)':>15} | {'FIBA_W r(Q4Win)':>15} | {'NBA_12m r(Q4Win)':>16} | {'FIBA_M r(Q4Tot)':>15} | {'FIBA_M r(FTWin)':>15}")
    print("-" * 132)
    for j in range(len(FEATURE_NAMES_CLEAN)):
        fm = corr_results["fiba_men_10m"][j]
        fw = corr_results["fiba_women_10m"][j]
        nb = corr_results["nba_12m"][j]
        print(
            f"{j:2d} | {fm['feature']:<24} | {fm['r_q4_win']:+15.4f} | {fm['r_q4_margin']:+15.4f} | "
            f"{fw['r_q4_win']:+15.4f} | {nb['r_q4_win']:+16.4f} | {fm['r_q4_total']:+15.4f} | {fm['r_ft_win']:+15.4f}"
        )

    # Matriz de inter-correlación entre las 10 features principales en fiba_men_10m
    X_fm, _, _, _, _ = build_matrix_clean(recs_by_arch["fiba_men_10m"], snap_idx=1, apply_modality_dropout=False)
    key_idxs = [0, 5, 10, 13, 14, 17, 20, 26, 27, 29, 30, 31]
    print("\n--- MATRIZ DE INTER-CORRELACIÓN ENTRE FEATURES CLAVE (fiba_men_10m) ---")
    key_names = [FEATURE_NAMES_CLEAN[k][:12] for k in key_idxs]
    header = f"{'Feature':<14} | " + " | ".join(f"{n:>10}" for n in key_names)
    print(header)
    print("-" * len(header))
    for idx_a in key_idxs:
        row_s = f"{FEATURE_NAMES_CLEAN[idx_a][:14]:<14} | "
        vals = []
        for idx_b in key_idxs:
            c_ab = float(np.corrcoef(X_fm[:, idx_a], X_fm[:, idx_b])[0, 1])
            vals.append(f"{c_ab:+10.3f}")
        print(row_s + " | ".join(vals))

    return corr_results, recs_by_arch


def compute_full_live_minute_by_minute_effectiveness(recs_by_arch):
    print("\n" + "=" * 95)
    print("2. EFECTIVIDAD MINUTO A MINUTO EN VIVO (DESDE INICIO DE Q3 HASTA +2 MIN DE Q4)")
    print("=" * 95)

    # Además de los 7 snapshots pre-extraídos (m26..m32 en 10m / m31..m38 en 12m),
    # interpolamos los minutos tempranos de Q3 (m21..m25 en 10m) usando ht_d -> cur_d(m26)
    # y evaluamos el modelo campeón real en Test Out-of-Time para cada minuto.
    minute_tables = {}

    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        recs = recs_by_arch[arch]
        n = len(recs)
        i_va = int(n * 0.85)
        te_recs = recs[i_va:]

        model_path = ROOT / "models" / "m27_v4" / "model_outputs" / f"model_v4_{arch}.pkl"
        with open(model_path, "rb") as f:
            art = pickle.load(f)

        w_xgb, w_hgb, w_lr = art.get("weights", (0.45, 0.40, 0.15))

        def eval_matrix(X_mat, y_win, y_m):
            p_xgb = art["xgb_q4_win"].predict_proba(X_mat)[:, 1]
            p_hgb = art["hgb_q4_win"].predict_proba(X_mat)[:, 1]
            p_lr = art["lr_q4_win"].predict_proba(X_mat)[:, 1]
            p_raw = w_xgb * p_xgb + w_hgb * p_hgb + w_lr * p_lr
            p_cal = np.clip(art["iso_q4_win"].predict(p_raw), 0.02, 0.98)
            m_pred = art["reg_q4_margin"].predict(X_mat)

            auc = float(roc_auc_score(y_win, p_cal))
            pick_h = p_cal >= 0.5
            conf = np.where(pick_h, p_cal, 1.0 - p_cal)
            hit = np.where(pick_h, y_win == 1, y_win == 0).astype(float)
            exp_m_fav = np.where(pick_h, m_pred, -m_pred)
            real_m_fav = np.where(pick_h, y_m, -y_m)

            c_xgb = np.where(pick_h, p_xgb, 1.0 - p_xgb)
            c_hgb = np.where(pick_h, p_hgb, 1.0 - p_hgb)
            c_lr = np.where(pick_h, p_lr, 1.0 - p_lr)
            min_sub = np.minimum(np.minimum(c_xgb, c_hgb), c_lr)

            m65 = conf >= 0.65
            m68 = (conf >= 0.68) & (exp_m_fav >= 1.8)
            m70 = (conf >= 0.70) & (exp_m_fav >= 1.8)
            m72 = (conf >= 0.72) & (exp_m_fav >= 3.0) & (min_sub >= 0.66)

            return {
                "auc": round(auc, 4),
                "acc_50": round(float(hit.mean() * 100.0), 2),
                "wr_65": round(float(hit[m65].mean() * 100.0), 2) if m65.sum() else 0.0,
                "n_65": int(m65.sum()),
                "wr_68": round(float(hit[m68].mean() * 100.0), 2) if m68.sum() else 0.0,
                "n_68": int(m68.sum()),
                "wr_70": round(float(hit[m70].mean() * 100.0), 2) if m70.sum() else 0.0,
                "n_70": int(m70.sum()),
                "wr_72": round(float(hit[m72].mean() * 100.0), 2) if m72.sum() else 0.0,
                "n_72": int(m72.sum()),
                "avg_real_m_70": round(float(real_m_fav[m70].mean()), 2) if m70.sum() else 0.0,
            }

        X_s0, y_win, y_m, _, _ = build_matrix_clean(te_recs, snap_idx=0, apply_modality_dropout=False)
        rows_arch = []

        # Minutos tempranos de Q3 (frac_q3 = 0.10, 0.20, 0.30, 0.40, 0.50 -> m21..m25 en 10m; m25..m30 en 12m)
        early_fracs = [
            (0.10, "m21 (Q3 09:00 rest)" if arch != "nba_12m" else "m25 (Q3 11:00 rest)"),
            (0.20, "m22 (Q3 08:00 rest)" if arch != "nba_12m" else "m26 (Q3 10:00 rest)"),
            (0.30, "m23 (Q3 07:00 rest)" if arch != "nba_12m" else "m28 (Q3 08:00 rest)"),
            (0.40, "m24 (Q3 06:00 rest)" if arch != "nba_12m" else "m29 (Q3 07:00 rest)"),
            (0.50, "m25 (Q3 05:00 rest)" if arch != "nba_12m" else "m30 (Q3 06:00 rest)"),
        ]
        for frac, label in early_fracs:
            X_early = X_s0.copy()
            ratio = frac / 0.60
            # Interpolar cur_d desde ht_d hasta cur_d(0.60)
            ht_d = X_s0[:, 2]
            q3_part = (X_s0[:, 0] - ht_d) * ratio
            cur_d = ht_d + q3_part
            X_early[:, 0] = cur_d
            X_early[:, 1] = np.abs(cur_d)
            X_early[:, 5] = q3_part
            X_early[:, 6] = q3_part - 0.5 * ht_d
            X_early[:, 7] = X_s0[:, 7] * ratio
            X_early[:, 10] = ht_d + (X_s0[:, 10] - ht_d) * ratio
            X_early[:, 12] = frac
            X_early[:, 26] = X_s0[:, 26] * ratio
            X_early[:, 27] = X_s0[:, 27] * ratio
            X_early[:, 28] = X_s0[:, 28] * ratio
            X_early[:, 31] = X_early[:, 21] - cur_d
            X_early[:, 33] = (np.abs(cur_d) >= 22.0).astype(np.float32)
            st = eval_matrix(X_early, y_win, y_m)
            st["minute_label"] = label
            rows_arch.append(st)

        snap_labels_10m = [
            "m26 (Q3 04:00 rest)",
            "m27 (Q3 03:00 rest) ⭐",
            "m28 (Q3 02:00 rest)",
            "m29 (Q3 01:00 rest)",
            "m30 (Fin Q3 00:00)",
            "m31 (Q4 +1:00 jugado)",
            "m32 (Q4 +2:00 jugado)",
        ]
        snap_labels_12m = [
            "m31 (Q3 05:00 rest)",
            "m33 (Q3 03:00 rest) ⭐",
            "m34 (Q3 02:00 rest)",
            "m35 (Q3 01:00 rest)",
            "m36 (Fin Q3 00:00)",
            "m37 (Q4 +1:00 jugado)",
            "m38 (Q4 +2:00 jugado)",
        ]
        labels = snap_labels_12m if arch == "nba_12m" else snap_labels_10m
        for s_idx, label in enumerate(labels):
            X_s, y_w, y_mg, _, _ = build_matrix_clean(te_recs, snap_idx=s_idx, apply_modality_dropout=False)
            st = eval_matrix(X_s, y_w, y_mg)
            st["minute_label"] = label
            rows_arch.append(st)

        minute_tables[arch] = rows_arch
        print(f"\n--- CURVA DE EFECTIVIDAD POR MINUTO EN VIVO: {arch} (N_test = {len(te_recs):,}) ---")
        print(f"{'Minuto / Estado':<25} | {'ROC AUC':>7} | {'Acc Global':>10} | {'WR @0.65 (N)':>16} | {'WR @0.68 BET (N)':>18} | {'WR @0.70 STRONG (N)':>20} | {'WR @0.72 SNIPER (N)':>20}")
        print("-" * 130)
        for r in rows_arch:
            print(
                f"{r['minute_label']:<25} | {r['auc']:7.4f} | {r['acc_50']:9.2f}% | "
                f"{r['wr_65']:6.2f}% ({r['n_65']:4d}) | "
                f"{r['wr_68']:6.2f}% ({r['n_68']:4d})   | "
                f"{r['wr_70']:6.2f}% ({r['n_70']:4d})     | "
                f"{r['wr_72']:6.2f}% ({r['n_72']:4d})"
            )

    return minute_tables


def analyze_quarters_players_and_shots_by_league(recs_by_arch):
    print("\n" + "=" * 95)
    print("3. ANÁLISIS POR LIGA DE CADA CUARTO (Q1..Q4), DESCANSO DE ESTRELLAS, ELO DE JUGADORES Y TIROS")
    print("=" * 95)

    uri = f"file:{ROOT / 'matches.db'}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    cur = conn.cursor()

    # 3.1 Mapear match_id -> league y calcular dinámicas de Q1, Q2, Q3, Q4 por liga
    mid_to_lg = {r[0]: (r[1] or "Unknown") for r in cur.execute("SELECT match_id, league FROM matches").fetchall()}

    all_recs = []
    for arch, rlist in recs_by_arch.items():
        for r in rlist:
            r_copy = dict(r)
            r_copy["arch"] = arch
            r_copy["league"] = mid_to_lg.get(r["mid"], "Unknown")
            all_recs.append(r_copy)

    # 3.2 Analizar desde `player_stats` y `match_events`:
    #   a) ELO cronológico individual de jugadores (Titulares Top 5 vs Banca #6-#10, Volatilidad individual, Dependencia de Estrella)
    #   b) Tasa real de participación de las Estrellas en el Q4 (Star Q4 Active Rate) cuando hay Blowout (+15 pts al Q3) vs Partido Cerrado (<=10 pts)
    print("\n[A] Calculando ELO Cronológico Individual de Jugadores y Volatilidad sobre player_stats (1.88M filas)...")
    t0 = time.time()

    # Ordenar partidos cronológicamente
    all_recs.sort(key=lambda x: x["mid"])
    mid_order = {r["mid"]: idx for idx, r in enumerate(all_recs)}

    # Cargar player_stats: match_id, team, player_id/player_name, minutes_played, points, plus_minus, sofascore_rating, starter
    ps_cols = [c[1] for c in cur.execute("PRAGMA table_info(player_stats)").fetchall()]
    print(f"    Columnas en player_stats: {ps_cols}")

    # Cargar eventos de anotación en match_events con player_name y period para medir exactamente si las estrellas descansan en Q4!
    ev_cols = [c[1] for c in cur.execute("PRAGMA table_info(match_events)").fetchall()]
    print(f"    Columnas en match_events: {ev_cols}")

    # Extraer por partido y equipo los jugadores y su impacto
    # Para velocidad, procesamos player_stats en un solo barrido ordenado por fecha de partido
    ps_rows = cur.execute("""
        SELECT ps.match_id, ps.team, ps.player_name, ps.minutes_played, ps.points,
               ps.plus_minus, ps.sofascore_rating, ps.is_starter,
               ps.two_points_made, ps.two_points_attempted,
               ps.three_points_made, ps.three_points_attempted,
               ps.free_throws_made, ps.free_throws_attempted
        FROM player_stats ps
        JOIN matches m ON ps.match_id = m.match_id
        ORDER BY m.date ASC, ps.match_id ASC
    """).fetchall()
    print(f"    Cargadas {len(ps_rows):,} filas de player_stats en {time.time()-t0:.1f}s")

    # Agrupar por match_id
    from itertools import groupby
    player_elo = defaultdict(lambda: 1500.0)
    player_pm_hist = defaultdict(list)
    player_pts_hist = defaultdict(list)

    match_player_elo_feats = {}
    match_top_stars = {}  # match_id -> (set of top2 home players, set of top2 away players)

    # Mapear match_id -> (home_team, away_team, q4_win, q4_m, q3_end_d)
    mid_meta_quick = {}
    for r in cur.execute("SELECT match_id, home_team, away_team FROM matches").fetchall():
        mid_meta_quick[r[0]] = (r[1], r[2])
    rec_by_mid = {r["mid"]: r for r in all_recs}

    for mid, grp in groupby(ps_rows, key=lambda x: x[0]):
        if mid not in rec_by_mid:
            continue
        r_rec = rec_by_mid[mid]
        ht_name, at_name = mid_meta_quick.get(mid, ("", ""))
        rows_m = list(grp)

        h_players = []
        a_players = []
        for row in rows_m:
            _, team_s, pname, mins_s, pts, pm, rat, is_st, fg2m, fg2a, fg3m, fg3a, ftm, fta = row
            if not pname:
                continue
            # Parse minutes
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
            pm_h = player_pm_hist[p_key]
            pts_h = player_pts_hist[p_key]
            pre_pm_std = float(np.std(pm_h[-10:])) if len(pm_h) >= 3 else 5.0
            pre_pts_avg = float(np.mean(pts_h[-10:])) if len(pts_h) >= 1 else 0.0

            p_entry = {
                "p_key": p_key,
                "pname": pname,
                "mins": mins,
                "pts": pts_v,
                "pm": pm_v,
                "rat": rat_v,
                "pre_elo": pre_elo,
                "pre_pm_std": pre_pm_std,
                "pre_pts_avg": pre_pts_avg,
            }
            if team_s == ht_name or team_s == "home":
                h_players.append(p_entry)
            else:
                a_players.append(p_entry)

        if len(h_players) >= 5 and len(a_players) >= 5:
            # Ordenar por ELO/puntos pre-partido (conocidos ANTES del partido, 0 leakage)
            h_sorted_pre = sorted(h_players, key=lambda x: (x["pre_elo"], x["pre_pts_avg"]), reverse=True)
            a_sorted_pre = sorted(a_players, key=lambda x: (x["pre_elo"], x["pre_pts_avg"]), reverse=True)

            h_top5_elo = float(np.mean([p["pre_elo"] for p in h_sorted_pre[:5]]))
            a_top5_elo = float(np.mean([p["pre_elo"] for p in a_sorted_pre[:5]]))
            h_bench_elo = float(np.mean([p["pre_elo"] for p in h_sorted_pre[5:10]])) if len(h_sorted_pre) > 5 else 1500.0
            a_bench_elo = float(np.mean([p["pre_elo"] for p in a_sorted_pre[5:10]])) if len(a_sorted_pre) > 5 else 1500.0

            h_star_dep = h_sorted_pre[0]["pre_elo"] - h_top5_elo
            a_star_dep = a_sorted_pre[0]["pre_elo"] - a_top5_elo

            h_vol = float(np.mean([p["pre_pm_std"] for p in h_sorted_pre[:5]]))
            a_vol = float(np.mean([p["pre_pm_std"] for p in a_sorted_pre[:5]]))

            match_player_elo_feats[mid] = {
                "d_starter_elo": h_top5_elo - a_top5_elo,
                "d_bench_elo": h_bench_elo - a_bench_elo,
                "d_star_dep": h_star_dep - a_star_dep,
                "d_player_vol": h_vol - a_vol,
                "sum_player_vol": h_vol + a_vol,
            }
            match_top_stars[mid] = (
                {p["pname"] for p in h_sorted_pre[:2]},
                {p["pname"] for p in a_sorted_pre[:2]},
            )

        # Actualizar ELO post-partido de cada jugador según su plus_minus y minutos jugados
        ft_win_h = r_rec["ft_win"]
        for p in h_players:
            if p["mins"] > 0:
                w_m = min(1.0, p["mins"] / 28.0)
                perf = 1.0 if (p["pm"] > 2.0 or (p["pm"] == 0 and ft_win_h == 1)) else (0.0 if p["pm"] < -2.0 else 0.5)
                exp_p = 1.0 / (1.0 + 10.0 ** ((1500.0 - p["pre_elo"]) / 400.0))
                player_elo[p["p_key"]] += 12.0 * w_m * (perf - exp_p)
                player_pm_hist[p["p_key"]].append(p["pm"])
                player_pts_hist[p["p_key"]].append(p["pts"])
        for p in a_players:
            if p["mins"] > 0:
                w_m = min(1.0, p["mins"] / 28.0)
                perf = 1.0 if (p["pm"] > 2.0 or (p["pm"] == 0 and ft_win_h == 0)) else (0.0 if p["pm"] < -2.0 else 0.5)
                exp_p = 1.0 / (1.0 + 10.0 ** ((1500.0 - p["pre_elo"]) / 400.0))
                player_elo[p["p_key"]] += 12.0 * w_m * (perf - exp_p)
                player_pm_hist[p["p_key"]].append(p["pm"])
                player_pts_hist[p["p_key"]].append(p["pts"])

    # Evaluar correlaciones de las features de Player ELO con Q4 Win y Q4 Margin
    p_elo_recs = [(rec_by_mid[m], f) for m, f in match_player_elo_feats.items() if m in rec_by_mid]
    print(f"    Partidos con Player ELO calculado: {len(p_elo_recs):,}")
    if p_elo_recs:
        y_w_p = np.array([r["q4_win"] for r, _ in p_elo_recs])
        y_m_p = np.array([r["q4_m"] for r, _ in p_elo_recs])
        y_ft_p = np.array([r["ft_win"] for r, _ in p_elo_recs])
        for k in ["d_starter_elo", "d_bench_elo", "d_star_dep", "d_player_vol", "sum_player_vol"]:
            arr = np.array([f[k] for _, f in p_elo_recs])
            rw = float(np.corrcoef(arr, y_w_p)[0, 1])
            rm = float(np.corrcoef(arr, y_m_p)[0, 1])
            rft = float(np.corrcoef(arr, y_ft_p)[0, 1])
            print(f"      * {k:<18} -> r(Q4_Win) = {rw:+.4f} | r(Q4_Margin) = {rm:+.4f} | r(FT_Win) = {rft:+.4f}")

    # 3.3 Medir en `match_events` si las estrellas descansan en el Q4 cuando van ganando por paliza (+15) vs juego cerrado (<=10)
    print("\n[B] Midiendo en `match_events` el Descanso Real de las Estrellas en Q4 (Close Game vs Blowout)...")
    ev_star_rows = cur.execute("""
        SELECT match_id, quarter, team_side, player_name, points_scored
        FROM match_events
        WHERE event_type = 'score' AND player_name IS NOT NULL AND player_name != ''
    """).fetchall()

    star_pts_q13 = defaultdict(lambda: [0.0, 0.0])  # mid -> [star_pts_leader, total_pts_leader] in Q1-Q3
    star_pts_q4 = defaultdict(lambda: [0.0, 0.0])   # mid -> [star_pts_leader, total_pts_leader] in Q4

    for mid, qtr, side, pname, pts in ev_star_rows:
        if mid not in match_top_stars or mid not in rec_by_mid:
            continue
        r_rec = rec_by_mid[mid]
        leader_is_home = r_rec["q3_end_d"] >= 0
        if (leader_is_home and side != "home") or (not leader_is_home and side != "away"):
            continue
        stars_set = match_top_stars[mid][0] if leader_is_home else match_top_stars[mid][1]
        p_val = float(pts or 0.0)
        is_star = pname in stars_set
        if qtr in ("Q1", "Q2", "Q3"):
            star_pts_q13[mid][1] += p_val
            if is_star:
                star_pts_q13[mid][0] += p_val
        elif qtr == "Q4":
            star_pts_q4[mid][1] += p_val
            if is_star:
                star_pts_q4[mid][0] += p_val

    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        close_q13, close_q4, blow_q13, blow_q4 = [], [], [], []
        star_zero_q4_close, star_zero_q4_blow = [], []
        for mid, (s13, t13) in star_pts_q13.items():
            r_rec = rec_by_mid[mid]
            if r_rec["arch"] != arch or t13 < 20:
                continue
            s4, t4 = star_pts_q4.get(mid, [0.0, 0.0])
            if t4 < 6:
                continue
            lead_q3 = abs(r_rec["q3_end_d"])
            sh13 = s13 / t13 * 100.0
            sh4 = s4 / t4 * 100.0
            if lead_q3 <= 8:
                close_q13.append(sh13)
                close_q4.append(sh4)
                star_zero_q4_close.append(1.0 if s4 == 0 else 0.0)
            elif lead_q3 >= 18:
                blow_q13.append(sh13)
                blow_q4.append(sh4)
                star_zero_q4_blow.append(1.0 if s4 == 0 else 0.0)
        if close_q13 and blow_q13:
            print(
                f"  * {arch:<15} | Juego Cerrado (<=8p): Top-2 Estrellas anotan {np.mean(close_q13):.1f}% en Q1-Q3 y {np.mean(close_q4):.1f}% en Q4 (Ausencia total Q4: {np.mean(star_zero_q4_close)*100:.1f}%)\n"
                f"                    | Paliza Blowout (>=18p): Top-2 Estrellas anotan {np.mean(blow_q13):.1f}% en Q1-Q3 y CAEN a {np.mean(blow_q4):.1f}% en Q4 (Ausencia total Q4: {np.mean(star_zero_q4_blow)*100:.1f}%)"
            )

    # 3.4 Anatomía de cada cuarto (Q1, Q2, Q3, Q4) por Arquetipo y por Liga Principal
    print("\n[C] Anatomía y Teoría de Cada Cuarto (Q1, Q2, Q3, Q4) por Arquetipo y Top Ligas...")
    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        sub = [r for r in all_recs if r["arch"] == arch]
        q1 = np.array([r["q1_d"] for r in sub])
        q2 = np.array([r["q2_d"] for r in sub])
        q3 = np.array([r["q3_end_d"] - r["ht_d"] for r in sub])
        ht = np.array([r["ht_d"] for r in sub])
        q13 = np.array([r["q3_end_d"] for r in sub])
        q4 = np.array([r["q4_m"] for r in sub])
        elo = np.array([r["d_elo_ft"] for r in sub])

        # Tasa en que el líder por >=15 pts al Q3 pierde el Q4 (Descanso de titulares / Garbage time)
        mask_lead15 = np.abs(q13) >= 15
        leader_loses_q4_15 = float(np.mean(np.sign(q13[mask_lead15]) != np.sign(q4[mask_lead15])) * 100.0)
        # Tasa en que el favorito ELO (>75) que va perdiendo al Q3 remonta y gana el Q4
        mask_fav_tr = ((elo > 75) & (q13 < 0)) | ((elo < -75) & (q13 > 0))
        fav_wins_q4_tr = float(np.mean(np.sign(elo[mask_fav_tr]) == np.sign(q4[mask_fav_tr])) * 100.0)

        print(
            f"\n  >>> ARQUETIPO {arch} (N={len(sub):,}):\n"
            f"      - Correlación Fuerza ELO con cada cuarto: r(ELO, Q1)={np.corrcoef(elo,q1)[0,1]:+.4f} | r(ELO, Q2)={np.corrcoef(elo,q2)[0,1]:+.4f} | r(ELO, Q3)={np.corrcoef(elo,q3)[0,1]:+.4f} | r(ELO, Q4)={np.corrcoef(elo,q4)[0,1]:+.4f}\n"
            f"      - Autocorrelación hacia el Q4          : r(Q1, Q4)={np.corrcoef(q1,q4)[0,1]:+.4f} | r(Q2, Q4)={np.corrcoef(q2,q4)[0,1]:+.4f} | r(Q3, Q4)={np.corrcoef(q3,q4)[0,1]:+.4f} | r(AcumQ3, Q4)={np.corrcoef(q13,q4)[0,1]:+.4f}\n"
            f"      - Volatilidad std por cuarto (pts)     : std(Q1)={np.std(q1):.2f} | std(Q2)={np.std(q2):.2f} | std(Q3)={np.std(q3):.2f} | std(Q4)={np.std(q4):.2f}\n"
            f"      - Líder +15 pts al Q3 PIERDE el Q4 (Efecto Descanso Titulares): {leader_loses_q4_15:.1f}%\n"
            f"      - Favorito ELO perdiendo al Q3 GANA el Q4 (Efecto Remontada)  : {fav_wins_q4_tr:.1f}%"
        )

    # Tabla por Liga específica (Top 18 ligas más representativas + ligas tóxicas)
    print("\n  >>> COMPORTAMIENTO DE CUARTOS (Q1..Q4), DESCANSO DE TITULARES Y VOLATILIDAD POR LIGA:")
    print(f"  {'Liga':<42} | {'N':>5} | {'r(ELO,Q1)':>9} | {'r(ELO,Q3)':>9} | {'r(ELO,Q4)':>9} | {'r(Q13,Q4)':>9} | {'Líder+15 PierdeQ4':>17} | {'FavPerd GanaQ4':>14} | {'std(Q4)':>7}")
    print("  " + "-" * 138)
    by_lg = defaultdict(list)
    for r in all_recs:
        by_lg[r["league"]].append(r)

    target_lgs = sorted(by_lg.keys(), key=lambda k: -len(by_lg[k]))[:22]
    for extra in ["Euroleague: regular season", "China CBA", "Austrian Basketball Superliga", "PGE Basket Liga", "Premijer ženska liga", "Spain - Liga ACB", "Italy - Lega A", "Turkey - BSL", "Germany - BBL", "France - Pro A"]:
        for k in by_lg:
            if extra.lower() in k.lower() and k not in target_lgs:
                target_lgs.append(k)

    for lg in target_lgs[:28]:
        sub = by_lg[lg]
        if len(sub) < 60:
            continue
        q1 = np.array([r["q1_d"] for r in sub])
        q3 = np.array([r["q3_end_d"] - r["ht_d"] for r in sub])
        q13 = np.array([r["q3_end_d"] for r in sub])
        q4 = np.array([r["q4_m"] for r in sub])
        elo = np.array([r["d_elo_ft"] for r in sub])
        m15 = np.abs(q13) >= 15
        lose15 = float(np.mean(np.sign(q13[m15]) != np.sign(q4[m15])) * 100.0) if m15.sum() >= 5 else np.nan
        m_ft = ((elo > 60) & (q13 < 0)) | ((elo < -60) & (q13 > 0))
        fav_cb = float(np.mean(np.sign(elo[m_ft]) == np.sign(q4[m_ft])) * 100.0) if m_ft.sum() >= 5 else np.nan
        print(
            f"  {lg[:42]:<42} | {len(sub):5d} | {np.corrcoef(elo,q1)[0,1]:+9.3f} | {np.corrcoef(elo,q3)[0,1]:+9.3f} | "
            f"{np.corrcoef(elo,q4)[0,1]:+9.3f} | {np.corrcoef(q13,q4)[0,1]:+9.3f} | "
            f"{lose15:16.1f}% | {fav_cb:13.1f}% | {np.std(q4):7.2f}"
        )

    # 3.5 Patrones y Correlaciones de Tiros (2PT, 3PT, FT)
    print("\n[D] Patrones de Tiros (2PT vs 3PT vs FT) y Sostenibilidad hacia el Q4...")
    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        sub = [r for r in all_recs if r["arch"] == arch]
        d2 = np.array([r["pbp_2pt_d"] for r in sub])
        d3 = np.array([r["pbp_3pt_d"] for r in sub])
        dft = np.array([r["pbp_ft_d"] for r in sub])
        q13 = np.array([r["q3_end_d"] for r in sub])
        q4 = np.array([r["q4_m"] for r in sub])
        q4w = np.array([r["q4_win"] for r in sub])

        # Cuando un equipo lidera al Q3 gracias a Pintura+TL (d2+dft alto, d3 bajo) vs cuando lidera solo por lluvia de Triples (d3 alto, d2<=0)
        paint_lead = ((q13 >= 4) & (q13 <= 12) & (d2 + dft >= q13 * 0.8)) | ((q13 <= -4) & (q13 >= -12) & (d2 + dft <= q13 * 0.8))
        three_lead = ((q13 >= 4) & (q13 <= 12) & (d3 >= q13 * 0.8) & (d2 <= 0)) | ((q13 <= -4) & (q13 >= -12) & (d3 <= q13 * 0.8) & (d2 >= 0))

        wr_paint = float(np.mean(np.sign(q13[paint_lead]) == np.sign(q4[paint_lead])) * 100.0) if paint_lead.sum() else 0.0
        wr_three = float(np.mean(np.sign(q13[three_lead]) == np.sign(q4[three_lead])) * 100.0) if three_lead.sum() else 0.0

        print(
            f"  * {arch:<15} | r(2PT_Q13, Q4_Win)={np.corrcoef(d2,q4w)[0,1]:+.4f} | r(FT_Q13, Q4_Win)={np.corrcoef(dft,q4w)[0,1]:+.4f} | r(3PT_Q13, Q4_Win)={np.corrcoef(d3,q4w)[0,1]:+.4f}\n"
            f"                    | Líder Q3 (4..12p) por PINTURA+TL gana Q4: {wr_paint:5.2f}% (N={int(paint_lead.sum())}) vs Líder por RACHA DE TRIPLES gana Q4: {wr_three:5.2f}% (N={int(three_lead.sum())})"
        )

    conn.close()


def main():
    match_meta, odds_map, ts_map, gp_map, ev_snaps = extract_full_dataset()
    corr_results, recs_by_arch = compute_all_feature_correlations_and_minute_curves(
        match_meta, odds_map, ts_map, gp_map, ev_snaps
    )
    compute_full_live_minute_by_minute_effectiveness(recs_by_arch)
    analyze_quarters_players_and_shots_by_league(recs_by_arch)


if __name__ == "__main__":
    main()
