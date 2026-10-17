"""
Análisis de Ventanas en Vivo (Minuto a Minuto: m25 a m34), Resiliencia ante Datos
Incompletos en Vivo y Auditoría de Errores (Error Analysis) en Apuestas de Alta Confianza.

Propósito:
1. Evaluar minuto a minuto (m25, m26, m27, m28, m29, m30 [Descanso Q3], m31, m32, m33)
   el rendimiento predictivo (AUC, Accuracy, Precisión @ >=0.68, >=0.70, >=0.72, Yield)
   para determinar qué ventana es más conveniente y eficiente en vivo.
2. Comparar un modelo entrenado en un minuto fijo vs un Modelo Continuo Time-Aware
   (entrenado en múltiples snapshots por partido con features normalizadas por tiempo)
   cuando en vivo la captura ocurre con jitter/desfase de tiempo.
3. Realizar Error Analysis sobre los falsos positivos de alta confianza (P >= 0.70 que pierden):
   - Efecto Garbage Time / Blowout (ventaja > 15 o > 20 pts al final del Q3 donde el favorito sienta titulares).
   - Volatilidad de triples en Q4.
   - Partidos sin cuotas pre-partido o sin historial suficiente.
4. Evaluar Resiliencia ante Condiciones No Idóneas en Vivo (Ablation / Missing Modality Test):
   - ¿Qué pasa si en vivo falta `match_odds`, `team_statistics`, `graph_points` o `lineups`?
"""
import sys
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 30:
        return float("nan")
    xm, ym = x[mask], y[mask]
    if np.std(xm) < 1e-9 or np.std(ym) < 1e-9:
        return 0.0
    return float(np.corrcoef(xm, ym)[0, 1])


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("1. Calculando ELO cronológico 2015-2026...")
    cur.execute("""
        SELECT m.match_id, m.date, m.time, m.league, m.home_team, m.away_team,
               q1.home, q1.away,
               q2.home, q2.away,
               q3.home, q3.away,
               q4.home, q4.away,
               CASE
                   WHEN lc.quarter_duration_minutes = 12 THEN 'nba_12m'
                   WHEN lc.is_women = 1 THEN 'fiba_women_10m'
                   WHEN lc.quarter_duration_minutes = 10 THEN 'fiba_men_10m'
                   ELSE 'UNCLASSIFIED'
               END
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        LEFT JOIN leagues_classification lc ON m.league = lc.league
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)
    all_matches = cur.fetchall()

    elo_ft = defaultdict(lambda: 1500.0)
    elo_q4 = defaultdict(lambda: 1500.0)
    team_q4_hist = defaultdict(list)
    team_ft_hist = defaultdict(list)

    prematch_priors = {}
    for row in all_matches:
        mid, dt, tm, lg, ht, at, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a, arch = row
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue
        fth = q1h + q2h + q3h + q4h
        fta = q1a + q2a + q3a + q4a
        ft_margin = fth - fta
        q4_margin = q4h - q4a

        r_h_ft, r_a_ft = elo_ft[ht], elo_ft[at]
        r_h_q4, r_a_q4 = elo_q4[ht], elo_q4[at]
        h_q4_l15 = np.mean(team_q4_hist[ht][-15:]) if len(team_q4_hist[ht]) >= 3 else 0.0
        a_q4_l15 = np.mean(team_q4_hist[at][-15:]) if len(team_q4_hist[at]) >= 3 else 0.0
        h_wr_l15 = np.mean([1.0 if m > 0 else 0.0 for m in team_ft_hist[ht][-15:]]) if len(team_ft_hist[ht]) >= 3 else 0.5
        a_wr_l15 = np.mean([1.0 if m > 0 else 0.0 for m in team_ft_hist[at][-15:]]) if len(team_ft_hist[at]) >= 3 else 0.5

        prematch_priors[mid] = (
            r_h_ft - r_a_ft,
            r_h_q4 - r_a_q4,
            h_q4_l15 - a_q4_l15,
            h_wr_l15 - a_wr_l15,
        )

        exp_h_ft = 1.0 / (1.0 + 10.0 ** ((r_a_ft - (r_h_ft + 35.0)) / 400.0))
        act_h_ft = 1.0 if ft_margin > 0 else (0.0 if ft_margin < 0 else 0.5)
        mov_mult = np.log(abs(ft_margin) + 1.0) * (2.2 / ((abs(r_h_ft - r_a_ft) * 0.001) + 2.2))
        elo_ft[ht] += 20.0 * mov_mult * (act_h_ft - exp_h_ft)
        elo_ft[at] += 20.0 * mov_mult * ((1.0 - act_h_ft) - (1.0 - exp_h_ft))

        exp_h_q4 = 1.0 / (1.0 + 10.0 ** ((r_a_q4 - r_h_q4) / 400.0))
        act_h_q4 = 1.0 if q4_margin > 0 else (0.0 if q4_margin < 0 else 0.5)
        elo_q4[ht] += 16.0 * (act_h_q4 - exp_h_q4)
        elo_q4[at] += 16.0 * ((1.0 - act_h_q4) - (1.0 - exp_h_q4))

        team_q4_hist[ht].append(q4_margin)
        team_q4_hist[at].append(-q4_margin)
        team_ft_hist[ht].append(ft_margin)
        team_ft_hist[at].append(-ft_margin)

    print("2. Cargando match_odds y team_strength...")
    cur.execute("""
        SELECT match_id, home_value, away_value
        FROM match_odds
        WHERE odds_type = 'Home/Away' AND market_period = 'Match'
          AND home_value > 1.0 AND away_value > 1.0
    """)
    odds_map = {}
    for mid, ho, ao in cur.fetchall():
        p_h = (1.0 / ho) / ((1.0 / ho) + (1.0 / ao))
        odds_map[mid] = (p_h, float(np.log(ao / ho)))

    cur.execute("SELECT match_id, team_name, wins, losses FROM team_strength")
    ts_raw = {}
    for mid, tname, w, l in cur.fetchall():
        w, l = w or 0, l or 0
        wp = (w / (w + l)) if (w + l) >= 3 else 0.5
        ts_raw[(mid, tname)] = wp
    ts_map = {}

    print("3. Extrayendo snapshots minuto a minuto (m25 a m34) de match_events (FIBA 10m)...")
    # Seleccionamos partidos fiba_men_10m que tienen match_events
    fiba_matches = [
        r for r in all_matches
        if r[14] == "fiba_men_10m" and None not in r[6:14] and r[12] != r[13]
    ]
    fiba_ids = set(r[0] for r in fiba_matches)
    print(f"   Total partidos fiba_men_10m sin empate Q4: {len(fiba_ids):,}")

    # Extraemos el marcador en segundos exactos:
    # m25=1500, m26=1560, m27=1620, m28=1680, m29=1740, m30=1800, m31=1860, m32=1920, m33=1980, m34=2040
    # Y puntos recientes (últimos 120s) en cada minuto
    # Para hacerlo ultrarrápido en SQLite, agrupamos los eventos con score entre 1380s y 2040s
    cur.execute("""
        SELECT match_id, time_seconds, home_score, away_score
        FROM match_events
        WHERE time_seconds BETWEEN 1200 AND 2040
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        ORDER BY match_id, time_seconds ASC, id ASC
    """)

    # Procesamos en streaming por match_id
    minutes_list = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34]
    sec_targets = [m * 60 for m in minutes_list]

    # Guardaremos para cada match_id: array de diffs en cada minuto m25..m34 y array de recent_2m_diff
    snaps_diff = {}
    snaps_tot = {}
    snaps_run2m = {}

    curr_mid = None
    events_buf = []

    def process_match_events(mid, evs):
        if mid not in fiba_ids or not evs:
            return
        # evs is list of (t, hs, as) sorted by t
        diffs = np.full(len(minutes_list), np.nan, dtype=np.float32)
        tots = np.full(len(minutes_list), np.nan, dtype=np.float32)
        runs = np.full(len(minutes_list), np.nan, dtype=np.float32)

        idx = 0
        n = len(evs)
        cur_hs, cur_as = None, None

        # pointer for t - 120s
        idx_prev = 0
        prev_hs, prev_as = None, None

        for m_i, target_s in enumerate(sec_targets):
            while idx < n and evs[idx][0] <= target_s:
                cur_hs, cur_as = evs[idx][1], evs[idx][2]
                idx += 1
            while idx_prev < n and evs[idx_prev][0] <= (target_s - 120):
                prev_hs, prev_as = evs[idx_prev][1], evs[idx_prev][2]
                idx_prev += 1
            if cur_hs is not None:
                diffs[m_i] = cur_hs - cur_as
                tots[m_i] = cur_hs + cur_as
                if prev_hs is not None:
                    runs[m_i] = (cur_hs - cur_as) - (prev_hs - prev_as)
                else:
                    runs[m_i] = 0.0
        if not np.isnan(diffs[2]):  # has m27
            snaps_diff[mid] = diffs
            snaps_tot[mid] = tots
            snaps_run2m[mid] = runs

    for mid, tsec, hs, a_s in cur:
        if mid != curr_mid:
            if curr_mid is not None:
                process_match_events(curr_mid, events_buf)
            curr_mid = mid
            events_buf = [(tsec, hs, a_s)]
        else:
            events_buf.append((tsec, hs, a_s))
    if curr_mid is not None:
        process_match_events(curr_mid, events_buf)

    print(f"   Partidos con curva minuto-a-minuto (m25..m34): {len(snaps_diff):,}")

    # Construimos dataset ordenado cronológicamente
    records = []
    for row in fiba_matches:
        mid, dt, tm, lg, ht, at, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a, arch = row
        if mid not in snaps_diff or mid not in prematch_priors:
            continue
        diffs = snaps_diff[mid]
        tots = snaps_tot[mid]
        runs = snaps_run2m[mid]
        # Asegurar que m30 exacto sea el fin oficial de Q3 (q1+q2+q3)
        ht_diff = (q1h + q2h) - (q1a + q2a)
        q3_end_diff = (q1h + q2h + q3h) - (q1a + q2a + q3a)
        q3_end_tot = (q1h + q2h + q3h) + (q1a + q2a + q3a)
        diffs[5] = q3_end_diff  # index 5 is m30
        tots[5] = q3_end_tot

        d_elo_ft, d_elo_q4, d_q4_l15, d_wr_l15 = prematch_priors[mid]
        has_odds = 1.0 if mid in odds_map else 0.0
        odds_prob, odds_logr = odds_map.get(mid, (np.nan, np.nan))
        ts_d_m, ts_d_q4 = ts_map.get(mid, (0.0, 0.0))

        exp_ft_margin = d_elo_ft * 0.032 + (odds_logr * 4.5 if not np.isnan(odds_logr) else d_elo_ft * 0.015)

        records.append({
            "mid": mid,
            "dt": dt,
            "lg": lg,
            "q1_diff": q1h - q1a,
            "q2_diff": q2h - q2a,
            "ht_diff": ht_diff,
            "q3_end_diff": q3_end_diff,
            "q4_margin": q4h - q4a,
            "q4_home_win": 1 if q4h > q4a else 0,
            "q4_total": q4h + q4a,
            "d_elo_ft": d_elo_ft,
            "d_elo_q4": d_elo_q4,
            "d_q4_l15": d_q4_l15,
            "d_wr_l15": d_wr_l15,
            "has_odds": has_odds,
            "odds_prob": odds_prob,
            "odds_logr": odds_logr,
            "ts_d_m": ts_d_m,
            "ts_d_q4": ts_d_q4,
            "exp_ft_margin": exp_ft_margin,
            "diffs": diffs,
            "tots": tots,
            "runs": runs,
        })

    N = len(records)
    n_train = int(N * 0.70)
    n_val = int(N * 0.85)
    print(f"   Dataset final para análisis de ventanas: N={N:,} (Train={n_train:,}, Val={n_val-n_train:,}, Test={N-n_val:,})")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss

    def build_features_for_min(rec_list, m_idx):
        minute_val = minutes_list[m_idx]
        X = np.zeros((len(rec_list), 16), dtype=np.float32)
        for i, r in enumerate(rec_list):
            cur_diff = r["diffs"][m_idx]
            cur_tot = r["tots"][m_idx]
            cur_run = r["runs"][m_idx]
            # Cuanto del partido va jugado en el Q3/Q4:
            q3_partial_diff = cur_diff - r["ht_diff"]
            # Si minute_val > 30, ya empezó el Q4: cuántos puntos van en el Q4 hasta este minuto
            in_q4_elapsed = max(0, minute_val - 30)
            q4_partial_diff = (cur_diff - r["q3_end_diff"]) if minute_val > 30 else 0.0
            eff_q3_diff = r["q3_end_diff"] - r["ht_diff"] if minute_val >= 30 else q3_partial_diff

            X[i, 0] = cur_diff
            X[i, 1] = abs(cur_diff)
            X[i, 2] = r["ht_diff"]
            X[i, 3] = eff_q3_diff
            X[i, 4] = eff_q3_diff - (r["ht_diff"] / 2.0)  # q3 momentum vs HT
            X[i, 5] = cur_run
            X[i, 6] = cur_tot / max(1.0, float(minute_val))  # pace pts/min
            X[i, 7] = r["d_elo_ft"]
            X[i, 8] = r["d_elo_q4"]
            X[i, 9] = r["d_q4_l15"]
            X[i, 10] = r["d_wr_l15"]
            X[i, 11] = r["odds_logr"] if not np.isnan(r["odds_logr"]) else 0.0
            X[i, 12] = r["has_odds"]
            X[i, 13] = r["exp_ft_margin"] - cur_diff  # Rubber-band gap
            X[i, 14] = float(minute_val)
            X[i, 15] = q4_partial_diff
        return X

    y_all = np.array([r["q4_home_win"] for r in records], dtype=np.int32)
    y_train, y_val, y_test = y_all[:n_train], y_all[n_train:n_val], y_all[n_val:]

    print("\n" + "=" * 95)
    print("A. BENCHMARK MINUTO A MINUTO (Min 25 a Min 34 FIBA 10m) — ¿Qué ventana es mejor en vivo?")
    print("=" * 95)
    print(f"{'Minuto':<18} | {'Test AUC':>8} | {'Acc':>6} | {'WR >=0.68 (N)':>16} | {'WR >=0.70 (N)':>16} | {'WR >=0.72 (N)':>16} | {'Yield @0.70':>11}")
    print("-" * 105)

    window_results = {}
    models_by_min = {}
    for m_idx, m_val in enumerate(minutes_list):
        X_m = build_features_for_min(records, m_idx)
        clf = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=80, random_state=42
        )
        clf.fit(X_m[:n_train], y_train)
        models_by_min[m_val] = clf
        p_test = clf.predict_proba(X_m[n_val:])[:, 1]
        auc = roc_auc_score(y_test, p_test)
        acc = accuracy_score(y_test, (p_test >= 0.5).astype(int))
        conf = np.maximum(p_test, 1.0 - p_test)
        pick = (p_test >= 0.5).astype(int)
        hit = (pick == y_test).astype(int)

        stats_t = {}
        for t in [0.68, 0.70, 0.72]:
            mask = conf >= t
            cnt = int(mask.sum())
            wr = float(hit[mask].mean()) if cnt > 0 else 0.0
            yld = (wr * 1.40 - 1.0) * 100.0 if cnt > 0 else 0.0
            stats_t[t] = (wr, cnt, yld)

        label = f"Min {m_val}"
        if m_val == 27:
            label += " (3m Q3)"
        elif m_val == 30:
            label += " (FIN Q3)"
        elif m_val > 30:
            label += f" (+{m_val-30}m Q4)"

        print(
            f"{label:<18} | {auc:>8.4f} | {acc*100:>5.2f}% | "
            f"{stats_t[0.68][0]*100:>5.1f}% ({stats_t[0.68][1]:>4}) | "
            f"{stats_t[0.70][0]*100:>5.1f}% ({stats_t[0.70][1]:>4}) | "
            f"{stats_t[0.72][0]*100:>5.1f}% ({stats_t[0.72][1]:>4}) | "
            f"{stats_t[0.70][2]:>+9.2f}%"
        )
        window_results[m_val] = {
            "auc": float(auc),
            "acc": float(acc),
            "t68_wr": stats_t[0.68][0], "t68_n": stats_t[0.68][1],
            "t70_wr": stats_t[0.70][0], "t70_n": stats_t[0.70][1], "t70_yield": stats_t[0.70][2],
            "t72_wr": stats_t[0.72][0], "t72_n": stats_t[0.72][1],
        }

    # B. RESILIENCIA A JITTER TEMPORAL: Modelo de Minuto Fijo (m27) vs Modelo Multi-Snapshot Continuo (m26..m30)
    print("\n" + "=" * 95)
    print("B. RESILIENCIA TEMPORAL EN VIVO: Modelo Fijo (solo m27) vs Modelo Multi-Snapshot (m26..m30)")
    print("=" * 95)
    # Entrenamos un modelo Multi-Snapshot sobre m26, m27, m28, m29, m30 del set de train
    X_multi_tr = np.vstack([build_features_for_min(records[:n_train], idx) for idx in [1, 2, 3, 4, 5]])
    y_multi_tr = np.tile(y_train, 5)
    clf_multi = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=100, random_state=42
    )
    clf_multi.fit(X_multi_tr, y_multi_tr)

    # También entrenamos un modelo Fase 2 (Early Q4: m30, m31, m32)
    X_q4_tr = np.vstack([build_features_for_min(records[:n_train], idx) for idx in [5, 6, 7]])
    y_q4_tr = np.tile(y_train, 3)
    clf_q4_multi = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=100, random_state=42
    )
    clf_q4_multi.fit(X_q4_tr, y_q4_tr)

    for idx, m_val in [(1, 26), (2, 27), (3, 28), (4, 29), (5, 30)]:
        X_te = build_features_for_min(records[n_val:], idx)
        # si usamos el modelo fijo m27 sin saber que cambió el minuto (o pasándole el estado actual)
        p_fixed = models_by_min[27].predict_proba(X_te)[:, 1]
        p_multi = clf_multi.predict_proba(X_te)[:, 1]
        auc_f = roc_auc_score(y_test, p_fixed)
        auc_m = roc_auc_score(y_test, p_multi)
        print(f"Evaluado en Min {m_val}: Modelo Fijo m27 AUC={auc_f:.4f} vs Modelo Multi-Snapshot (m26..m30) AUC={auc_m:.4f} (Delta={auc_m-auc_f:+.4f})")

    # C. ANÁLISIS DE ERRORES (ERROR ANALYSIS) EN APUESTAS DE ALTA CONFIANZA
    print("\n" + "=" * 95)
    print("C. AUDITORÍA DE ERRORES (ERROR ANALYSIS) — ¿Por qué fallan las señales de alta confianza?")
    print("=" * 95)
    X_te_m30 = build_features_for_min(records[n_val:], 5)  # m30
    p_te_m30 = clf_multi.predict_proba(X_te_m30)[:, 1]
    conf_m30 = np.maximum(p_te_m30, 1.0 - p_te_m30)
    pick_m30 = (p_te_m30 >= 0.5).astype(int)
    hit_m30 = (pick_m30 == y_test).astype(int)
    test_recs = records[n_val:]

    # 1. Rendimiento según el margen acumulado al final del Q3 (Blowout / Garbage Time analysis)
    print("\n1) Impacto del Margen al final de Q3 (Efecto Garbage Time / Rotación de Banca):")
    brackets = [
        ("Partido Cerrado (|diff| 0-5 pts)", 0, 5),
        ("Ventaja Moderada (|diff| 6-11 pts)", 6, 11),
        ("Ventaja Amplia (|diff| 12-18 pts)", 12, 18),
        ("Paliza Moderada (|diff| 19-25 pts)", 19, 25),
        ("Blowout / Garbage Time (|diff| > 25 pts)", 26, 200),
    ]
    for b_name, lo, hi in brackets:
        abs_d = np.abs(X_te_m30[:, 0])
        b_mask = (abs_d >= lo) & (abs_d <= hi)
        hc_mask = b_mask & (conf_m30 >= 0.68)
        auc_b = roc_auc_score(y_test[b_mask], p_te_m30[b_mask]) if b_mask.sum() > 50 else float("nan")
        wr_hc = hit_m30[hc_mask].mean() * 100.0 if hc_mask.sum() > 0 else 0.0
        # ¿Quién gana el Q4 cuando el líder va ganando por ese margen?
        leader_is_home = X_te_m30[:, 0] > 0
        leader_wins_q4 = np.where(leader_is_home, y_test == 1, y_test == 0)
        leader_q4_wr = leader_wins_q4[b_mask].mean() * 100.0 if b_mask.sum() > 0 else 0.0
        print(
            f"   {b_name:<42}: N={b_mask.sum():>4} | AUC={auc_b:.4f} | "
            f"WR @>=0.68: {wr_hc:>5.1f}% (n={hc_mask.sum():>3}) | Líder Q3 gana Q4: {leader_q4_wr:>5.1f}%"
        )

    # 2. ¿Cuándo apuesta el modelo al que va PERDIENDO (Rubber-Band) vs al que va GANANDO?
    pick_is_trailing = ((pick_m30 == 1) & (X_te_m30[:, 0] < 0)) | ((pick_m30 == 0) & (X_te_m30[:, 0] > 0))
    hc = conf_m30 >= 0.68
    hc_trail = hc & pick_is_trailing
    hc_lead = hc & (~pick_is_trailing)
    print(f"\n2) Desglose por Dirección táctica en Confianza >= 0.68:")
    print(f"   - Apostando al FAVORITO QUE VA PERDIENDO (Rubber-Band): WR = {hit_m30[hc_trail].mean()*100:.2f}% (N={hc_trail.sum()})")
    print(f"   - Apostando al FAVORITO QUE VA GANANDO (Dominancia):    WR = {hit_m30[hc_lead].mean()*100:.2f}% (N={hc_lead.sum()})")

    # 3. Impacto de disponibilidad de Cuotas Pre-partido (has_odds)
    with_odds = (X_te_m30[:, 12] == 1.0) & (conf_m30 >= 0.68)
    no_odds = (X_te_m30[:, 12] == 0.0) & (conf_m30 >= 0.68)
    print(f"\n3) Resiliencia cuando FALTAN cuotas pre-partido (match_odds):")
    print(f"   - Con match_odds disponible: WR @>=0.68 = {hit_m30[with_odds].mean()*100:.2f}% (N={with_odds.sum()}, AUC total={roc_auc_score(y_test[X_te_m30[:,12]==1], p_te_m30[X_te_m30[:,12]==1]):.4f})")
    print(f"   - Sin match_odds (solo ELO): WR @>=0.68 = {hit_m30[no_odds].mean()*100:.2f}% (N={no_odds.sum()}, AUC total={roc_auc_score(y_test[X_te_m30[:,12]==0], p_te_m30[X_te_m30[:,12]==0]):.4f})")

    # 4. Prueba de Degradación (Simulando caída de endpoints en vivo en el Test Set)
    print("\n4) Simulación de Fallo de Endpoints en Vivo (Modality Dropout Evaluation):")
    # ¿Qué pasa si un partido que tenía cuotas pierde el endpoint de cuotas en vivo?
    X_te_no_odds = X_te_m30.copy()
    X_te_no_odds[:, 11] = 0.0
    X_te_no_odds[:, 12] = 0.0
    X_te_no_odds[:, 13] = X_te_no_odds[:, 7] * 0.047 - X_te_no_odds[:, 0]  # fallback exp_ft_margin solo con ELO
    p_drop_odds = clf_multi.predict_proba(X_te_no_odds)[:, 1]
    print(f"   - Modelo normal en Test Set:                AUC = {roc_auc_score(y_test, p_te_m30):.4f}")
    print(f"   - Si cae endpoint de Odds en vivo (Fallback ELO): AUC = {roc_auc_score(y_test, p_drop_odds):.4f}")

    X_te_no_pbp_run = X_te_m30.copy()
    X_te_no_pbp_run[:, 5] = 0.0  # sin recent_2m_run (solo marcador por cuartos)
    p_drop_pbp = clf_multi.predict_proba(X_te_no_pbp_run)[:, 1]
    print(f"   - Si cae PBP detallado (solo Quarter Scores):     AUC = {roc_auc_score(y_test, p_drop_pbp):.4f}")

    # 5. Análisis de los errores específicos en Confianza >= 0.70 (¿Qué pasó en el Q4 cuando perdimos?)
    hc70_losses = (conf_m30 >= 0.70) & (hit_m30 == 0)
    hc70_wins = (conf_m30 >= 0.70) & (hit_m30 == 1)
    q4_margins_losses = np.array([ abs(test_recs[i]["q4_margin"]) for i in range(len(test_recs)) if hc70_losses[i] ])
    near_miss_pct = np.mean(q4_margins_losses <= 3) * 100.0 if len(q4_margins_losses) > 0 else 0.0
    near_miss_5_pct = np.mean(q4_margins_losses <= 5) * 100.0 if len(q4_margins_losses) > 0 else 0.0
    print(f"\n5) Anatomía de las derrotas en Confianza >= 0.70 (Total derrotas: {hc70_losses.sum()} de {(conf_m30>=0.70).sum()} apuestas):")
    print(f"   - Derrotas por 1 a 3 puntos (1 posesión / triple final en Q4): {near_miss_pct:.1f}% de las derrotas")
    print(f"   - Derrotas por <= 5 puntos en Q4:                              {near_miss_5_pct:.1f}% de las derrotas")
    print(f"   - Margen medio de derrota en Q4 cuando falla:                  {np.mean(q4_margins_losses):.2f} pts")

    conn.close()


if __name__ == "__main__":
    main()
