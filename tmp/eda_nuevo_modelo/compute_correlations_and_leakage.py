"""
Script: compute_correlations_and_leakage.py
Ubicación: tmp/eda_nuevo_modelo/compute_correlations_and_leakage.py
Propósito: 
  1. Demostrar cuantitativamente las 2 fugas de datos (Data Leakage) presentes en modelos/propuestas previas:
     - Leakage A: En models/m27_v3/train.py, g["dt"] (00:00:00) < sample.dt (HH:MM:00) incluía el propio partido actual
       en el historial H2H cuando time != '00:00', filtrando el ganador FT a h2h_last_home_won.
     - Leakage B: En tmp/team_strength_analysis/measure_quarter_impact.py (usado para PROPUESTA_NUEVO_MODELO_M27_V4.md),
       se usó ps.sofascore_rating y ps.points del partido actual (calculados al final del partido tras Q4) en vez del
       rolling histórico pre-partido de los jugadores.
  2. Calcular el EDA completo y la matriz de correlaciones 100% LIBRE DE LEAKAGE sobre el nuevo dataset histórico
     (227,028 partidos, 2015-2026) para todos los bloques de features:
     - Bloque ELO Dinámico (FT, Q4, Q1-Q3) + Déficit ELO en vivo al min 27
     - Bloque Cuotas Bookmaker (match_odds: prob implícita, spread proxy, Déficit del Favorito al min 27)
     - Bloque Roster Rolling Pre-Partido (Titulares, Banquillo, Plus-Minus, Profundidad, Ausencias/Bajas de titulares)
     - Bloque H2H Limpio (FT y Q4 específico)
     - Bloque Team Strength & Descanso (Win%, Form, Back-to-Back, Días de descanso)
     - Bloque Momentum & Marcador en Vivo al Min 27 (PBP 1m/2m/3m, Graph Points, Faltas/Bonus, Pérdidas, Rebotes)
"""
import sys
import sqlite3
import math
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"

def fast_metrics(x_list, y_bin_list, y_margin_list):
    x = np.asarray(x_list, dtype=float)
    yb = np.asarray(y_bin_list, dtype=float)
    ym = np.asarray(y_margin_list, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(yb) & ~np.isnan(ym)
    x = x[mask]
    yb = yb[mask]
    ym = ym[mask]
    n = len(x)
    if n < 200 or np.std(x) < 1e-12:
        return {"n": n, "r_margin": 0.0, "r_win": 0.0, "auc": 0.5, "dir_acc": 50.0}
    r_margin = float(np.corrcoef(x, ym)[0, 1])
    r_win = float(np.corrcoef(x, yb)[0, 1])
    try:
        auc = float(roc_auc_score(yb.astype(int), x))
        if auc < 0.5:
            auc_eff = 1.0 - auc
        else:
            auc_eff = auc
    except Exception:
        auc = 0.5
        auc_eff = 0.5
    # Directional accuracy for non-zero x
    nz = x != 0
    if np.sum(nz) > 100:
        if r_win >= 0:
            hits = np.sum((x[nz] > 0) == (yb[nz] == 1))
        else:
            hits = np.sum((x[nz] < 0) == (yb[nz] == 1))
        dir_acc = float(hits * 100.0 / np.sum(nz))
    else:
        dir_acc = 50.0
    return {
        "n": int(n),
        "r_margin": round(r_margin, 4),
        "r_win": round(r_win, 4),
        "auc": round(auc, 4),
        "auc_abs": round(auc_eff, 4),
        "dir_acc": round(dir_acc, 2),
    }

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("1. Cargando matches y quarter_scores (227k partidos)...", flush=True)
    cur.execute("""
        SELECT m.match_id, m.date, m.time, m.league, m.home_team, m.away_team,
               m.home_score, m.away_score, m.status_type
        FROM matches m
        WHERE m.status_type = 'finished' AND m.home_score > 0 AND m.away_score > 0
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)
    matches = cur.fetchall()
    print(f"   Partidos finalizados válidos: {len(matches):,}", flush=True)

    cur.execute("SELECT match_id, quarter, home, away FROM quarter_scores")
    qs_map = defaultdict(dict)
    for mid, q, h, a in cur.fetchall():
        if h is not None and a is not None:
            qs_map[mid][q] = (int(h), int(a))

    # --- VERIFICACIÓN DE LEAKAGE A: H2H en m27_v3/train.py ---
    print("\n2. Auditando Leakage A (h2h_last_home_won en m27_v3/train.py)...", flush=True)
    # En m27_v3/train.py: g["dt"] = YYYY-MM-DD 00:00:00, sample.dt = YYYY-MM-DD HH:MM:00
    # Cuando time != '00:00', g["dt"] < sample.dt incluye el mismo partido!
    leaked_h2h = []
    clean_h2h = []
    clean_h2h_q4 = []
    y_q4_all = []
    m_q4_all = []

    pair_history = defaultdict(list) # pair -> list of (date_str, mid, ht, at, ft_h, ft_a, q1_d, q4_d)
    for mid, dt, tm, lg, ht, at, hs, aws, st in matches:
        qs = qs_map.get(mid)
        if not qs or "Q4" not in qs or "Q1" not in qs:
            continue
        q4h, q4a = qs["Q4"]
        if q4h == q4a:
            continue
        q1h, q1a = qs["Q1"]
        d_clean = dt.split(" ")[0] if dt else ""
        tm_clean = tm if tm else "00:00"
        pair = tuple(sorted([ht, at]))
        hist = pair_history[pair]

        # Clean H2H: strictly date < d_clean
        past_clean = [g for g in hist if g[0] < d_clean]
        if past_clean:
            last_g = past_clean[-1]
            is_h = (last_g[2] == ht)
            won_last = int((last_g[4] > last_g[5]) == is_h)
            clean_h2h.append(1.0 if won_last else -1.0)
            # Q4 diff historical
            q4_diffs = [(g[7] if g[2] == ht else -g[7]) for g in past_clean]
            clean_h2h_q4.append(sum(q4_diffs) / len(q4_diffs))
        else:
            clean_h2h.append(np.nan)
            clean_h2h_q4.append(np.nan)

        # Leaked H2H (como hacía m27_v3/train.py cuando time != '00:00')
        if tm_clean != "00:00":
            # El partido actual tiene fecha d_clean 00:00 < d_clean HH:MM, por lo que past[-1] es él mismo!
            leaked_h2h.append(1.0 if hs > aws else 0.0)
        else:
            if past_clean:
                last_g = past_clean[-1]
                is_h = (last_g[2] == ht)
                leaked_h2h.append(1.0 if ((last_g[4] > last_g[5]) == is_h) else 0.0)
            else:
                leaked_h2h.append(0.0)

        y_q4_all.append(1 if q4h > q4a else 0)
        m_q4_all.append(q4h - q4a)

        hist.append((d_clean, mid, ht, at, int(hs), int(aws), q1h - q1a, q4h - q4a))

    print("   -> H2H CON BUG (m27_v3 train.py):", fast_metrics(leaked_h2h, y_q4_all, m_q4_all), flush=True)
    print("   -> H2H LIMPIO (estricto < date): ", fast_metrics(clean_h2h, y_q4_all, m_q4_all), flush=True)
    print("   -> H2H Q4 Diff Promedio Limpio:  ", fast_metrics(clean_h2h_q4, y_q4_all, m_q4_all), flush=True)

    conn.close()

if __name__ == "__main__":
    main()
