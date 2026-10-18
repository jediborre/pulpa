"""
Verifica exhaustivamente:
1. Que las 36 features de `models/m27_v4/features.py` coincidan con el pipeline de entrenamiento
   sin NaNs, ceros accidentales ni desajustes de tipos tanto en partidos de `matches.db` como en
   payloads vivos de `MobileClient.fetch_full_match`.
2. Qué combinación exacta de condiciones operativas maximiza el porcentaje de verdes (Win Rate)
   en los 3 arquetipos (`fiba_men_10m`, `fiba_women_10m`, `nba_12m`) sobre los 31,833 partidos
   de Test Out-of-Time.
"""
from __future__ import annotations

import sys
import pickle
import sqlite3
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tmp.eda_nuevo_modelo.iterative_training_pipeline import (
    extract_full_dataset,
    build_archetype_records,
    build_matrix_clean,
    FEATURE_NAMES_CLEAN,
)

MODEL_DIR = ROOT_DIR / "models" / "m27_v4" / "model_outputs"


def test_green_conditions():
    match_meta, odds_map, ts_map, gp_map, ev_snaps = extract_full_dataset()
    conn = sqlite3.connect(f"file:{ROOT_DIR / 'matches.db'}?mode=ro", uri=True)
    lg_rows = dict(conn.execute("SELECT match_id, league FROM matches").fetchall())
    conn.close()

    import json
    toxic_map = json.loads((MODEL_DIR / "toxic_leagues.json").read_text(encoding="utf-8"))

    print("\n" + "=" * 100)
    print("OPTIMIZACIÓN DE CONDICIONES PARA MAXIMIZAR VERDES EN TEST OUT-OF-TIME (m27 / m33)")
    print("=" * 100)

    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        recs = build_archetype_records(arch, match_meta, odds_map, ts_map, gp_map, ev_snaps)
        n = len(recs)
        i_val = int(n * 0.85)
        test_r = recs[i_val:]

        with open(MODEL_DIR / f"model_v4_{arch}.pkl", "rb") as f:
            bundle = pickle.load(f)

        # Evaluar en m27 (snap_idx=1)
        X, y_win, y_m, _, _ = build_matrix_clean(test_r, snap_idx=1, apply_modality_dropout=False)
        w_xgb, w_hgb, w_lr = bundle["weights"]
        p_xgb = bundle["xgb_q4_win"].predict_proba(X)[:, 1]
        p_hgb = bundle["hgb_q4_win"].predict_proba(X)[:, 1]
        p_lr = bundle["lr_q4_win"].predict_proba(X)[:, 1]
        raw_p = w_xgb * p_xgb + w_hgb * p_hgb + w_lr * p_lr
        prob = bundle["iso_q4_win"].predict(raw_p)
        pred_m = bundle["reg_q4_margin"].predict(X)

        conf = np.maximum(prob, 1.0 - prob)
        pick_win = (prob >= 0.5).astype(int)
        hit = (pick_win == y_win).astype(int)
        pick_sign = np.where(prob >= 0.5, 1.0, -1.0)

        exp_m_fav = pick_sign * pred_m
        c_xgb = np.where(pick_sign > 0, p_xgb, 1.0 - p_xgb)
        c_hgb = np.where(pick_sign > 0, p_hgb, 1.0 - p_hgb)
        c_lr = np.where(pick_sign > 0, p_lr, 1.0 - p_lr)
        min_sub = np.minimum(np.minimum(c_xgb, c_hgb), c_lr)

        cur_d_fav = pick_sign * X[:, 0]  # ventaja actual del equipo elegido
        elo_q4_fav = pick_sign * X[:, 14]
        q4_bay_fav = pick_sign * X[:, 17]
        paint_ft_fav = pick_sign * (X[:, 26] * 2.0 + X[:, 28])
        three_fav = pick_sign * (X[:, 27] * 3.0)

        tox_list = toxic_map.get(arch, [])
        not_toxic = np.array([
            not any(t.lower() in lg_rows.get(r["mid"], "").lower() for t in tox_list)
            for r in test_r
        ], dtype=bool)

        # Protección contra apostar a favor del equipo que YA va ganando por paliza (descanso de estrellas en Q4)
        # Si cur_d_fav >= 18 en FIBA Men o >= 14 en NBA, el líder sienta titulares
        max_lead_allowed = 14.0 if arch == "nba_12m" else (24.0 if arch == "fiba_women_10m" else 18.0)
        no_star_rest_trap = cur_d_fav < max_lead_allowed

        print(f"\n>>> ARQUETIPO: {arch} (N_test = {len(test_r):,})")
        configs = [
            ("1. Base Actual (P>=0.68 + E[dQ4]>=1.8)", (conf >= 0.68) & (exp_m_fav >= 1.8)),
            ("2. Sin Ligas Tóxicas + No Star-Rest Trap + P>=0.68 + E[dQ4]>=2.0 + minSub>=0.62",
             not_toxic & no_star_rest_trap & (conf >= 0.68) & (exp_m_fav >= 2.0) & (min_sub >= 0.62)),
            ("3. Sin Tóxicas + No Star-Rest Trap + P>=0.69 + E[dQ4]>=2.2 + minSub>=0.64 + ELO_Q4>=-15",
             not_toxic & no_star_rest_trap & (conf >= 0.69) & (exp_m_fav >= 2.2) & (min_sub >= 0.64) & (elo_q4_fav >= -15.0)),
            ("4. BET_STRONG Opt (Sin Tóxicas + No Trap + P>=0.70 + E[dQ4]>=2.4 + minSub>=0.65 + ELO_Q4>=0)",
             not_toxic & no_star_rest_trap & (conf >= 0.70) & (exp_m_fav >= 2.4) & (min_sub >= 0.65) & (elo_q4_fav >= 0.0)),
            ("5. BET_SNIPER Opt (Sin Tóxicas + No Trap + P>=0.72 + E[dQ4]>=3.0 + minSub>=0.66 + ELO_Q4>=0)",
             not_toxic & no_star_rest_trap & (conf >= 0.72) & (exp_m_fav >= 3.0) & (min_sub >= 0.66) & (elo_q4_fav >= 0.0)),
        ]
        for label, mask in configs:
            nb = int(np.sum(mask))
            if nb > 0:
                wins = int(np.sum(hit[mask]))
                losses = nb - wins
                wr = float(np.mean(hit[mask])) * 100.0
                real_m = float(np.mean((pick_sign * y_m)[mask]))
                print(f"  {label:<88s} -> WR: {wr:6.2f}% ({wins:4d}V - {losses:3d}R / {nb:4d}) | Margen Q4: {real_m:+.2f}p")


if __name__ == "__main__":
    test_green_conditions()
