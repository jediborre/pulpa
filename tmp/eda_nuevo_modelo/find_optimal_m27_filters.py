"""
tmp/eda_nuevo_modelo/find_optimal_m27_filters.py
================================================
Propósito:
Analizar exactamente qué distingue a las apuestas GANADAS vs PERDIDAS en `P >= 0.70`
estrictamente en el minuto 27 (`m27` en FIBA 10m / `m33` en NBA 12m, cero segundos de Q4)
y encontrar la regla de filtrado pre-Q4 (calibrada en Val y verificada en Test) que
maximiza la efectividad (Win Rate 79% - 85%) sin esperar al minuto 32.
"""
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmp.eda_nuevo_modelo.iterative_training_pipeline import (
    build_archetype_records,
    build_matrix_clean,
    extract_full_dataset,
)


def main():
    uri = f"file:{ROOT / 'matches.db'}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    lg_rows = conn.execute("SELECT match_id, league FROM matches").fetchall()
    conn.close()
    mid_to_lg = {r[0]: (r[1] or "Unknown") for r in lg_rows}

    match_meta, odds_map, ts_map, gp_map, ev_snaps = extract_full_dataset()

    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        recs = build_archetype_records(arch, match_meta, odds_map, ts_map, gp_map, ev_snaps)
        snap_opt = 33 if arch == "nba_12m" else 27
        X, y_win, y_margin, y_tot, y_ft = build_matrix_clean(recs, snap_idx=1, apply_modality_dropout=False)

        n = len(y_win)
        i_tr = int(n * 0.70)
        i_va = int(n * 0.85)

        model_path = ROOT / "models" / "m27_v4" / "model_outputs" / f"model_v4_{arch}.pkl"
        with open(model_path, "rb") as f:
            art = pickle.load(f)

        w_xgb, w_hgb, w_lr = art.get("weights", (0.45, 0.40, 0.15))
        p_xgb = art["xgb_q4_win"].predict_proba(X)[:, 1]
        p_hgb = art["hgb_q4_win"].predict_proba(X)[:, 1]
        p_lr = art["lr_q4_win"].predict_proba(X)[:, 1]
        p_raw = w_xgb * p_xgb + w_hgb * p_hgb + w_lr * p_lr
        p_cal = np.clip(art["iso_q4_win"].predict(p_raw), 0.02, 0.98)
        pred_margin = art["reg_q4_margin"].predict(X)

        rows = []
        for idx in range(i_tr, n):
            mid = recs[idx]["mid"]
            lg = mid_to_lg.get(mid, "Unknown")
            pc = float(p_cal[idx])
            pick_home = pc >= 0.5
            s = 1.0 if pick_home else -1.0
            conf = pc if pick_home else (1.0 - pc)
            won = int(y_win[idx] == 1) if pick_home else int(y_win[idx] == 0)
            exp_m = s * float(pred_margin[idx])
            real_m = s * float(y_margin[idx])

            # Acuerdo entre submodelos (dispersión entre XGB, HGB y LR)
            c_xgb = float(p_xgb[idx]) if pick_home else (1.0 - float(p_xgb[idx]))
            c_hgb = float(p_hgb[idx]) if pick_home else (1.0 - float(p_hgb[idx]))
            c_lr = float(p_lr[idx]) if pick_home else (1.0 - float(p_lr[idx]))
            min_submodel_conf = min(c_xgb, c_hgb, c_lr)
            submodel_std = float(np.std([c_xgb, c_hgb, c_lr]))

            rows.append({
                "split": "val" if idx < i_va else "test",
                "league": lg,
                "pick_home": int(pick_home),
                "conf": conf,
                "won": won,
                "exp_m": exp_m,
                "real_m": real_m,
                "min_sub": min_submodel_conf,
                "sub_std": submodel_std,
                "cur_d_fav": s * float(X[idx, 0]),
                "q3_eff_fav": s * float(X[idx, 5]),
                "elo_ft_fav": s * float(X[idx, 13]),
                "elo_q4_fav": s * float(X[idx, 14]),
                "q4_bayes_fav": s * float(X[idx, 17]),
                "h2h_q4_fav": s * float(X[idx, 23]),
                "rol_pm_fav": s * float(X[idx, 29]),
                "pbp_3pt_fav": s * float(X[idx, 27]),
            })

        df = pd.DataFrame(rows)
        df_val = df[df["split"] == "val"]
        df_test = df[df["split"] == "test"]

        # Calcular WR por liga en Train+Val en umbral >= 0.65 para identificar ligas verdaderamente tóxicas con buena muestra
        # Usemos Val para medir ligas tóxicas en apuestas >= 0.65
        val_65 = df_val[df_val["conf"] >= 0.65]
        lg_val = val_65.groupby("league").agg(n_b=("won", "count"), wr=("won", "mean"))
        toxic_lgs = set(lg_val[(lg_val["n_b"] >= 5) & (lg_val["wr"] < 0.60)].index)
        strong_lgs = set(lg_val[(lg_val["n_b"] >= 5) & (lg_val["wr"] >= 0.72)].index)

        print("\n" + "=" * 95)
        print(f"ARQUETIPO: {arch} (Corte estricto en m{snap_opt} Pre-Q4 | Ligas Tóxicas detectadas en Val: {len(toxic_lgs)})")
        print("=" * 95)

        strategies = [
            ("0. Base V4.4 (P >= 0.70 + ExpM >= 1.8)",
             lambda d: (d["conf"] >= 0.70) & (d["exp_m"] >= 1.8)),
            ("1. Base + Sin Ligas Tóxicas en Apuestas (Val WR@0.65 < 60%)",
             lambda d: (d["conf"] >= 0.70) & (d["exp_m"] >= 1.8) & (~d["league"].isin(toxic_lgs))),
            ("2. Filtro 1 + Unanimidad de los 3 Modelos (min(XGB,HGB,LR) >= 0.64)",
             lambda d: (d["conf"] >= 0.70) & (d["exp_m"] >= 2.2) & (d["min_sub"] >= 0.64) & (~d["league"].isin(toxic_lgs))),
            ("3. Filtro 2 + Fuerza Histórica Q4 a favor (elo_q4_fav > 0 & q4_bayes_fav >= 0)",
             lambda d: (d["conf"] >= 0.70) & (d["exp_m"] >= 2.5) & (d["min_sub"] >= 0.64) & (d["elo_q4_fav"] > 0) & (d["q4_bayes_fav"] >= 0) & (~d["league"].isin(toxic_lgs))),
            ("4. Alta Precisión m27 (P >= 0.72 + ExpM >= 3.0 + min_sub >= 0.66 + Sin Tóxicas)",
             lambda d: (d["conf"] >= 0.72) & (d["exp_m"] >= 3.0) & (d["min_sub"] >= 0.66) & (d["elo_q4_fav"] > 0) & (~d["league"].isin(toxic_lgs))),
            ("5. Ultra-Precisión m27 (P >= 0.74 + ExpM >= 3.8 + min_sub >= 0.68 + Sin Tóxicas)",
             lambda d: (d["conf"] >= 0.74) & (d["exp_m"] >= 3.8) & (d["min_sub"] >= 0.68) & (d["elo_q4_fav"] > 0) & (~d["league"].isin(toxic_lgs))),
            ("6. Francotirador Ligas Fuertes (P >= 0.72 + ExpM >= 3.0 + Ligas Fuertes Val)",
             lambda d: (d["conf"] >= 0.72) & (d["exp_m"] >= 3.0) & (d["min_sub"] >= 0.65) & (d["league"].isin(strong_lgs))),
        ]

        for name, fn in strategies:
            sv = df_val[fn(df_val)]
            st = df_test[fn(df_test)]
            wr_v = sv["won"].mean() * 100.0 if len(sv) > 0 else 0.0
            wr_t = st["won"].mean() * 100.0 if len(st) > 0 else 0.0
            rm_t = st["real_m"].mean() if len(st) > 0 else 0.0
            be_t = 100.0 / wr_t if wr_t > 0 else 99.0
            y140 = (wr_t / 100.0 * 1.40 - 1.0) * 100.0 if len(st) > 0 else 0.0
            print(
                f"  * {name:<76}\n"
                f"      -> VAL : WR={wr_v:5.2f}% ({len(sv):4d} bets) | "
                f"TEST: WR={wr_t:5.2f}% ({len(st):4d} bets) | MargenRealQ4: +{rm_t:4.2f}p | BreakEven: {be_t:.2f} | ROI@1.40: {y140:+5.2f}%"
            )


if __name__ == "__main__":
    main()
