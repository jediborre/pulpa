"""
tmp/eda_nuevo_modelo/league_and_filter_optimization.py
======================================================
Propósito:
1. Auditar la tabla `match_odds` en `matches.db` (modo solo lectura) para verificar qué
   mercados (`market_name`, `market_group`, `market_period`, `is_live`) entrega realmente
   la API de SofaScore.
2. Evaluar la efectividad del modelo campeón `models/m27_v4` ESTRICTAMENTE en `m27` (antes
   del Q4, sin tocar `m32`) desglosada POR LIGA en Validación y Test Out-of-Time, listando:
   - Las ligas donde el modelo tiene excelente efectividad (Élite).
   - Las ligas donde el modelo de plano anda muy mal (Ligas Tóxicas / Ruidosas).
3. Diseñar y probar filtros pre-Q4 (`m27`) calibrados en Validación (`Val`) y probados en
   Test Out-of-Time (`Test`) para elevar el Win Rate de `m27` sin depender de `m32`.
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


def audit_match_odds_table():
    print("=" * 90)
    print("1. AUDITORÍA DE MERCADOS DE MOMIOS EN matches.db (match_odds)")
    print("=" * 90)
    uri = f"file:{ROOT / 'matches.db'}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    cols = [r[1] for r in conn.execute("PRAGMA table_info(match_odds)").fetchall()]
    print(f"  Columnas de match_odds: {cols}")
    rows = conn.execute("""
        SELECT odds_type, market_name, market_period, COUNT(*) as n_rows, COUNT(DISTINCT match_id) as n_matches
        FROM match_odds
        GROUP BY odds_type, market_name, market_period
        ORDER BY n_matches DESC
        LIMIT 25
    """).fetchall()
    for o_type, m_name, m_per, nr, nm in rows:
        print(f"  - type={str(o_type):<10} | market_name={str(m_name):<24} | period={str(m_per):<12} -> {nm:6d} partidos ({nr:7d} filas)")
    conn.close()


def evaluate_leagues_and_filters():
    print("\n" + "=" * 90)
    print("2. CARGANDO DATASET CRONOLÓGICO Y MODELOS m27_v4 (ESTRICTAMENTE EN MINUTO 27 / 33 PRE-Q4)")
    print("=" * 90)

    uri = f"file:{ROOT / 'matches.db'}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=60)
    lg_rows = conn.execute("""
        SELECT m.match_id, m.league, lc.is_youth, lc.is_college, lc.competition_type, m.date
        FROM matches m
        LEFT JOIN leagues_classification lc ON m.league = lc.league
    """).fetchall()
    conn.close()

    match_info = {
        r[0]: {
            "league": r[1] or "Unknown",
            "is_youth": int(r[2] or 0),
            "is_college": int(r[3] or 0),
            "comp_type": str(r[4] or ""),
            "date": str(r[5] or ""),
        }
        for r in lg_rows
    }

    match_meta, odds_map, ts_map, gp_map, ev_snaps = extract_full_dataset()

    for arch in ["fiba_men_10m", "fiba_women_10m", "nba_12m"]:
        recs = build_archetype_records(arch, match_meta, odds_map, ts_map, gp_map, ev_snaps)
        snap_opt = 33 if arch == "nba_12m" else 27
        # snap_idx=1 corresponde exactamente a m27 (FIBA 10m) y m33 (NBA 12m), 3 minutos antes de Q4
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

        records = []
        for idx in range(i_tr, n):
            r = recs[idx]
            mid = r["mid"]
            meta = match_info.get(mid, {"league": r.get("league", "Unknown"), "is_youth": 0, "is_college": 0, "comp_type": "", "date": ""})
            pc = float(p_cal[idx])
            pick_home = pc >= 0.5
            conf = pc if pick_home else (1.0 - pc)
            won = int(y_win[idx] == 1) if pick_home else int(y_win[idx] == 0)
            exp_m = float(pred_margin[idx]) if pick_home else float(-pred_margin[idx])
            real_m = float(y_margin[idx]) if pick_home else float(-y_margin[idx])

            s_sign = 1.0 if pick_home else -1.0
            elo_fav = s_sign * float(X[idx, 13])       # d_elo_ft
            elo_q4_fav = s_sign * float(X[idx, 14])    # d_elo_q4
            roll_pm_fav = s_sign * float(X[idx, 29])   # d_rol_pm
            q3_lead_fav = s_sign * float(X[idx, 0])    # cur_d at m27
            lead_abs_val = float(X[idx, 1])            # abs(cur_d)
            has_odds = float(X[idx, 19]) > 0.5
            odds_logr_fav = s_sign * float(X[idx, 20]) # log(p_home/p_away) * sign

            records.append({
                "split": "val" if idx < i_va else "test",
                "match_id": mid,
                "date": meta["date"],
                "league": meta["league"],
                "is_youth": meta["is_youth"],
                "is_college": meta["is_college"],
                "comp_type": meta["comp_type"],
                "conf": conf,
                "won": won,
                "exp_margin": exp_m,
                "real_margin": real_m,
                "elo_fav": elo_fav,
                "elo_q4_fav": elo_q4_fav,
                "roll_pm_fav": roll_pm_fav,
                "q3_lead_fav": q3_lead_fav,
                "lead_abs": lead_abs_val,
                "has_odds": has_odds,
                "odds_logr_fav": odds_logr_fav,
            })

        df = pd.DataFrame(records)
        df_val = df[df["split"] == "val"]
        df_test = df[df["split"] == "test"]

        print("\n" + "#" * 90)
        print(f"ARQUETIPO: {arch} (Corte estricto en m{snap_opt} Pre-Q4 | Val={len(df_val):,} | Test={len(df_test):,})")
        print("#" * 90)

        lg_stats = []
        for lg, grp in df_test.groupby("league"):
            n_tot = len(grp)
            acc_all = grp["won"].mean() * 100.0
            g68 = grp[grp["conf"] >= 0.68]
            g70 = grp[(grp["conf"] >= 0.70) & (grp["exp_margin"] >= 1.8)]
            if n_tot >= 30 or len(g68) >= 8:
                lg_stats.append({
                    "league": lg,
                    "n_matches": n_tot,
                    "acc_all": acc_all,
                    "n_68": len(g68),
                    "wr_68": g68["won"].mean() * 100.0 if len(g68) > 0 else np.nan,
                    "n_70": len(g70),
                    "wr_70": g70["won"].mean() * 100.0 if len(g70) > 0 else np.nan,
                })
        df_lg = pd.DataFrame(lg_stats)
        df_lg_active = df_lg[df_lg["n_68"] >= 8].sort_values("wr_68", ascending=False)

        print(f"\n  >>> TOP 15 MEJORES LIGAS EN TEST ({arch}, con >=8 apuestas @ P>=0.68 en m{snap_opt}):")
        for _, r in df_lg_active.head(15).iterrows():
            w70_s = f"{r['wr_70']:5.1f}% ({int(r['n_70']):3d}b)" if not np.isnan(r["wr_70"]) and r["n_70"] > 0 else "  N/A     "
            print(f"    + {r['league'][:48]:<48} | Partidos:{int(r['n_matches']):4d} (Acc:{r['acc_all']:5.1f}%) | @0.68: {r['wr_68']:5.1f}% ({int(r['n_68']):3d}b) | @0.70: {w70_s}")

        print(f"\n  >>> TOP 15 PEORES LIGAS / TÓXICAS EN TEST ({arch}, con >=8 apuestas @ P>=0.68 en m{snap_opt}):")
        for _, r in df_lg_active.tail(15).sort_values("wr_68", ascending=True).iterrows():
            w70_s = f"{r['wr_70']:5.1f}% ({int(r['n_70']):3d}b)" if not np.isnan(r["wr_70"]) and r["n_70"] > 0 else "  N/A     "
            print(f"    - {r['league'][:48]:<48} | Partidos:{int(r['n_matches']):4d} (Acc:{r['acc_all']:5.1f}%) | @0.68: {r['wr_68']:5.1f}% ({int(r['n_68']):3d}b) | @0.70: {w70_s}")

        # Aprender Blacklist / Whitelist de ligas ESTRICTAMENTE en Validación (Val)
        val_lg_acc = df_val.groupby("league").agg(
            n_val=("won", "count"),
            acc_val=("won", "mean"),
        )
        bad_leagues_val = set(val_lg_acc[(val_lg_acc["n_val"] >= 15) & (val_lg_acc["acc_val"] < 0.56)].index)
        elite_leagues_val = set(val_lg_acc[(val_lg_acc["n_val"] >= 20) & (val_lg_acc["acc_val"] >= 0.59)].index)

        print(f"\n  >>> IMPACTO DE FILTROS ADICIONALES EN m{snap_opt} PRE-Q4 (CALIBRADOS EN VAL, PROBADOS EN TEST):")
        filters = [
            ("0. Base V4.4 (P >= 0.70 + Gate Margen >= 1.8)", (df_test["conf"] >= 0.70) & (df_test["exp_margin"] >= 1.8)),
            ("1. Base + Poda Ligas Tóxicas (Val<56%) y sin Youth/Friendly", (df_test["conf"] >= 0.70) & (df_test["exp_margin"] >= 1.8) & (~df_test["league"].isin(bad_leagues_val)) & (df_test["is_youth"] == 0) & (df_test["comp_type"] != "friendly")),
            ("2. Filtro 1 + Consenso Triple (ELO>0 + RollingPM>0 + Margen>=2.5)", (df_test["conf"] >= 0.70) & (df_test["exp_margin"] >= 2.5) & (df_test["elo_fav"] > 0) & (df_test["roll_pm_fav"] >= 0) & (~df_test["league"].isin(bad_leagues_val)) & (df_test["is_youth"] == 0)),
            ("3. Filtro 2 + Anti-Garbage-Time (q3_lead_fav <= 14 pts)", (df_test["conf"] >= 0.70) & (df_test["exp_margin"] >= 2.5) & (df_test["elo_fav"] > 0) & (df_test["roll_pm_fav"] >= 0) & (df_test["q3_lead_fav"] <= 14) & (~df_test["league"].isin(bad_leagues_val)) & (df_test["is_youth"] == 0)),
            ("4. Modo Élite Pre-Q4 (P >= 0.72 + Margen >= 3.0 + Anti-Blowout)", (df_test["conf"] >= 0.72) & (df_test["exp_margin"] >= 3.0) & (df_test["elo_fav"] > 0) & (df_test["q3_lead_fav"] <= 14) & (~df_test["league"].isin(bad_leagues_val)) & (df_test["is_youth"] == 0)),
            ("5. Modo Sniper Pre-Q4 (P >= 0.74 + Margen >= 3.5 + Ligas Élite)", (df_test["conf"] >= 0.74) & (df_test["exp_margin"] >= 3.5) & (df_test["elo_fav"] > 0) & (df_test["league"].isin(elite_leagues_val))),
        ]
        for fname, mask in filters:
            sub = df_test[mask]
            nb = len(sub)
            if nb == 0:
                print(f"    * {fname:<66} -> 0 apuestas")
                continue
            wr = sub["won"].mean() * 100.0
            avg_real_m = sub["real_margin"].mean()
            be_odds = 100.0 / wr if wr > 0 else 99.0
            y_140 = (wr / 100.0 * 1.40 - 1.0) * 100.0
            y_135 = (wr / 100.0 * 1.35 - 1.0) * 100.0
            print(
                f"    * {fname:<66} -> WR: {wr:5.2f}% ({nb:4d} bets) | "
                f"RealQ4: +{avg_real_m:4.2f}p | BreakEven: {be_odds:.2f} | "
                f"ROI@1.40: {y_140:+5.2f}% | ROI@1.35: {y_135:+5.2f}%"
            )


if __name__ == "__main__":
    audit_match_odds_table()
    evaluate_leagues_and_filters()
