# =====================================================================
# v5_4 (F1/F2/F3): ¿APORTAN LAS FEATURES DE ESTADO Y DE "MOTIVACION"?
# ---------------------------------------------------------------------
# FEATURES A PROBAR (sin leakage: solo con lo que se sabe en el snapshot):
#   F2  q_won_home / q_won_away   : cuantos cuartos COMPLETADOS lleva ganados
#                                   cada equipo en el momento del snapshot
#   F2  sweep_home / sweep_away   : ese equipo gano TODOS los cuartos
#                                   completados (2-0 al min 27, 3-0 al 30)
#   F3  lead_extreme              : la ventaja esta en el extremo de la
#                                   distribucion de ESA liga (percentil 90)
#   F1  prior_estado              : P(local gana Q4 | liga x bucket de
#                                   ventaja), estimado SOLO en el bloque A y
#                                   encogido hacia la base de la liga
#
# COMPARA: V0 (features actuales) vs V1 (+F2/F3) vs V2 (+F1).
# Mide en el bloque C (test fuera de muestra): AUC, Brier, LogLoss, tabla de
# umbrales (apuestas, WR, ROI@1.40, ROI@1.83).
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/v54_features_experiment.py
# =====================================================================

import json
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

from models.m27_v5_3.features import mirror  # noqa: E402
from train_m27_v5_3_pipeline import (  # noqa: E402
    fit_calibrator, fit_ensemble, green, margin_sym, metrics, p_sym, roi, tier_table,
)

W = {"xgb": 0.40, "hgb": 0.35, "lr": 0.25}
THR = 0.75
OUT = ROOT / "tmp" / "m27_v5_3_audit" / "v54_features_experiment.json"


def block(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d), float((c + r) / d)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", help="V0 | V1 | V2 | all")
    args = ap.parse_args()

    print("Cargando dataset y cuartos...")
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
    anchor = np.array([m["anchor"] for m in meta])
    lg = np.array([m["lg"] for m in meta])
    mid = np.array([m["mid"] for m in meta])
    tt = np.array([float(m["t"]) for m in meta])

    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row
    quarters = {}
    for r in conn.execute(
        "SELECT match_id, q1_home, q1_away, q2_home, q2_away, q3_home, q3_away "
        "FROM quarter_scores_v2 WHERE q1_home IS NOT NULL AND q2_home IS NOT NULL "
        "AND q3_home IS NOT NULL"
    ):
        quarters[r["match_id"]] = (r["q1_home"], r["q1_away"], r["q2_home"], r["q2_away"],
                                   r["q3_home"], r["q3_away"])
    conn.close()
    print(f"  cuartos disponibles para {len(quarters):,} partidos")

    # ------------------------------------------------------------------
    # Features de estado (solo informacion disponible en el snapshot)
    # ------------------------------------------------------------------
    n_rows = len(meta)
    q_won_home = np.zeros(n_rows, dtype=np.float32)
    q_won_away = np.zeros(n_rows, dtype=np.float32)
    sweep_home = np.zeros(n_rows, dtype=np.float32)
    sweep_away = np.zeros(n_rows, dtype=np.float32)
    n_done_arr = np.zeros(n_rows, dtype=np.float32)
    missing = 0
    for i in range(n_rows):
        q = quarters.get(mid[i])
        if q is None:
            missing += 1
            continue
        t = tt[i]
        n_done = min(3, int(t // 10))  # cuartos COMPLETADOS en el snapshot
        n_done_arr[i] = n_done
        if n_done <= 0:
            continue
        won = 0
        for k in range(n_done):
            h, a = q[2 * k], q[2 * k + 1]
            if h is None or a is None:
                break
            if h > a:
                won += 1
        q_won_home[i] = won
        q_won_away[i] = n_done - won
        if n_done >= 2:
            sweep_home[i] = 1.0 if won == n_done else 0.0
            sweep_away[i] = 1.0 if won == 0 else 0.0
    print(f"  filas sin cuartos: {missing:,} ({missing * 100 / n_rows:.1f} %)")

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowA = (mrank >= 0) & (mrank < a_end)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end
    vals, inv = np.unique(mrank, return_inverse=True)
    cnt = np.bincount(inv)[inv]
    w_all = 1.0 / cnt

    # ------------------------------------------------------------------
    # F3: umbral de ventaja extrema POR LIGA (percentil 90 del |lead|)
    #    (estimado en el bloque A; el lead del snapshot es X[:, 1])
    # ------------------------------------------------------------------
    lead = X[:, 1].astype(np.float64)
    thr_lead = {}
    by_lg = defaultdict(list)
    for i in np.where(rowA & anchor)[0]:
        by_lg[lg[i]].append(abs(lead[i]))
    for l, v in by_lg.items():
        thr_lead[l] = float(np.percentile(np.array(v), 90)) if len(v) >= 40 else None
    default_thr = float(np.percentile(np.abs(lead[rowA & anchor]), 90))
    extreme = np.array([
        1.0 if (thr_lead.get(lg[i]) is not None and abs(lead[i]) >= thr_lead[lg[i]])
        else (1.0 if (thr_lead.get(lg[i]) is None and abs(lead[i]) >= default_thr) else 0.0)
        for i in range(n_rows)], dtype=np.float32)

    # ------------------------------------------------------------------
    # F1: prior P(local gana Q4 | liga x bucket de ventaja), encogido
    # ------------------------------------------------------------------
    # bucket de ventaja: quintiles del lead dentro de cada liga (en el ancla)
    lead_q = {}
    buckets = np.zeros(n_rows, dtype=np.int8)
    lead_by_lg = defaultdict(list)
    for i in np.where(rowA & anchor)[0]:
        lead_by_lg[lg[i]].append(lead[i])
    for l, v in lead_by_lg.items():
        if len(v) >= 40:
            lead_q[l] = np.percentile(np.array(v), [20, 40, 60, 80])
    for i in range(n_rows):
        q = lead_q.get(lg[i])
        if q is None:
            buckets[i] = 2
        else:
            buckets[i] = int(np.searchsorted(q, lead[i]))
    # tabla en el bloque A (anclas)
    cnt_cell = defaultdict(int)
    win_cell = defaultdict(int)
    cnt_lg = defaultdict(int)
    win_lg = defaultdict(int)
    for i in np.where(rowA & anchor)[0]:
        cell = (lg[i], int(buckets[i]))
        cnt_cell[cell] += 1
        win_cell[cell] += int(y[i] == 1)
        cnt_lg[lg[i]] += 1
        win_lg[lg[i]] += int(y[i] == 1)
    base_p = float(y[rowA & anchor].mean())
    PRIOR_K = 30.0  # peso de encogimiento (muestra equivalente)
    prior = np.zeros(n_rows, dtype=np.float32)
    for i in range(n_rows):
        l = lg[i]
        p_lg = (win_lg[l] + PRIOR_K * base_p) / (cnt_lg[l] + PRIOR_K) if cnt_lg.get(l) else base_p
        cell = (l, int(buckets[i]))
        n_c = cnt_cell.get(cell, 0)
        w_c = win_cell.get(cell, 0)
        p_cell = (w_c + PRIOR_K * p_lg) / (n_c + PRIOR_K) if n_c else p_lg
        prior[i] = p_cell
    print(f"  prior por liga x bucket: {len(cnt_cell):,} celdas con datos "
          f"(liga x 5 buckets) | base {base_p * 100:.1f} %")

    EXTRA = {
        "F2/F3": np.column_stack([q_won_home, q_won_away, sweep_home, sweep_away, extreme]),
        "F1": prior.reshape(-1, 1),
    }

    # ------------------------------------------------------------------
    # Espejo extendido: las features nuevas no son todas "negar".
    #   q_won_home <-> q_won_away   (se intercambian)
    #   sweep_home  <-> sweep_away   (se intercambian)
    #   extreme                     (|ventaja|: invariante)
    #   prior                       (P(local gana Q4) -> 1 - p)
    # ------------------------------------------------------------------
    N_BASE = X.shape[1]

    def mirror_ext(Xin: np.ndarray) -> np.ndarray:
        base = mirror(Xin[:, :N_BASE])
        extra = Xin[:, N_BASE:].copy()
        if extra.shape[1] >= 4:
            a, b = extra[:, 0].copy(), extra[:, 1].copy()
            extra[:, 0], extra[:, 1] = b, a
            c, d = extra[:, 2].copy(), extra[:, 3].copy()
            extra[:, 2], extra[:, 3] = d, c
        if extra.shape[1] >= 6:
            extra[:, 5] = 1.0 - extra[:, 5]
        return np.column_stack([base, extra])

    import train_m27_v5_3_pipeline as PIPE
    PIPE.mirror = mirror_ext

    results = []
    results_path = ROOT / "tmp" / "m27_v5_3_audit" / "v54_features_partial.json"
    if results_path.exists():
        results = json.loads(results_path.read_text(encoding="utf-8"))

    def run(name: str, Xuse: np.ndarray) -> None:
        if any(r["variant"] == name for r in results):
            print(f"[{name}] ya calculado, se omite")
            return
        t0 = time.time()
        models = fit_ensemble(Xuse[rowA], y[rowA], q4d[rowA], w_all[rowA], quick=False)
        pB = p_sym(models, Xuse[rowB])
        iso = fit_calibrator(pB, y[rowB])
        pC = iso.predict(p_sym(models, Xuse[rowC]))
        mC = margin_sym(models, Xuse[rowC])
        yC = y[rowC]
        anchorC = anchor[rowC]
        mt = metrics(pC, yC)
        tiers = tier_table(pC, yC, mC, thrs=(0.70, 0.72, 0.75, 0.80))
        tiers_anchor = tier_table(pC[anchorC], yC[anchorC], mC[anchorC], thrs=(0.70, 0.72, 0.75, 0.80))
        res = {"variant": name, "n_features": int(Xuse.shape[1]),
               "auc_test": round(mt["auc"], 4), "brier_test": round(mt["brier"], 4),
               "logloss_test": round(mt["logloss"], 4), "acc_test": round(mt["acc"], 4),
               "tiers": tiers, "tiers_anchor": tiers_anchor, "segundos": round(time.time() - t0)}
        results.append(res)
        results_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[{name}] ({Xuse.shape[1]} features) en {res['segundos']}s")
        print(f"   AUC {mt['auc']:.4f} | Brier {mt['brier']:.4f} | LogLoss {mt['logloss']:.4f} "
              f"| Acc {mt['acc'] * 100:.2f} %")
        for t in tiers_anchor:
            print(f"      ANCLA thr {t['thr']:.2f}: apuestas {t['bets']:>6,} | WR {t['wr']:>5.2f}% "
                  f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi_140']:>+6.2f}% | ROI183 {t['roi_183']:>+6.2f}%")

    if args.only in ("all", "V0"):
        block("V0: FEATURES ACTUALES")
        run("V0 actual", X)
    if args.only in ("all", "V1"):
        block("V1: + F2 (patron de cuartos) + F3 (ventaja extrema por liga)")
        run("V1 +F2/F3", np.column_stack([X, EXTRA["F2/F3"]]))
    if args.only in ("all", "V2"):
        block("V2: + F1 (prior contextual por liga x estado)")
        run("V2 +F1", np.column_stack([X, EXTRA["F2/F3"], EXTRA["F1"]]))

    if len(results) < 3:
        print(f"\n  (calculadas {len(results)}/3 variantes; sigue con --only para las demás)")
        return

    block("COMPARATIVA")
    base = results[0]
    for r in results[1:]:
        print(f"  {r['variant']} vs V0: dAUC {r['auc_test'] - base['auc_test']:+.4f} | "
              f"dBrier {r['brier_test'] - base['brier_test']:+.4f} | "
              f"dLogLoss {r['logloss_test'] - base['logloss_test']:+.4f} | "
              f"dAcc {(r['acc_test'] - base['acc_test']) * 100:+.2f} pp")
        for t0, t1 in zip(base["tiers_anchor"], r["tiers_anchor"]):
            print(f"      thr {t0['thr']:.2f}: apuestas {t0['bets']:>6,} -> {t1['bets']:>6,} | "
                  f"WR {t0['wr']:>5.2f}% -> {t1['wr']:>5.2f}% | "
                  f"ROI140 {t0['roi_140']:>+6.2f}% -> {t1['roi_140']:>+6.2f}% | "
                  f"ROI183 {t0['roi_183']:>+6.2f}% -> {t1['roi_183']:>+6.2f}%")

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nInforme: {OUT}")


if __name__ == "__main__":
    main()
