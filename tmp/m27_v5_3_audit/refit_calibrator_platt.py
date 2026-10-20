# =====================================================================
# REFIT DEL CALIBRADOR DESPLEGADO: isotonica -> Platt (continua)
# ---------------------------------------------------------------------
# Por que es legitimo refitear SOLO el calibrador:
#   El ensamble (xgb/hgb/lr/reg) se entrena en el bloque A con semilla fija
#   (random_state=42) sobre el mismo dataset -> es determinista. El
#   calibrador se ajusta en el bloque B. Por tanto, refitear el calibrador
#   con los MISMOS datos y la MISMA regla produce exactamente el mismo
#   artefacto que haria un reentrenamiento completo.
#
# QUE HACE:
#   1. Backup del calibrador.pkl actual (isotonica).
#   2. Recalcula p_raw en el bloque B con los pesos desplegados.
#   3. Ajusta el PlattCalibrator simetrico y lo guarda.
#   4. Verifica en el bloque C: AUC, Brier, LogLoss y tabla de umbrales.
#   5. Comprueba que la inferencia desplegada sigue funcionando.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/refit_calibrator_platt.py
# =====================================================================

import json
import pickle
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

from models.common.calibration import PlattCalibrator  # noqa: E402
from models.m27_v5_3.features import mirror  # noqa: E402
from train_m27_v5_3_pipeline import p_sym  # noqa: E402

MD = ROOT / "models" / "m27_v5_3" / "model_outputs"
W = {"xgb": 0.40, "hgb": 0.35, "lr": 0.25}


def block(t: str) -> None:
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def wilson(w, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d), float((c + r) / d)


def tiers(p, y, thrs=(0.70, 0.72, 0.74, 0.75, 0.76, 0.78, 0.80, 0.85)):
    out = []
    conf = np.maximum(p, 1 - p)
    pick_home = p >= 0.5
    for t in thrs:
        b = conf >= t
        n = int(b.sum())
        if n == 0:
            continue
        w = int((pick_home[b] == (y[b] == 1)).sum())
        lo, hi = wilson(w, n)
        out.append({"thr": t, "bets": n, "wr": round(w * 100 / n, 2),
                    "wr_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
                    "roi140": round((w * 1.40 - n) / n * 100, 2),
                    "roi183": round((w * 1.83 - n) / n * 100, 2),
                    "u140": round(n * (w * 1.40 - n) / n, 1),
                    "u183": round(n * (w * 1.83 - n) / n, 1)})
    return out


def main() -> None:
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, meta = D["X"], D["y"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
    anchor = np.array([m["anchor"] for m in meta])
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowB = (mrank >= a_end) & (mrank < b_end)
    rowC = mrank >= b_end

    models = {}
    for key, fname in (("xgb", "xgb_model.pkl"), ("hgb", "hgb_model.pkl"),
                       ("lr", "lr_model.pkl"), ("reg", "reg_model.pkl"),
                       ("calibrator", "calibrator.pkl")):
        models[key] = pickle.load(open(MD / fname, "rb"))
    print(f"  calibrador actual: {models['calibrator']!r}")

    print("  recalculando p_raw en el bloque B con los pesos desplegados...")
    pB = p_sym(models, X[rowB])
    yB = y[rowB]

    idxC = np.where(rowC)[0]
    pC_raw = p_sym(models, X[idxC])
    yC = y[idxC]
    anchorC = anchor[idxC]

    block("ANTES (isotonica desplegada)")
    pC_iso = np.clip(models["calibrator"].predict(pC_raw), 0.03, 0.97)
    print(f"  valores distintos: {len(np.unique(np.round(pC_iso, 4))):,}")
    for t in tiers(pC_iso, yC):
        print(f"    thr {t['thr']:.2f}: apuestas {t['bets']:>6,} | WR {t['wr']:>5.2f}% | "
              f"ROI140 {t['roi140']:>+6.2f}% | u@1.40 {t['u140']:>6.0f} | u@1.83 {t['u183']:>6.0f}")

    block("AJUSTANDO EL PLATT SIMETRICO (mismos datos: bloque B)")
    cal = PlattCalibrator().fit(pB, yB)
    print(f"  {cal!r}")
    backup = MD / "calibrator_isotonic_backup.pkl"
    if not backup.exists():
        shutil.copy2(MD / "calibrator.pkl", backup)
        print(f"  backup: {backup.name}")
    with open(MD / "calibrator.pkl", "wb") as f:
        pickle.dump(cal, f)
    print("  calibrator.pkl actualizado")

    block("DESPUES (Platt, mismo conjunto de modelos)")
    pC_platt = np.clip(cal.predict(pC_raw), 0.03, 0.97)
    print(f"  valores distintos: {len(np.unique(np.round(pC_platt, 4))):,}")
    for t in tiers(pC_platt, yC):
        print(f"    thr {t['thr']:.2f}: apuestas {t['bets']:>6,} | WR {t['wr']:>5.2f}% "
              f"(IC95 {t['wr_ci95']}) | ROI140 {t['roi140']:>+6.2f}% | u@1.40 {t['u140']:>6.0f} | u@1.83 {t['u183']:>6.0f}")

    block("COMPARATIVA EN EL BLOQUE C COMPLETO (98.476 filas)")
    for nombre, p in (("isotonica", pC_iso), ("platt", pC_platt)):
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yC, p)
        brier = float(np.mean((p - yC) ** 2))
        ll = float(-np.mean(yC * np.log(p) + (1 - yC) * np.log(1 - p)))
        print(f"  {nombre:<12} AUC {auc:.4f} | Brier {brier:.4f} | LogLoss {ll:.4f} | "
              f"valores distintos {len(np.unique(np.round(p, 4))):,}")

    block("INFERENCIA DESPLEGADA (comprobacion de humo)")
    from models.common.data_loader import get_db_connection, load_match
    from models.m27_v5_3 import predict as predict_v53
    db = get_db_connection()
    db.row_factory = sqlite3.Row
    row = db.execute(
        "SELECT match_id FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3' "
        "AND signal_type LIKE '%BET%' ORDER BY created_at DESC LIMIT 1").fetchone()
    mid = row["match_id"]
    payload = load_match(db, mid)
    for minute in (27.0, 30.0):
        res = predict_v53(mid, match_data=payload, conn=db, snapshot_minute=minute)
        rj = res.raw_json or {}
        print(f"  min {minute}: available={res.available} signal={res.signal} "
              f"p_cal={rj.get('p_cal')} conf={res.confidence} thr={rj.get('min_conf_threshold')}")
    db.close()


if __name__ == "__main__":
    main()
