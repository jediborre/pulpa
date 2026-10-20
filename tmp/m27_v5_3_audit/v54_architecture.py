# =====================================================================
# v5_4 (arquitectura): las DOS cabezas del modelo se contradicen?
#   - clasificador: p_cal (P(local gana Q4))
#   - regresor:      margin (margen de Q4 esperado)
#   La regla de disparo exige p_cal >= 0.75 Y margin >= 0. Si las dos
#   cabezas apuntan a lados distintos, la arquitectura de dos cabezas esta
#   desperdiciando senal (y explica la meseta de la isotonica).
#   Ademas: sesgo de calibracion por categoria de fase (calibracion global
#   vs calibracion condicional).
# =====================================================================

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.competition_stage import classify_stage  # noqa: E402
from models.m27_v5_3.features import mirror  # noqa: E402

MD = ROOT / "models" / "m27_v5_3" / "model_outputs"
W = {"xgb": 0.40, "hgb": 0.35, "lr": 0.25}


def block(t: str) -> None:
    print(f"\n{'=' * 92}\n{t}\n{'=' * 92}")


def main() -> None:
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, meta = D["X"], D["y"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
    anchor = np.array([m["anchor"] for m in meta])
    lg = np.array([m["lg"] for m in meta])
    n_el = len(elig)
    b_end = int(n_el * 0.82)
    rowC = mrank >= b_end
    aC = anchor[rowC]
    idxC = np.where(rowC)[0]

    models = {}
    for key, fname in (("xgb", "xgb_model.pkl"), ("hgb", "hgb_model.pkl"),
                       ("lr", "lr_model.pkl"), ("reg", "reg_model.pkl"),
                       ("calibrator", "calibrator.pkl")):
        models[key] = pickle.load(open(MD / fname, "rb"))
    cfg = json.loads((MD / "model_config.json").read_text(encoding="utf-8"))
    thr = float(cfg["min_conf"])

    Xc = X[idxC]
    p_raw = np.empty(len(Xc))
    margin = np.empty(len(Xc))
    for i in range(0, len(Xc), 20000):
        xb = Xc[i:i + 20000]
        xm = mirror(xb)
        e1 = (W["xgb"] * models["xgb"].predict_proba(xb)[:, 1]
              + W["hgb"] * models["hgb"].predict_proba(xb)[:, 1]
              + W["lr"] * models["lr"].predict_proba(xb)[:, 1])
        e2 = (W["xgb"] * models["xgb"].predict_proba(xm)[:, 1]
              + W["hgb"] * models["hgb"].predict_proba(xm)[:, 1]
              + W["lr"] * models["lr"].predict_proba(xm)[:, 1])
        p_raw[i:i + 20000] = 0.5 * (e1 + 1.0 - e2)
        margin[i:i + 20000] = 0.5 * (models["reg"].predict(xb) - models["reg"].predict(xm))
    lo, hi = cfg["p_clip"]
    p = np.clip(models["calibrator"].predict(p_raw), lo, hi)
    yc = y[idxC]
    lgc = lg[idxC]
    ac = aC

    block("1. LAS DOS CABEZAS: ¿CLASSIFIER Y MARGEN APUNTAN AL MISMO LADO?")
    side_p = np.where(p >= 0.5, 1, -1)
    side_m = np.where(margin >= 0, 1, -1)
    dis = side_p != side_m
    print(f"  filas de C: {len(p):,} | desacuerdos: {int(dis.sum()):,} "
          f"({dis.sum() * 100 / len(p):.1f} %)")
    print(f"  en las anclas t=30: desacuerdos {int(dis[ac].sum()):,} "
          f"({dis[ac].sum() * 100 / ac.sum():.1f} %)")
    # Entre las que la regla de disparo acepta, ¿cuantas se contradicen?
    bet = (np.maximum(p, 1 - p) >= thr) & (np.abs(margin) >= 0)
    if bet.sum():
        print(f"  entre las que el modelo APUESTA (thr {thr}): {int(bet.sum()):,}, "
              f"de ellas {int((dis & bet).sum()):,} tienen las cabezas en lados "
              f"opuestos ({(dis & bet).sum() * 100 / bet.sum():.1f} %)")
        # ¿acierten mejor las que coinciden?
        for name, m in (("cabezas coinciden", bet & ~dis), ("cabezas se contradicen", bet & dis)):
            if m.sum() == 0:
                continue
            w = int(((side_p[m] > 0) == (yc[m] == 1)).sum())
            print(f"     {name:<24} n={int(m.sum()):>6,}  WR={w * 100 / m.sum():>5.1f} %")

    block("2. SESGO DE CALIBRACION POR FASE (calibracion global vs condicional)")
    cats = np.array([classify_stage(x)[0] for x in lgc])
    print(f"  {'categoria':<16}{'n':>9}{'apuestas':>9}{'prob media':>12}{'WR real':>9}{'sesgo':>9}{'WR si recalibra':>17}")
    for cat in sorted(set(cats.tolist())):
        m = (cats == cat) & ac
        if m.sum() < 200:
            continue
        pm, ym = p[m], yc[m]
        pick = np.maximum(pm, 1 - pm)
        betm = pick >= thr
        if betm.sum() < 20:
            continue
        prob = float(pick[betm].mean())
        wr = float(((pm[betm] >= 0.5) == (ym[betm] == 1)).mean())
        # recalibracion por la misma categoria (isotonico sobre las prob de la categoria)
        from sklearn.isotonic import IsotonicRegression
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.03, y_max=0.95)
        iso.fit(pick[betm], ((pm[betm] >= 0.5) == (ym[betm] == 1)).astype(float))
        pc = np.clip(iso.predict(pick[betm]), 0.03, 0.97)
        wr2 = float(((pc >= 0.5) == (ym[betm] == 1)).mean())
        print(f"  {cat:<16}{int(m.sum()):>9,}{int(betm.sum()):>9,}{prob:>12.3f}"
              f"{wr * 100:>8.1f}%{prob - wr:>+9.3f}{wr2 * 100:>16.1f}%")
    print("\n  (WR si recalibra = isotonica POR CATEGORIA aplicada a las mismas apuestas)")
    print("  -> si la columna 'WR si recalibra' supera a 'WR real', la calibracion")
    print("     global esta dejando dinero: se puede corregir sin tocar el modelo.")


if __name__ == "__main__":
    main()
