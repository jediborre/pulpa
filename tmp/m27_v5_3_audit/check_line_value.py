# =====================================================================
# ¿SON INFORMATIVAS LAS LINEAS Y SON UNA POBLACION DISTINTA?
#   Usa el modelo DESPLIEGADO (m27_v5_3 en disco) sobre el dataset.
# =====================================================================

import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import IDX, mirror  # noqa: E402
from models.m27_v5_3.predict import _load_production_models, _ensemble  # noqa: E402

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, y, meta = D["X"], D["y"], D["meta"]
elig = D["eligible_ord"]
rank = {int(o): i for i, o in enumerate(elig)}
mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
n_el = len(elig)
a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
rowA = (mrank >= 0) & (mrank < a_end)
rowB = (mrank >= a_end) & (mrank < b_end)
rowC = mrank >= b_end

models = _load_production_models()
print("claves del modelo:", sorted(models.keys()))
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p_raw = 0.5 * (pe[:n] + 1.0 - pe[n:])
p = np.clip(models["calibrator"].predict(p_raw), lo, hi)

has_sp = ~np.isnan(X[:, IDX["mkt_margin_spread"]])
has_ml = X[:, IDX["has_ml"]] > 0.5
p_mkt = X[:, IDX["mkt_p_home"]]
sp = X[:, IDX["mkt_margin_spread"]]
conf = np.maximum(p, 1 - p)
pick_home = p >= 0.5
won = np.where(pick_home, y == 1, y == 0)

print()
print("1) ¿LA LINEA DE SPREAD PREDICE EL RESULTADO DEL Q4 POR SI SOLA? (bloque C)")
m = rowC & has_sp
p_line = 1.0 / (1.0 + np.exp(-(sp[m] * 0.10)))
print(f"   AUC linea->resultado (n={int(m.sum()):,}): {roc_auc_score(y[m], p_line):.4f}")
print(f"   AUC mercado mkt_p_home (n={int(m.sum()):,}): "
      f"{roc_auc_score(y[m], np.where(np.isnan(p_mkt[m]), 0.5, p_mkt[m])):.4f}")
print(f"   AUC del modelo:                              {roc_auc_score(y[m], p[m]):.4f}")

print()
print("2) POBLACION CON LINEA vs SIN LINEA (umbral operativo 0.75):")
for nombre, mask in (("C completo", rowC), ("C con linea", rowC & has_sp),
                      ("C sin linea", rowC & ~has_sp),
                      ("C con linea y con mercado", rowC & has_sp & has_ml),
                      ("C con linea y sin mercado", rowC & has_sp & ~has_ml)):
    s = mask & (conf >= 0.75)
    if s.sum() == 0:
        print(f"   {nombre:<30} sin apuestas")
        continue
    w = won[s].mean()
    k = int(s.sum())
    u140 = k * (1.40 * w - 1.0)
    print(f"   {nombre:<30} apuestas {k:>6,} | WR {w * 100:>5.2f}% | u@1.40 {u140:>7.0f}")

print()
print("3) ¿SUMAR LA LINEA AL MODELO MEJORA EL AUC? (filas con linea, bloque C)")
mC = rowC & has_sp
p_mkt_ok = np.where(np.isnan(p_mkt), 0.5, p_mkt)
for nombre, score in (("modelo", p[mC]), ("mercado", p_mkt_ok[mC]),
                      ("linea->prob", p_line), ("modelo+linea (0.8/0.2)",
                                               0.8 * p[mC] + 0.2 * p_line)):
    print(f"   {nombre:<24} AUC {roc_auc_score(y[mC], score):.4f}")
print(f"   (n = {int(mC.sum()):,})")

print()
print("4) ¿Y EN LAS FILAS SIN LINEA? (control)")
mC2 = rowC & ~has_sp
for nombre, score in (("modelo", p[mC2]), ("mercado", p_mkt_ok[mC2])):
    print(f"   {nombre:<24} AUC {roc_auc_score(y[mC2], score):.4f}")
print(f"   (n = {int(mC2.sum()):,})")

print()
print("5) ¿EN EL BLOQUE DE ENTRENAMIENTO (A) CUANTAS HAY CON LINEA?")
mA = rowA & has_sp
sA = mA & (conf >= 0.75)
print(f"   filas con linea: {int(mA.sum()):,} ({mA.sum() * 100 / rowA.sum():.1f} %) | "
      f"apuestas sobre ellas: {int(sA.sum()):,} | WR {won[sA].mean() * 100:.2f} %"
      if sA.sum() else "   sin apuestas")
