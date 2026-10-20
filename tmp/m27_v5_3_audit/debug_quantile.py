# =====================================================================
# DEBUG de P2: ¿por que el modelo de cuantiles predice una constante?
# =====================================================================

import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from models.m27_v5_3.features import mirror  # noqa: E402

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
elig = D["eligible_ord"]
rank = {int(o): i for i, o in enumerate(elig)}
mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
anchor = np.array([m["anchor"] for m in meta])
n_el = len(elig)
a_end = int(n_el * 0.70)
rowA = (mrank >= 0) & (mrank < a_end)
rowC = mrank >= int(n_el * 0.82)
vals, inv = np.unique(mrank, return_inverse=True)
w_all = 1.0 / np.bincount(inv)[inv]

Xtr, ytr, qtr, wtr = X[rowA], y[rowA], q4d[rowA], w_all[rowA]
print(f"margen Q4 en el train: media {qtr.mean():.2f} | std {qtr.std():.2f} | "
      f"min {qtr.min():.0f} max {qtr.max():.0f} | % positivos {(qtr > 0).mean() * 100:.1f} %")

Xa = np.vstack([Xtr, mirror(Xtr)])
ya = np.concatenate([ytr, 1 - ytr])
qa = np.concatenate([qtr, -qtr])
wa = np.concatenate([wtr, wtr])
print(f"datos aumentados: {Xa.shape} | objetivo con espejo: media {qa.mean():.3f} "
      f"| % positivos {(qa > 0).mean() * 100:.1f} %")

idxC = np.where(rowC & anchor)[0][:3000]
print("\najustando un modelo de cuantil 0.5 (como el regresor de margen actual):")
t0 = time.time()
m = HistGradientBoostingRegressor(loss="quantile", quantile=0.5, max_iter=200, learning_rate=0.06,
                                  max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=40.0,
                                  random_state=42)
m.fit(Xa, ya, sample_weight=wa)
print(f"  ({time.time() - t0:.0f}s)")
pred = m.predict(X[idxC])
print(f"  predicciones: media {pred.mean():.2f} std {pred.std():.2f} "
      f"min {pred.min():.1f} max {pred.max():.1f}")
print(f"  % predicciones > 0: {(pred > 0).mean() * 100:.1f} %")

print("\ncomparando con el regresor de margen DE PRODUCCION (conjunto simetrico):")
from train_m27_v5_3_pipeline import fit_ensemble, margin_sym
t0 = time.time()
ens = fit_ensemble(X[rowA], y[rowA], q4d[rowA], w_all[rowA], quick=False)
ms = margin_sym(ens, X[idxC])
print(f"  ({time.time() - t0:.0f}s)  media {ms.mean():.2f} std {ms.std():.2f} "
      f"| % > 0: {(ms > 0).mean() * 100:.1f} %")
print(f"  correlacion con el cuantil 0.5: {np.corrcoef(pred, ms)[0, 1]:.4f}")
