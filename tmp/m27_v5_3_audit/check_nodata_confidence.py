# =====================================================================
# ¿CONFIA EL MODELO REALMENTE EN LAS LIGAS SIN DATOS, O ADIVINA?
#   1) Calibracion por tramo de confianza DENTRO de cada grupo:
#      (a) liga con 0 filas en el train, (b) ambos equipos con 0 filas,
#      (c) liga con 1-49 filas, (d) resto.
#      Si lo que predice coincide con lo que ocurre, la confianza es real.
#   2) Curva de beneficio por umbral en cada grupo: ¿hay edge extra para
#      apostar mas donde no hay datos de entrenamiento?
# =====================================================================

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import mirror  # noqa: E402
from models.m27_v5_3.predict import _load_production_models, _ensemble  # noqa: E402

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, y, meta = D["X"], D["y"], D["meta"]
elig = D["eligible_ord"]
rank = {int(o): i for i, o in enumerate(elig)}
mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
n_el = len(elig)
rowA = (mrank >= 0) & (mrank < int(n_el * 0.70))
rowC = mrank >= int(n_el * 0.82)
lg_all = np.array([m["lg"] for m in meta])
ht_all = np.array([m["ht"] for m in meta])
at_all = np.array([m["at"] for m in meta])

cov_liga = Counter(lg_all[rowA].tolist())
cov_equipo = Counter()
for h, a in zip(ht_all[rowA].tolist(), at_all[rowA].tolist()):
    cov_equipo[h] += 1
    cov_equipo[a] += 1

cov_l = np.array([cov_liga.get(l, 0) for l in lg_all])
cov_e = np.array([min(cov_equipo.get(h, 0), cov_equipo.get(a, 0))
                  for h, a in zip(ht_all, at_all)])

GRUPOS = {
    "liga 0 filas + equipos 0": (cov_l == 0) & (cov_e == 0),
    "liga 0 filas": cov_l == 0,
    "liga 1-49": (cov_l >= 1) & (cov_l < 50),
    "liga 50+": cov_l >= 50,
}

models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p = np.clip(models["calibrator"].predict(0.5 * (pe[:n] + 1.0 - pe[n:])), lo, hi)
conf = np.maximum(p, 1 - p)
pick_home = p >= 0.5
won = np.where(pick_home, y == 1, y == 0)

print("1) CALIBRACION POR GRUPO (bloque C): lo que dice el modelo vs lo que pasa")
print(f"   {'grupo':<24}{'banda':<14}{'n':>8}{'p med':>9}{'WR real':>9}{'gap':>8}")
for nombre, mask_grupo in GRUPOS.items():
    for banda, etiq in (((0.50, 0.60), "0.50-0.60"), ((0.60, 0.65), "0.60-0.65"),
                        ((0.65, 0.70), "0.65-0.70"), ((0.70, 0.75), "0.70-0.75"),
                        ((0.75, 1.01), "0.75+")):
        m = rowC & mask_grupo & (conf >= banda[0]) & (conf < banda[1])
        if m.sum() < 30:
            continue
        wr = won[m].mean()
        pm = conf[m].mean()
        print(f"   {nombre:<24}{etiq:<14}{int(m.sum()):>8,}{pm:>9.3f}"
              f"{wr * 100:>8.1f}%{(wr - pm) * 100:>+7.1f}pp")

print()
print("2) CURVA DE BENEFICIO POR GRUPO (bloque C)")
print(f"   {'grupo':<24}{'umbral':>8}{'n':>8}{'WR':>9}{'u@1.40':>10}{'u@1.83':>10}")
for nombre, mask_grupo in GRUPOS.items():
    for t in (0.65, 0.70, 0.72, 0.75):
        m = rowC & mask_grupo & (conf >= t)
        if m.sum() == 0:
            continue
        w = won[m].sum()
        k = int(m.sum())
        print(f"   {nombre:<24}{t:>8.2f}{k:>8,}{w / k * 100:>8.1f}%"
              f"{w * 1.40 - k:>10.0f}{w * 1.83 - k:>10.0f}")

print()
print("3) ¿QUE SIGNIFICA '0 FILAS EN LA LIGA'? (¿mismo partido con otro nombre?)")
mism = [l for l in cov_liga if cov_liga[l] == 0]
print(f"   ligas en el test que no aparecen en el train: {len(mism)}")
for l in list(mism)[:12]:
    print(f"      {l!r}")
