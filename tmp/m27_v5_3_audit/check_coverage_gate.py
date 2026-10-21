# =====================================================================
# ¿EL MODELO DISPARA CON ALTA CONFIANZA EN LIGAS/EQUIPOS QUE NO CONOCE?
#   En el backtest (bloque C) a 0.75: WR segun la cobertura de la liga en
#   el bloque de ENTRENAMIENTO y segun si el equipo tiene elo/h2h.
#   Si ahi tambien se hunde, el fix es un filtro de cobertura.
# =====================================================================

import pickle
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

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
rowA = (mrank >= 0) & (mrank < int(n_el * 0.70))
rowC = mrank >= int(n_el * 0.82)
lg_all = np.array([m["lg"] for m in meta])
ht_all = np.array([m["ht"] for m in meta])
at_all = np.array([m["at"] for m in meta])

cov_liga = Counter(lg_all[rowA].tolist())
cov_equipo = Counter()
for l, h, a in zip(lg_all[rowA].tolist(), ht_all[rowA].tolist(), at_all[rowA].tolist()):
    cov_equipo[h] += 1
    cov_equipo[a] += 1

models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p = np.clip(models["calibrator"].predict(0.5 * (pe[:n] + 1.0 - pe[n:])), lo, hi)
conf = np.maximum(p, 1 - p)
pick_home = p >= 0.5
won = np.where(pick_home, y == 1, y == 0)


def b(v, cortes=(0, 1, 6, 51, 201, 1001, 10 ** 9)):
    for i in range(len(cortes) - 1):
        if cortes[i] <= v < cortes[i + 1]:
            return f"{cortes[i]}-{cortes[i + 1] - 1}" if cortes[i + 1] < 10 ** 9 else f"{cortes[i]}+"
    return "?"


print("1) BACKTEST (bloque C) A 0.75: WR POR COBERTURA DE LA LIGA EN EL ENTRENAMIENTO")
selC = rowC & (conf >= 0.75)
print(f"   {'cobertura':<14}{'apuestas':>10}{'WR':>9}{'conf med':>11}")
buckets = np.array([b(cov_liga.get(l, 0)) for l in lg_all])
for etiq in ("0-0", "1-5", "6-50", "51-200", "201-1000", "1001+"):
    m = selC & (buckets == etiq)
    if m.sum() == 0:
        print(f"   {etiq:<14}{0:>10}")
        continue
    print(f"   {etiq:<14}{int(m.sum()):>10,}{won[m].mean() * 100:>8.1f}%{conf[m].mean():>11.3f}")

print()
print("2) BACKTEST (bloque C) A 0.75: WR POR COBERTURA DEL EQUIPO (minimo de los dos)")
mincov = np.array([min(cov_equipo.get(h, 0), cov_equipo.get(a, 0))
                   for h, a in zip(ht_all, at_all)])
buckets_e = np.array([b(v) for v in mincov])
for etiq in ("0-0", "1-5", "6-50", "51-200", "201-1000", "1001+"):
    m = selC & (buckets_e == etiq)
    if m.sum() == 0:
        print(f"   {etiq:<14}{0:>10}")
        continue
    print(f"   {etiq:<14}{int(m.sum()):>10,}{won[m].mean() * 100:>8.1f}%{conf[m].mean():>11.3f}")

print()
print("3) LAS 6 APUESTAS VIVAS DE HOY A 0.75, CON LA COBERTURA DE SUS EQUIPOS:")
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
info = {}
for r in conn.execute("SELECT match_id, home_team, away_team, league FROM matches"):
    info[str(r["match_id"])] = (r["home_team"], r["away_team"], r["league"])
for r in conn.execute("""
    SELECT i.match_id, i.pick, i.confidence, q.q4_home, q.q4_away
    FROM model_inference_v3 i JOIN quarter_scores_v2 q
      ON CAST(q.match_id AS INTEGER)=i.match_id
    WHERE i.model_version='m27_v5_3' AND i.game_minute<=30
      AND i.confidence>=0.75 AND q.q4_home IS NOT NULL"""):
    mid = str(r["match_id"])
    if mid not in info:
        continue
    h, a, lg = info[mid]
    ch = cov_equipo.get(h, 0)
    ca = cov_equipo.get(a, 0)
    dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
           else (r["q4_away"] - r["q4_home"]))
    print(f"   {lg[:30]:<30} | {str(h)[:18]:<18} ({ch:>4} filas) vs {str(a)[:18]:<18} ({ca:>4}) "
          f"| conf {r['confidence']:.3f} | {'GANADA' if dif > 0 else 'PERDIDA'}")

print()
print("4) ¿QUE PASA SI EXIGIMOS UN MINIMO DE FILAS POR EQUIPO? (backtest, 0.75)")
for minimo in (0, 1, 6, 20, 51):
    m = selC & (mincov >= minimo)
    if m.sum() == 0:
        continue
    print(f"   equipos con >= {minimo:>3} filas de train: {int(m.sum()):>7,} apuestas | "
          f"WR {won[m].mean() * 100:>5.1f}% | conf media {conf[m].mean():.3f}")
conn.close()
