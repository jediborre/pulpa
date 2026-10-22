# =====================================================================
# ¿LAS POBLACIONES QUE EL ARREGLO BLOQUEA RINDEN MAL?
#   (si rinden bien, bloquearlas cuesta dinero y hay que pensarlo)
#   - A: la ronda polaca "Runda Zasadnicza" (1.370 partidos, antes REGULAR)
#   - B: los "Relegation/Promotion Playoffs" (372, antes ELIMINATORIA)
# =====================================================================

import pickle
import sys
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
rowC = mrank >= int(len(elig) * 0.82)
idxC = np.where(rowC)[0]
lg_all = np.array([m["lg"] for m in meta])

models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X[idxC], mirror(X[idxC])])
pe = _ensemble(models, X2)
p = np.clip(models["calibrator"].predict(0.5 * (pe[:len(idxC)] + 1.0 - pe[len(idxC):])), lo, hi)
conf = np.maximum(p, 1 - p)
won = np.where(p >= 0.5, y[idxC] == 1, y[idxC] == 0)
lgC = lg_all[idxC]

GRUPOS = {
    "A) ronda polaca (zasadnicza)": np.array(["zasadnicza" in str(l).lower() for l in lgC]),
    "B) releg/prom playoffs": np.array(
        [any(k in str(l).lower() for k in ("relegation playoff", "rel/prom", "rel./prom",
                                           "promotion playoff", "promotion play-off",
                                           "placement playoff"))
         for l in lgC]),
    "C) el resto": np.ones(len(lgC), bool),
}

print("WR A 0.75 (bloque C, modelo desplegado) POR POBLACION:")
for etiq, mask in GRUPOS.items():
    m = mask & (conf >= 0.75)
    if m.sum() == 0:
        print(f"   {etiq:<32} sin apuestas")
        continue
    w = int(won[m].sum())
    k = int(m.sum())
    wr = w / k
    print(f"   {etiq:<32} n={k:>5,} | WR {wr * 100:>5.1f}% | "
          f"u@1.40 {w * 1.40 - k:>+6.0f} | u@1.83 {w * 1.83 - k:>+6.0f} | "
          f"{'rentable' if wr >= 1 / 1.40 else 'POR DEBAJO DEL EQUILIBRIO 1.40'}")

print()
print("Y EN LA SERIE VIVA (apuestas reales de m27_v5_3, por categoria):")
import sqlite3  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
liga_de = {str(r["match_id"]): r["league"] for r in
           conn.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL")}
from models.common.competition_stage import classify_stage  # noqa: E402
from models.m27_v5_3.features import BLOCKED_STAGE_CATEGORIES  # noqa: E402
cats = Counter()
for r in conn.execute("""
    SELECT match_id, result FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'
      AND result IN ('win','loss')"""):
    cat, _ = classify_stage(liga_de.get(str(r["match_id"])) or "")
    cats[(cat, r["result"])] += 1
por_cat = defaultdict(lambda: [0, 0])
for (cat, res), n in cats.items():
    por_cat[cat][0] += n
    por_cat[cat][1] += res == "win"
for cat, (n, w) in sorted(por_cat.items(), key=lambda kv: -kv[1][0]):
    marca = "  <-- AHORA BLOQUEADA" if cat in BLOCKED_STAGE_CATEGORIES else ""
    print(f"   {cat:<14} {n:>3} bets | WR {w * 100:>5.1f}%{marca}")
conn.close()
