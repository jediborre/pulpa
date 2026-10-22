# =====================================================================
# ¿EL MODELO PIERDE EN LOS PARTIDOS DE "RUNDA ZASADNICZA"?
#   (ronda de promocion/descenso de Polonia: el clasificador la marca
#    como REGULAR, pero es el equivalente polaco de "championship round",
#    que SI se bloquea)
# =====================================================================

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

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

# 1) El marcador polaco
MARCA = "zasadnicza"
es_z = np.array([MARCA in str(l).lower() for l in lgC])
print(f"1) MARCADOR POLACO '{MARCA}':")
print(f"   filas en el bloque C: {int(es_z.sum()):,} ({es_z.sum() * 100 / len(es_z):.2f} %)")
print(f"   partidos distintos: {len({str(l) for l, z in zip(lgC, es_z) if z}):,}")

print()
print("2) WR A 0.75 CON Y SIN ESE FORMATO (bloque C, modelo desplegado):")
print(f"   {'grupo':<28}{'n':>8}{'WR':>9}{'u@1.40':>10}{'u@1.83':>10}")
for etiq, mask in (("Runda Zasadnicza", es_z), ("resto de ligas", ~es_z)):
    m = mask & (conf >= 0.75)
    if m.sum() == 0:
        print(f"   {etiq:<28}{0:>8}")
        continue
    w = int(won[m].sum())
    k = int(m.sum())
    print(f"   {etiq:<28}{k:>8,}{w / k * 100:>8.1f}%{w * 1.40 - k:>10.0f}{w * 1.83 - k:>10.0f}")

print()
print("3) LO MISMO EN LA SERIE VIVA (model_inference_v3, con el Q4 real):")
import sqlite3  # noqa: E402
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
liga_de = {str(r["match_id"]): r["league"] for r in
           conn.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL")}
z_n = z_w = o_n = o_w = 0
for r in conn.execute("""
    SELECT i.match_id, i.pick, i.confidence, q.q4_home, q4_away
    FROM model_inference_v3 i JOIN quarter_scores_v2 q
      ON CAST(q.match_id AS INTEGER)=i.match_id
    WHERE i.model_version='m27_v5_3' AND i.game_minute<=30
      AND i.confidence>=0.75 AND q.q4_home IS NOT NULL"""):
    lg = (liga_de.get(str(r["match_id"])) or "").lower()
    dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
           else (r["q4_away"] - r["q4_home"]))
    if dif == 0:
        continue
    if MARCA in lg:
        z_n += 1
        z_w += dif > 0
    else:
        o_n += 1
        o_w += dif > 0
print(f"   Runda Zasadnicza : {z_w}/{z_n} = {z_w * 100 / max(z_n, 1):.1f} %")
print(f"   resto de ligas  : {o_w}/{o_n} = {o_w * 100 / max(o_n, 1):.1f} %")

print()
print("4) ¿QUE OTRAS FASES SIN RECONOCER QUEDAN EN EL ENTRENAMIENTO?")
from models.common.competition_stage import classify_stage  # noqa: E402
from models.m27_v5_3.features import BLOCKED_STAGE_CATEGORIES  # noqa: E402
cand = Counter()
for lg in set(lg_all.tolist()):
    cat, _ = classify_stage(lg)
    if cat == "REGULAR":
        cand[lg] += 1
# phases that LOOK like a phase (contain a comma or a phase-ish word) but classify as REGULAR
import re  # noqa: E402
sospechosas = [(lg, n) for lg, n in cand.items()
                if re.search(r",\s*(runda|round|fase|playoff|play-off|runde|runde|"
                             r"rondes|turn|round|final|kop| playoff)", str(lg), re.I)]
print(f"   ligas con 'fase' en el nombre pero clasificadas como REGULAR: {len(sospechosas)}")
for lg, n in sorted(sospechosas, key=lambda kv: -kv[1])[:25]:
    cat, mot = classify_stage(lg)
    print(f"      {str(lg)[:56]:<58} n={n:>5} | {cat:<14} {mot[:34]}")
conn.close()
