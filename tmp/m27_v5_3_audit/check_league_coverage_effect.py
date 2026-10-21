# =====================================================================
# ¿EL MODELO FALLA MÁS EN LAS LIGAS CON POCO HISTORIAL?
#   Para cada partido vivo: cuantas filas de entrenamiento (bloque A) tiene
#   su liga, y como se comporta el modelo en cada tramo de cobertura.
#   Si el backtest tambien degrada en las ligas cortas, el WR global del
#   backtest esta inflado por las ligas grandes y la poblacion viva
#   (ligas chicas) es genuinamente peor -> el filtro es de cobertura.
# =====================================================================

import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
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

# cobertura por liga en el bloque de entrenamiento
cov = Counter(lg_all[rowA].tolist())
cov_c = Counter(lg_all[rowC].tolist())


def bucket(v: int) -> str:
    return "<50" if v < 50 else ("50-200" if v < 200 else ("200-1000" if v < 1000 else ">1000"))


buckets = np.array([bucket(cov.get(l, 0)) for l in lg_all])  # por fila del dataset

models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p = np.clip(models["calibrator"].predict(0.5 * (pe[:n] + 1.0 - pe[n:])), lo, hi)
conf = np.maximum(p, 1 - p)
won = np.where(p >= 0.5, y == 1, y == 0)

print("1) BACKTEST (bloque C): WR a 0.75 por tramo de cobertura de la liga")
print(f"   {'cobertura':<22}{'ligas':>8}{'apuestas':>10}{'WR':>9}")
for etiq in ("<50", "50-200", "200-1000", ">1000"):
    mask_b = buckets == etiq
    nb = len({l for l, v in cov.items() if bucket(v) == etiq})
    m = rowC & (conf >= 0.75) & mask_b
    if m.sum() == 0:
        print(f"   {etiq:<22}{nb:>8}{0:>10}")
        continue
    print(f"   {etiq:<22}{nb:>8}{int(m.sum()):>10,}{won[m].mean() * 100:>8.1f}%")

print()
print("2) VIVO: acierto por tramo de cobertura (todas las predicciones)")
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
liga_de = {str(r["match_id"]): r["league"] for r in conn.execute("SELECT match_id, league FROM matches")}
filas = conn.execute("""
    SELECT i.match_id, i.game_minute, i.pick, i.confidence,
           q.q4_home, q.q4_away
    FROM model_inference_v3 i
    JOIN quarter_scores_v2 q ON CAST(q.match_id AS INTEGER) = i.match_id
    WHERE i.model_version='m27_v5_3' AND i.game_minute <= 30
      AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL
    ORDER BY i.match_id, i.game_minute""").fetchall()
por_partido = {}
for r in filas:
    d = dict(r)
    d["league"] = liga_de.get(str(r["match_id"]), "")
    por_partido[r["match_id"]] = d
print(f"   partidos vivos evaluables: {len(por_partido)}")
print(f"   {'cobertura':<22}{'partidos':>10}{'acierto pick':>14}{'conf media':>12}{'a 0.75+':>9}")
for etiq in ("<50", "50-200", "200-1000", ">1000"):
    sel = [r for r in por_partido.values()
           if bucket(cov.get(r["league"], 0)) == etiq and r["pick"] in ("HOME", "AWAY")]
    if not sel:
        print(f"   {etiq:<22}{0:>10}")
        continue
    w = 0
    nsel = 0
    for r in sel:
        dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
               else (r["q4_away"] - r["q4_home"]))
        if dif != 0:
            w += dif > 0
            nsel += 1
    confs = [r["confidence"] for r in sel]
    n75 = sum(1 for c in confs if c >= 0.75)
    print(f"   {etiq:<22}{len(sel):>10}{w / max(nsel, 1) * 100:>13.1f}%"
          f"{np.mean(confs):>12.3f}{n75:>9}")

print()
print("3) DISTRIBUCION DE LA COBERTURA: backtest vs vivo")
c_v = Counter(bucket(cov.get(r["league"], 0)) for r in por_partido.values())
for k in ("<50", "50-200", "200-1000", ">1000"):
    nb = len({l for l, v in cov.items() if bucket(v) == k})
    print(f"   {k:<10} ligas: backtest {nb:>5} | partidos vivos: {c_v.get(k, 0):>4}")

print()
print("4) LAS 6 APUESTAS DE HOY A 0.75, CON SU COBERTURA:")
for mid, r in por_partido.items():
    if r["confidence"] >= 0.75:
        dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
               else (r["q4_away"] - r["q4_home"]))
        res = "GANADA" if dif > 0 else ("PERDIDA" if dif < 0 else "empate")
        print(f"   {mid} | {str(r['league'])[:36]:<36} | conf {r['confidence']:.3f} | "
              f"margen Q4 {dif:>4} | {res:<7} | filas de train en la liga: "
              f"{cov.get(r['league'], 0):>5}")
conn.close()
