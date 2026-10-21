# =====================================================================
# LOS PARTIDOS VIVOS DE LAS LIGAS "GRANDES" (>=200 filas de train):
#   acierto real, confianza y detalle. ¿Es un bloque real o ruido?
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

D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, y, meta = D["X"], D["y"], D["meta"]
elig = D["eligible_ord"]
rank = {int(o): i for i, o in enumerate(elig)}
mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
rowA = (mrank >= 0) & (mrank < int(len(elig) * 0.70))
lg_all = np.array([m["lg"] for m in meta])
cov = Counter(lg_all[rowA].tolist())

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
info = {str(r["match_id"]): (r["home_team"], r["away_team"], r["league"])
        for r in conn.execute("SELECT match_id, home_team, away_team, league FROM matches")}

filas = conn.execute("""
    SELECT i.match_id, i.pick, i.confidence, i.margin, q.q4_home, q.q4_away
    FROM model_inference_v3 i JOIN quarter_scores_v2 q
      ON CAST(q.match_id AS INTEGER)=i.match_id
    WHERE i.model_version='m27_v5_3' AND i.game_minute<=30
      AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL
    ORDER BY i.match_id, i.game_minute""").fetchall()
por_partido = {}
for r in filas:
    por_partido[r["match_id"]] = dict(r)

print("PARTIDOS VIVOS CON LA LIGA EN EL TRAMO 'GRANDE' (>=200 filas de train):")
print(f"   {'liga':<30}{'conf':>7}{'margen':>8}{'pick':>6}{'Q4':>10}{'resultado':>10}")
w_g = n_g = 0
for mid, r in por_partido.items():
    h, a, lg = info.get(mid, ("", "", ""))
    if cov.get(lg, 0) < 200:
        continue
    dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
           else (r["q4_away"] - r["q4_home"]))
    res = "GANADA" if dif > 0 else ("PERDIDA" if dif < 0 else "empate")
    if dif != 0:
        n_g += 1
        w_g += dif > 0
    print(f"   {str(lg)[:29]:<30}{r['confidence']:>7.3f}{r['margin']:>8.2f}"
          f"{r['pick']:>6}{str(r['q4_home']) + '-' + str(r['q4_away']):>10}{res:>10}"
          f"   [train {cov.get(lg, 0)}]")
print(f"   -> acierto en ligas grandes: {w_g}/{n_g} = {w_g * 100 / max(n_g, 1):.1f} %")

print()
print("Y EN EL BACKTEST (bloque C), A TODAS LAS PREDICCIONES, POR TRAMO DE COBERTURA:")
from models.m27_v5_3.features import mirror  # noqa: E402
from models.m27_v5_3.predict import _load_production_models, _ensemble  # noqa: E402

models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p = np.clip(models["calibrator"].predict(0.5 * (pe[:n] + 1.0 - pe[n:])), lo, hi)
conf = np.maximum(p, 1 - p)
won = np.where(p >= 0.5, y == 1, y == 0)
rowC = mrank >= int(len(elig) * 0.82)


def bkt(v):
    return "<200" if v < 200 else "200-1000" if v < 1000 else ">1000"


bk = np.array([bkt(cov.get(l, 0)) for l in lg_all])
for etiq in ("<200", "200-1000", ">1000"):
    m = rowC & (bk == etiq)
    if m.sum() == 0:
        continue
    print(f"   {etiq:<10} filas {int(m.sum()):>7,} | acierto del pick {won[m].mean() * 100:>5.1f}% "
          f"| conf media {conf[m].mean():.3f}")
conn.close()
