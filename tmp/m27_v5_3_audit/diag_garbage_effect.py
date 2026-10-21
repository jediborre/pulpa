# =====================================================================
# ¿POR QUE HOY VA MAL?  ->  EFECTO "GARBAGE" (partido ya decidido)
#   1) WR del modelo desplegado a 0.75 por magnitud del margen de Q3
#   2) WR historico vivo de m27_v5_3 (todas sus apuestas)
#   3) ¿hoy es un dia atipico o el patron de siempre?
# =====================================================================

import json
import pickle
import sqlite3
import sys
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
rowC = mrank >= int(n_el * 0.82)

models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p = np.clip(models["calibrator"].predict(0.5 * (pe[:n] + 1.0 - pe[n:])), lo, hi)

q3 = X[:, IDX["cur_score_diff"]]
absq3 = np.abs(q3)
conf = np.maximum(p, 1 - p)
pick_home = p >= 0.5
won = np.where(pick_home, y == 1, y == 0)

print("1) WR DEL MODELO A 0.75 SEGUN EL MARGEN DE Q3 (bloque C, la misma regla):")
print(f"   {'margen Q3':<16}{'apuestas':>10}{'WR':>9}{'u@1.40':>10}")
for lo_b, hi_b, etiqueta in ((0, 4, "empate +-4"), (5, 9, "5-9"), (10, 14, "10-14"),
                              (15, 19, "15-19"), (20, 24, "20-24"), (25, 99, "25+ (decidido)")):
    m = rowC & (absq3 >= lo_b) & (absq3 < hi_b + 1) & (conf >= 0.75)
    if m.sum() == 0:
        print(f"   {etiqueta:<16}{0:>10}")
        continue
    w = won[m].sum()
    k = int(m.sum())
    print(f"   {etiqueta:<16}{k:>10,}{w * 100 / k:>8.1f}%{k * (1.4 * w / k - 1):>10.0f}")

m = rowC & (absq3 >= 25) & (conf >= 0.75)
print(f"   -> en partidos ya decididos (25+): {int(m.sum()):,} apuestas, "
      f"WR {won[m].mean() * 100:.1f} %")

print()
print("2) ¿Y LA CONFIANZA ES EXAGERADA EN ESOS PARTIDOS?")
m = rowC & (absq3 >= 25)
for t in (0.70, 0.75, 0.80, 0.85):
    s = m & (conf >= t)
    if s.sum() == 0:
        continue
    w = won[s].mean()
    print(f"   conf>={t:.2f}: n={int(s.sum()):>6,} | conf media {conf[s].mean():.3f} "
          f"| WR real {w * 100:>5.1f}% | sobre/confianza {w * 100 - conf[s].mean() * 100:>+5.1f} pp")
m2 = rowC & (absq3 < 15)
for t in (0.70, 0.75, 0.80, 0.85):
    s = m2 & (conf >= t)
    if s.sum() == 0:
        continue
    w = won[s].mean()
    print(f"   [partidos _{int(m2.sum()):,}_] conf>={t:.2f}: n={int(s.sum()):>6,} | "
          f"WR real {w * 100:>5.1f}% | sobre/confianza {w * 100 - conf[s].mean() * 100:>+5.1f} pp")

print()
print("3) HISTORICO VIVO DE m27_v5_3 (todas sus apuestas reales):")
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT b.*, m.league FROM bet_monitor_log_v3 b "
    "LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id "
    "WHERE b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%' "
    "AND b.result IN ('win','loss') ORDER BY b.id").fetchall()
wins = sum(1 for r in rows if r["result"] == "win")
print(f"   apuestas resueltas: {len(rows)} | ganadas {wins} | perdidas {len(rows) - wins} | "
      f"WR {wins * 100 / max(len(rows), 1):.1f} %")
by_day = {}
for r in rows:
    d = r["created_at"][:10]
    by_day.setdefault(d, []).append(r["result"])
print("   por dia:")
for d, res in sorted(by_day.items()):
    w = sum(1 for x in res if x == "win")
    print(f"     {d}: {len(res):>3} apuestas | WR {w * 100 / len(res):>5.1f}%  "
          f"[{', '.join(x[0].upper() for x in res)}]")

print()
print("4) LAS APUESTAS DE HOY CON SU MARGEN DE Q3 (para contrastar):")
for r in conn.execute(
        "SELECT b.match_id, b.picked_side, b.confidence, b.result, b.created_at, m.league "
        "FROM bet_monitor_log_v3 b LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id "
        "WHERE b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%' "
        "AND substr(b.created_at,1,10)='2026-10-20' ORDER BY b.id"):
    qs = conn.execute("SELECT q3_home, q3_away FROM quarter_scores_v2 WHERE match_id=?",
                      (r["match_id"],)).fetchone()
    d = (qs["q3_home"] - qs["q3_away"]) if qs else None
    print(f"   {r['created_at'][11:19]} | {str(r['league'])[:30]:<30} | {r['picked_side']:<5} "
          f"conf {r['confidence']:.4f} | Q3 dif {d:>4} | {r['result']}")
conn.close()
