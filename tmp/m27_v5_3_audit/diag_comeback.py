# =====================================================================
# ULTIMA PIEZA: ¿las apuestas "de remontada" (elegir al que va
# perdiendo) son sistematicamente PEORES?
#   1) En el backtest (bloque C) a 0.75: WR por situacion
#   2) En vivo: WR por situacion
#   3) ¿El modelo es honesto en su confianza dentro de cada situacion?
# =====================================================================

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
rowC = mrank >= int(len(elig) * 0.82)

models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p = np.clip(models["calibrator"].predict(0.5 * (pe[:n] + 1.0 - pe[n:])), lo, hi)

q3 = X[:, IDX["cur_score_diff"]]
conf = np.maximum(p, 1 - p)
pick_home = p >= 0.5
won = np.where(pick_home, y == 1, y == 0)
# situacion: el elegido va ganando (leading) o perdiendo (trailing)
chosen_diff = np.where(pick_home, q3, -q3)
trailing = chosen_diff < 0
even = chosen_diff == 0


def tabla(nombre, mask_base, conf_min=0.75):
    print(f"   {nombre}")
    print(f"      {'situacion':<34}{'n':>8}{'WR':>9}{'conf med':>10}{'sobre/conf':>12}")
    for etiq, m in (("el elegido va GANANDO", mask_base & ~trailing & ~even),
                    ("empate", mask_base & even),
                    ("el elegido va PERDIENDO", mask_base & trailing)):
        s = m & (conf >= conf_min)
        if s.sum() == 0:
            print(f"      {etiq:<34}{0:>8}")
            continue
        w = won[s].mean()
        cm = conf[s].mean()
        print(f"      {etiq:<34}{int(s.sum()):>8,}{w * 100:>8.1f}%{cm:>10.3f}"
              f"{w * 100 - cm * 100:>+11.1f}pp")


print("1) BACKTEST (bloque C) A 0.75:")
tabla("todas las apuestas", rowC)

print()
print("   por magnitud del|diferencia| del elegido:")
for lo_b, hi_b, etiq in ((0, 2, "0-2 (empate技术)"), (3, 6, "3-6"), (7, 12, "7-12"),
                         (13, 20, "13-20"), (21, 99, "21+ (decidido)")):
    m = rowC & (np.abs(chosen_diff) >= lo_b) & (np.abs(chosen_diff) <= hi_b) & (conf >= 0.75)
    if m.sum() == 0:
        continue
    w = won[m].mean()
    cm = conf[m].mean()
    print(f"      {etiq:<34}{int(m.sum()):>8,}{w * 100:>8.1f}%{cm:>10.3f}"
          f"{w * 100 - cm * 100:>+11.1f}pp")

print()
print("2) VIVO (m27_v5_3, 45 apuestas):")
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
rows = conn.execute("""
    SELECT b.*, m.league FROM bet_monitor_log_v3 b
    LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id
    WHERE b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%'
      AND b.result IN ('win','loss')""").fetchall()
buckets = {"ganando": [], "empate": [], "perdiendo": []}
for r in rows:
    qs = conn.execute("SELECT q3_home, q3_away FROM quarter_scores_v2 WHERE match_id=?",
                      (r["match_id"],)).fetchone()
    if not qs:
        continue
    d = (qs["q3_home"] - qs["q3_away"]) * (1 if r["picked_side"] == "HOME" else -1)
    key = "ganando" if d > 0 else ("empate" if d == 0 else "perdiendo")
    buckets[key].append(r)
for k, v in buckets.items():
    if not v:
        print(f"      {k:<34}{0:>8}")
        continue
    w = sum(1 for x in v if x["result"] == "win")
    cm = sum(x["confidence"] for x in v) / len(v)
    print(f"      {k:<34}{len(v):>8}{w * 100 / len(v):>8.1f}%{cm:>10.3f}"
          f"{w * 100 / len(v) - cm * 100:>+11.1f}pp")

print()
print("3) VIVO: WR por banda de confianza (m27_v5_3, 45 apuestas):")
for lo_b, hi_b in ((0.75, 0.80), (0.80, 0.85), (0.85, 1.01)):
    v = [x for x in rows if lo_b <= x["confidence"] < hi_b]
    if not v:
        continue
    w = sum(1 for x in v if x["result"] == "win")
    cm = sum(x["confidence"] for x in v) / len(v)
    print(f"      conf {lo_b:.2f}-{hi_b:.2f}{'':<12}{len(v):>8}{w * 100 / len(v):>8.1f}%{cm:>10.3f}"
          f"{w * 100 / len(v) - cm * 100:>+11.1f}pp")

print()
print("4) ¿CUANTAS APUESTAS DE HOY SON DE LAS PEORES CATEGORIAS?")
for r in rows:
    if not r["created_at"].startswith("2026-10-20"):
        continue
    qs = conn.execute("SELECT q3_home, q3_away FROM quarter_scores_v2 WHERE match_id=?",
                      (r["match_id"],)).fetchone()
    d = (qs["q3_home"] - qs["q3_away"]) * (1 if r["picked_side"] == "HOME" else -1)
    print(f"      {r['created_at'][11:19]} | {str(r['league'])[:32]:<32} | {r['picked_side']:<5} "
          f"conf {r['confidence']:.3f} | dif elegido {d:>4} | {r['result']}")
conn.close()
