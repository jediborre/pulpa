# =====================================================================
# HIPOTESIS FINAL: la poblacion viva (mezcla de ligas) es mas debil
#   -> WR del backtest a 0.75 restringido a las ligas que aparecen
#      en las apuestas vivas, vs el WR global
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

from models.m27_v5_3.features import mirror  # noqa: E402
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
conf = np.maximum(p, 1 - p)
pick_home = p >= 0.5
won = np.where(pick_home, y == 1, y == 0)
lg = np.array([m["lg"] for m in meta])

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
vivas = conn.execute("""
    SELECT b.match_id, b.picked_side, b.confidence, b.result, b.created_at, m.league
    FROM bet_monitor_log_v3 b LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id
    WHERE b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%'
      AND b.result IN ('win','loss')""").fetchall()
ligas_vivas = Counter(r["league"] for r in vivas if r["league"])
print(f"ligas distintas en las {len(vivas)} apuestas vivas: {len(ligas_vivas)}")
for lgname, cnt in ligas_vivas.most_common():
    print(f"   {cnt} | {lgname}")

m_seleccion = rowC & (conf >= 0.75)
m_ligas_vivas = m_seleccion & np.isin(lg, list(ligas_vivas.keys()))
print()
print("WR DEL BACKTEST a 0.75 (bloque C):")
print(f"   todas las ligas          : {int(m_seleccion.sum()):>7,} apuestas | "
      f"WR {won[m_seleccion].mean() * 100:>5.1f}%")
print(f"   solo las ligas de las vivas: {int(m_ligas_vivas.sum()):>7,} apuestas | "
      f"WR {won[m_ligas_vivas].mean() * 100:>5.1f}%")
print(f"   peso de las ligas vivas dentro del backtest: "
      f"{m_ligas_vivas.sum() * 100 / m_seleccion.sum():.1f} % de las apuestas")

print()
print("LAS 5 APUESTAS DE HOY, WR historico de cada liga en el backtest (bloque C, 0.75):")
hoy = [r for r in vivas if r["created_at"].startswith("2026-10-20")]
for r in hoy:
    m = m_seleccion & (lg == r["league"])
    if m.sum() == 0:
        # buscar por prefijo
        pref = (r["league"] or "").split(",")[0].strip()
        m = m_seleccion & np.char.startswith(lg.astype(str), pref)
    if m.sum() == 0:
        print(f"   {str(r['league'])[:40]:<40} sin datos en el backtest a 0.75 | "
              f"conf {r['confidence']:.3f} {r['result']}")
    else:
        print(f"   {str(r['league'])[:40]:<40} n={int(m.sum()):>5,} | "
              f"WR {won[m].mean() * 100:>5.1f}% | conf {r['confidence']:.3f} | {r['result']}")

print()
print("TAMAÑO DE LAS LIGAS VIVAS vs EL BACKTEST:")
tam = {}
for r in conn.execute("""
    SELECT league, COUNT(*) n FROM matches WHERE date >= '2024-01-01'
    GROUP BY league"""):
    tam[r["league"]] = r["n"]
small = {k for k, v in tam.items() if v < 500}
m_small = m_seleccion & np.isin(lg, list(small))
m_big = m_seleccion & ~np.isin(lg, list(small))
print(f"   ligas con <500 partidos desde 2024: WR {won[m_small].mean() * 100:.1f}% "
      f"({int(m_small.sum()):,} apuestas)")
print(f"   ligas con >=500 partidos          : WR {won[m_big].mean() * 100:.1f}% "
      f"({int(m_big.sum()):,} apuestas)")
vivas_small = sum(c for k, c in ligas_vivas.items() if k in small)
print(f"   de las {len(vivas)} apuestas vivas, {vivas_small} son de ligas pequenas")
conn.close()
