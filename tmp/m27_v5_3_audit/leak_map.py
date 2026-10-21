# =====================================================================
# MAPA DE FUGAS: donde se escapa el acierto vivo
#   1) ¿Los features VIVOS son identicos a los OFFLINE en el mismo
#      partido y minuto? (si NO, ahi esta la causa)
#   2) Expectativa del backtest para cada apuesta viva (por banda de
#      confianza y situacion) vs lo que realmente paso
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
names = list(D["feature_names"])
print("features del dataset:", len(names))

idx_by = {}
for i, m in enumerate(meta):
    idx_by.setdefault(str(m["mid"]), []).append(i)

models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X, mirror(X)])
pe = _ensemble(models, X2)
n = len(X)
p_off = np.clip(models["calibrator"].predict(0.5 * (pe[:n] + 1.0 - pe[n:])), lo, hi)

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

print()
print("1) FEATURES VIVOS vs OFFLINE (mismo partido, mismo minuto):")
rows = conn.execute("""
    SELECT i.match_id, i.game_minute, i.snapshot_minute, i.p_cal, i.pick, i.confidence,
           i.features_json
    FROM model_inference_v3 i
    WHERE i.model_version='m27_v5_3' AND i.features_json IS NOT NULL
    ORDER BY i.match_id, i.game_minute""").fetchall()
print(f"   filas con features: {len(rows)}")

peores = {}
n_cmp = 0
n_exacto = 0
for r in rows:
    mid = str(r["match_id"])
    if mid not in idx_by:
        continue
    try:
        feats = json.loads(r["features_json"])
    except Exception:
        continue
    # el vector vivo se guarda como dict {feature: valor}
    if not isinstance(feats, dict):
        continue
    # fila offline mas cercana en tiempo
    objetivo = float(r["snapshot_minute"] or r["game_minute"])
    mejor = min(idx_by[mid], key=lambda i: abs(float(meta[i]["t"]) - objetivo))
    t_off = float(meta[mejor]["t"])
    if abs(t_off - objetivo) > 0.6:
        continue
    n_cmp += 1
    diffs = {}
    for fname, v in feats.items():
        if fname not in IDX or not isinstance(v, (int, float)):
            continue
        off = X[mejor, IDX[fname]]
        if isinstance(off, np.floating) and np.isnan(off):
            continue
        d = abs(float(v) - float(off))
        diffs[fname] = d
        if d > 1e-6:
            peores[fname] = max(peores.get(fname, 0.0), d)
    if all(d <= 1e-6 for d in diffs.values()) and diffs:
        n_exacto += 1
    elif not diffs:
        n_exacto += 1

print(f"   comparados con su fila offline: {n_cmp} | identicos: {n_exacto} "
      f"({n_exacto * 100 / max(n_cmp, 1):.1f} %)")
if peores:
    print("   features con mayor diferencia maxima:")
    for k, v in sorted(peores.items(), key=lambda kv: -kv[1])[:12]:
        print(f"      {k:<24} max|diff| = {v:.6f}")
else:
    print("   NINGUNA diferencia: los features vivos son identicos a los offline")

print()
print("2) EXPECTATIVA DEL BACKTEST PARA CADA APUESTA VIVA:")
vivas = conn.execute("""
    SELECT b.*, m.league FROM bet_monitor_log_v3 b
    LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id
    WHERE b.model_version='m27_v5_3' AND b.signal_type LIKE 'BET%'
      AND b.result IN ('win','loss') ORDER BY b.id""").fetchall()

exp_total = 0.0
real_total = 0
por_banda = {}
for r in vivas:
    mid = str(r["match_id"])
    conf = float(r["confidence"])
    # expectativa: el WR historico del backtest para la banda de confianza
    banda = (0.75, 0.80) if conf < 0.80 else ((0.80, 0.85) if conf < 0.85 else (0.85, 1.01))
    m_b = (np.maximum(p_off, 1 - p_off) >= banda[0]) & (np.maximum(p_off, 1 - p_off) < banda[1])
    if m_b.sum() == 0:
        continue
    wr_esp = float((np.where(p_off[m_b] >= 0.5, y[m_b] == 1, y[m_b] == 0)).mean())
    real = 1 if r["result"] == "win" else 0
    exp_total += wr_esp
    real_total += real
    d = por_banda.setdefault(banda, {"n": 0, "exp": 0.0, "real": 0})
    d["n"] += 1
    d["exp"] += wr_esp
    d["real"] += real

print(f"   apuestas: {len(vivas)} | ganadas esperadas (backtest por banda): {exp_total:.1f} "
      f"| ganadas reales: {real_total}")
print(f"   -> diferencia: {real_total - exp_total:+.1f} apuestas "
      f"({(real_total - exp_total) * 100 / max(len(vivas), 1):+.1f} pp de acierto)")
print()
print("   por banda de confianza:")
for banda, d in sorted(por_banda.items()):
    print(f"      {banda[0]:.2f}-{banda[1]:.2f}: n={d['n']:>3} | esperadas {d['exp']:>5.1f} "
          f"| reales {d['real']:>3} | WR real {d['real'] * 100 / d['n']:>5.1f}% "
          f"| WR esperada {d['exp'] * 100 / d['n']:>5.1f}% | gap "
          f"{(d['real'] - d['exp']) * 100 / d['n']:>+5.1f} pp")

print()
print("3) Y SI EL BACKTEST SE RESTRINGE A LAS MISMAS LIGAS QUE LAS APUESTAS VIVAS:")
ligas = sorted({r["league"] for r in vivas if r["league"]})
m_lig = np.isin(np.array([m["lg"] for m in meta]), ligas)
conf_all = np.maximum(p_off, 1 - p_off)
won_all = np.where(p_off >= 0.5, y == 1, y == 0)
for nombre, mask in (("todas las ligas", np.ones(len(y), bool)), ("solo las ligas vivas", m_lig)):
    m = mask & (conf_all >= 0.75)
    print(f"   {nombre:<22} n={int(m.sum()):>6,} | WR {won_all[m].mean() * 100:>5.1f}%")
conn.close()
