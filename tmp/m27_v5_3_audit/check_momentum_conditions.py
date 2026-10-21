# =====================================================================
# ¿LAS CONDICIONES DE MOMENTUM (m27_v3) SEPARAN EL ACIERTO?
# ---------------------------------------------------------------------
# El.dispatch por condicion que propone v5_4 solo vale si, en los
# partidos donde la condicion se cumple, el modelo acierta MAS.
#
# Se calculan tick a tick (hasta el min 27) desde match_events:
#   C1 racha activa (>=4 puntos seguidos del mismo equipo)
#   C2 racha maxima en el partido (>=8 puntos)
#   C3 el que va perdiendo viene con racha en los ultimos 3 min
#   C4 densidad de eventos (>=30 eventos hasta el min 27)
#   C5 partido competitive (|dif| <= 8 en el min 27)
#   C6 el trailer de medio tiempo esta cortando en Q3
# y se mide el WR de m27_v5_3 a 0.75 con la condicion presente vs ausente,
# en el backtest (bloque C) y en la serie viva.
# =====================================================================

import pickle
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import IDX, mirror  # noqa: E402
from models.m27_v5_3.predict import _load_production_models, _ensemble  # noqa: E402

MINUTO = 27
CORTE = MINUTO * 60

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

# ── eventos de anotacion con tiempo global ──
print("cargando eventos...", flush=True)
ev = defaultdict(list)
for r in conn.execute(
        "SELECT CAST(match_id AS INTEGER) mid, time_seconds, team, points, home_score, away_score "
        "FROM match_events WHERE points > 0 AND time_seconds IS NOT NULL "
        "AND team IN ('home','away') AND time_seconds <= ?", (CORTE,)):
    ev[r["mid"]].append((float(r["time_seconds"]), r["team"], int(r["points"] or 0),
                         r["home_score"], r["away_score"]))
for k in ev:
    ev[k].sort(key=lambda e: e[0])
print(f"   partidos con eventos hasta el min {MINUTO}: {len(ev):,}")

# ── marcador de medio tiempo (para C6) ──
ht = {}
for r in conn.execute(
        "SELECT CAST(match_id AS INTEGER) mid, q1_home, q1_away, q2_home, q2_away "
        "FROM quarter_scores_v2"):
    if r["q1_home"] is None:
        continue
    ht[r["mid"]] = ((r["q1_home"] or 0) + (r["q2_home"] or 0),
                    (r["q1_away"] or 0) + (r["q2_away"] or 0))


def condiciones(mid: int) -> dict | None:
    evs = ev.get(mid)
    if not evs or len(evs) < 5:
        return None
    dif = (evs[-1][3] or 0) - (evs[-1][4] or 0)
    # racha actual (desde el final hacia atras)
    run_h = run_a = 0
    for t, team, pts, _, _ in reversed(evs):
        if team == "home" and pts > 0:
            run_h += pts
        else:
            break
    for t, team, pts, _, _ in reversed(evs):
        if team == "away" and pts > 0:
            run_a += pts
        else:
            break
    # racha maxima (se reinicia cuando anota el otro equipo)
    best = 0
    run = 0
    ultimo = None
    for t, team, pts, _, _ in evs:
        if pts > 0:
            if ultimo is not None and team != ultimo:
                run = 0
            run += pts
            ultimo = team
            best = max(best, run)
        else:
            run = 0
            ultimo = team
    # ventana de 3 min
    r3 = sum(p if team == "home" else -p for t, team, p, _, _ in evs if t > CORTE - 180)
    h = ht.get(mid)
    ht_diff = (h[0] - h[1]) if h else None
    c6 = None
    if ht_diff is not None:
        if ht_diff < 0 and dif > 0:
            c6 = 1
        elif ht_diff > 0 and dif < 0:
            c6 = 1
        else:
            c6 = 0
    return {
        "dif": dif,
        "run_h": run_h, "run_a": run_a,
        "run_activa": max(run_h, run_a),
        "racha_max": best,
        "n_ev": len(evs),
        "r3": r3,
        "c3": int((dif < 0 and r3 > 0) or (dif > 0 and r3 < 0)),
        "c6": c6,
    }


# ══════════ BACKTEST (bloque C, ancla t=30) ══════════
print("calculando el modelo sobre el dataset...", flush=True)
D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
X, y, meta = D["X"], D["y"], D["meta"]
elig = D["eligible_ord"]
rank = {int(o): i for i, o in enumerate(elig)}
mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
rowC = mrank >= int(len(elig) * 0.82)
idxC = np.where(rowC)[0]
models = _load_production_models()
lo, hi = models["cfg"]["p_clip"]
X2 = np.vstack([X[idxC], mirror(X[idxC])])
pe = _ensemble(models, X2)
pC = np.clip(models["calibrator"].predict(0.5 * (pe[:len(idxC)] + 1.0 - pe[len(idxC):])), lo, hi)
confC = np.maximum(pC, 1 - pC)
wonC = np.where(pC >= 0.5, y[idxC] == 1, y[idxC] == 0)

# una fila por partido: la del ancla (t=30) o la de mayor t
midsC = [str(meta[i]["mid"]) for i in idxC]
tsC = [float(meta[i]["t"]) for i in idxC]
mejor = {}
for j, (m, t) in enumerate(zip(midsC, tsC)):
    if m not in mejor or t > mejor[m][0]:
        mejor[m] = (t, j)

sel_C = []
for m, (t, j) in mejor.items():
    if confC[j] < 0.75:
        continue
    c = condiciones(int(m))
    if c is None:
        continue
    sel_C.append((m, c, bool(wonC[j])))
print(f"   backtest: {len(sel_C):,} apuestas a 0.75 con condiciones calculables "
      f"(de {len(mejor):,} partidos del bloque C)")

# ══════════ VIVO ══════════
filas = conn.execute("""
    SELECT i.match_id, i.pick, i.confidence, q.q4_home, q.q4_away
    FROM model_inference_v3 i JOIN quarter_scores_v2 q
      ON CAST(q.match_id AS INTEGER)=i.match_id
    WHERE i.model_version='m27_v5_3' AND i.game_minute<=30
      AND i.confidence>=0.75 AND q.q4_home IS NOT NULL
    ORDER BY i.match_id, i.game_minute""").fetchall()
vivo = {}
for r in filas:
    d = (r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME" else (r["q4_away"] - r["q4_home"])
    if d == 0:
        continue
    vivo[str(r["match_id"])] = (condiciones(int(r["match_id"])), d > 0)
print(f"   vivo: {len(vivo)} apuestas a 0.75 con Q4")

CONDS = {
    "C1 racha activa >=4": lambda c: c["run_activa"] >= 4,
    "C2 racha max >=8": lambda c: c["racha_max"] >= 8,
    "C3 el que pierde viene con racha 3m": lambda c: c["c3"] == 1,
    "C4 densidad >=30 eventos": lambda c: c["n_ev"] >= 30,
    "C5 competitive (|dif|<=8)": lambda c: abs(c["dif"]) <= 8,
    "C6 trailer de medio tiempo corta": lambda c: c["c6"] == 1,
}

for nombre, datos, etiqueta in (("BACKTEST (bloque C, 0.75)", sel_C, "bt"),
                                 ("VIVO (0.75)", list(vivo.items()), "vivo")):
    print()
    print("=" * 100)
    print(f"{nombre}")
    print("=" * 100)
    print(f"   {'condicion':<32}{'con la condicion':>28}{'sin la condicion':>28}")
    for etiq, fn in CONDS.items():
        con = [(c, w) for _m, c, w in datos] if etiqueta == "bt" else [
            (c, w) for _m, (c, w) in datos if c is not None and fn(c)]
        sin = [(c, w) for _m, c, w in datos if not fn(c)] if etiqueta == "bt" else [
            (c, w) for _m, (c, w) in datos if c is not None and not fn(c)]
        def fmt(lst):
            if not lst:
                return "n=0"
            w = sum(1 for _c, ww in lst if ww)
            return f"n={len(lst):>5,} WR {w / len(lst) * 100:>5.1f}%"
        print(f"   {etiq:<32}{fmt(con):>28}{fmt(sin):>28}")
    if etiqueta == "vivo":
        print()
        print("   detalle de las apuestas vivas:")
        for m, (c, w) in vivo.items():
            if c is None:
                print(f"      {m}: sin eventos hasta el min 27")
                continue
            print(f"      {m}: dif {c['dif']:>4} | run_act {c['run_activa']:>2} | "
                  f"racha_max {c['racha_max']:>2} | ev {c['n_ev']:>3} | r3 {c['r3']:>3} | "
                  f"C3 {c['c3']} C6 {c['c6']} | {'GANADA' if w else 'PERDIDA'}")

# ══════════ C1 (racha activa) x BANDA DE CONFIANZA (backtest) ══════════
print()
print("=" * 100)
print("C1 (racha activa >=4) x BANDA DE CONFIANZA — donde se dispatchearia")
print("=" * 100)
print(f"   {'banda':<16}{'C1 si':>26}{'C1 no':>26}")


def fmt(lst):
    if not lst:
        return "n=0"
    w = sum(1 for _c, ww in lst if ww)
    return f"n={len(lst):>5,} WR {w / len(lst) * 100:>5.1f}%"


for lo_b, hi_b in ((0.75, 0.80), (0.80, 0.85), (0.85, 1.01)):
    sub = []
    for m, (t, j) in mejor.items():
        if not (lo_b <= confC[j] < hi_b):
            continue
        c = condiciones(int(m))
        if c is None:
            continue
        sub.append((m, c, bool(wonC[j])))
    con = [(c, w) for _m, c, w in sub if c["run_activa"] >= 4]
    sin = [(c, w) for _m, c, w in sub if c["run_activa"] < 4]
    print(f"   {lo_b:.2f}-{hi_b:.2f}{'':<4}{fmt(con):>26}{fmt(sin):>26}")

print()
print("   ¿Y la probabilidad MEDIA de la banda C1 vs no C1 (calibracion)?")
for etiq, fn in (("C1 si", lambda c: c["run_activa"] >= 4), ("C1 no", lambda c: c["run_activa"] < 4)):
    sel = []
    for m, (t, j) in mejor.items():
        if confC[j] < 0.75:
            continue
        c = condiciones(int(m))
        if c is None or not fn(c):
            continue
        sel.append((confC[j], bool(wonC[j])))
    if not sel:
        continue
    pm = np.mean([s[0] for s in sel])
    wr = np.mean([1.0 if s[1] else 0.0 for s in sel])
    print(f"      {etiq:<8} n={len(sel):>5,} | conf media {pm:.4f} | WR real {wr * 100:.1f}% "
          f"| gap {(wr - pm) * 100:+.1f} pp")
conn.close()
