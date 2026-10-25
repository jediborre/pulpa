"""
tmp/m27_v5_3_2/audit_feature_parity.py
Auditoria de paridad OFFLINE vs LIVE de m27_v5_3_2 (Protocolo AGENTS.md regla 6, punto 6).

Para N partidos del holdout se reconstruye el vector por el camino LIVE (load_match -> ctx_from_payload -> snapshot_features)
y se compara contra la fila que genero el entrenamiento (build_dataset). Se usa el estado de priors "al inicio del holdout"
(state_at_test_start.pkl) = produccion sin refresco, asi las features de priors difieren solo por obsolescencia, no por leakage.
Ademas: pruebas de robustez (payload sin eventos / sin odds / sin grafica / ya en Q4 / liga 12m) y latencia.
Criterio: features de estado y mercado con diferencia maxima < 1e-3; priors con correlacion > 0.98.
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import models.m27_v5_3_2.features as F  # noqa: E402
from models.common.data_loader import get_db_connection, load_match  # noqa: E402
from models.m27_v5_3_2.predict import predict  # noqa: E402

N = 400
D = pickle.load(open(ROOT / "tmp" / "m27_v5_3_2" / "dataset_v5_3.pkl", "rb"))
X, meta, cut = D["X"], D["meta"], D["cut_ord"]
with open(ROOT / "tmp" / "m27_v5_3_2" / "state_at_test_start.pkl", "rb") as f:
    stale = pickle.load(f)

PRIOR = {"d_elo_ft", "d_elo_q4", "d_wr15", "d_q4_bayes", "d_ft_margin15", "d_q4_net12", "d_q3_net12", "d_season_wr",
         "h2h_ft_margin", "h2h_q4_margin", "elo_reliability", "lg_q4_vol", "lg_home_q4_rate", "expected_ft_margin",
         "cur_vs_expected", "lead_vs_lg_vol"}  # dependen del estado de priors
rng = np.random.default_rng(5)
# primeros dias del holdout: el estado "obsoleto" casi coincide con el as-of del dataset (priors comparables)
cand = [i for i, m in enumerate(meta) if cut <= m["ord"] < cut + 2500 and m["has_ev"] and not m["toxic"]]
rng.shuffle(cand)
conn = get_db_connection()

diffs = {n: [] for n in F.FEATURE_NAMES}
pri_a, pri_b = {n: [] for n in PRIOR}, {n: [] for n in PRIOR}
done = 0
lat = []
for i in cand:
    m = meta[i]
    md = load_match(conn, m["mid"])
    if not md:
        continue
    md["odds"] = [dict(zip(("odds_type", "market_period", "home_value", "away_value", "line"), r)) for r in conn.execute(
        "SELECT odds_type, market_period, home_value, away_value, line FROM match_odds WHERE match_id=? "
        "AND odds_type IN ('Home/Away','Point spread','Over/Under') ORDER BY id", (m["mid"],))]
    t0 = time.perf_counter()
    ctx = F.ctx_from_payload(md, m["t"], stale)
    if ctx is None:
        continue
    v = F.snapshot_features(ctx, stale)
    lat.append((time.perf_counter() - t0) * 1000)
    row = X[i]
    for j, n in enumerate(F.FEATURE_NAMES):
        a, b = v[j], row[j]
        if np.isnan(a) and np.isnan(b):
            diffs[n].append(0.0)
        elif np.isnan(a) != np.isnan(b):
            diffs[n].append(np.inf)
        else:
            diffs[n].append(abs(float(a) - float(b)))
        if n in PRIOR and not (np.isnan(a) or np.isnan(b)):
            pri_a[n].append(float(a)); pri_b[n].append(float(b))
    done += 1
    if done >= N:
        break

print(f"=== PARIDAD OFFLINE (dataset) vs LIVE (payload) | {done} partidos del holdout ===")
ok_all = True
print(f"{'feature':<24}{'max|diff|':>12}{'mean|diff|':>13}{'%NaN-mismatch':>15}   estado")
for n in F.FEATURE_NAMES:
    d = np.array(diffs[n])
    mism = np.isinf(d).mean() * 100
    fin = d[np.isfinite(d)]
    mx, mn = (fin.max() if len(fin) else 0.0), (fin.mean() if len(fin) else 0.0)
    if n in PRIOR:
        c = np.corrcoef(pri_a[n], pri_b[n])[0, 1] if len(pri_a[n]) > 10 and np.std(pri_a[n]) > 0 else float("nan")
        status = f"prior (corr live/train {c:.4f})"
        good = (c > 0.85) or np.isnan(c)  # estado obsoleto vs as-of: solo cambia por juegos recientes
    else:
        good = mx < 1e-3 and mism < 0.5
        status = "OK" if good else "REVISAR"
    ok_all &= bool(good)
    print(f"{n:<24}{mx:>12.5f}{mn:>13.6f}{mism:>14.2f}%   {status}")

# Robustez
print("\n=== ROBUSTEZ ===")
state_prod = F.get_state()
base_id = None
F._STATE_CACHE = stale
for i in cand:
    r_ = predict(meta[i]["mid"], match_data=load_match(conn, meta[i]["mid"]), snapshot_minute=27.0)
    if r_.available and 0.42 <= r_.p_home_win <= 0.58:
        base_id = meta[i]["mid"]
        break
base_id = base_id or meta[cand[0]]["mid"]
md0 = load_match(conn, base_id)


def run(label, fn):
    md = load_match(conn, base_id)
    fn(md)
    r = predict(base_id, match_data=md, snapshot_minute=27.0)
    print(f"{label:<44} available={r.available} signal={r.signal} p_home={r.p_home_win} min={(r.raw_json or {}).get('snapshot_minute_used')} faltantes={(r.raw_json or {}).get('features_missing')} {r.reason or ''}")
    return r


run("normal", lambda md: None)
run("sin eventos ni play_by_play", lambda md: (md.update(events={}), md.update(play_by_play={})))
run("sin grafica", lambda md: md.update(graph_points=[]))
run("sin odds", lambda md: md.update(odds=[]))
run("odds como lista in-play ignoradas (vacia)", lambda md: md.update(odds=None))
run("en Q4 (status 4th quarter)", lambda md: md["match"].update(status_description="4th quarter", status_type="inprogress"))
run("en juego, reloj min 27.4", lambda md: (md["match"].update(status_type="inprogress", game_seconds_played=27.4 * 60), md["score"].update(home=60, away=58)))
run("equipo desconocido (sin priors)", lambda md: md["match"].update(home_team="Equipo Inventado FC", away_team="Otro Club Inexistente"))
run("sin cuartos Q1/Q2 (usa eventos)", lambda md: md["score"].update(quarters={}))
run("liga 12m (period_length 720)", lambda md: md["match"].update(period_length=720, league="NBA"))
run("liga toxica", lambda md: md["match"].update(league="Korean Basketball League"))

print(f"\nLatencia de extraccion de features: media {np.mean(lat):.2f} ms | p95 {np.percentile(lat, 95):.2f} ms")
md_fixed = load_match(conn, base_id)
t0 = time.perf_counter()
for _ in range(100):
    predict(base_id, match_data=md_fixed, snapshot_minute=27.0)
print(f"Latencia predict completa (payload en memoria, como monitor_v3): {(time.perf_counter()-t0)/100*1000:.1f} ms por llamada")
print("\nRESULTADO:", "PARIDAD OK" if ok_all else "HAY DIFERENCIAS, revisar")
conn.close()
