"""
tmp/m27_v5_3_3/build_dataset.py
Construye el dataset de entrenamiento de m27_v5_3_3 usando EXACTAMENTE el mismo extractor que el monitor en vivo
(models/m27_v5_3_3/features.py::snapshot_features) y reproduce el PriorState cronologicamente (as-of).

Filas por partido:
  * ANCHOR  t=30.0 (cierre de Q3) para todos los partidos con Q1..Q4 (marcador exacto desde quarter_scores).
  * INTRA   2 snapshots t ~ U(25.5, 29.9) para partidos con eventos de gol validos (match_events con time_seconds,
            2022+). El marcador al minuto t sale del marcador corriente de cada evento.
Esto entrena al modelo para ser resiliente al minuto exacto en que el monitor lo consulta.

Salidas: tmp/m27_v5_3_3/dataset_v5_3.pkl (X, y, q4_diff, meta) y models/m27_v5_3_3/model_outputs/v5_3_state.pkl
Hipotesis a validar: el skew train/live desaparece por construccion y el modelo no pierde calidad entre min 25.5 y 30.
"""
from __future__ import annotations

import copy
import pickle
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3_3.features import (  # noqa: E402
    FEATURE_NAMES, SNAPSHOT_MAX, fallback_info, SNAPSHOT_MIN, STATE_PATH, PriorState, date_ord, extract_market,
    is_excluded_stage, is_toxic_league, score_at, snapshot_features,
)

DB_PATH = ROOT / "matches.db"
OUT = ROOT / "tmp" / "m27_v5_3_3" / "dataset_v5_3_3.pkl"
N_INTRA = 2
rng = np.random.default_rng(20261018)


def load_events(conn) -> dict[int, tuple]:
    t0 = time.time()
    print("[1/5] Cargando eventos de gol con time_seconds...", flush=True)
    cur = conn.execute(
        "SELECT CAST(match_id AS INTEGER), time_seconds, CASE team WHEN 'home' THEN 1 WHEN 'away' THEN -1 END, "
        "points, home_score, away_score FROM match_events "
        "WHERE points > 0 AND time_seconds IS NOT NULL AND home_score IS NOT NULL AND away_score IS NOT NULL "
        "AND team IN ('home','away')"
    )
    chunks = []
    while True:
        rows = cur.fetchmany(1_000_000)
        if not rows:
            break
        chunks.append(np.array(rows, dtype=np.float64))
        print(f"   ... {sum(len(c) for c in chunks):,} eventos ({time.time()-t0:.0f}s)", flush=True)
    A = np.concatenate(chunks)
    o = np.lexsort((A[:, 4] + A[:, 5], A[:, 1], A[:, 0]))
    A = A[o]
    mids, starts = np.unique(A[:, 0].astype(np.int64), return_index=True)
    ends = np.append(starts[1:], len(A))
    out = {}
    for m, s, e in zip(mids, starts, ends):
        blk = A[s:e]
        out[int(m)] = (blk[:, 1], blk[:, 3] * blk[:, 2], blk[:, 4], blk[:, 5])
    print(f"   -> {len(out):,} partidos con eventos ({time.time()-t0:.0f}s)", flush=True)
    return out


def load_graph(conn) -> dict[int, list]:
    print("[2/5] Cargando graph_points...", flush=True)
    g: dict[int, list] = defaultdict(list)
    for mid, mn, v in conn.execute(
        "SELECT CAST(match_id AS INTEGER), minute, value FROM graph_points WHERE minute <= 31 ORDER BY match_id, seq"
    ):
        g[int(mid)].append((mn, v))
    print(f"   -> {len(g):,} partidos con grafica", flush=True)
    return g


def load_odds(conn) -> dict[int, list[dict]]:
    print("[3/5] Cargando cuotas pre-partido (ML / spread / total)...", flush=True)
    o: dict[int, list[dict]] = defaultdict(list)
    for mid, ot, mp, hv, av, ln in conn.execute(
        "SELECT CAST(match_id AS INTEGER), odds_type, market_period, home_value, away_value, line FROM match_odds "
        "WHERE odds_type IN ('Home/Away','Point spread','Over/Under') ORDER BY id"
    ):
        o[int(mid)].append({"odds_type": ot, "market_period": mp, "home_value": hv, "away_value": av, "line": ln})
    print(f"   -> {len(o):,} partidos con cuotas", flush=True)
    return o


def main() -> None:
    t0 = time.time()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=120)
    ev_map = load_events(conn)
    gp_map = load_graph(conn)
    odds_map = load_odds(conn)

    state = PriorState()
    # Agrupar por clean_name: variantes de una misma liga comparten priors.
    agg_lc = {}
    for lg, cn, qd, w, ft, q4 in conn.execute(
        "SELECT league, clean_name, quarter_duration_minutes, is_women, avg_total_points, avg_q4_total_points FROM leagues_classification"
    ):
        key = cn or lg
        qd = int(qd or 10)
        cur = agg_lc.setdefault(key, {"q_dur": qd, "is_w": 0, "lg_avg_ft": [], "lg_avg_q4": [], "n": 0})
        cur["q_dur"] = qd
        if cur["n"] == 0:
            cur["is_w"] = int(w or 0)
        cur["lg_avg_ft"].append(float(ft) if ft is not None else (215.0 if qd == 12 else 158.0))
        cur["lg_avg_q4"].append(float(q4) if q4 is not None else (53.0 if qd == 12 else 39.5))
        cur["n"] += 1
    for key, v in agg_lc.items():
        state.lc_map[key] = {
            "q_dur": v["q_dur"], "is_w": v["is_w"],
            "lg_avg_ft": float(np.mean(v["lg_avg_ft"])),
            "lg_avg_q4": float(np.mean(v["lg_avg_q4"])),
        }
    clean_map = {}
    for lg, cn in conn.execute("SELECT league, clean_name FROM leagues_classification"):
        clean_map[lg] = cn or lg

    print("[4/5] Cargando partidos y cuartos...", flush=True)
    # quarter_scores_v2 es la tabla CANONICA (formato ancho, 1 fila por
    # partido): la escribe el monitor y la migracion de 2026-10-20 la dejo
    # completa (250.098 partidos con el dato DEFINITIVO de fin de partido).
    # Antes se leia `quarter_scores` (formato largo) y se hacia un pivot en
    # memoria; esa tabla sigue existiendo como respaldo historico.
    rows = conn.execute(
        "SELECT m.match_id, m.date, m.league, m.home_team, m.away_team, "
        "q.q1_home, q.q1_away, q.q2_home, q.q2_away, q.q3_home, q.q3_away, "
        "q.q4_home, q.q4_away "
        "FROM matches m JOIN quarter_scores_v2 q ON q.match_id = m.match_id "
        "WHERE m.status_type='finished' AND m.date IS NOT NULL "
        "  AND q.q1_home IS NOT NULL AND q.q1_away IS NOT NULL "
        "  AND q.q2_home IS NOT NULL AND q.q2_away IS NOT NULL "
        "  AND q.q3_home IS NOT NULL AND q.q3_away IS NOT NULL "
        "  AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL "
        "ORDER BY m.date ASC, m.match_id ASC"
    ).fetchall()
    conn.close()
    print(f"   -> {len(rows):,} partidos terminados con cuartos ({time.time()-t0:.0f}s)", flush=True)

    print("[5/5] Replay cronologico + extraccion de features...", flush=True)
    # Corte del holdout (18% final de partidos elegibles: 10m, sin empate en Q4, liga no toxica), igual que v5_1/v5_2.
    # `is_excluded_stage` descarta ademas los formatos de torneo no estandar
    # (amistosos, segunda fase, promotion, preliminares, colocacion): es la
    # MISMA regla que aplica el bloqueo en inferencia, para que train y
    # operacion no diverjan (decision 2026-10-20).
    eligible = [
        i for i, r in enumerate(rows)
        if None not in r[5:13] and r[3] and r[4] and int(r[11]) != int(r[12])
        and (state.lc_map.get(clean_map.get(r[2], r[2])) or {"q_dur": 10})["q_dur"] != 12
        and not is_toxic_league(r[2]) and not is_excluded_stage(r[2])
    ]
    cut_i = eligible[int(len(eligible) * 0.82)]
    state_test_start = None
    print(f"   elegibles {len(eligible):,} | holdout desde partido #{cut_i} ({rows[cut_i][1]})", flush=True)
    X, y, q4d, meta = [], [], [], []
    n_valid_ev = n_bad_ev = n_skip_12m = 0
    for i, r in enumerate(rows):
        if i == cut_i:
            state_test_start = copy.deepcopy(state)  # estado "obsoleto" para simular produccion sin refresco
        mid_s, dt, lg, ht, at = r[0], r[1], r[2], r[3], r[4]
        q = r[5:13]
        if None in q or not ht or not at:
            continue
        q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a = [int(v) for v in q]
        dord = date_ord(dt)
        clean = clean_map.get(lg, lg)
        info = state.lc_map.get(clean) or fallback_info(clean, ht, at)
        mid = int(mid_s)

        if info["q_dur"] == 12:
            n_skip_12m += 1
        elif q4h != q4a:
            ev = ev_map.get(mid)
            cur3 = (q1h + q2h + q3h, q1a + q2a + q3a)
            ev_ok = False
            if ev is not None and len(ev[0]) >= 40:
                sh, sa = score_at(ev, 1800.0)
                ev_ok = abs(sh - cur3[0]) <= 4 and abs(sa - cur3[1]) <= 4 and score_at(ev, 1200.0)[0] > 0
            n_valid_ev += ev_ok
            n_bad_ev += (ev is not None) and not ev_ok

            ml_h, ml_a, sp, tl = extract_market(odds_map.get(mid))
            base = {
                "ht": ht, "at": at, "league": clean, "dord": dord, "q1": (q1h, q1a), "q2": (q2h, q2a), "info": info,
                "gp": gp_map.get(mid), "spread_line": sp, "total_line": tl,
                "ml": (ml_h, ml_a) if ml_h == ml_h else None,
            }

            snaps = [(SNAPSHOT_MAX, cur3, True)]
            if ev_ok:
                for tt in rng.uniform(SNAPSHOT_MIN, 29.9, N_INTRA):
                    sh, sa = score_at(ev, float(tt) * 60.0)
                    snaps.append((float(tt), (sh, sa), False))
            for tt, cur, anchor in snaps:
                ctx = dict(base, minute=tt, cur=cur, ev=ev if ev_ok else None)
                X.append(snapshot_features(ctx, state))
                y.append(1 if q4h > q4a else 0)
                q4d.append(q4h - q4a)
                meta.append({
                    "mid": mid_s, "dt": str(dt).split(" ")[0], "yr": int(str(dt)[:4]), "lg": clean, "ht": ht, "at": at,
                    "t": tt, "anchor": anchor, "has_ev": bool(ev_ok), "toxic": is_toxic_league(lg),
                    "women": bool(info["is_w"]), "ord": i,
                })

        state.update(ht, at, dord, clean, q1h - q1a, q2h - q2a, q3h - q3a, q4h - q4a)
        state.built_through = (str(dt), str(mid_s))
        if (i + 1) % 25000 == 0:
            print(f"   {i+1:,}/{len(rows):,} partidos | filas {len(X):,} | {time.time()-t0:.0f}s", flush=True)

    X = np.array(X, dtype=np.float32)
    print(f"Dataset: X={X.shape} | partidos con eventos validos {n_valid_ev:,} | eventos descartados por incoherencia {n_bad_ev:,} | 12m omitidos {n_skip_12m:,}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as f:
        pickle.dump({"X": X, "y": np.array(y, dtype=np.int32), "q4_diff": np.array(q4d, dtype=np.float32),
                     "meta": meta, "feature_names": FEATURE_NAMES, "cut_ord": cut_i,
                     "eligible_ord": np.array(eligible, dtype=np.int64)}, f, protocol=pickle.HIGHEST_PROTOCOL)
    state.save(STATE_PATH)
    state_test_start.save(OUT.parent / "state_at_test_start.pkl")
    print(f"Guardado: {OUT} y {STATE_PATH} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
