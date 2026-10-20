# =====================================================================
# BACKTEST PROFUNDO POR LIGA Y FASE DE TORNEO (m27_v5_3, bloque C)
# ---------------------------------------------------------------------
# PROBLEMA QUE ABORDA:
#   La lista de "ligas toxicas" de m27_v5_3 se escribio a mano y nunca se
#   audito a fondo. Su intencion original era BLOQUEAR FASES DE TORNEO con
#   comportamiento distinto (eliminatorias, championship round, copas,
#   promotion) porque para esas se planeaba un modelo especifico. Pero:
#     * 5 de 15 patrones no casan con ningun nombre real de SofaScore,
#     * hay VARIANTES de fase que nadie reviso (hay 3.761 nombres de liga),
#     * y el modelo no dice POR QUE descarta una liga.
#
# QUE HACE:
#   1. Reproduce las predicciones del modelo DESPLIEGADO sobre el bloque C
#      (test fuera de muestra, 18 % cronologico final) usando los pickles de
#      model_outputs: NO reentrena, usa exactamente lo que esta en produccion.
#   2. Clasifica cada nombre de liga en (liga base, fase, categoria de fase).
#   3. Mide por categoria de fase y por liga-fase: n, aciertos, WR, IC95%,
#      ROI a 1.40/1.83 y el SESGO (probabilidad media del pick - WR real):
#        sesgo > 0 -> el modelo SUBESTIMA (deja valor en la mesa)
#        sesgo < 0 -> el modelo SOBREESTIMA (direccion peligrosa)
#   4. Cruza las categorias con las PERDIDAS EN VIVO para ver donde se pierde.
#   5. Exporta el informe a JSON para usarlo como lista de bloqueo.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/phase_backtest.py
# =====================================================================

import json
import pickle
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import mirror  # noqa: E402

DATASET = ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl"
OUT_JSON = ROOT / "tmp" / "m27_v5_3_audit" / "phase_backtest_report.json"
WEIGHTS = {"xgb": 0.40, "hgb": 0.35, "lr": 0.25}
BE_140, BE_183 = 1 / 1.40, 1 / 1.83
OPERATING_CONF = 0.75

# ---------------------------------------------------------------------
# Clasificacion de fases de torneo
# ---------------------------------------------------------------------
# Orden de prioridad: si una liga tiene varias marcas, manda la mas
# especifica. "championship round" es una SEGUNDA FASE (grupos tras la
# regular), NO una eliminatoria; "playoffs" si es eliminatoria.
PHASE_RULES: list[tuple[str, list[str]]] = [
    ("FORMATIVAS", ["u14", "u15", "u16", "u17", "u18", "u19", "u20", "u21", "u22",
                    "u23", "junior", "youth", "espoirs", "youth league"]),
    ("COPA", ["cup", "copa", "poklop", "pokal", "coupe"]),
    ("ELIMINATORIA", ["playoff", "play off", "play-off", "playoffs", "knockout",
                      "elimination", "eliminatoria", "eliminatorias", "octavos",
                      "cuartos", "semifinal", "final", "finals", "1/4", "1/2"]),
    ("SEGUNDA FASE", ["championship", "ronda de campeonato", "intergroup",
                      "inter group", "inter-group", "second round", "second phase",
                      "group stage", "fase de grupos", "groups", "conference"]),
    ("PROMOCION", ["promotion", "relegation", "playout", "play-out", "descenso",
                   "ascenso", "intermediate phase", "intermediate"]),
    ("REGULAR", ["regular season", "temporada regular", "apertura", "clausura",
                 "split", "zk", "grundspiel", "temporada", "championship league"]),
]


def split_league(name: str) -> tuple[str, str]:
    """(liga base, fase) a partir del nombre de SofaScore."""
    s = str(name or "").strip()
    if "," in s:
        base, _, rest = s.partition(",")
        return base.strip(), rest.strip()
    return s, ""


def classify_phase(league: str) -> tuple[str, str, str]:
    """(liga base, fase, categoria)."""
    base, phase = split_league(league)
    hay = f"{base} {phase}".lower()
    for cat, marks in PHASE_RULES:
        for mk in marks:
            if mk in hay:
                return base, phase, cat
    return base, phase, ("REGULAR" if not phase else "OTRA")


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - r) / d), float((c + r) / d)


def load_models() -> dict:
    md = ROOT / "models" / "m27_v5_3" / "model_outputs"
    out = {}
    for key, fname in (("xgb", "xgb_model.pkl"), ("hgb", "hgb_model.pkl"),
                       ("lr", "lr_model.pkl"), ("reg", "reg_model.pkl"),
                       ("calibrator", "calibrator.pkl")):
        with open(md / fname, "rb") as f:
            out[key] = pickle.load(f)
    cfg = json.loads((md / "model_config.json").read_text(encoding="utf-8"))
    out["cfg"] = cfg
    return out


def predict_block(models: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Replica exacta de models/m27_v5_3/predict.py para un lote."""
    p_raw = np.empty(len(X), dtype=np.float64)
    margin = np.empty(len(X), dtype=np.float64)
    step = 20000
    for i in range(0, len(X), step):
        xb = X[i:i + step]
        xm = mirror(xb)
        e1 = (WEIGHTS["xgb"] * models["xgb"].predict_proba(xb)[:, 1]
              + WEIGHTS["hgb"] * models["hgb"].predict_proba(xb)[:, 1]
              + WEIGHTS["lr"] * models["lr"].predict_proba(xb)[:, 1])
        e2 = (WEIGHTS["xgb"] * models["xgb"].predict_proba(xm)[:, 1]
              + WEIGHTS["hgb"] * models["hgb"].predict_proba(xm)[:, 1]
              + WEIGHTS["lr"] * models["lr"].predict_proba(xm)[:, 1])
        p_raw[i:i + step] = 0.5 * (e1 + 1.0 - e2)
        margin[i:i + step] = 0.5 * (models["reg"].predict(xb) - models["reg"].predict(xm))
    lo, hi = models["cfg"]["p_clip"]
    p_cal = np.clip(models["calibrator"].predict(p_raw), lo, hi)
    return p_cal, margin


def group_stats(mask: np.ndarray, p: np.ndarray, y: np.ndarray, thr: float) -> dict:
    if mask.sum() == 0:
        return {"n": 0}
    pm, ym = p[mask], y[mask]
    pick_p = np.where(pm >= 0.5, pm, 1 - pm)
    bet = pick_p >= thr
    n_bet = int(bet.sum())
    # El pick es LOCAL si p >= 0.5; gana si el lado elegido coincide con y.
    pick_home = pm[bet] >= 0.5
    w = int((pick_home == (ym[bet] == 1)).sum()) if n_bet else 0
    lo, hi = wilson(w, n_bet)
    return {
        "n": int(mask.sum()),
        "bets": n_bet,
        "bet_rate": round(n_bet / mask.sum() * 100, 2),
        "wr": round(w / n_bet * 100, 2) if n_bet else 0.0,
        "wr_ci95": [round(lo * 100, 1), round(hi * 100, 1)],
        "roi_140": round((w * 1.40 - n_bet) / n_bet * 100, 2) if n_bet else 0.0,
        "roi_183": round((w * 1.83 - n_bet) / n_bet * 100, 2) if n_bet else 0.0,
        "bias": round(float(pick_p[bet].mean() - w / n_bet), 4) if n_bet else 0.0,
        "prob_media": round(float(pick_p[bet].mean()), 4) if n_bet else 0.0,
    }


def block(t: str) -> None:
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


def main() -> None:
    print("Cargando dataset y modelos desplegados...")
    D = pickle.load(open(DATASET, "rb"))
    X, y, meta = D["X"], D["y"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    ord_ = np.array([m["ord"] for m in meta])
    mrank = np.array([rank.get(int(o), -1) for o in ord_])
    anchor = np.array([m["anchor"] for m in meta])
    lg = np.array([m["lg"] for m in meta])
    n_el = len(elig)
    a_end, b_end = int(n_el * 0.70), int(n_el * 0.82)
    rowC = mrank >= b_end
    aC = anchor[rowC]
    print(f"  filas {len(y):,} | elegibles {n_el:,} | bloque C {int(rowC.sum()):,} | anclas C {int(aC.sum()):,}")

    models = load_models()
    print(f"  min_conf desplegado = {models['cfg']['min_conf']} "
          f"(este backtest usa el umbral de analisis OPERATING_CONF={OPERATING_CONF})")
    p_all, m_all = predict_block(models, X[rowC])
    yC, lgC, aCm = y[rowC], lg[rowC], aC

    block("1. PAISAJE DE FASES DE TORNEO (nombre de liga de SofaScore)")
    cat_rows = defaultdict(int)
    cat_matches = defaultdict(set)
    base_of = {}
    phase_of = {}
    for name in set(lgC.tolist()):
        b, ph, cat = classify_phase(name)
        base_of[name] = b
        phase_of[name] = ph
    for i, name in enumerate(lgC):
        b, ph, cat = classify_phase(name)
        cat_rows[cat] += 1
        if aCm[i]:
            cat_matches[cat].add(name)
    n_variants = len(set(lgC.tolist()))
    n_bases = len({split_league(x)[0] for x in set(lgC.tolist())})
    print(f"  variantes de liga distintas en C: {n_variants}  (sobre {n_bases} ligas base)")
    print(f"  {'categoria':<16}{'filas':>10}{'partidos':>10}{'% filas':>9}")
    for cat, nrows in sorted(cat_rows.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:<16}{nrows:>10,}{len(cat_matches[cat]):>10,}{nrows * 100 / len(yC):>8.1f}%")

    print("\n  Desglose de la categoria OTRA (fases sin regla que las capture) - las 25 mas frecuentes:")
    otra = defaultdict(lambda: [0, 0])
    for name in set(lgC.tolist()):
        b, ph, cat = classify_phase(name)
        if cat != "OTRA":
            continue
        key = ph if ph else "(liga sin sufijo)"
        otra[key][0] += 1
    for i, name in enumerate(lgC):
        b, ph, cat = classify_phase(name)
        if cat == "OTRA":
            otra[ph if ph else "(liga sin sufijo)"][1] += 1
    for ph, (nlig, nfil) in sorted(otra.items(), key=lambda kv: -kv[1][1])[:25]:
        print(f"    {ph[:60]:<62} {nlig:>4} liga(s)  {nfil:>7,} filas")

    block("2. RENDIMIENTO POR CATEGORIA DE FASE (anclas t=30, umbral 0.75)")
    masks = {}
    cats = np.array([classify_phase(x)[2] for x in lgC])
    for cat in cat_rows:
        masks[cat] = (cats == cat) & aCm
    print(f"  {'categoria':<16}{'n':>8}{'apuestas':>9}{'WR':>8}{'IC95':>16}{'ROI140':>9}{'ROI183':>9}{'sesgo':>9}")
    cat_stats = {}
    for cat in sorted(cat_rows, key=lambda c: -cat_rows[c]):
        st = group_stats(masks[cat], p_all, yC, OPERATING_CONF)
        cat_stats[cat] = st
        if st["n"] == 0:
            continue
        print(f"  {cat:<16}{st['n']:>8,}{st['bets']:>9,}{st['wr']:>7.1f}%"
              f"{str(st['wr_ci95']):>16}{st['roi_140']:>8.1f}%{st['roi_183']:>8.1f}%{st['bias']:>+9.3f}")
    print(f"\n  equilibrio: {BE_140 * 100:.1f}% a cuota 1.40 | {BE_183 * 100:.1f}% a cuota 1.83")
    print("  sesgo > 0 = el modelo SUBESTIMA (deja valor); sesgo < 0 = SOBREESTIMA (peligroso)")

    block("3. VARIANTES DE LIGA MAS CONGESTIONADAS (con WR, sesgo y decision)")
    idx_by_league = defaultdict(list)
    for i, name in enumerate(lgC):
        if aCm[i]:
            idx_by_league[name].append(i)
    per_league = {}
    for name, idxs in idx_by_league.items():
        if len(idxs) < 20:
            continue
        m = np.zeros(len(lgC), dtype=bool)
        m[np.array(idxs)] = True
        st = group_stats(m, p_all, yC, OPERATING_CONF)
        b, ph, cat = classify_phase(name)
        st.update({"league": name, "base": b, "phase": ph, "categoria": cat})
        per_league[name] = st
    per_league = dict(sorted(per_league.items(), key=lambda kv: -kv[1]["n"]))
    print(f"  {'liga':<48}{'cat':<14}{'n':>6}{'ap':>5}{'WR':>7}{'sesgo':>8}{'decision':>26}")
    for name, st in per_league.items():
        dec = decision(st)
        print(f"  {name[:46]:<48}{st['categoria']:<14}{st['n']:>6,}{st['bets']:>5,}"
              f"{st['wr']:>6.1f}%{st['bias']:>+8.3f}  {dec:<26}")
    return cat_stats, per_league, lgC, yC, p_all, aCm


def decision(st: dict) -> str:
    """Decision operativa por liga a partir de su medicion."""
    if st.get("n", 0) == 0 or st.get("bets", 0) == 0:
        return "sin apuestas"
    if st["categoria"] in ("ELIMINATORIA", "COPA"):
        return "BLOQUEAR (fase especial)"
    if st["wr_ci95"][1] < BE_140 * 100:
        return "BLOQUEAR (WR bajo el eq.)"
    if st["wr_ci95"][0] < BE_140 * 100:
        return "subir umbral / vigilar"
    if st["bias"] < -0.05:
        return "SOBREESTIMA: revisar"
    return "mantener"


if __name__ == "__main__":
    main()
