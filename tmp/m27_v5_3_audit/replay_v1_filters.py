# =====================================================================
# REPLAY DE LA PILA DE FILTROS DEL MONITOR v1 (monitor_v1/bet_monitor.py)
# ---------------------------------------------------------------------
# Objetivo: medir si las "capas que se perdieron" (blacklist de ligas,
# ligas volatiles, penalizacion al local, +5 % en fases eliminatorias,
# filtro de edge contra el break-even y stake por Kelly) mejoran el WR y
# el beneficio, en vivo y en el backtest.
#
# NO se construye ningun modelo: solo se aplica el filtro viejo a datos
# que ya existen.
#
# Reglas recuperadas de:
#   monitor_v1/bet_monitor.py:2301-2329  (reglas 6,7,8 del filtro v2)
#   monitor_v1/bet_monitor.py:2382-2420  (filtro v6_2: p_pick, edge, Kelly)
#   monitor_v1/bet_monitor.py:2582-2636  (blacklist, +5 % por fase, por liga)
#   monitor_v1/FILTROS_LIGAS.md:52-102     (patrones SQL del monitor)
# =====================================================================

import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tmp" / "m27_v5_3"))

from models.m27_v5_3.features import IDX, mirror  # noqa: E402
from train_m27_v5_3_pipeline import fit_calibrator, fit_ensemble, p_sym  # noqa: E402

ODDS = 1.40

# ── monitor_v1/FILTROS_LIGAS.md (superconjunto operativo del monitor) ──
LIGAS_EXCLUIDAS = [
    "wnba", "women", "feminina", "femenina", "liga femenina", "lf challenge",
    "playoff", "play off", "playoff", "play out", "u21 espoirs elite",
    "polish basketball league", "supersport premijer liga",
    "prvenstvo hrvatske za d", "aba liga", "argentina liga nacional",
    "basketligaen", "lite 2", "eybl", "i b mckl", "liga 1 masculin",
    "liga nationala", "nbl1", "pba commissioner", "rapid league",
    "stoiximan gbl", "superleague", "superliga", "swedish basketball superettan",
    "swiss cup", "финал", "turkish basketball super league", "nba",
    "big v", "egyptian basketball super league", "lega a basket", "liga e par",
    "liga ouro", "señal", "lnb", "meridianbet kls", "mpbl", "nationale 1",
    "poland 2nd basketball league", "portugal lbp", "portugal proliga",
    "saku i liiga", "serie a2", "slovenian second basketball", "super league",
    "united cup", "united league",
]

# ── bet_monitor.py:2306-2314 (reglas volatiles y penalizacion al local) ──
VOLATILE = ("puerto rico bsn", "cibacopa", "brazil nbb", "israeli national",
            "primera feb", "indonesian bl", "new zealand nbl", "slb")
HOME_PENALTY = ("brazil nbb", "puerto rico bsn", "china cba", "new zealand nbl")
STRONG = ("euroleague", "b1 league", "b2 league", "china cba", "germany bbl", "bnxt")

# ── bet_monitor.py:2588-2599 (fase eliminatoria: +5 % de umbral) ──
PHASE_PLUS5 = ("playoffs", "play offs", "play-in", "knockout", "quarterfinal",
               "quarter-finals", "semifinal", "semi-finals", "final")

# ── bet_monitor.py:2570-2584 (blacklist del filtro v2) ──
BANNED_LEAGUES = ("playoff", "play-in", "knockout", "cup", "final", "segunda vuelta")


def v1_filter(league: str, confidence: float, pick: str, odds: float = ODDS) -> tuple[bool, str]:
    """Replica de la pila del monitor v1. Devuelve (acepta, motivo)."""
    ll = (league or "").lower()

    # 0) blacklist operativa del monitor (superconjunto)
    for pat in LIGAS_EXCLUIDAS:
        if pat in ll:
            return False, f"liga_excluida_v1:{pat}"

    # 1) fases eliminatorias: +5 % de umbral
    phase_extra = 0.0
    for pp in PHASE_PLUS5:
        if pp in ll:
            if "euroleague" in ll and pp == "play-in":
                break
            phase_extra = 0.05
            break

    # 2) reglas por liga (bet_monitor.py:2601-2640)
    if "b1 league" in ll:
        if 0.50 <= confidence <= 0.70:
            return False, "b1_league_zona_50_70"
        return True, "b1_league_ok"
    if "b2 league" in ll:
        if confidence < 0.30 + phase_extra:
            return False, "b2_league_bajo_umbral"
        return True, "b2_league_ok"
    if "primera feb" in ll:
        if confidence < 0.60 + phase_extra:
            return False, "primera_feb_bajo_umbral"
        return True, "primera_feb_ok"
    if "lmb" in ll:
        if confidence < 0.60 + phase_extra:
            return False, "lmb_bajo_umbral"
        return True, "lmb_ok"
    if "colombia lpb" in ll:
        min_conf = (0.50 + phase_extra) if pick == "home" else (0.35 + phase_extra)
        if confidence < min_conf:
            return False, "colombia_lpb_bajo_umbral"
        return True, "colombia_lpb_ok"

    # 3) localVisitante en ligas fuertes (regla 6)
    if pick == "away" and 0.30 <= confidence <= 0.45 and any(s in ll for s in STRONG):
        return True, "visitante_liga_fuerte_stake_reducido"

    # 4) ligas volatiles (regla 7) + penalizacion al local (regla 5)
    is_volatile = any(v in ll for v in VOLATILE)
    min_conf = 0.55 if is_volatile else 0.30
    if any(h in ll for h in HOME_PENALTY) and pick == "home":
        min_conf = max(min_conf, 0.30) + 0.05
    if confidence < min_conf + phase_extra:
        return False, f"volatil_o_umbral:{min_conf + phase_extra:.2f}"
    if is_volatile and confidence >= 0.80:
        return False, "volatil_sobreconfianza"
    return True, "aceptado"


def v62_filter(confidence: float, odds: float = ODDS) -> tuple[bool, str]:
    """Filtro v6_2 (bet_monitor.py:2382-2390): p_pick minimo y edge positivo."""
    if confidence < 0.58:
        return False, "p_pick_below_min:0.58"
    be = 1.0 / odds
    if confidence - be <= 0:
        return False, f"edge_no_positivo:{be:.3f}"
    return True, "edge_ok"


def main() -> None:
    # ─────────── BACKTEST (bloque C) ───────────
    D = pickle.load(open(ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl", "rb"))
    X, y, q4d, meta = D["X"], D["y"], D["q4_diff"], D["meta"]
    elig = D["eligible_ord"]
    rank = {int(o): i for i, o in enumerate(elig)}
    mrank = np.array([rank.get(int(m["ord"]), -1) for m in meta])
    n_el = len(elig)
    rowA = (mrank >= 0) & (mrank < int(n_el * 0.70))
    rowB = (mrank >= a_end) if False else ((mrank >= int(n_el * 0.70)) & (mrank < int(n_el * 0.82)))
    rowC = mrank >= int(n_el * 0.82)
    vals, inv = np.unique(mrank, return_inverse=True)
    w_all = 1.0 / np.bincount(inv)[inv]
    lg_all = np.array([m["lg"] for m in meta])

    models = fit_ensemble(X[rowA], y[rowA], q4d[rowA], w_all[rowA], quick=False)
    idxB = np.where(rowB)[0]
    idxC = np.where(rowC)[0]
    cal = fit_calibrator(p_sym(models, X[idxB]), y[idxB])
    pC = np.clip(cal.predict(p_sym(models, X[idxC])), 0.03, 0.97)
    conf = np.maximum(pC, 1 - pC)
    pick_home = pC >= 0.5
    won = np.where(pick_home, y[idxC] == 1, y[idxC] == 0)
    lgs = lg_all[idxC]

    print("=" * 104)
    print("A) BACKTEST (bloque C) A 0.75: SIN FILTROS vs CON LA PILA DEL v1")
    print("=" * 104)
    base = conf >= 0.75
    print(f"   {'variante':<34}{'apuestas':>10}{'WR':>9}{'u@1.40':>10}{'u@1.83':>10}")
    w = int(won[base].sum())
    k = int(base.sum())
    print(f"   {'SIN filtros (0.75)':<34}{k:>10,}{w / k * 100:>8.1f}%"
          f"{w * 1.40 - k:>10.0f}{w * 1.83 - k:>10.0f}")

    for etiqueta, funcion in (("pila v1 completa", lambda lg, c, p: v1_filter(lg, c, p)),
                              ("solo blacklist de ligas del v1",
                               lambda lg, c, p: (False, "x") if any(
                                   x in (lg or "").lower() for x in LIGAS_EXCLUIDAS) else (True, "ok")),
                              ("solo filtro de edge (v6_2)", lambda lg, c, p: v62_filter(c)),
                              ("pila v1 + edge", lambda lg, c, p: (
                                  v1_filter(lg, c, p) if v1_filter(lg, c, p)[0]
                                  and v62_filter(c)[0] else (False, "x")))):
        keep = np.zeros(len(idxC), bool)
        for i in np.where(base)[0]:
            lg = lgs[i]
            pick = "home" if pick_home[i] else "away"
            ok, _ = funcion(lg, float(conf[i]), pick)
            if ok:
                keep[i] = True
        kk = int(keep.sum())
        if kk == 0:
            print(f"   {etiqueta:<34}{0:>10}")
            continue
        ww = int(won[keep].sum())
        print(f"   {etiqueta:<34}{kk:>10,}{ww / kk * 100:>8.1f}%"
              f"{ww * 1.40 - kk:>10.0f}{ww * 1.83 - kk:>10.0f}")

    print()
    print("   ¿QUE DESCARTA LA PILA v1? (motivos, bloque C a 0.75)")
    motivos = Counter()
    for i in np.where(base)[0]:
        pick = "home" if pick_home[i] else "away"
        ok, mot = v1_filter(lgs[i], float(conf[i]), pick)
        motivos[mot.split(":")[0]] += 1
    for mot, n in motivos.most_common(12):
        print(f"      {mot:<34}{n:>8,}")

    # ─────────── VIVO ───────────
    print()
    print("=" * 104)
    print("B) VIVO (model_inference_v3): que habria apostado la pila del v1")
    print("=" * 104)
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row
    liga_de = {str(r["match_id"]): r["league"] for r in conn.execute("SELECT match_id, league FROM matches")}
    filas = conn.execute("""
        SELECT i.match_id, i.game_minute, i.pick, i.confidence, q.q4_home, q.q4_away
        FROM model_inference_v3 i JOIN quarter_scores_v2 q
          ON CAST(q.match_id AS INTEGER)=i.match_id
        WHERE i.model_version='m27_v5_3' AND i.game_minute<=30
          AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL
        ORDER BY i.match_id, i.game_minute""").fetchall()
    por_partido = {}
    for r in filas:
        por_partido[r["match_id"]] = dict(r)

    sets = {"Sin filtros (0.75)": [], "Pila v1 completa": [], "Solo blacklist ligas": []}
    for mid, r in por_partido.items():
        if r["pick"] not in ("HOME", "AWAY") or r["confidence"] < 0.75:
            continue
        dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
               else (r["q4_away"] - r["q4_home"]))
        if dif == 0:
            continue
        lg = liga_de.get(mid, "")
        pick = "home" if r["pick"] == "HOME" else "away"
        sets["Sin filtros (0.75)"].append(dif > 0)
        ok, mot = v1_filter(lg, r["confidence"], pick)
        if ok:
            sets["Pila v1 completa"].append(dif > 0)
        if not any(x in (lg or "").lower() for x in LIGAS_EXCLUIDAS):
            sets["Solo blacklist ligas"].append(dif > 0)

    print(f"   {'variante':<34}{'apuestas':>10}{'WR':>9}{'u@1.40':>10}{'u@1.83':>10}")
    for etiqueta, res in sets.items():
        kk = len(res)
        if kk == 0:
            print(f"   {etiqueta:<34}{0:>10}")
            continue
        ww = sum(res)
        print(f"   {etiqueta:<34}{kk:>10}{ww / kk * 100:>8.1f}%"
              f"{ww * 1.40 - kk:>10.0f}{ww * 1.83 - kk:>10.0f}")

    print()
    print("   LAS APUESTAS DE HOY QUE LA PILA v1 DESCARTARIA:")
    for mid, r in sorted(por_partido.items(), key=lambda kv: kv[1]["created_at"] if "created_at" in kv[1] else ""):
        if r["pick"] not in ("HOME", "AWAY") or r["confidence"] < 0.75:
            continue
        lg = liga_de.get(mid, "")
        pick = "home" if r["pick"] == "HOME" else "away"
        ok, mot = v1_filter(lg, r["confidence"], pick)
        dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
               else (r["q4_away"] - r["q4_home"]))
        res = "GANADA" if dif > 0 else "PERDIDA"
        print(f"      {str(lg)[:30]:<30} conf {r['confidence']:.3f} | {res:<8} | "
              f"{'PASA' if ok else 'DESCARTADA: ' + mot}")
    conn.close()


from collections import Counter  # noqa: E402

if __name__ == "__main__":
    main()
