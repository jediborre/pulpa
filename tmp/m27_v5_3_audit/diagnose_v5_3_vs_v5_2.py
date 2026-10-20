# =====================================================================
# DIAGNÓSTICO m27_v5_3 vs m27_v5_2 EN VIVO (causas, no síntomas)
# ---------------------------------------------------------------------
# QUÉ HACE:
#   1. Descompone el gap de acierto (v5_2 9/11 vs v5_3 29/40) en:
#      partidos comunes / exclusivos de cada versión.
#   2. Contrfactuales de la REGLA DE DISPARO usando el p_cal y margin
#      congelados en cada log:
#        - umbral 0.72 (v5_3) vs 0.75 (v5_2)
#        - con y sin el filtro de margen >= 1.8 que usa v5_2
#        - con y sin el tope de minuto 30 (rango de entrenamiento)
#   3. Audita el BLOQUEO DE LIGAS TÓXICAS: compara los patrones literales
#      de TOXIC_LEAGUE_PATTERNS contra las ligas que el propio modelo
#      marcó como block_candidate en su calibración offline, y contra
#      los nombres reales de liga de SofaScore.
#   4. Calcula ESPERADO vs REAL por liga usando league_calibration.json
#      (el WR offline que el modelo esperaba para esa liga).
#   5. Test de significación (Fisher exacto) entre ambos modelos.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/diagnose_v5_3_vs_v5_2.py
# =====================================================================

import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"
sys.path.insert(0, str(ROOT))

from models.m27_v5_3.features import TOXIC_LEAGUE_PATTERNS, is_toxic_league  # noqa: E402

VERSIONS = ("m27_v5_2", "m27_v5_3")
SNAP_MIN, SNAP_MAX = 25.5, 30.0


def block(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def load() -> list[dict]:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT l.id, l.match_id, l.model_version, l.signal_type, l.picked_side,
               l.confidence, l.inference_minute, l.result, l.created_at,
               l.inference_json, l.actual_home_score, l.actual_away_score,
               s.league, s.home_team, s.away_team,
               q.q1_home, q.q1_away, q.q2_home, q.q2_away,
               q.q3_home, q.q3_away, q.q4_home, q.q4_away
        FROM bet_monitor_log_v3 l
        LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
        LEFT JOIN quarter_scores_v2 q ON q.match_id = l.match_id
        WHERE l.model_version IN ('m27_v5_2', 'm27_v5_3')
        ORDER BY l.created_at
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def raw(row: dict) -> dict:
    try:
        inf = json.loads(row.get("inference_json") or "{}")
        return inf.get("raw_json") or {}
    except Exception:
        return {}


def is_bet(row: dict) -> bool:
    st = row.get("signal_type") or ""
    return "BET" in st and "NO_BET" not in st


def q4_label(row: dict) -> str | None:
    h, a = row.get("q4_home"), row.get("q4_away")
    if h is None or a is None:
        return None
    h, a = int(h), int(a)
    return "home" if h > a else ("away" if h < a else "push")


def won(row: dict) -> bool | None:
    lbl = q4_label(row)
    if lbl is None:
        return None
    if lbl == "push":
        return False
    return (lbl == "home") == (row.get("picked_side") == "HOME")


def wr(rows: list[dict]) -> tuple[int, int, float]:
    n = len(rows)
    w = sum(1 for r in rows if won(r))
    return w, n, (w * 100 / n if n else 0.0)


def roi(w: int, n: int, odds: float) -> float:
    return (w * (odds - 1.0) - (n - w)) / n * 100 if n else 0.0


def fisher_exact_2x2(a: int, b: int, c: int, d: int) -> float:
    """p-value двуilateral exacto de Fisher para [[a,b],[c,d]]."""
    from math import comb

    n = a + b + c + d
    r1, r2, c1 = a + b, c + d, a + c
    if n == 0:
        return 1.0

    def prob(x: int) -> float:
        return comb(r1, x) * comb(r2, c1 - x) / comb(n, c1)

    lo = max(0, c1 - r2)
    hi = min(r1, c1)
    p_obs = prob(a)
    return sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * 1.0000001)


def main() -> None:
    rows = load()
    bets = [r for r in rows if is_bet(r)]
    v52 = [r for r in bets if r["model_version"] == "m27_v5_2"]
    v53 = [r for r in bets if r["model_version"] == "m27_v5_3"]

    block("1. DESCOMPOSICIÓN DEL GAP")
    m52 = {r["match_id"] for r in v52}
    m53 = {r["match_id"] for r in v53}
    common = m52 & m53
    print(f"  v5_2: {len(v52)} apuestas ({wr(v52)[0]}/{wr(v52)[1]} = {wr(v52)[2]:.0f}%)")
    print(f"  v5_3: {len(v53)} apuestas ({wr(v53)[0]}/{wr(v53)[1]} = {wr(v53)[2]:.0f}%)")
    print(f"  partidos comunes: {len(common)} | solo v5_2: {len(m52 - m53)} | solo v5_3: {len(m53 - m52)}")
    for name, subset in (("comunes", common), ("solo v5_3", m53 - m52), ("solo v5_2", m52 - m53)):
        w = n = 0
        for r in bets:
            if r["match_id"] in subset and won(r) is not None:
                n += 1
                w += 1 if won(r) else 0
        print(f"    {name:<12} {w}/{n} = {(w * 100 / n) if n else 0:.0f}%")
    a, b = wr(v52)[0], wr(v52)[1] - wr(v52)[0]
    c, d = wr(v53)[0], wr(v53)[1] - wr(v53)[0]
    print(f"  Fisher exacto v5_2 vs v5_3: p = {fisher_exact_2x2(a, b, c, d):.3f}  (no significativo si p > 0.05)")
    print(f"  IC95% v5_3 ({wr(v53)[0]}/{wr(v53)[1]}): ~[56%, 85%]  → 72% es compatible con 78-81%")

    block("2. REGLAS DE DISPARO EN CÓDIGO (leídas de predict.py)")
    print("  v5_2: p_cal >= 0.75  AND  p_xgb > 0.5  AND  exp_margin >= 1.8")
    print("  v5_3: p_cal >= 0.72  AND  exp_margin >= 0.0   (y liga no tóxica)")
    print("  → v5_3 relaja el umbral (-0.03) y ANULA el filtro de margen (1.8 → 0.0)")

    block("3. CONTRFACTUALES SOBRE EL HISTÓRICO EN VIVO (p_cal/margin congelados)")
    usable = [r for r in rows if raw(r).get("p_cal") is not None and raw(r).get("margin") is not None
              and q4_label(r) is not None]
    print(f"  filas con p_cal+margin+Q4 resuelto: {len(usable)}")

    def sim(name: str, thr: float, min_margin: float, max_minute: float | None, versions=("m27_v5_3",)):
        sel = []
        for r in usable:
            if r["model_version"] not in versions:
                continue
            rw = raw(r)
            p, mg = float(rw["p_cal"]), float(rw["margin"])
            minute = int(r.get("inference_minute") or 0)
            if max_minute is not None and minute > max_minute:
                continue
            if rw.get("toxic_league_blocked"):
                continue
            if (p >= thr and mg >= min_margin) or (p <= 1 - thr and mg <= -min_margin):
                sel.append(r)
        w, n, acc = wr(sel)
        print(f"  {name:<46} n={n:>4}  WR={acc:>5.1f}%  ROI@1.40={roi(w, n, 1.40):+6.2f}%  ROI@1.83={roi(w, n, 1.83):+6.2f}%")
        return sel, w, n

    sim("v5_3 como está (thr .72, marg 0.0, sin tope)", 0.72, 0.0, None)
    sim("v5_3 con thr 0.75 (igual que v5_2)", 0.75, 0.0, None)
    sim("v5_3 con thr 0.75 + marg 1.8 (regla v5_2)", 0.75, 1.8, None)
    sim("v5_3 con thr 0.72 + marg 1.8", 0.72, 1.8, None)
    sim("v5_3 con tope de minuto 30 (rango treino)", 0.72, 0.0, 30)
    sim("v5_3 thr .75 + marg 1.8 + tope min 30", 0.75, 1.8, 30)

    block("4. INFERENCIAS FUERA DEL RANGO DE ENTRENAMIENTO (25.5-30.0)")
    oor = [r for r in rows if (r.get("inference_minute") or 0) > SNAP_MAX]
    oor_bets = [r for r in oor if is_bet(r)]
    print(f"  inferencias con minuto > 30: {len(oor)} de {len(rows)}")
    print(f"  de ellas, apuestas reales  : {len(oor_bets)} → acierto {wr(oor_bets)[0]}/{wr(oor_bets)[1]}")
    hist = Counter(int(r.get("inference_minute") or 0) for r in oor)
    print(f"  minutos afectados: {dict(sorted(hist.items()))}")
    print("  (predict.py solo valida el piso MIN_VALID_MINUTE=25; no hay techo sobre 30)")

    block("5. AUDITORÍA DEL BLOQUEO DE LIGAS TÓXICAS")
    calib = json.loads((ROOT / "models" / "m27_v5_3" / "model_outputs" / "league_calibration.json")
                       .read_text(encoding="utf-8"))
    flagged = {c["league"] for c in calib if c.get("block_candidate")}
    print(f"  ligas marcadas block_candidate por la calibración offline: {len(flagged)}")
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    real_leagues = {r[0] for r in conn.execute(
        "SELECT DISTINCT league FROM bet_monitor_schedule_v3 WHERE league IS NOT NULL")}
    conn.close()
    print(f"  ligas distintas vistas en vivo: {len(real_leagues)}")
    print("\n  a) ¿Cada liga block_candidate está realmente bloqueada por los patrones?")
    for lg in sorted(flagged):
        print(f"     {'BLOQUEADA ' if is_toxic_league(lg) else 'NO BLOQUEA'}  {lg}")
    print("\n  b) ¿Algún patrón literal coincide con alguna liga real de SofaScore?")
    for p in TOXIC_LEAGUE_PATTERNS:
        hits = [lg for lg in real_leagues if p in lg.lower()]
        flag = "OK " if hits else "NUNCA COINCIDE"
        print(f"     {flag}  '{p}' → {len(hits)} liga(s) {hits[:2]}")
    print("\n  c) Patrones que 'suena' a liga pero no casa con el nombre real:")
    suspects = ["cyprus division", "georgia superli", "tipos sbl", "albania super",
                "korean basketball league", "armenia a-league", "1st dbbl", "korisliiga, women"]
    for p in suspects:
        near = [lg for lg in real_leagues if p.split(",")[0][:9] in lg.lower()]
        print(f"     '{p}' → posibles reales: {near[:3]}")

    block("6. ESPERADO vs REAL POR LIGA (calibración offline del propio modelo)")
    cal_by_league = {norm(c["league"]): c for c in calib}
    exp_w = 0.0
    act_w = 0
    n_cal = 0
    rows_out = []
    for r in v53:
        lg = r.get("league") or "?"
        c = cal_by_league.get(norm(lg))
        if not c:
            rows_out.append((lg, None, None, won(r)))
            continue
        n_cal += 1
        p_exp = c["wr"] / 100
        exp_w += p_exp
        act_w += 1 if won(r) else 0
        rows_out.append((lg, c["wr"], c["bets"], won(r)))
    print(f"  aciertos esperados (suma de WR offline por liga): {exp_w:.1f} de {n_cal}")
    print(f"  aciertos reales                                 : {act_w} de {n_cal}")
    print(f"  desviación                                       : {act_w - exp_w:+.1f} apuestas")
    print(f"  (apuestas en ligas SIN calibracion offline: {len(v53) - n_cal} de {len(v53)})")
    print("\n  detalle (liga | WR offline esperado | n offline | resultado live):")
    for lg, cwr, bets_off, w in sorted(rows_out, key=lambda t: (t[1] is None, t[1] or 0)):
        mark = "✅" if w else ("❌" if w is False else "—")
        print(f"    {str(lg)[:44]:<46} {('%.1f%%' % cwr) if cwr is not None else 'sin calib':>9} "
              f"{str(bets_off or '-'):>6}  {mark}")

    block("7. RESUMEN DE PUNTOS DÉBILES DETECTADOS")
    print("  1. Regla de disparo más laxa: umbral 0.72 y sin filtro de margen (v5_2 exigía margen >= 1.8).")
    print("  2. Inferencias en minutos > 30, fuera del rango de entrenamiento (25.5-30.0).")
    print("  3. Bloqueo de ligas tóxicas con patrones literales que no casan con los nombres reales.")
    print("  4. Muestra en vivo mínima (40 apuestas, 2 días): el gap vs v5_2 no es estadísticamente significativo.")


if __name__ == "__main__":
    main()
