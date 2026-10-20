# =====================================================================
# HEAD-TO-HEAD EN VIVO: m27_v5_2 vs m27_v5_3 (bet_monitor_log_v3)
# ---------------------------------------------------------------------
# QUÉ HACE:
#   Compara apples-to-apples el desempeño en vivo de ambas versiones:
#   - volumen, acierto, ROI @1.40 y @1.83
#   - distribución de resultados (incluye cómo se liquidaron los empates
#     de Q4: 'push' vs 'loss', porque la regla cambió en el commit 9121cb6)
#   - umbral de confianza y de margen que cada versión usó en vivo
#   - solapamiento: partidos donde AMBAS versionesemanager apostaron
#     (mismo partido, distinta decisión) -> desacopla "mismo set de partidos"
#     de "distinto set de partidos"
#   - banda de confianza: dónde se concentran las pérdidas de cada una
#   - calidad por liga
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/head_to_head.py
# =====================================================================

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"
VERSIONS = ("m27_v5_2", "m27_v5_3")


def block(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


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


def q4_result(row: dict) -> str | None:
    """Reetiqueta según el parcial de Q4 con la regla ACTUAL (empate = pérdida)."""
    h, a = row.get("q4_home"), row.get("q4_away")
    if h is None or a is None:
        return None
    h, a = int(h), int(a)
    if h == a:
        return "push"
    return "home" if h > a else "away"


def is_bet(row: dict) -> bool:
    st = row.get("signal_type") or ""
    return "BET" in st and "NO_BET" not in st


def won(row: dict, label: str | None) -> bool | None:
    if label is None:
        return None
    pick = row.get("picked_side")
    if label == "push":
        return False  # regla vigente: el empate cuenta como pérdida
    return (label == "home") == (pick == "HOME")


def roi(wins: int, n: int, odds: float) -> float | None:
    if not n:
        return None
    return (wins * (odds - 1.0) - (n - wins)) / n * 100


def summarize(rows: list[dict], version: str, use_logged: bool) -> dict:
    """use_logged=True usa el campo `result` guardado; False recalcula con Q4."""
    bets = [r for r in rows if r["model_version"] == version and is_bet(r)]
    out = {"bets": bets, "n": len(bets)}
    labels = []
    for r in bets:
        lbl = q4_result(r)
        labels.append(lbl)
    w_logged = sum(1 for r in bets if str(r.get("result")) in ("win", "hit"))
    w_recalc = sum(1 for r, l in zip(bets, labels) if won(r, l))
    out["w_logged"] = w_logged
    out["w_recalc"] = w_recalc
    out["resolved_logged"] = sum(1 for r in bets if r.get("result") in ("win", "hit", "loss", "miss", "push"))
    out["resolved_recalc"] = sum(1 for l in labels if l is not None)
    return out


def main() -> None:
    rows = load()
    print(f"Filas totales en bet_monitor_log_v3 (v5_2 + v5_3): {len(rows)}")
    print(f"\nFilas por versión: {dict(Counter(r['model_version'] for r in rows))}")
    print(f"\nTipos de señal: {dict(Counter((r['model_version'], r['signal_type']) for r in rows))}")

    block("1. DESGLOSE DE RESULTADOS GUARDADOS (campo `result`)")
    for v in VERSIONS:
        c = Counter(r.get("result") for r in rows
                    if r["model_version"] == v and is_bet(r))
        print(f"  {v:<12} {dict(c)}")

    block("2. DESGLOSE REETIQUETADO CON Q4 (regla vigente: empate = pérdida)")
    for v in VERSIONS:
        c = Counter(q4_result(r) or "sin_dato" for r in rows
                    if r["model_version"] == v and is_bet(r))
        print(f"  {v:<12} {dict(c)}")

    block("3. RENDIMIENTO GLOBAL EN VIVO (recalculado con Q4)")
    for v in VERSIONS:
        bets = [r for r in rows if r["model_version"] == v and is_bet(r)]
        labeled = [(r, q4_result(r)) for r in bets]
        labeled = [(r, l) for r, l in labeled if l is not None]
        n = len(labeled)
        w = sum(1 for r, l in labeled if won(r, l))
        print(f"  {v}")
        print(f"    apuestas        : {len(bets)}   resueltas: {n}")
        print(f"    acierto         : {w}/{n} = {round(w * 100 / n) if n else 0}%")
        print(f"    ROI @1.40       : {roi(w, n, 1.40):+.2f}%" if n else "    ROI @1.40       : —")
        print(f"    ROI @1.83       : {roi(w, n, 1.83):+.2f}%" if n else "    ROI @1.83       : —")
        if bets:
            print(f"    periodo         : {bets[0]['created_at'][:10]} → {bets[-1]['created_at'][:10]}")
            print(f"    picks           : {dict(Counter(r['picked_side'] for r in bets))}")
            print(f"    minutos         : {dict(sorted(Counter(int(r['inference_minute'] or 0) for r in bets).items()))}")
            thr = Counter(raw(r).get("min_conf_threshold") for r in bets)
            mm = Counter(raw(r).get("min_margin") for r in bets)
            print(f"    min_conf usado  : {dict(thr)}")
            print(f"    min_margin usado: {dict(mm)}")

    block("4. SOLAPAMIENTO DE PARTIDOS (mismos partidos, ambas versiones)")
    by_match = defaultdict(dict)
    for r in rows:
        if is_bet(r):
            by_match[r["match_id"]][r["model_version"]] = r
    both = {m: d for m, d in by_match.items() if len(d) == 2}
    print(f"  partidos con apuesta en AMBAS versiones: {len(both)}")
    agree = 0
    for m, d in both.items():
        p2, p3 = d["m27_v5_2"].get("picked_side"), d["m27_v5_3"].get("picked_side")
        if p2 == p3:
            agree += 1
    if both:
        print(f"  de esos, misma dirección de pick: {agree}/{len(both)}")
    print(f"  partidos solo v5_2: {sum(1 for d in by_match.values() if 'm27_v5_2' in d and 'm27_v5_3' not in d)}")
    print(f"  partidos solo v5_3: {sum(1 for d in by_match.values() if 'm27_v5_3' in d and 'm27_v5_2' not in d)}")

    block("5. SOLO PARTIDOS COMUNES: ¿quién acertó más?")
    for v in VERSIONS:
        w = n = 0
        for m, d in both.items():
            lbl = q4_result(d[v])
            if lbl is None:
                continue
            n += 1
            if won(d[v], lbl):
                w += 1
        print(f"  {v:<12} {w}/{n} = {round(w * 100 / n) if n else 0}%")

    block("6. ACIERTO POR BANDA DE CONFIANZA (recalculado)")
    bands = [(0.60, 0.68), (0.68, 0.72), (0.72, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 1.01)]
    print(f"  {'banda':<14}" + "".join(f"{v:>22}" for v in VERSIONS))
    for lo, hi in bands:
        line = f"  {lo:.2f}-{hi:.2f}   "
        for v in VERSIONS:
            sel = [(r, q4_result(r)) for r in rows
                   if r["model_version"] == v and is_bet(r) and q4_result(r) is not None]
            sel = [(r, l) for r, l in sel if lo <= float(r.get("confidence") or 0) < hi]
            n = len(sel)
            w = sum(1 for r, l in sel if won(r, l))
            line += f"{f'{w}/{n} ({round(w * 100 / n) if n else 0}%)':>22}"
        print(line)

    block("7. ACIERTO POR LIGA (v5_2 vs v5_3, ligas con ≥2 apuestas en alguna)")
    league_sets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if not is_bet(r):
            continue
        lbl = q4_result(r)
        if lbl is None:
            continue
        league_sets[r.get("league") or "?"][r["model_version"]].append((r, lbl))
    interesting = [lg for lg, d in league_sets.items()
                   if len(d["m27_v5_2"]) + len(d["m27_v5_3"]) >= 2]
    interesting.sort(key=lambda lg: -(len(league_sets[lg]["m27_v5_2"]) + len(league_sets[lg]["m27_v5_3"])))
    for lg in interesting:
        parts = []
        for v in VERSIONS:
            sel = league_sets[lg][v]
            n = len(sel)
            w = sum(1 for r, l in sel if won(r, l))
            parts.append(f"{v}: {w}/{n}" if n else f"{v}: —")
        print(f"  {lg[:52]:<54} {' | '.join(parts)}")

    block("8. MOMENTO DE LA APUESTA (inferencias no-BET = rechazos)")
    for v in VERSIONS:
        vrows = [r for r in rows if r["model_version"] == v]
        nb = [r for r in vrows if not is_bet(r)]
        print(f"  {v:<12} total filas {len(vrows):>4} | BET {len(vrows) - len(nb):>4} | no-BET {len(nb):>4}")


if __name__ == "__main__":
    main()
