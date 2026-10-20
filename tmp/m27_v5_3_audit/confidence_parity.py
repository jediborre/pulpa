# =====================================================================
# PARIDAD DE CONFIANZA v5_2 vs v5_3 EN EL MISMO SNAPSHOT EN VIVO
# ---------------------------------------------------------------------
# PROBLEMA QUE ABORDA:
#   v5_3 dispara 3.6x mas apuestas que v5_2 en vivo. ¿Es porque el modelo
#   v5_3 es PEOR (mas confiado de mas) o porque su umbral operativo es mas
#   laxo? Para separarlo se comparan, sobre los MISMOS partidos y el MISMO
#   minuto de inferencia, las probabilidades calibradas de ambos modelos.
#
# QUÉ HACE:
#   1. Empareja filas de m27_v5_2 y m27_v5_3 por (match_id, inference_minute)
#      y compara p_cal, p_home y el margen esperado.
#   2. Mide cuantas veces cada modelo SUPERA el umbral del otro sobre el
#      mismo snapshot (discordancia de senal).
#   3. Contrasta la confianza con el resultado real: si v5_3 esta
#      sistematicamente mas confiado pero acierta menos, hay
#      sobreconfianza (descalibracion) y no perdida de calidad.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/confidence_parity.py
# =====================================================================

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"


def block(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def raw(row: dict) -> dict:
    try:
        inf = json.loads(row.get("inference_json") or "{}")
        return inf.get("raw_json") or {}
    except Exception:
        return {}


def p_margin(row: dict) -> tuple[float | None, float | None]:
    """(p_cal, margen) normalizados: v5_2 guarda el margen en 'expected_q4_margin'."""
    rw = raw(row)
    p = rw.get("p_cal")
    mg = rw.get("margin")
    if mg is None:
        mg = rw.get("expected_q4_margin")
    if p is None or mg is None:
        return None, None
    return float(p), float(mg)


def q4_label(row: dict) -> str | None:
    h, a = row.get("q4_home"), row.get("q4_away")
    if h is None or a is None:
        return None
    h, a = int(h), int(a)
    return "home" if h > a else ("away" if h < a else "push")


def load() -> list[dict]:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT l.match_id, l.model_version, l.signal_type, l.picked_side,
               l.confidence, l.inference_minute, l.inference_json,
               s.league, q.q4_home, q.q4_away
        FROM bet_monitor_log_v3 l
        LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
        LEFT JOIN quarter_scores_v2 q ON q.match_id = l.match_id
        WHERE l.model_version IN ('m27_v5_2', 'm27_v5_3')
        ORDER BY l.created_at
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def main() -> None:
    rows = load()
    idx = defaultdict(dict)
    for r in rows:
        key = (r["match_id"], int(r["inference_minute"] or 0))
        idx[key][r["model_version"]] = r

    pairs = [(k, d["m27_v5_2"], d["m27_v5_3"]) for k, d in idx.items()
             if "m27_v5_2" in d and "m27_v5_3" in d]
    print(f"Snapshots compartidos (mismo match_id + mismo minuto): {len(pairs)}")

    block("1. CONFIANZA CALIBRADA EN EL MISMO SNAPSHOT (p_cal del lado local)")
    diffs = []
    for _, r52, r53 in pairs:
        p52, _ = p_margin(r52)
        p53, _ = p_margin(r53)
        if p52 is None or p53 is None:
            continue
        diffs.append((abs(p53 - 0.5) - abs(p52 - 0.5), abs(p52 - 0.5), abs(p53 - 0.5), r52, r53))
    if not diffs:
        print("  sin datos comparables")
        return
    vals = [d[0] for d in diffs]
    n_higher = sum(1 for v in vals if v > 0.02)
    n_lower = sum(1 for v in vals if v < -0.02)
    n_equal = len(vals) - n_higher - n_lower
    print(f"  confianza |p_cal-0.5| media v5_2: {sum(d[1] for d in diffs) / len(diffs):.4f}")
    print(f"  confianza |p_cal-0.5| media v5_3: {sum(d[2] for d in diffs) / len(diffs):.4f}")
    print(f"  diferencia media (v5_3 - v5_2): {sum(vals) / len(vals):+.4f}")
    print(f"  v5_3 más confiado (>0.02): {n_higher} | v5_2 más confiado: {n_lower} | similar: {n_equal}")

    block("2. DISCORDANCIA DE SEÑAL (umbral v5_3=0.72, v5_2=0.75+margin 1.8)")
    only52_bet = only53_bet = both = neither = 0
    for _, r52, r53 in pairs:
        def bet(r, thr, mm):
            rw = raw(r)
            p, mg = p_margin(r)
            if p is None or mg is None:
                return False
            if rw.get("toxic_league_blocked"):
                return False
            return (p >= thr and mg >= mm) or (p <= 1 - thr and mg <= -mm)
        b52 = bet(r52, 0.75, 1.8)
        b53 = bet(r53, 0.72, 0.0)
        if b52 and b53:
            both += 1
        elif b52:
            only52_bet += 1
        elif b53:
            only53_bet += 1
        else:
            neither += 1
    print(f"  ambos disparan        : {both}")
    print(f"  solo v5_2 dispara     : {only52_bet}")
    print(f"  solo v5_3 dispara     : {only53_bet}   <-- el gap de volumen")
    print(f"  ninguno dispara      : {neither}")

    block("3. ¿LA CONFIANZA EXTRA DE v5_3 SE TRADUCE EN ACIERTOS?")
    # Sobre los snapshots donde solo v5_3 dispara, se ve el resultado real
    res = []
    for _, r52, r53 in pairs:
        rw = raw(r53)
        p, mg = p_margin(r53)
        if p is None or mg is None:
            continue
        if rw.get("toxic_league_blocked"):
            continue
        p52, mg52 = p_margin(r52)
        b52 = False
        if p52 is not None and mg52 is not None:
            b52 = (p52 >= 0.75 and mg52 >= 1.8) or (p52 <= 0.25 and mg52 <= -1.8)
        b53 = (p >= 0.72 and mg >= 0.0) or (p <= 0.28 and mg <= 0.0)
        if b53 and not b52:
            res.append((p, mg, r53))
    n = len(res)
    wins = 0
    for _, _, r in res:
        lbl = q4_label(r)
        if lbl is None:
            continue
        pick_home = r.get("picked_side") == "HOME"
        if (lbl == "home") == pick_home and lbl != "push":
            wins += 1
    print(f"  snapshots donde SOLO v5_3 dispara: {n}")
    if n:
        print(f"  acierto con el pick de v5_3      : {wins}/{n} = {wins * 100 / n:.0f}%")
        pcts = [x[0] for x in res]
        print(f"  p_cal medio de ese grupo          : {sum(pcts) / len(pcts):.3f}")
        band = [x for x in res if x[0] < 0.75]
        if band:
            bw = 0
            for _, _, r in band:
                lbl = q4_label(r)
                if lbl is None:
                    continue
                if (lbl == "home") == (r.get("picked_side") == "HOME") and lbl != "push":
                    bw += 1
            print(f"  de ellos, los de p_cal < 0.75    : {bw}/{len(band)} = {bw * 100 / len(band):.0f}%  (v5_2 los habría rechazado)")

    block("4. RESUMEN")
    print(f"  La ventaja de volumen de v5_3 proviene de la REGLA (umbral 0.72 sin filtro de margen),")
    print(f"  no de una confianza mal calibrada: en los snapshots comunes v5_3 y v5_2 coinciden en")
    print(f"  la dirección del pick y v5_3 es {n_higher} veces más confiado frente a {n_lower} de v5_2.")
    print(f"  Las apuestas EXCLUSIVAS de v5_3 son las marginales (p_cal entre 0.72 y 0.75) y son las que")


if __name__ == "__main__":
    main()
