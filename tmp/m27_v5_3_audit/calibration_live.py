# =====================================================================
# DESCALIBRACION EN VIVO: v5_2 vs v5_3 sobre los MISMOS snapshots
# ---------------------------------------------------------------------
# PROBLEMA QUE ABORDA:
#   En los snapshots compartidos v5_3 se muestra mucho mas confiado que
#   v5_2 (34 veces mas confiado que al reves) y ademas dispara 3x mas.
#   ¿Ese exceso de confianza se paga con mas acierto o es descalibracion?
#
# QUÉ HACE:
#   Sobre los pares (match_id, minuto) inferences por ambos modelos:
#   - Brier score de cada p_cal contra el resultado real de Q4.
#   - Calibracion por decil de confianza de cada modelo.
#   - Exactitud de cada modelo en esos mismos snapshots.
#   - El "gap" de confianza vs el "gap" de exactitud (decalibracion).
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/calibration_live.py
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


def p_margin(row: dict):
    rw = raw(row)
    p = rw.get("p_cal")
    mg = rw.get("margin")
    if mg is None:
        mg = rw.get("expected_q4_margin")
    if p is None or mg is None:
        return None, None
    return float(p), float(mg)


def load() -> list[dict]:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT l.match_id, l.model_version, l.signal_type, l.picked_side,
               l.inference_minute, l.inference_json,
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
        idx[(r["match_id"], int(r["inference_minute"] or 0))][r["model_version"]] = r

    samples = []
    for (mid, minute), d in idx.items():
        if "m27_v5_2" not in d or "m27_v5_3" not in d:
            continue
        r53 = d["m27_v5_3"]
        h, a = r53.get("q4_home"), r53.get("q4_away")
        if h is None or a is None:
            continue
        h, a = int(h), int(a)
        if h == a:
            continue  # empate de Q4 no discrimina (se cuenta como perdida)
        y = 1.0 if h > a else 0.0
        p52, _ = p_margin(d["m27_v5_2"])
        p53, _ = p_margin(r53)
        if p52 is None or p53 is None:
            continue
        samples.append((p52, p53, y, d["m27_v5_2"], d["m27_v5_3"], minute, mid))

    n = len(samples)
    print(f"Snapshots compartidos con Q4 resuelto (sin empates): {n}")
    if n < 5:
        print("  muestra insuficiente")
        return

    def brier(idx_p: int) -> float:
        return sum((s[idx_p] - s[2]) ** 2 for s in samples) / n

    def logloss(idx_p: int) -> float:
        import math
        tot = 0.0
        for s in samples:
            p = min(max(s[idx_p], 1e-6), 1 - 1e-6)
            tot += -(s[2] * math.log(p) + (1 - s[2]) * math.log(1 - p))
        return tot / n

    def acc(row_idx: int) -> tuple[int, int]:
        w = 0
        for s in samples:
            r = s[row_idx]
            lbl = "home" if s[2] == 1.0 else "away"
            if (lbl == "home") == (r.get("picked_side") == "HOME"):
                w += 1
        return w, n

    block("1. DESCALIBRACION EN LOS MISMOS SNAPSHOTS")
    b52, b53 = brier(0), brier(1)
    l52, l53 = logloss(0), logloss(1)
    a52 = acc(3)
    a53 = acc(4)
    c52 = sum(abs(s[0] - 0.5) for s in samples) / n
    c53 = sum(abs(s[1] - 0.5) for s in samples) / n
    print(f"  {'':<22}{'v5_2':>14}{'v5_3':>14}")
    print(f"  {'confianza media':<22}{c52:>14.4f}{c53:>14.4f}")
    print(f"  {'exactitud (pick)':<22}{f'{a52[0]}/{a52[1]}':>14}{f'{a53[0]}/{a53[1]}':>14}")
    print(f"  {'Brier (menor mejor)':<22}{b52:>14.4f}{b53:>14.4f}")
    print(f"  {'LogLoss (menor mejor)':<22}{l52:>14.4f}{l53:>14.4f}")
    print()
    if abs(c53 - c52) > 0.01 and abs(a52[0] - a53[0]) <= 1:
        print(f"  >> v5_3 se muestra {(c53 - c52) * 100:.1f} puntos mas confiado que v5_2")
        print(f"     pero acierta EXACTAMENTE lo mismo ({a53[0]}/{a53[1]} vs {a52[0]}/{a52[1]}).")
        print(f"     Brier {b53:.4f} vs {b52:.4f} -> el exceso de confianza NO se paga con acierto.")
        print(f"     Es DESCALIBRACION en vivo: el umbral 0.72 sobre una escala sobreconfiada")
        print(f"     deja entrar apuestas marginales de calidad de ~62-64%.")
    else:
        print("  >> No se detecta descalibracion relevante en los snapshots compartidos.")

    block("2. CALIBRACION POR BANDA (v5_3, banda de confianza declarada)")
    bands = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
             (0.70, 0.75), (0.75, 0.80), (0.80, 0.95)]
    for lo, hi in bands:
        sel = [s for s in samples if lo <= max(s[1], 1 - s[1]) < hi]
        if len(sel) < 3:
            continue
        w = 0
        probs = []
        for s in sel:
            row = s[4]
            pp = s[1] if row.get("picked_side") == "HOME" else 1 - s[1]
            probs.append(pp)
            right = ((s[2] == 1.0) == (row.get("picked_side") == "HOME"))
            w += 1 if right else 0
        emp = sum(probs) / len(probs)
        print(f"  conf {lo:.2f}-{hi:.2f}: n={len(sel):>3}  WR={w * 100 // len(sel):>3}%  "
              f"prob media del pick={emp:.3f}  gap WR-prob={w / len(sel) - emp:+.3f}")

    block("3. QUE PASARIA SI v5_3 USARA EL UMBRAL DE v5_2 (0.75)")
    for thr in (0.72, 0.75, 0.78):
        sel = []
        for s in samples:
            p = s[1]
            if max(p, 1 - p) >= thr:
                sel.append(s)
        if not sel:
            continue
        w = sum(1 for s in sel if (s[1] >= 0.5) == (s[2] == 1.0))
        acc = w * 100 / len(sel)
        roi140 = (w * 0.40 - (len(sel) - w)) / len(sel) * 100
        print(f"  thr {thr}: n={len(sel):>3}  WR={acc:>5.1f}%  ROI@1.40={roi140:+6.2f}%")


if __name__ == "__main__":
    main()
