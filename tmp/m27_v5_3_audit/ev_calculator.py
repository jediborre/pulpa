# =====================================================================
# CALCULO DEL EV REAL DE LAS APUESTAS (cuota del mercado Q4 in-play)
# ---------------------------------------------------------------------
# POR QUE ESTA TABLA:
#   La API NO publica el mercado "4th quarter" en PRE-partido (solo
#   "1st half" y "1st quarter"), asi que el EV de una apuesta de Q4 NO se
#   puede backtestear: solo se puede medir HACIA ADELANTE, con la cuota
#   in-play del mercado de Q4 que el monitor ya captura en
#   live_snapshots_v3.odds_live_json (q4_ml_home / q4_ml_away).
#
# QUE HACE:
#   1. Cruza cada apuesta registrada con la cuota del mercado Q4 que había
#      en el minuto de la inferencia.
#   2. Calcula el EV real: EV = p_model * cuota - 1 (para el lado apostado).
#   3. Reporta el WR y el ROI a la cuota REAL (no a la hipotetica 1.40).
#   4. Contrfactual: qué apuestas habría tomado un umbral de EV en vez de
#      un umbral de probabilidad, y cuánto habría cambiado el resultado.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/ev_calculator.py
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"


def block(t: str) -> None:
    print(f"\n{'=' * 96}\n{t}\n{'=' * 96}")


def dec_frac(v) -> float | None:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and "/" in v:
        a, b = v.split("/")
        try:
            return round(int(a) / int(b) + 1.0, 3)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    block("1. APUESTAS REGISTRADAS Y CUOTA DEL MERCADO Q4 EN ESE MINUTO")
    bets = [dict(r) for r in conn.execute(
        """
        SELECT l.id, l.match_id, l.model_version, l.signal_type, l.picked_side,
               l.confidence, l.inference_minute, l.result, l.created_at, l.inference_json,
               s.league
        FROM bet_monitor_log_v3 l
        LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
        WHERE l.signal_type LIKE '%BET%' AND l.signal_type NOT LIKE '%NO_BET%'
        ORDER BY l.created_at DESC
        """)]
    print(f"  apuestas registradas: {len(bets):,}")

    con_cuota = []
    for b in bets:
        snap = conn.execute(
            "SELECT odds_live_json FROM live_snapshots_v3 WHERE match_id=? AND game_minute<=? "
            "ORDER BY game_minute DESC LIMIT 1", (b["match_id"], b["inference_minute"] or 0)).fetchone()
        if not snap:
            continue
        try:
            odds = json.loads(snap["odds_live_json"] or "{}")
        except Exception:
            continue
        h, a = odds.get("q4_ml_home"), odds.get("q4_ml_away")
        if not h or not a:
            continue
        cuota = dec_frac(h if b["picked_side"] == "HOME" else a)
        if not cuota:
            continue
        p = b["confidence"]
        try:
            p = float(p)
            if p > 1.0:
                p /= 100.0
        except (TypeError, ValueError):
            p = None
        if p is None:
            continue
        ev = p * cuota - 1.0
        b["cuota_q4"] = cuota
        b["p_model"] = p
        b["ev"] = ev
        con_cuota.append(b)

    print(f"  con cuota del mercado Q4 en el momento de apostar: {len(con_cuota):,}")
    if not con_cuota:
        print("  (todavia no hay datos: el mercado Q4 in-play se captura desde el 2026-10-20)")
        conn.close()
        return

    block("2. RESULTADO CON LA CUOTA REAL (no la hipotetica de 1.40)")
    res = [b for b in con_cuota if b["result"] in ("win", "loss")]
    if res:
        w = sum(1 for b in res if b["result"] == "win")
        n = len(res)
        stake = 100.0
        profit = sum((b["cuota_q4"] - 1) * stake if b["result"] == "win" else -stake for b in res)
        print(f"  resueltas: {n} | WR {w * 100 / n:.1f} % | cuota media {sum(b['cuota_q4'] for b in res) / n:.2f}")
        print(f"  ROI real a la cuota del mercado Q4: {profit / (n * stake) * 100:+.2f} % "
              f"(frente al -6.4 % que daria la hipotetica de 1.40 con ese WR)")
        print(f"  equilibrio a la cuota media: {100 / (sum(b['cuota_q4'] for b in res) / n):.1f} % de acierto")

    block("3. CONTRFACTUAL: UMBRAL DE EV EN VEZ DE UMBRAL DE PROBABILIDAD")
    for ev_thr in (0.0, 0.02, 0.05, 0.10):
        sel = [b for b in con_cuota if b["ev"] > ev_thr and b["result"] in ("win", "loss")]
        if not sel:
            continue
        w = sum(1 for b in sel if b["result"] == "win")
        stake = 100.0
        profit = sum((b["cuota_q4"] - 1) * stake if b["result"] == "win" else -stake for b in sel)
        print(f"  EV > {ev_thr * 100:>4.0f}%: n={len(sel):>3} | WR {w * 100 / len(sel):>5.1f} % | "
              f"ROI real {profit / (len(sel) * stake) * 100:>+6.2f} % | "
              f"cuota media {sum(b['cuota_q4'] for b in sel) / len(sel):.2f}")

    block("4. BANDAS DE CONFIANZA vs EV (¿la probabilidad predice bien el precio?)")
    bands = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 1.01)]
    print(f"  {'banda p_model':<16}{'n':>5}{'WR real':>10}{'cuota media':>14}{'p media':>10}{'cuota implicita':>17}")
    for lo, hi in bands:
        sel = [b for b in con_cuota if lo <= b["p_model"] < hi and b["result"] in ("win", "loss")]
        if not sel:
            continue
        w = sum(1 for b in sel if b["result"] == "win")
        cu = sum(b["cuota_q4"] for b in sel) / len(sel)
        print(f"  {lo:.2f}-{hi:.2f}{'':<8}{len(sel):>5}{w * 100 / len(sel):>9.1f} %"
              f"{cu:>14.2f}{sum(b['p_model'] for b in sel) / len(sel):>10.3f}{100 / cu:>16.1f} %")
    print("\n  (la 'cuota implicita' es el acierto que el mercado estaba descontando)")
    conn.close()


if __name__ == "__main__":
    main()
