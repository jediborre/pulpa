# =====================================================================
# BARRIDO DE min_conf PARA m27_v5_3 (¿qué pasa si bajo el umbral?)
# ---------------------------------------------------------------------
# QUÉ HACE:
#   Reconstruye la decisión del modelo con umbrales alternativos usando los
#   valores YA GUARDADOS en cada log (inference_json.raw_json):
#       p_cal  → probabilidad calibrada del local
#       margin → margen esperado en Q4
#       min_conf_threshold / min_margin → umbrales vigentes al apostar
#   Regla real del modelo:
#       BET si (p_cal >= thr y margin >= mm) o (p_cal <= 1-thr y margin <= -mm)
#   El resultado (win/loss) se toma del parcial de Q4 ya liquidado.
#
# POR QUÉ ASÍ Y NO RE-INFIRIENDO:
#   El payload guardado está congelado en el minuto de la inferencia y
#   `ctx_from_payload` deriva el minuto del reloj en vivo; re-inferir con otro
#   snapshot_minute daría exactamente lo mismo. Con p_cal/margin guardados sí
#   se puede simular el umbral con fidelidad.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/threshold_sweep.py
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"

THRESHOLDS = [0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85]


def load() -> list[dict]:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT l.match_id, l.signal_type, l.picked_side, l.confidence, l.result,
               l.inference_json, l.inference_minute, s.league,
               q.q4_home, q.q4_away
        FROM bet_monitor_log_v3 l
        LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
        LEFT JOIN quarter_scores_v2 q ON q.match_id = l.match_id
        WHERE l.model_version = 'm27_v5_3'
          AND q.q4_home IS NOT NULL AND q.q4_away IS NOT NULL
        ORDER BY l.created_at
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows if dict(r).get("q4_home") is not None]


def _model_raw(row: dict) -> dict:
    try:
        inf = json.loads(row.get("inference_json") or "{}")
        return inf.get("raw_json") or {}
    except Exception:
        return {}


def main() -> None:
    rows = load()
    # Solo filas con la info del modelo necesaria para simular
    data = []
    for r in rows:
        raw = _model_raw(r)
        p_cal, margin = raw.get("p_cal"), raw.get("margin")
        if p_cal is None or margin is None:
            continue
        q4h, q4a = int(r["q4_home"]), int(r["q4_away"])
        if q4h == q4a:
            continue  # empate: ahora cuenta como pérdida, no aporta al barrido
        home_won = q4h > q4a
        data.append(
            {
                "match_id": r["match_id"],
                "league": r.get("league") or "?",
                "p_cal": float(p_cal),
                "margin": float(margin),
                "thr": float(raw.get("min_conf_threshold") or 0.72),
                "mm": float(raw.get("min_margin") or 0),
                "home_won": home_won,
                "minute": int(r.get("inference_minute") or 0),
            }
        )

    print(f"Filas utilizables (con p_cal + margin + Q4 resuelto, sin empates): {len(data)}")
    if not data:
        return
    orig = [d for d in data if ("BET" in "BET") and True]
    print(f"Umbral vigente en producción: {sorted({d['thr'] for d in data})}")
    print(f"Margen mínimo vigente      : {sorted({d['mm'] for d in data})}")

    print("\n" + "=" * 78)
    print(f"{'min_conf':>8} | {'apuestas':>8} {'acierto':>9} | {'yield@1.40':>11} {'yield@1.83':>11} | {'nuevas':>7} {'acierto nuevas':>14}")
    print("=" * 78)

    def simulate(thr: float) -> list[dict]:
        out = []
        for d in data:
            mm = d["mm"]
            bet = (d["p_cal"] >= thr and d["margin"] >= mm) or (
                d["p_cal"] <= 1.0 - thr and d["margin"] <= -mm
            )
            if not bet:
                continue
            pick_home = d["p_cal"] >= 0.5
            won = d["home_won"] == pick_home
            out.append({**d, "won": won, "pick_home": pick_home})
        return out

    def yield_pct(sim: list[dict], odds: float) -> str:
        n = len(sim)
        if not n:
            return "—"
        w = sum(1 for s in sim if s["won"])
        l = n - w
        value = (w * (odds - 1.0) - l) / n * 100
        return f"{value:+.1f}%"

    base = {d["match_id"]: d for d in simulate(0.72)}
    for thr in THRESHOLDS:
        sim = simulate(thr)
        n = len(sim)
        w = sum(1 for s in sim if s["won"])
        acc = round(w * 100 / n) if n else 0
        new = [s for s in sim if s["match_id"] not in base]
        nw = sum(1 for s in new if s["won"])
        nacc = f"{round(nw * 100 / len(new))}% ({nw}✅/{len(new) - nw}❌)" if new else "—"
        print(
            f"{thr:>8.2f} | {n:>8} {acc:>8}% | {yield_pct(sim, 1.40):>11} {yield_pct(sim, 1.83):>11} "
            f"| {len(new):>7} {nacc:>14}"
        )

    print("\n" + "=" * 78)
    print("DETALLE DE LAS APUESTAS QUE SE AÑADIRÍAN al bajar a 0.68 / 0.65")
    print("=" * 78)
    for thr in (0.68, 0.65):
        sim = simulate(thr)
        new = [s for s in sim if s["match_id"] not in base]
        print(f"\nmin_conf = {thr}: +{len(new)} apuestas")
        for s in new:
            print(f"  p_cal={s['p_cal']:.3f} margin={s['margin']:+.2f} min={s['minute']:>2} "
                  f"{'✅' if s['won'] else '❌'}  {s['league'][:50]}")

    print("\n" + "=" * 78)
    print("ACierto por banda de p_cal (solo filas que el modelo evaluó, sin empates)")
    print("=" * 78)
    bands = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 1.01)]
    for lo, hi in bands:
        band = [d for d in data if lo <= max(d["p_cal"], 1 - d["p_cal"]) < hi]
        if not band:
            continue
        # Solo tiene sentido si el modelo habría apostado en esa banda con thr=lo
        wins = sum(1 for d in band if (d["home_won"] == (d["p_cal"] >= 0.5)))
        print(f"  conf {lo:.2f}-{hi:.2f}: {wins}/{len(band)} ({round(wins * 100 / len(band))}%) "
              f"→ si se apostara todo: {'rentable' if wins * 100 / len(band) > 52.4 else 'no rentable'} "
              f"(break-even 52.4% @1.91)")


if __name__ == "__main__":
    main()
