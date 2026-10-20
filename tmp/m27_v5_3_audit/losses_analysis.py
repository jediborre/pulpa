# =====================================================================
# ANÁLISIS DE PÉRDIDAS DE m27_v5_3 (findings + hipótesis + guard)
# ---------------------------------------------------------------------
# QUÉ HACE:
#   Carga todas las apuestas registradas de m27_v5_3 desde bet_monitor_log_v3,
#   las cruza con el itinerario (liga) y con quarter_scores_v2 (parciales), y
#   desglosa las pérdidas para buscar patrones accionables:
#     - por liga, por lado (local/visitante), por confianza, por minuto,
#     - por margen acumulado al momento de la inferencia (sin leakage),
#     - por disponibilidad de datos de mercado (has_ml / has_spread),
#     - por margen esperado del modelo (expected_q4_margin).
#   Después evalúa GUARDS candidatos: cuántas pérdidas evita y cuántas
#   ganadas sacrifica (siempre usando solo información disponible al apostar).
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/losses_analysis.py
# =====================================================================

import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"


def load_rows() -> list[dict]:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT l.id, l.match_id, l.signal_type, l.picked_side, l.confidence,
               l.inference_minute, l.graph_points_count, l.result,
               l.actual_home_score, l.actual_away_score, l.created_at,
               l.inference_json,
               s.league, s.home_team, s.away_team, s.event_date,
               q.q1_home, q.q1_away, q.q2_home, q.q2_away,
               q.q3_home, q.q3_away, q.q4_home, q.q4_away
        FROM bet_monitor_log_v3 l
        LEFT JOIN bet_monitor_schedule_v3 s ON s.match_id = l.match_id
        LEFT JOIN quarter_scores_v2 q ON q.match_id = l.match_id
        WHERE l.model_version = 'm27_v5_3'
          AND l.signal_type LIKE '%BET%'
          AND l.signal_type NOT LIKE '%NO_BET%'
        ORDER BY l.created_at
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _raw(row: dict) -> dict:
    try:
        inf = json.loads(row.get("inference_json") or "{}")
        return inf.get("raw_json") or {}
    except Exception:
        return {}


def _q4_margin(row: dict) -> int | None:
    h, a = row.get("q4_home"), row.get("q4_away")
    return None if h is None or a is None else int(h) - int(a)


def _wins_on(row: dict) -> bool:
    return str(row.get("result")) in ("win", "hit")


def _rate(rows: list[dict]) -> str:
    w = sum(1 for r in rows if _wins_on(r))
    n = len(rows)
    return f"{w}/{n} ({round(w * 100 / n) if n else 0}%)" if n else "0/0"


def block(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    rows = load_rows()
    resolved = [r for r in rows if str(r.get("result")) in ("win", "hit", "loss", "miss")]
    losses = [r for r in resolved if not _wins_on(r)]

    block("0. VOLUMEN")
    print(f"apuestas m27_v5_3 (BET) totales : {len(rows)}")
    print(f"resueltas                        : {len(resolved)}  → {_rate(resolved)}")
    if not resolved:
        return
    print(f"pérdidas                         : {len(losses)}")
    print(f"periodo                          : {resolved[0]['created_at'][:10]} → {resolved[-1]['created_at'][:10]}")
    print(f"picks: {dict(Counter(r['picked_side'] for r in resolved))}")
    print(f"resultados: {dict(Counter(r['result'] for r in resolved))}")

    block("1. PÉRDIDAS POR LIGA (ligas con ≥3 apuestas)")
    by_league = defaultdict(list)
    for r in resolved:
        by_league[r.get("league") or "?"].append(r)
    for league, items in sorted(by_league.items(), key=lambda kv: -len(kv[1])):
        if len(items) < 3:
            continue
        w = sum(1 for r in items if _wins_on(r))
        print(f"  {w}/{len(items)}  {_rate(items):>14}  {league[:55]}")

    block("2. POR LADO DEL PICK")
    for side in ("HOME", "AWAY"):
        items = [r for r in resolved if r["picked_side"] == side]
        print(f"  {side:<5} {_rate(items)}")

    block("3. POR CONFIANZA (buckets)")
    buckets = [(0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]
    for lo, hi in buckets:
        items = [r for r in resolved if lo <= float(r["confidence"] or 0) < hi]
        if items:
            print(f"  {lo:.2f}-{hi:.2f}: {_rate(items)}")

    block("4. POR MINUTO DE INFERENCIA")
    by_min = defaultdict(list)
    for r in resolved:
        by_min[int(r["inference_minute"] or 0)].append(r)
    for minute, items in sorted(by_min.items()):
        print(f"  min {minute:>2}: {_rate(items)}")

    block("5. MARGEN ACUMULADO AL APOSTAR (actual_home - actual_away)")
    # Positivo = gana el local en ese momento. Se compara con el lado elegido.
    def margin_bucket(r: dict) -> str:
        try:
            diff = int(r["actual_home_score"]) - int(r["actual_away_score"])
        except (TypeError, ValueError):
            return "sin dato"
        signed = diff if r["picked_side"] == "HOME" else -diff
        if signed <= -10:
            return "pick va -10 o peor"
        if signed <= -5:
            return "pick va -5..-9"
        if signed < 0:
            return "pick va -1..-4"
        if signed == 0:
            return "empate"
        if signed <= 4:
            return "pick va +1..+4"
        if signed <= 9:
            return "pick va +5..+9"
        return "pick va +10 o más"

    by_margin = defaultdict(list)
    for r in resolved:
        by_margin[margin_bucket(r)].append(r)
    for bucket in ("pick va +10 o más", "pick va +5..+9", "pick va +1..+4", "empate",
                   "pick va -1..-4", "pick va -5..-9", "pick va -10 o peor", "sin dato"):
        items = by_margin.get(bucket, [])
        if items:
            print(f"  {bucket:<20} {_rate(items)}")

    block("6. DATOS DE MERCADO EN LA INFERENCIA (has_ml / has_spread / expected margin)")
    for flag in ("has_ml", "has_spread"):
        yes = [r for r in resolved if _raw(r).get(flag)]
        no = [r for r in resolved if not _raw(r).get(flag)]
        print(f"  {flag}=True : {_rate(yes)}")
        print(f"  {flag}=False: {_rate(no)}")
    margins = [(float(_raw(r).get("expected_q4_margin") or 0), r) for r in resolved]
    strong = [r for m, r in margins if abs(m) >= 3]
    weak = [r for m, r in margins if abs(m) < 3]
    print(f"  |expected_q4_margin| ≥3: {_rate(strong)}")
    print(f"  |expected_q4_margin| <3: {_rate(weak)}")

    block("7. CONTEXTO DE PARTIDO (sin leakage: solo Q1-Q2 al minuto 27)")
    def half_margin(r: dict) -> int | None:
        try:
            return (int(r["q1_home"]) + int(r["q2_home"])) - (int(r["q1_away"]) + int(r["q2_away"]))
        except (TypeError, ValueError):
            return None

    def half_bucket(r: dict) -> str:
        hm = half_margin(r)
        if hm is None:
            return "sin dato"
        signed = hm if r["picked_side"] == "HOME" else -hm
        if signed <= -10:
            return "pick -10 o peor al descanso"
        if signed < 0:
            return "pick va abajo al descanso"
        if signed == 0:
            return "empate al descanso"
        if signed <= 9:
            return "pick +1..+9 al descanso"
        return "pick +10 o más al descanso"

    by_half = defaultdict(list)
    for r in resolved:
        by_half[half_bucket(r)].append(r)
    for bucket, items in by_half.items():
        print(f"  {bucket:<30} {_rate(items)}")

    block("8. MARCADOR FINAL DEL CUARTO (para entender el tipo de derrota)")
    diffs = [_q4_margin(r) for r in losses if _q4_margin(r) is not None]
    if diffs:
        print(f"  Q4 (local-visita) en las pérdidas: min {min(diffs)}, max {max(diffs)}")
        small = sum(1 for d in diffs if abs(d) <= 3)
        print(f"  derrotas por ≤3 puntos en Q4: {small}/{len(diffs)}")

    block("9. GUARDS CANDIDATOS (coste/beneficio sobre el histórico)")
    def evaluate(name: str, keep) -> None:
        kept = [r for r in resolved if keep(r)]
        dropped = [r for r in resolved if not keep(r)]
        dw = sum(1 for r in dropped if _wins_on(r))
        dl = sum(1 for r in dropped if not _wins_on(r))
        print(f"  {name:<52} queda {_rate(kept):>12}  | quita {dw}✅/{dl}❌")

    evaluate("baseline (sin guard)", lambda r: True)
    evaluate("conf ≥ 0.75", lambda r: float(r["confidence"] or 0) >= 0.75)
    evaluate("conf ≥ 0.80", lambda r: float(r["confidence"] or 0) >= 0.80)
    evaluate("conf ≥ 0.85", lambda r: float(r["confidence"] or 0) >= 0.85)
    evaluate("|expected_q4_margin| ≥ 2", lambda r: abs(float(_raw(r).get("expected_q4_margin") or 0)) >= 2)
    evaluate("|expected_q4_margin| ≥ 3", lambda r: abs(float(_raw(r).get("expected_q4_margin") or 0)) >= 3)
    evaluate("pick NO va perdiendo por ≥5 al apostar",
             lambda r: not (
                 (int(r["actual_home_score"] or 0) - int(r["actual_away_score"] or 0))
                 * (1 if r["picked_side"] == "HOME" else -1) <= -5
             ))
    evaluate("has_ml=True (hay cuota de mercado)", lambda r: bool(_raw(r).get("has_ml")))

    block("10. LIGAS DONDE HA PERDIDO (detalle)")
    for r in losses:
        print(f"  {r['created_at'][:10]} {str(r['match_id']):>9} {r['picked_side']:<4} "
              f"conf={float(r['confidence'] or 0):.2f} Q4={r.get('q4_home')}-{r.get('q4_away')} "
              f"| {(r.get('home_team') or '?')[:22]} vs {(r.get('away_team') or '?')[:22]} "
              f"| {(r.get('league') or '?')[:38]}")


if __name__ == "__main__":
    main()
