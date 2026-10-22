# =====================================================================
# RESUMEN EN VIVO DEL MODELO (para el bot de Telegram)
# ---------------------------------------------------------------------
# Same numbers as the report in docs/RESUMEN_LIVE_M27_V5_3.md, computed
# from the DB on demand:
#   - apuestas reales por dia y acumulado (con IC95 y unidades)
#   - por banda de confianza
#   - serie por minuto (acierto y calibracion)
#   - curva de umbral vivo (todas las predicciones, no solo las apuestas)
#   - las ligas con muestra
# =====================================================================

from __future__ import annotations

import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

# Equilibrio por cuota: WR_equilibrio = 1 / cuota
ODDS_REF = (1.40, 1.83)
BREAKEVEN_140 = 1.0 / 1.40
EXPECTED_WR_075 = 0.807   # backtest del modelo desplegado a 0.75
UMBRALES_CURVA = (0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80)


def _wilson(wins: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def compute_live_summary(conn: sqlite3.Connection,
                         model_version: str = "m27_v5_3") -> dict[str, Any]:
    """Todos los numeros del resumen en vivo del modelo."""
    out: dict[str, Any] = {"model": model_version}

    # ── 1. Apuestas reales por dia y acumulado ──
    por_dia = []
    for r in conn.execute("""
        SELECT substr(created_at,1,10) d, COUNT(*) n,
               SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w,
               AVG(confidence) c
        FROM bet_monitor_log_v3
        WHERE model_version=? AND signal_type LIKE 'BET%'
          AND result IN ('win','loss')
        GROUP BY d ORDER BY d""", (model_version,)):
        lo, hi = _wilson(r["w"], r["n"])
        por_dia.append({"dia": r["d"], "n": r["n"], "w": r["w"],
                        "wr": r["w"] / r["n"], "ci": (lo, hi),
                        "conf": r["c"] or 0.0})
    out["por_dia"] = por_dia

    tot = conn.execute("""
        SELECT COUNT(*) n, SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w
        FROM bet_monitor_log_v3
        WHERE model_version=? AND signal_type LIKE 'BET%'
          AND result IN ('win','loss')""", (model_version,)).fetchone()
    n_tot, w_tot = int(tot["n"] or 0), int(tot["w"] or 0)
    lo, hi = _wilson(w_tot, n_tot)
    out["acumulado"] = {
        "n": n_tot, "w": w_tot, "wr": (w_tot / n_tot) if n_tot else 0.0,
        "ci": (lo, hi),
        "u140": w_tot * 1.40 - n_tot, "u183": w_tot * 1.83 - n_tot,
        "z": ((w_tot / n_tot) - EXPECTED_WR_075)
             / math.sqrt(EXPECTED_WR_075 * (1 - EXPECTED_WR_075) / n_tot) if n_tot else 0.0,
    }

    # ── 2. Por banda de confianza ──
    bandas = []
    for r in conn.execute("""
        SELECT CASE WHEN confidence<0.80 THEN '0.75-0.80'
                    WHEN confidence<0.85 THEN '0.80-0.85' ELSE '0.85+' END banda,
               COUNT(*) n, SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) w
        FROM bet_monitor_log_v3
        WHERE model_version=? AND signal_type LIKE 'BET%'
          AND result IN ('win','loss')
        GROUP BY banda ORDER BY banda""", (model_version,)):
        bandas.append({"banda": r["banda"], "n": r["n"], "w": r["w"],
                       "wr": r["w"] / r["n"] if r["n"] else 0.0,
                       "u140": r["w"] * 1.40 - r["n"], "u183": r["w"] * 1.83 - r["n"]})
    out["bandas"] = bandas

    # ── 3. Serie por minuto ──
    filas = conn.execute("""
        SELECT i.match_id, i.game_minute, i.pick, i.confidence,
               q.q4_home, q.q4_away
        FROM model_inference_v3 i JOIN quarter_scores_v2 q
          ON CAST(q.match_id AS INTEGER)=i.match_id
        WHERE i.model_version=? AND q.q4_home IS NOT NULL
        ORDER BY i.match_id, i.game_minute""", (model_version,)).fetchall()
    por_min: dict[float, list[tuple[int, float]]] = defaultdict(list)
    for r in filas:
        dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
               else (r["q4_away"] - r["q4_home"]))
        por_min[r["game_minute"]].append((dif, r["confidence"] or 0.0))
    minutos = []
    for mn in sorted(por_min):
        sel = por_min[mn]
        w = sum(1 for d, _c in sel if d > 0)
        n = sum(1 for d, _c in sel if d != 0)
        if n < 10:      # con menos de 10 el dato no informa de nada
            continue
        conf = sum(c for _d, c in sel) / len(sel)
        minutos.append({"min": mn, "n": n, "w": w, "wr": w / n, "conf": conf,
                        "gap": (w / n - conf) * 100})
    out["minutos"] = minutos

    # ── 4. Curva de umbral (una fila por partido: la del ultimo minuto) ──
    mejor: dict[str, tuple[float, Any]] = {}
    for r in filas:
        m = str(r["match_id"])
        if m not in mejor or r["game_minute"] > mejor[m][0]:
            mejor[m] = (r["game_minute"], r)
    curva = []
    for t in UMBRALES_CURVA:
        sel = [r for _m, (_g, r) in mejor.items()
               if r["pick"] in ("HOME", "AWAY") and (r["confidence"] or 0) >= t]
        w = n = 0
        for r in sel:
            dif = ((r["q4_home"] - r["q4_away"]) if r["pick"] == "HOME"
                   else (r["q4_away"] - r["q4_home"]))
            if dif != 0:
                n += 1
                w += dif > 0
        if n == 0:
            continue
        lo, hi = _wilson(w, n)
        curva.append({"thr": t, "n": n, "w": w, "wr": w / n, "ci": (lo, hi),
                      "u140": w * 1.40 - n, "u183": w * 1.83 - n,
                      "ok140": (w / n) >= BREAKEVEN_140})
    out["curva"] = curva
    out["partidos_evaluados"] = len(mejor)

    # ── 5. Por liga (solo las que tienen muestra) ──
    liga_de = {str(r["match_id"]): r["league"] for r in
               conn.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL")}
    por_liga: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in conn.execute("""
        SELECT match_id, result FROM bet_monitor_log_v3
        WHERE model_version=? AND signal_type LIKE 'BET%'
          AND result IN ('win','loss')""", (model_version,)):
        lg = liga_de.get(str(r["match_id"]), "?")
        por_liga[lg][0] += 1
        por_liga[lg][1] += r["result"] == "win"
    out["por_liga"] = sorted(
        ({"liga": k, "n": v[0], "w": v[1], "wr": v[1] / v[0] if v[0] else 0.0}
         for k, v in por_liga.items() if v[0] >= 2),
        key=lambda d: -d["n"])
    return out


def render_live_summary_text(data: dict[str, Any]) -> str:
    """Texto HTML del resumen en vivo (para Telegram)."""
    L: list[str] = []
    L.append(f"📈 <b>Resumen en vivo · {data['model']}</b>")
    L.append("")

    acc = data["acumulado"]
    L.append("<b>Por día</b>")
    for d in data["por_dia"]:
        L.append(f"  {d['dia']}: {d['n']} apuestas | {d['w']} ganadas | "
                 f"WR {d['wr'] * 100:.1f}%")
    L.append("")
    lo, hi = acc["ci"]
    L.append(f"<b>Acumulado</b>: {acc['n']} apuestas | {acc['w']} ganadas | "
             f"WR <b>{acc['wr'] * 100:.1f}%</b> (IC95 {lo * 100:.0f}-{hi * 100:.0f}%)")
    L.append(f"  u@1.40: {acc['u140']:+.0f} | u@1.83: {acc['u183']:+.0f} | "
             f"esperado por el backtest: 80.7% (z = {acc['z']:+.2f})")
    L.append(f"  equilibrio a 1.40: <b>{BREAKEVEN_140 * 100:.1f}%</b> · a 1.83: 54.6%")
    L.append("")

    if data["bandas"]:
        L.append("<b>Por banda de confianza</b>")
        for b in data["bandas"]:
            L.append(f"  {b['banda']}: {b['n']} | WR {b['wr'] * 100:.1f}% | "
                     f"u@1.40 {b['u140']:+.0f} | u@1.83 {b['u183']:+.0f}")
        L.append("")

    if data["minutos"]:
        L.append("<b>Serie por minuto</b> (todas las predicciones)")
        for m in data["minutos"]:
            L.append(f"  min {m['min']:.0f}: {m['n']} | WR {m['wr'] * 100:.1f}% | "
                     f"conf {m['conf']:.3f} ({m['gap']:+.1f} pp)")
        L.append("")

    if data["curva"]:
        L.append(f"<b>Curva de umbral viva</b> ({data['partidos_evaluados']} partidos)")
        for c in data["curva"]:
            L.append(f"  {c['thr']:.2f}: {c['n']:>4} | WR {c['wr'] * 100:>5.1f}% | "
                     f"u@1.40 {c['u140']:>+4.0f} | u@1.83 {c['u183']:>+4.0f}"
                     + ("  ✅" if c["ok140"] else ""))
        L.append("")

    if data["por_liga"]:
        L.append("<b>Ligas con muestra</b>")
        for l in data["por_liga"][:8]:
            L.append(f"  {l['liga'][:36]}: {l['n']} | WR {l['wr'] * 100:.0f}%")
    return "\n".join(L)
