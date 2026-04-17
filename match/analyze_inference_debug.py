#!/usr/bin/env python3
"""
analyze_inference_debug.py — Analiza historial de inferencias del monitor de apuestas.

Uso:
    python match/analyze_inference_debug.py                        # todos los matches (resumen)
    python match/analyze_inference_debug.py 15736636               # match específico en detalle
    python match/analyze_inference_debug.py --flips                # solo matches con cambio de señal
    python match/analyze_inference_debug.py --date 2026-04-15      # matches de una fecha
    python match/analyze_inference_debug.py --all                  # todos con detalle
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "matches.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _fmt_ts(ts: str | None) -> str:
    if not ts:
        return "?"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M:%S")
    except Exception:
        return str(ts)


def _signal_emoji(signal: str) -> str:
    s = (signal or "").upper()
    if s == "BET":
        return "🟢 BET"
    if s == "UNAVAILABLE":
        return "⚪ UNAVAIL"
    if s.startswith("NO") or s == "NO_BET":
        return "🔴 NO_BET"
    return f"❓ {signal}"


def _parse_pred(json_str: str | None) -> dict[str, Any]:
    if not json_str:
        return {}
    try:
        return json.loads(json_str)
    except Exception:
        return {}


def _extract_key_pred_fields(pred: dict, target: str) -> dict:
    """Extract the most relevant fields from inference result JSON."""
    predictions = pred.get("predictions", {})
    t = predictions.get(target, {}) or {}
    return {
        "available": t.get("available", False),
        "final_recommendation": t.get("final_recommendation") or t.get("bet_signal") or "",
        "predicted_winner": t.get("predicted_winner") or "",
        "confidence": t.get("confidence"),
        "reasoning": t.get("reasoning") or "",
        "predicted_home": t.get("predicted_home"),
        "predicted_away": t.get("predicted_away"),
        "predicted_total": t.get("predicted_total"),
    }


def get_inference_rows(
    match_id: str | None = None,
    date_filter: str | None = None,
) -> list[sqlite3.Row]:
    conn = _conn()
    try:
        if match_id:
            rows = conn.execute(
                "SELECT * FROM inference_debug_log WHERE match_id=? ORDER BY created_at ASC",
                (match_id,),
            ).fetchall()
        elif date_filter:
            rows = conn.execute(
                "SELECT * FROM inference_debug_log WHERE date(created_at)=? ORDER BY match_id, target, created_at ASC",
                (date_filter,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM inference_debug_log ORDER BY match_id, target, created_at ASC"
            ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return rows


def get_monitor_log_matching(match_ids: set[str]) -> dict[str, list[sqlite3.Row]]:
    """Load bet_monitor_log rows for the given match IDs."""
    if not match_ids:
        return {}
    conn = _conn()
    try:
        placeholders = ",".join("?" for _ in match_ids)
        rows = conn.execute(
            f"SELECT * FROM bet_monitor_log WHERE match_id IN ({placeholders}) ORDER BY match_id, target, created_at ASC",
            list(match_ids),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    result: dict[str, list] = {}
    for r in rows:
        mid = r["match_id"]
        result.setdefault(mid, []).append(r)
    return result


def get_eval_results(match_ids: set[str]) -> dict[str, sqlite3.Row]:
    if not match_ids:
        return {}
    conn = _conn()
    try:
        placeholders = ",".join("?" for _ in match_ids)
        rows = conn.execute(
            f"SELECT * FROM eval_match_results WHERE match_id IN ({placeholders})",
            list(match_ids),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {r["match_id"]: r for r in rows}


# ─────────────────────────────── Detail view ──────────────────────────────────

def print_match_detail(match_id: str) -> None:
    rows = get_inference_rows(match_id=match_id)
    if not rows:
        print(f"  ⚠️  No se encontraron filas en inference_debug_log para match {match_id}")
        return

    # Separate by target
    by_target: dict[str, list] = {}
    for r in rows:
        by_target.setdefault(r["target"], []).append(r)

    eval_results = get_eval_results({match_id})
    eval_row = eval_results.get(match_id)
    monitor_rows_d = get_monitor_log_matching({match_id})
    monitor_rows = monitor_rows_d.get(match_id, [])

    # Header
    if rows:
        home = away = league = "?"
        for mr in monitor_rows:
            if mr["home_team"]:
                home = mr["home_team"]
                away = mr["away_team"]
                league = mr["league"] or "?"
                break
        first_created = rows[0]["created_at"]
        print(f"\n{'='*70}")
        print(f"Match ID : {match_id}")
        print(f"Equipos  : {home} vs {away}")
        print(f"Liga     : {league}")
        print(f"1er log  : {first_created}")
        if eval_row:
            q3w = eval_row["q3_winner"] or "?"
            q4w = eval_row["q4_winner"] or "?"
            q3s = f"{eval_row['q3_home_score']}-{eval_row['q3_away_score']}"
            q4s = f"{eval_row['q4_home_score']}-{eval_row['q4_away_score']}"
            print(f"Resultado: Q3: {q3s} ({q3w})  |  Q4: {q4s} ({q4w})")
        print(f"{'='*70}")

    for target in ["q3", "q4"]:
        target_rows = by_target.get(target, [])
        if not target_rows:
            continue

        print(f"\n── {target.upper()} ({len(target_rows)} inferencias) ──")
        signals = [r["signal"] or "?" for r in target_rows]
        unique_signals = list(dict.fromkeys(signals))  # order preserved

        # Detect flip
        has_flip = len(unique_signals) > 1
        if has_flip:
            flip_str = " → ".join(unique_signals)
            print(f"  ⚠️  CAMBIO DE SEÑAL: {flip_str}")
        else:
            print(f"  Señal consistente: {unique_signals[0]}")

        for r in target_rows:
            pred = _parse_pred(r["inference_json"])
            fields = _extract_key_pred_fields(pred, target)
            sig_emoji = _signal_emoji(r["signal"])
            conf_txt = f" {float(r['confidence']) * 100:.0f}%" if r["confidence"] else ""
            gp_txt = f" | gp={r['gp_count']}" if r["gp_count"] is not None else ""
            min_txt = f" | min={r['scraped_minute']}" if r["scraped_minute"] is not None else ""
            ts_txt = _fmt_ts(r["created_at"])

            reason = fields.get("reasoning") or ""
            reason_short = reason[:80] + ("…" if len(reason) > 80 else "")
            pick = fields.get("predicted_winner") or ""
            conf_model = fields.get("confidence")
            proj_txt = ""
            ph = fields.get("predicted_home")
            pa = fields.get("predicted_away")
            pt = fields.get("predicted_total")
            if ph is not None and pa is not None:
                proj_txt = f"\n       Proyección: 🏠{ph:.1f} | ✈️{pa:.1f} | Total={pt:.1f}" if pt is not None else f"\n       Proyección: 🏠{ph:.1f} | ✈️{pa:.1f}"

            print(
                f"  [{ts_txt}]{min_txt}{gp_txt}  {sig_emoji}{conf_txt}"
                + (f"  pick={pick}" if pick else "")
                + (f"  conf_mod={conf_model * 100:.0f}%" if conf_model is not None else "")
            )
            if reason_short:
                print(f"       Razón: {reason_short}")
            if proj_txt:
                print(proj_txt)

    # Monitor log summary
    if monitor_rows:
        print(f"\n── bit\u00e1cora monitor ({len(monitor_rows)} entradas) ──")
        for mr in monitor_rows:
            tgt = (mr["target"] or "?").upper()
            sig = mr["signal"] or "?"
            res = mr["result"] or "pending"
            res_emoji = {"win": "✅", "loss": "❌", "push": "➖", "pending": "⏳"}.get(res, "⏳")
            conf = mr["confidence"]
            conf_txt = f" {float(conf) * 100:.0f}%" if conf else ""
            rec = (mr["recommendation"] or "")[:60]
            print(f"  {res_emoji} {tgt} | {_signal_emoji(sig)}{conf_txt} | {rec}")


# ─────────────────────────────── Summary view ─────────────────────────────────

def print_summary(
    rows: list[sqlite3.Row],
    flips_only: bool = False,
) -> None:
    # Group by (match_id, target)
    groups: dict[tuple[str, str], list] = {}
    for r in rows:
        key = (r["match_id"], r["target"])
        groups.setdefault(key, []).append(r)

    match_ids = {r["match_id"] for r in rows}
    monitor_rows_d = get_monitor_log_matching(match_ids)
    eval_results = get_eval_results(match_ids)

    print(f"\n{'='*80}")
    print(f"  RESUMEN: {len(groups)} combinaciones (match × cuarto)  |  {len(rows)} inferencias total")
    print(f"{'='*80}")

    flip_count = 0
    for (match_id, target), group_rows in sorted(groups.items(), key=lambda x: x[1][0]["created_at"]):
        signals = [r["signal"] or "?" for r in group_rows]
        unique_signals = list(dict.fromkeys(signals))
        has_flip = len(unique_signals) > 1

        if flips_only and not has_flip:
            continue

        if has_flip:
            flip_count += 1

        # Get names from monitor log
        mlog = monitor_rows_d.get(match_id, [])
        home = away = "?"
        for ml in mlog:
            if ml["home_team"]:
                home = ml["home_team"][:12]
                away = ml["away_team"][:12]
                break

        eval_row = eval_results.get(match_id)
        result_txt = ""
        if eval_row:
            if target == "q3":
                outcome = eval_row["q3_winner"] or "?"
                score = f"{eval_row['q3_home_score']}-{eval_row['q3_away_score']}"
                result_txt = f" → {score}({outcome})"
            else:
                outcome = eval_row["q4_winner"] or "?"
                score = f"{eval_row['q4_home_score']}-{eval_row['q4_away_score']}"
                result_txt = f" → {score}({outcome})"

        sig_txt = " → ".join(unique_signals)
        n_checks = len(group_rows)
        last_sig = group_rows[-1]["signal"]
        last_conf = group_rows[-1]["confidence"]
        last_min = group_rows[-1]["scraped_minute"]
        conf_txt = f" {float(last_conf) * 100:.0f}%" if last_conf else ""
        min_txt = f" min={last_min}" if last_min is not None else ""
        flip_tag = " ⚠️FLIP" if has_flip else ""

        first_ts = group_rows[0]["created_at"][:10]

        print(
            f"  [{first_ts}] {match_id} {target.upper():<3} | {home} vs {away}"
            f"\n             Señales({n_checks}): {sig_txt}"
            + conf_txt + min_txt
            + result_txt
            + flip_tag
        )

    if flips_only:
        print(f"\n  Total matches con FLIP: {flip_count}")

    print(f"{'='*80}")


# ─────────────────────────────── DB info ──────────────────────────────────────

def print_db_info() -> None:
    conn = _conn()
    try:
        total = conn.execute("SELECT COUNT(*) as n FROM inference_debug_log").fetchone()["n"]
        matches = conn.execute("SELECT COUNT(DISTINCT match_id) as n FROM inference_debug_log").fetchone()["n"]
        oldest = conn.execute("SELECT MIN(created_at) as t FROM inference_debug_log").fetchone()["t"]
        newest = conn.execute("SELECT MAX(created_at) as t FROM inference_debug_log").fetchone()["t"]
        print(f"\ninference_debug_log: {total} filas | {matches} matches únicos")
        print(f"  Rango: {oldest} → {newest}")
    except sqlite3.OperationalError as exc:
        print(f"  Error al leer inference_debug_log: {exc}")
    conn.close()


# ──────────────────────────────── Main ────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analiza historial de inferencias del monitor de apuestas"
    )
    parser.add_argument("match_id", nargs="?", help="ID del partido a analizar en detalle")
    parser.add_argument("--flips", action="store_true", help="Mostrar solo partidos con cambio de señal")
    parser.add_argument("--date", "-d", metavar="YYYY-MM-DD", help="Filtrar por fecha (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", dest="show_all", help="Mostrar detalle de todos")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"❌ Base de datos no encontrada: {DB_PATH}")
        sys.exit(1)

    print_db_info()

    if args.match_id:
        print_match_detail(args.match_id)
    elif args.show_all:
        rows = get_inference_rows(date_filter=args.date)
        match_ids = set(r["match_id"] for r in rows)
        for mid in sorted(match_ids):
            print_match_detail(mid)
    else:
        rows = get_inference_rows(date_filter=args.date)
        if not rows:
            print("\n⚠️  No se encontraron registros en inference_debug_log.")
            print("   El monitor debe correr al menos un partido para generar datos.")
        else:
            print_summary(rows, flips_only=args.flips)
            print("\nPara ver detalle de un partido específico:")
            print(f"  python match/analyze_inference_debug.py <match_id>")
            print("Para ver solo cambios de señal:")
            print(f"  python match/analyze_inference_debug.py --flips")


if __name__ == "__main__":
    main()
