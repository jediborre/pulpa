"""
test_roi.py - Script de prueba de ROI V16 con salida legible.

Corre inferencia real sobre el holdout (nunca visto por training) y resume:
  - ROI global
  - ROI por liga+target (coloreado / con flags PASS/FAIL)
  - ROI global con el TOP-N ligas rentables (cartera real)
  - Diagnostico leak: ligas con train_val_gap sospechoso

Uso:
    python -m training.v16.cli test-roi
    python -m training.v16.cli test-roi --odds 1.50 --top 10
    python -m training.v16.test_roi   # directo

Salidas:
  reports/test_roi_v16.csv  (tabla completa por liga+target)
  reports/test_roi_v16.json (resumen estructurado para consumir desde API)
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from training.v16 import config, dataset as ds
from training.v16.inference import V15Engine, SUMMARY_PATH


REPORTS_DIR = Path(__file__).parent / "reports"


# ============================================================================
# Colores ANSI para terminal
# ============================================================================

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"


def _roi_color(roi: float | None) -> str:
    if roi is None:
        return C.DIM
    if roi >= 0.05:
        return C.GREEN + C.BOLD
    if roi >= 0.0:
        return C.GREEN
    if roi >= -0.05:
        return C.YELLOW
    return C.RED


def _hit_color(hit: float | None) -> str:
    if hit is None:
        return C.DIM
    if hit >= config.TARGET_HIT_RATE:
        return C.GREEN + C.BOLD
    if hit >= config.MIN_ACCEPTABLE_HIT_RATE:
        return C.GREEN
    if hit >= 0.68:
        return C.YELLOW
    return C.RED


def _gap_color(gap: float | None) -> str:
    if gap is None:
        return C.DIM
    if gap > 0.20:
        return C.RED + C.BOLD
    if gap > 0.10:
        return C.YELLOW
    return C.GREEN


# ============================================================================
# Utilidades
# ============================================================================

def _reconstruct_qs(s: ds.Sample) -> dict[str, int]:
    f = s.features
    q1_total = int(f.get("q1_total", 0))
    q1_diff = int(f.get("q1_diff", 0))
    q1_home = (q1_total + q1_diff) // 2
    q2_total = int(f.get("q2_total", 0))
    q2_diff = int(f.get("q2_diff", 0))
    q2_home = (q2_total + q2_diff) // 2
    out = {
        "q1_home": q1_home, "q1_away": q1_total - q1_home,
        "q2_home": q2_home, "q2_away": q2_total - q2_home,
    }
    if s.target == "q4":
        q3_total = int(f.get("q3_total", 0))
        q3_diff = int(f.get("q3_diff", 0))
        q3_home = (q3_total + q3_diff) // 2
        out["q3_home"] = q3_home
        out["q3_away"] = q3_total - q3_home
    return out


@dataclass
class GroupStats:
    league: str
    target: str
    n_predictions: int = 0
    n_bets: int = 0
    wins: int = 0
    roi: float | None = None
    hit_rate: float | None = None
    pnl: float = 0.0
    coverage: float = 0.0
    train_val_gap: float | None = None
    reasons: dict[str, int] = None

    def to_dict(self) -> dict:
        return {
            "league": self.league,
            "target": self.target,
            "n_predictions": self.n_predictions,
            "n_bets": self.n_bets,
            "wins": self.wins,
            "roi": self.roi,
            "hit_rate": self.hit_rate,
            "pnl": self.pnl,
            "coverage": self.coverage,
            "train_val_gap": self.train_val_gap,
            "reasons": dict(self.reasons or {}),
        }


# ============================================================================
# Runner
# ============================================================================

def run(
    odds: float = config.DEFAULT_ODDS,
    top_n: int = 20,
    min_bets: int = 10,
    emit_csv: bool = True,
) -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        print(f"{C.RED}[error] no hay modelos entrenados. "
              f"Corre: python -m training.v16.cli train{C.RESET}")
        return {}

    # Windows console (cp1252) no soporta unicode completo; forzar utf-8
    try:
        import sys as _sys
        if hasattr(_sys.stdout, "reconfigure"):
            _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print(f"{C.BOLD}{C.CYAN}[v16/test-roi] odds={odds}  "
          f"break-even={1/odds:.4f}  target={config.TARGET_HIT_RATE:.0%}{C.RESET}")
    print(f"[v16/test-roi] cargando engine...")
    engine = V15Engine.load(SUMMARY_PATH)

    print(f"[v16/test-roi] cargando muestras y split...")
    samples, _ = ds.build_samples(use_cache=True, verbose=False)
    splits = ds.split_temporal(samples)
    holdout = [
        s for s in splits["holdout"]
        if (s.target == "q3" and s.snapshot_minute == config.Q3_GRAPH_CUTOFF)
        or (s.target == "q4" and s.snapshot_minute == config.Q4_GRAPH_CUTOFF)
    ]
    if not holdout:
        print(f"{C.RED}[error] holdout vacio. Verificar fechas en DB.{C.RESET}")
        return {}
    print(f"[v16/test-roi] {len(holdout)} muestras de holdout "
          f"(cutoffs q3={config.Q3_GRAPH_CUTOFF} q4={config.Q4_GRAPH_CUTOFF})")

    # Preload graph/pbp
    match_ids = sorted({s.match_id for s in holdout})
    conn = ds.get_db_connection()
    try:
        gp = ds.load_graph_points(conn, match_ids)
        pbp = ds.load_pbp_events(conn, match_ids)
    finally:
        conn.close()

    # Indice de gaps desde summary para anotar diagnostico
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)
    gap_map: dict[tuple[str, str], float] = {}
    for m in summary.get("models", []):
        if m.get("skipped"):
            continue
        gap = None
        tm = (m.get("train_metrics") or {}).get("f1")
        vm = (m.get("val_metrics") or {}).get("f1")
        if tm is not None and vm is not None:
            gap = tm - vm
        gap_map[(m["league"], m["target"])] = gap

    # Ejecutar predicciones
    print(f"[v16/test-roi] corriendo inferencia...")
    groups: dict[tuple[str, str], GroupStats] = {}
    all_preds: list[dict] = []
    for s in holdout:
        key = (s.league, s.target)
        grp = groups.setdefault(key, GroupStats(
            league=s.league, target=s.target,
            reasons=defaultdict(int),
            train_val_gap=gap_map.get(key),
        ))
        grp.n_predictions += 1
        quarter_scores = _reconstruct_qs(s)
        pred = engine.predict(
            match_id=s.match_id, target=s.target, league=s.league,
            quarter_scores=quarter_scores,
            graph_points=gp.get(s.match_id, []),
            pbp_events=pbp.get(s.match_id, []),
        )
        all_preds.append({
            "match_id": s.match_id, "league": s.league, "target": s.target,
            "date": s.date, "signal": pred.signal, "reason": pred.reason,
            "probability_home": pred.probability,
            "confidence": pred.confidence, "threshold": pred.threshold,
            "true_winner": s.target_winner,
        })
        if pred.signal in ("BET_HOME", "BET_AWAY"):
            pick = 1 if pred.signal == "BET_HOME" else 0
            grp.n_bets += 1
            if pick == s.target_winner:
                grp.wins += 1
        else:
            grp.reasons[pred.reason] += 1

    # Metricas por grupo
    for grp in groups.values():
        if grp.n_bets > 0:
            grp.hit_rate = grp.wins / grp.n_bets
            grp.pnl = grp.wins * (odds - 1) - (grp.n_bets - grp.wins)
            grp.roi = grp.pnl / grp.n_bets
            grp.coverage = grp.n_bets / grp.n_predictions

    # Global
    total_bets = sum(g.n_bets for g in groups.values())
    total_wins = sum(g.wins for g in groups.values())
    total_pnl = sum(g.pnl for g in groups.values())
    total_preds = sum(g.n_predictions for g in groups.values())
    global_hit = total_wins / total_bets if total_bets else None
    global_roi = total_pnl / total_bets if total_bets else None

    # Portfolio (top_n rentables con >= min_bets)
    ranked = sorted(
        [g for g in groups.values() if (g.n_bets or 0) >= min_bets],
        key=lambda g: -(g.roi or -99),
    )
    portfolio = ranked[:top_n]
    p_bets = sum(g.n_bets for g in portfolio)
    p_wins = sum(g.wins for g in portfolio)
    p_pnl = sum(g.pnl for g in portfolio)
    p_hit = p_wins / p_bets if p_bets else None
    p_roi = p_pnl / p_bets if p_bets else None

    # Imprimir
    _print_header(odds, len(holdout), total_preds)
    _print_global(global_roi, global_hit, total_bets, total_wins, total_pnl, odds)
    _print_portfolio(portfolio, p_roi, p_hit, p_bets, p_wins, p_pnl, odds, min_bets)
    _print_full_table(groups, odds)
    _print_leak_warnings(groups)
    _print_hint(global_roi, p_roi)

    # Persistencia
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    result = {
        "odds": odds,
        "n_predictions": total_preds,
        "global": {
            "n_bets": total_bets, "wins": total_wins,
            "hit_rate": global_hit, "roi": global_roi, "pnl": total_pnl,
        },
        "portfolio": {
            "top_n": top_n, "min_bets": min_bets,
            "n_bets": p_bets, "wins": p_wins, "hit_rate": p_hit,
            "roi": p_roi, "pnl": p_pnl,
            "leagues": [g.to_dict() for g in portfolio],
        },
        "per_group": [g.to_dict() for g in sorted(
            groups.values(), key=lambda g: -(g.roi if g.roi is not None else -99)
        )],
    }

    out_json = REPORTS_DIR / "test_roi_v16.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_jsonable)
    print(f"\n[v16/test-roi] resumen -> {out_json}")

    if emit_csv:
        out_csv = REPORTS_DIR / "test_roi_v16.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "league", "target", "n_predictions", "n_bets", "wins",
                "hit_rate", "roi", "pnl", "coverage", "train_val_gap",
            ])
            for g in sorted(groups.values(),
                             key=lambda g: -(g.roi if g.roi is not None else -99)):
                writer.writerow([
                    g.league, g.target, g.n_predictions, g.n_bets, g.wins,
                    _fmt(g.hit_rate), _fmt(g.roi), _fmt(g.pnl),
                    _fmt(g.coverage), _fmt(g.train_val_gap),
                ])
        print(f"[v16/test-roi] csv     -> {out_csv}")

    return result


# ============================================================================
# Impresion bonita
# ============================================================================

def _print_header(odds: float, n_holdout: int, n_preds: int) -> None:
    print()
    print(f"{C.BOLD}{'=' * 92}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  V16 - ROI TEST REPORT{C.RESET}")
    print(f"{C.BOLD}{'=' * 92}{C.RESET}")
    print(f"  muestras holdout      : {n_holdout}")
    print(f"  predicciones totales  : {n_preds}")
    print(f"  odds                  : {odds}")
    print(f"  break-even            : {1/odds:.4f}  ({100/odds:.2f}%)")
    print(f"  target hit rate       : {config.TARGET_HIT_RATE:.0%}")
    print(f"  min acceptable        : {config.MIN_ACCEPTABLE_HIT_RATE:.0%}")
    print(f"{'-' * 92}")


def _print_global(roi, hit, n_bets, wins, pnl, odds) -> None:
    print(f"{C.BOLD}GLOBAL (todas las ligas, todos los thresholds ya aplicados){C.RESET}")
    if n_bets == 0:
        print(f"  {C.RED}Sin apuestas. Threshold demasiado alto o sin modelos.{C.RESET}")
        print(f"{'-' * 92}")
        return
    profit_u = pnl
    print(f"  apuestas              : {n_bets}")
    print(f"  ganadas               : {wins}")
    print(f"  hit rate              : {_hit_color(hit)}{hit:.4f}{C.RESET} "
          f"(break-even={1/odds:.4f})")
    print(f"  ROI                   : {_roi_color(roi)}{roi:+.4f}{C.RESET}")
    sign = "+" if profit_u >= 0 else ""
    print(f"  P&L unidades          : {_roi_color(roi)}{sign}{profit_u:.2f}u{C.RESET}")
    print(f"{'-' * 92}")


def _print_portfolio(portfolio, roi, hit, n_bets, wins, pnl, odds, min_bets) -> None:
    print(f"{C.BOLD}PORTFOLIO recomendado ({len(portfolio)} ligas con ROI > 0 "
          f"y >= {min_bets} apuestas){C.RESET}")
    if not portfolio or n_bets == 0:
        print(f"  {C.YELLOW}Sin ligas rentables. Revisar threshold o agregar datos.{C.RESET}")
        print(f"{'-' * 92}")
        return
    print(f"  apuestas              : {n_bets}")
    print(f"  hit rate              : {_hit_color(hit)}{hit:.4f}{C.RESET}")
    print(f"  ROI                   : {_roi_color(roi)}{roi:+.4f}{C.RESET}")
    print(f"  P&L unidades          : {_roi_color(roi)}{pnl:+.2f}u{C.RESET}")
    print()
    print(f"  {'liga':30s} {'tgt':3s} {'bets':>5s} {'hit':>6s} {'ROI':>8s} "
          f"{'gap':>6s}  flag")
    for g in portfolio:
        flag = ""
        if g.hit_rate is not None and g.hit_rate >= config.TARGET_HIT_RATE:
            flag = f"{C.GREEN}TARGET{C.RESET}"
        elif g.hit_rate is not None and g.hit_rate >= config.MIN_ACCEPTABLE_HIT_RATE:
            flag = f"{C.GREEN}OK{C.RESET}"
        else:
            flag = f"{C.YELLOW}marginal{C.RESET}"
        print(
            f"  {g.league[:30]:30s} {g.target:3s} {g.n_bets:>5d} "
            f"{_hit_color(g.hit_rate)}{(g.hit_rate or 0):>6.3f}{C.RESET} "
            f"{_roi_color(g.roi)}{(g.roi or 0):>+8.3f}{C.RESET} "
            f"{_gap_color(g.train_val_gap)}{(g.train_val_gap or 0):>6.3f}{C.RESET}  "
            f"{flag}"
        )
    print(f"{'-' * 92}")


def _print_full_table(groups: dict, odds) -> None:
    print(f"{C.BOLD}DETALLE COMPLETO (todas las ligas evaluadas en holdout){C.RESET}")
    rows = sorted(groups.values(),
                  key=lambda g: -(g.roi if g.roi is not None else -99))
    print(f"  {'liga':32s} {'tgt':4s} {'pred':>5s} {'bets':>5s} "
          f"{'hit':>7s} {'ROI':>8s} {'cov':>6s} {'gap':>6s}")
    for g in rows:
        if g.n_bets == 0:
            top_reason = ""
            if g.reasons:
                items = sorted(g.reasons.items(), key=lambda kv: -kv[1])
                top_reason = f"  {C.DIM}no bet ({items[0][0][:30]}){C.RESET}"
            print(
                f"  {g.league[:32]:32s} {g.target:4s} {g.n_predictions:>5d} "
                f"{C.DIM}{g.n_bets:>5d}{C.RESET} {'--':>7s} {'--':>8s} "
                f"{'--':>6s} "
                f"{_gap_color(g.train_val_gap)}{(g.train_val_gap or 0):>6.3f}{C.RESET}"
                f"{top_reason}"
            )
            continue
        print(
            f"  {g.league[:32]:32s} {g.target:4s} {g.n_predictions:>5d} "
            f"{g.n_bets:>5d} "
            f"{_hit_color(g.hit_rate)}{(g.hit_rate or 0):>7.3f}{C.RESET} "
            f"{_roi_color(g.roi)}{(g.roi or 0):>+8.3f}{C.RESET} "
            f"{(g.coverage or 0):>6.3f} "
            f"{_gap_color(g.train_val_gap)}{(g.train_val_gap or 0):>6.3f}{C.RESET}"
        )
    print(f"{'-' * 92}")


def _print_leak_warnings(groups: dict) -> None:
    suspect = [g for g in groups.values()
               if g.train_val_gap is not None and g.train_val_gap > 0.15]
    if not suspect:
        print(f"{C.BOLD}LEAK CHECK{C.RESET}  {C.GREEN}OK "
              f"(ningun modelo con gap train-val > 0.15){C.RESET}")
        print(f"{'-' * 92}")
        return
    suspect.sort(key=lambda g: -g.train_val_gap)
    print(f"{C.BOLD}{C.YELLOW}LEAK CHECK  "
          f"{len(suspect)} modelos con gap sospechoso (>0.15){C.RESET}")
    for g in suspect[:15]:
        sev = ("LEAK" if g.train_val_gap > 0.20 else "overfit")
        color = C.RED if g.train_val_gap > 0.20 else C.YELLOW
        print(
            f"  {color}[{sev:7s}]{C.RESET} {g.league[:40]:40s} "
            f"{g.target} gap={g.train_val_gap:+.3f}"
        )
    print(f"{'-' * 92}")


def _print_hint(global_roi, portfolio_roi) -> None:
    print(f"{C.BOLD}SIGUIENTES PASOS{C.RESET}")
    if portfolio_roi is None:
        print(f"  {C.YELLOW}- Sin portfolio rentable. Bajar MIN_CONFIDENCE_BASE "
              f"o ampliar TRAIN_DAYS en config.{C.RESET}")
    elif portfolio_roi >= 0.05:
        print(f"  {C.GREEN}- Portfolio con ROI {portfolio_roi:+.3f}. "
              f"Ligas del portfolio listas para produccion.{C.RESET}")
        print(f"  - Correr {C.BOLD}python -m training.v16.cli plots{C.RESET} "
              f"para inspeccion grafica.")
        print(f"  - Editar league_overrides.py para las ligas borderline.")
    else:
        print(f"  {C.YELLOW}- ROI del portfolio ({portfolio_roi:+.3f}) "
              f"bajo el target de 5%.{C.RESET}")
        print(f"  - Revisar curvas de threshold en "
              f"{C.BOLD}reports/threshold_curves/{C.RESET}.")
        print(f"  - Considerar subir MIN_CONFIDENCE_BASE o forzar overrides "
              f"por liga problematica.")
    print(f"{'=' * 92}\n")


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.5f}"
    return str(v)


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, set):
        return list(x)
    return str(x)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--odds", type=float, default=config.DEFAULT_ODDS)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--min-bets", type=int, default=10)
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()
    run(
        odds=args.odds, top_n=args.top,
        min_bets=args.min_bets, emit_csv=not args.no_csv,
    )
