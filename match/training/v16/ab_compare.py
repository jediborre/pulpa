"""
ab_compare.py - Reporte comparativo A/B (y 3-way) de backends de forecasting.

Variantes soportadas: TFM (TimesFM), CHRONOS (Chronos Bolt), NOTFM (baseline Holt).
Rerunnable: agrega nuevas variantes poniendo el JSON en reports/ con el sufijo correcto.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
RPT  = BASE / "reports"
OUT  = BASE / "model_outputs"


def _load(name: str) -> dict | None:
    p = RPT / f"test_roi_v16_{name}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_summary(name: str) -> dict:
    p = OUT / f"training_summary_v16_{name}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def g(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d


def fmt_f(v, width=12, pct=False, sign=False) -> str:
    if v is None:
        return "-".rjust(width)
    if isinstance(v, float):
        if pct:
            s = f"{v*100:+.2f}%" if sign else f"{v*100:.2f}%"
        else:
            s = f"{v:+.4f}" if sign else f"{v:.4f}"
        return s.rjust(width)
    return str(v).rjust(width)


def fmt_i(v, width=8) -> str:
    if v is None:
        return "-".rjust(width)
    return str(int(v)).rjust(width)


VARIANTS = [
    ("TFM",     "TimesFM (Google)"),
    ("CHRONOS", "Chronos Bolt-tiny (Amazon)"),
    ("NOTFM",   "Holt baseline (sin DL)"),
]

data    = {k: _load(k) for k, _ in VARIANTS}
summary = {k: _load_summary(k) for k, _ in VARIANTS}
avail   = [(k, lbl) for k, lbl in VARIANTS if data[k] is not None]

W = 92
print("=" * W)
print(" " * 20 + "V16 A/B/C REPORT  -  TimesFM vs Chronos vs Holt baseline")
print("=" * W)
print()
print("Mismo holdout (375 muestras NO vistas), mismos thresholds por liga.")
print("Odds 1.40 | break-even 71.43% | target hit-rate 75%")
print()

# --- Header -----------------------------------------------------------------
print("-" * W)
header = f"{'metrica':<28}"
for k, lbl in avail:
    header += f" {lbl[:22]:>22}"
if len(avail) == 3:
    header += f"  {'mejor':>10}"
print(header)
print("-" * W)


def row(label, key, pct=False, higher_is_better=True, fmt_fn=None):
    vals = []
    for k, _ in avail:
        vals.append(g(data[k], "global", key))
    if fmt_fn is None:
        fmt_fn = lambda v: fmt_f(v, width=22, pct=pct)
    s = f"{label:<28}"
    for v in vals:
        s += f" {fmt_fn(v)}"
    if len(avail) == 3 and all(v is not None for v in vals):
        best_idx = (vals.index(max(vals)) if higher_is_better
                    else vals.index(min(vals)))
        s += f"  {avail[best_idx][1][:10]:>10}"
    print(s)


row("apuestas (bets)",    "n_bets", pct=False, higher_is_better=True,
    fmt_fn=lambda v: fmt_i(v, width=22))
row("wins",               "wins",   pct=False, higher_is_better=True,
    fmt_fn=lambda v: fmt_i(v, width=22))
row("hit rate",           "hit_rate", pct=True,  higher_is_better=True)
row("ROI",                "roi",    pct=True,  higher_is_better=True)
row("P&L (unidades)",     "pnl_units", pct=False, higher_is_better=True)

print("-" * W)
print()

# --- Modelos entrenados ------------------------------------------------------
print("-" * W)
s = f"{'modelos entrenados':<28}"
for k, _ in avail:
    trained = sum(1 for m in summary[k].get("models", []) if m.get("status") == "trained")
    s += f" {str(trained):>22}"
print(s)
s = f"{'tiempo train (s)':<28}"
for k, _ in avail:
    sec = summary[k].get("training_seconds")
    s += f" {(str(round(sec)) if sec else '-'):>22}"
print(s)
print("-" * W)
print()

# --- Detalle por liga --------------------------------------------------------
print("-" * W)
print("DETALLE POR LIGA (solo ligas con apuestas en al menos una variante):")
print("-" * W)

def _idx(d):
    out = {}
    for r in (d or {}).get("per_group", []) or []:
        key = (r.get("league",""), r.get("target",""))
        out[key] = r
    return out

idxs = {k: _idx(data[k]) for k, _ in avail}
all_keys = sorted(set().union(*[set(idxs[k].keys()) for k, _ in avail]))

col_w = 17
hdr2 = f"{'liga':<36} {'tgt':<4}"
for k, lbl in avail:
    col_lbl = lbl[:col_w]
    hdr2 += f"  {col_lbl:>{col_w}}"
print(hdr2)
print("-" * W)

for key in all_keys:
    bets_any = any((idxs[k].get(key, {}).get("n_bets") or 0) > 0 for k, _ in avail)
    if not bets_any:
        continue
    liga, tgt = key
    s = f"{liga[:36]:<36} {tgt:<4}"
    for k, _ in avail:
        r = idxs[k].get(key, {})
        nb = r.get("n_bets") or 0
        roi = r.get("roi")
        hit = r.get("hit_rate")
        if nb == 0:
            cell = "-"
        else:
            h_s = f"{hit*100:.0f}%" if hit is not None else "?%"
            r_s = f"{roi*100:+.0f}%" if roi is not None else "?"
            cell = f"{nb}b {h_s} {r_s}"
        s += f"  {cell:>{col_w}}"
    print(s)

print()

# --- Veredicto ---------------------------------------------------------------
print("=" * W)
print("VEREDICTO")
print("=" * W)
for k, lbl in avail:
    bets = g(data[k], "global", "n_bets") or 0
    hit  = g(data[k], "global", "hit_rate") or 0.0
    roi  = g(data[k], "global", "roi") or 0.0
    pnl  = g(data[k], "global", "pnl_units") or 0.0
    print(f"  {lbl:<28}  bets={bets}  hit={hit*100:.2f}%  ROI={roi*100:+.2f}%  P&L={pnl:+.2f}u")

print()
print("Notas:")
print("  - Con n<15 bets por variante, ningun resultado es estadisticamente significativo.")
print("  - El mismo holdout (7 dias) es demasiado corto para discriminar backends.")
print("  - Cronos entrenó 87 pares (vs 46 de TimesFM) porque no filtra por pbp minimo.")
print("  - Para decision definitiva se necesitan >= 30 dias de holdout acumulado.")
