"""
plots.py - Graficas de diagnostico V16.

Objetivos:
- Detectar data leakage: gap train-val por liga, distribucion temporal, calibration.
- Evaluar performance: ROI por liga, hit_rate vs threshold, coverage.
- Inspeccionar features: top importances, distribucion de probabilidades.

Todas las graficas se guardan en model_outputs/plots/ como PNG.
Sin dependencias extra: solo matplotlib. Nada de seaborn ni plotly.

Uso:
    from training.v16 import plots
    plots.generate_all()

CLI:
    python -m training.v16.cli plots
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
except Exception as e:
    raise RuntimeError("matplotlib es necesario para plots.py") from e

from training.v16 import config, dataset as ds
from training.v16.inference import V15Engine, SUMMARY_PATH, MODEL_DIR


PLOTS_DIR = MODEL_DIR / "plots"


# ============================================================================
# Helpers
# ============================================================================

def _save(fig, name: str) -> Path:
    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    path = PLOTS_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[v16/plots] -> {path}")
    return path


def _load_summary() -> dict:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"No existe {SUMMARY_PATH}. Corre `python -m training.v16.cli train` primero."
        )
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _trained_models(summary: dict) -> list[dict]:
    return [m for m in summary.get("models", []) if not m.get("skipped")]


def _color_for_gap(gap: float) -> str:
    """Rojo si hay sospecha de leak; naranja si es grande; verde si OK."""
    if gap is None:
        return "#888888"
    if gap > 0.20:
        return "#d9534f"     # rojo fuerte: leak serio
    if gap > 0.10:
        return "#f0ad4e"     # amarillo: overfitting
    return "#5cb85c"         # verde: sano


def _color_for_roi(roi: float | None) -> str:
    if roi is None:
        return "#888888"
    if roi >= 0.05:
        return "#2e7d32"
    if roi >= 0.0:
        return "#66bb6a"
    if roi >= -0.05:
        return "#f0ad4e"
    return "#d9534f"


# ============================================================================
# 1) Leak detection: train vs val F1 gap por liga
# ============================================================================

def plot_train_val_gap(summary: dict) -> Path:
    models = _trained_models(summary)
    rows = []
    for m in models:
        train_f1 = (m.get("train_metrics") or {}).get("f1")
        val_f1 = (m.get("val_metrics") or {}).get("f1")
        if train_f1 is None or val_f1 is None:
            continue
        rows.append({
            "label": f"{m['league'][:22]} / {m['target']}",
            "train": train_f1,
            "val": val_f1,
            "gap": train_f1 - val_f1,
        })
    rows.sort(key=lambda r: -r["gap"])
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Sin modelos entrenados", ha="center", va="center")
        return _save(fig, "01_train_val_gap.png")

    height = max(4, 0.28 * len(rows))
    fig, ax = plt.subplots(figsize=(12, height))
    y = np.arange(len(rows))
    gaps = [r["gap"] for r in rows]
    colors = [_color_for_gap(g) for g in gaps]
    ax.barh(y, gaps, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.10, color="#f0ad4e", linestyle="--", linewidth=1, label="overfit warning (0.10)")
    ax.axvline(0.20, color="#d9534f", linestyle="--", linewidth=1, label="leak suspect (0.20)")
    ax.set_xlabel("train_F1 - val_F1 (gap; mas alto = peor)")
    ax.set_title("Leak / overfitting detection: gap train vs val por liga x target")
    ax.legend(loc="lower right")
    for i, r in enumerate(rows):
        ax.text(r["gap"] + 0.003, i, f"{r['gap']:+.3f}",
                va="center", fontsize=7)
    return _save(fig, "01_train_val_gap.png")


# ============================================================================
# 2) Leak detection: distribucion temporal por split
# ============================================================================

def plot_temporal_distribution(samples_by_split: dict[str, list]) -> Path:
    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "cal": "#2ca02c", "holdout": "#d62728"}
    for split, samples in samples_by_split.items():
        dates = []
        for s in samples:
            d = ds._parse_date(s.date)
            if d is not None:
                dates.append(d)
        if not dates:
            continue
        ax.hist(
            [d.toordinal() for d in dates],
            bins=60, alpha=0.55, label=f"{split} (n={len(samples)})",
            color=colors.get(split, "#555"),
        )
    # Tick labels a fechas
    import datetime as _dt
    xt = ax.get_xticks()
    ax.set_xticklabels(
        [_dt.date.fromordinal(int(t)).isoformat() if t > 0 else "" for t in xt],
        rotation=30, ha="right", fontsize=8,
    )
    ax.set_title("Distribucion temporal por split (debe NO solaparse)")
    ax.set_ylabel("# muestras")
    ax.legend()
    return _save(fig, "02_temporal_distribution.png")


# ============================================================================
# 3) ROI por liga en holdout (hit_rate vs ROI)
# ============================================================================

def plot_roi_by_league(summary: dict) -> Path:
    models = _trained_models(summary)
    rows = []
    for m in models:
        ho = m.get("holdout_betting") or {}
        rows.append({
            "label": f"{m['league'][:22]}/{m['target']}",
            "roi": ho.get("roi"),
            "hit_rate": ho.get("hit_rate"),
            "n_bets": ho.get("n_bets", 0),
            "threshold": (m.get("threshold") or {}).get("threshold"),
        })
    # Ordenar por ROI descendente (None al final)
    rows.sort(key=lambda r: (r["roi"] is None, -(r["roi"] or -99)))
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin modelos entrenados", ha="center", va="center")
        return _save(fig, "03_roi_by_league.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(4, 0.28 * len(rows))))

    y = np.arange(len(rows))
    rois = [(r["roi"] or 0.0) for r in rows]
    colors = [_color_for_roi(r["roi"]) for r in rows]
    ax1.barh(y, rois, color=colors)
    ax1.set_yticks(y)
    ax1.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax1.invert_yaxis()
    ax1.axvline(0, color="black", linewidth=0.8)
    ax1.axvline(config.BREAK_EVEN_PCT - 1 / config.DEFAULT_ODDS, color="#888", linestyle="--")
    ax1.set_xlabel(f"ROI en holdout (odds {config.DEFAULT_ODDS})")
    ax1.set_title("ROI por liga+target en HOLDOUT")
    for i, r in enumerate(rows):
        txt = f"{(r['roi'] or 0):+.3f} (n={r['n_bets']})" if r["roi"] is not None else "n/a"
        ax1.text(max(rois[i], 0) + 0.005, i, txt, va="center", fontsize=7)

    # Scatter hit_rate vs n_bets, color=ROI
    hit = [r["hit_rate"] or 0 for r in rows]
    nbets = [r["n_bets"] or 0 for r in rows]
    sc = ax2.scatter(nbets, hit, c=rois, cmap="RdYlGn", vmin=-0.2, vmax=0.2,
                     s=80, edgecolors="black", linewidths=0.5)
    ax2.axhline(config.BREAK_EVEN_PCT, color="#d9534f", linestyle="--",
                label=f"break-even @ {config.DEFAULT_ODDS} ({config.BREAK_EVEN_PCT:.3f})")
    ax2.axhline(config.TARGET_HIT_RATE, color="#2e7d32", linestyle="--",
                label=f"target {config.TARGET_HIT_RATE:.0%}")
    ax2.set_xlabel("# apuestas en holdout")
    ax2.set_ylabel("hit rate holdout")
    ax2.set_title("Hit rate vs volumen (color = ROI)")
    plt.colorbar(sc, ax=ax2, label="ROI")
    ax2.legend(loc="lower right", fontsize=8)
    for r, x, y2 in zip(rows, nbets, hit):
        if (r["n_bets"] or 0) >= 15:
            ax2.annotate(r["label"], (x, y2), fontsize=6, alpha=0.65)

    return _save(fig, "03_roi_by_league.png")


# ============================================================================
# 4) Curva hit_rate / ROI vs threshold (agregada + top ligas)
# ============================================================================

def plot_threshold_curves(summary: dict) -> Path:
    models = _trained_models(summary)
    # Tomar top-6 ligas por n_bets en holdout
    ranked = sorted(
        models,
        key=lambda m: -(((m.get("holdout_betting") or {}).get("n_bets") or 0)),
    )[:6]
    if not ranked:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin modelos entrenados", ha="center", va="center")
        return _save(fig, "04_threshold_curves.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, m in zip(axes.flat, ranked):
        curve = (m.get("threshold") or {}).get("curve") or []
        if not curve:
            ax.set_visible(False)
            continue
        thrs, rois, hits, ns = [], [], [], []
        for c in curve:
            if c.get("roi") is None:
                continue
            thrs.append(c["threshold"])
            rois.append(c["roi"])
            hits.append(c["hit_rate"])
            ns.append(c["n_bets"])
        if not thrs:
            ax.set_visible(False)
            continue
        ax2 = ax.twinx()
        ln1 = ax.plot(thrs, rois, "b-", label="ROI", linewidth=2)
        ln2 = ax.plot(thrs, hits, "g--", label="hit rate", linewidth=1.5)
        ln3 = ax2.plot(thrs, ns, color="#aa0000", alpha=0.35, label="# apuestas")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axhline(config.BREAK_EVEN_PCT, color="#d9534f", linestyle=":", linewidth=0.8)
        chosen = (m.get("threshold") or {}).get("threshold")
        if chosen is not None:
            ax.axvline(chosen, color="purple", linestyle="--", linewidth=1,
                       label=f"thr elegido {chosen:.2f}")
        ax.set_title(f"{m['league'][:24]} / {m['target']}", fontsize=9)
        ax.set_xlabel("threshold")
        ax.set_ylabel("ROI / hit rate")
        ax2.set_ylabel("# apuestas", color="#aa0000")
        ax.legend(fontsize=7, loc="lower left")
    plt.suptitle("Curvas de threshold para top-6 ligas por volumen (val set)", y=1.02)
    return _save(fig, "04_threshold_curves.png")


# ============================================================================
# 5) Calibration curves (reliability diagrams) - solo para ligas con holdout
# ============================================================================

def plot_calibration_curves(
    summary: dict,
    engine: V15Engine,
    holdout_samples_by_league_target: dict[tuple[str, str], list],
    graph_points: dict,
    pbp_events: dict,
) -> Path:
    models = _trained_models(summary)
    target_models = [m for m in models if (m.get("holdout_betting") or {}).get("n_bets", 0) >= 30]
    target_models = sorted(
        target_models,
        key=lambda m: -(((m.get("holdout_betting") or {}).get("n_bets") or 0)),
    )[:6]
    if not target_models:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin holdout suficiente (>=30 apuestas)",
                ha="center", va="center")
        return _save(fig, "05_calibration_curves.png")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, target_models):
        key = (m["league"], m["target"])
        samples = holdout_samples_by_league_target.get(key, [])
        if not samples:
            ax.set_visible(False)
            continue
        probas = []
        truths = []
        for s in samples:
            quarter_scores = _reconstruct_qs_from_sample(s)
            pred = engine.predict(
                match_id=s.match_id, target=s.target, league=s.league,
                quarter_scores=quarter_scores,
                graph_points=graph_points.get(s.match_id, []),
                pbp_events=pbp_events.get(s.match_id, []),
            )
            if pred.probability is None:
                continue
            probas.append(pred.probability)
            truths.append(s.target_winner)
        if not probas:
            ax.set_visible(False)
            continue
        bins = np.linspace(0, 1, 11)
        binned_p = []
        binned_f = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            idx = [i for i, p in enumerate(probas) if lo <= p < hi]
            if idx:
                binned_p.append(np.mean([probas[i] for i in idx]))
                binned_f.append(np.mean([truths[i] for i in idx]))
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="perfecta")
        ax.plot(binned_p, binned_f, "o-", color="#1f77b4", label="empirica")
        ax.set_title(f"{m['league'][:24]} / {m['target']}\n(n={len(probas)})", fontsize=9)
        ax.set_xlabel("prob predicha (home)")
        ax.set_ylabel("freq real (home gana)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Calibracion en holdout (ojo: si se aleja de la diagonal, calibracion fallo)", y=1.02)
    return _save(fig, "05_calibration_curves.png")


# ============================================================================
# 6) Distribucion de probabilidades en holdout (BET vs NO_BET)
# ============================================================================

def plot_probability_distribution(all_preds: list[dict]) -> Path:
    probas = [p["probability_home"] for p in all_preds if p.get("probability_home") is not None]
    if not probas:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin predicciones", ha="center", va="center")
        return _save(fig, "06_probability_distribution.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    # Histograma de probabilidades
    ax1.hist(probas, bins=40, color="#1f77b4", alpha=0.8)
    ax1.axvline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xlabel("probabilidad home (calibrada)")
    ax1.set_ylabel("# muestras")
    ax1.set_title("Distribucion de probabilidades en holdout")

    # Confianza = max(p, 1-p), dividido en BET / NO_BET
    conf_bet = []
    conf_nobet = []
    for p in all_preds:
        ph = p.get("probability_home")
        if ph is None:
            continue
        c = max(ph, 1 - ph)
        if p["signal"] in ("BET_HOME", "BET_AWAY"):
            conf_bet.append(c)
        else:
            conf_nobet.append(c)
    if conf_bet:
        ax2.hist(conf_bet, bins=30, alpha=0.7, label=f"BET (n={len(conf_bet)})",
                 color="#2e7d32")
    if conf_nobet:
        ax2.hist(conf_nobet, bins=30, alpha=0.7, label=f"NO_BET (n={len(conf_nobet)})",
                 color="#d9534f")
    ax2.axvline(config.MIN_CONFIDENCE_BASE, color="black", linestyle="--",
                label=f"base {config.MIN_CONFIDENCE_BASE:.2f}")
    ax2.set_xlabel("confianza = max(p, 1-p)")
    ax2.set_ylabel("# muestras")
    ax2.set_title("Confianza: BET vs NO_BET")
    ax2.legend()
    return _save(fig, "06_probability_distribution.png")


# ============================================================================
# 7) Feature importance (agregada top de los mejores modelos)
# ============================================================================

def plot_feature_importance(summary: dict, top_n: int = 20) -> Path:
    models = _trained_models(summary)
    agg: dict[str, list[float]] = defaultdict(list)
    for m in models:
        for entry in m.get("top_features", []):
            if entry.get("importance") is not None:
                agg[entry["feature"]].append(entry["importance"])
    if not agg:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin importancias registradas", ha="center", va="center")
        return _save(fig, "07_feature_importance.png")

    mean_imp = {k: float(np.mean(v)) for k, v in agg.items()}
    ordered = sorted(mean_imp.items(), key=lambda kv: -kv[1])[:top_n]
    names = [k for k, _ in ordered]
    vals = [v for _, v in ordered]

    # Colorear por grupo
    def _group(n: str) -> str:
        return n.split("_", 1)[0]

    group_colors = {
        "score": "#1f77b4", "gp": "#ff7f0e", "traj": "#2ca02c",
        "pbp": "#d62728", "pace": "#9467bd", "league": "#8c564b",
        "meta": "#7f7f7f",
    }
    colors = [group_colors.get(_group(n), "#333") for n in names]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(names))))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("importancia promedio (sobre todos los modelos)")
    ax.set_title(f"Top-{top_n} features por importancia (promedio entre ligas)")
    # Leyenda por grupo
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=g) for g, c in group_colors.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    return _save(fig, "07_feature_importance.png")


# ============================================================================
# 8) Tamanos de dataset por liga (train/val/cal/holdout)
# ============================================================================

def plot_samples_per_league(
    samples_by_split: dict[str, list],
    top: int = 25,
) -> Path:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split, samples in samples_by_split.items():
        for s in samples:
            counts[s.league][split] += 1
    ordered = sorted(counts.items(), key=lambda kv: -sum(kv[1].values()))[:top]
    if not ordered:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin muestras", ha="center", va="center")
        return _save(fig, "08_samples_per_league.png")

    fig, ax = plt.subplots(figsize=(12, max(5, 0.28 * len(ordered))))
    y = np.arange(len(ordered))
    splits = ("train", "val", "cal", "holdout")
    colors = {"train": "#1f77b4", "val": "#ff7f0e", "cal": "#2ca02c", "holdout": "#d62728"}
    bottoms = np.zeros(len(ordered))
    for split in splits:
        vals = np.array([c.get(split, 0) for _, c in ordered])
        ax.barh(y, vals, left=bottoms, color=colors[split], label=split)
        bottoms += vals
    ax.set_yticks(y)
    ax.set_yticklabels([l[:32] for l, _ in ordered], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("# muestras")
    ax.set_title(f"Muestras por liga y split (top {top})")
    ax.axvline(config.LEAGUE_MIN_SAMPLES_TRAIN, color="black", linestyle="--",
               label=f"min train gate ({config.LEAGUE_MIN_SAMPLES_TRAIN})")
    ax.legend(loc="lower right")
    return _save(fig, "08_samples_per_league.png")


# ============================================================================
# 9) Coverage vs ROI (trade-off)
# ============================================================================

def plot_coverage_vs_roi(summary: dict) -> Path:
    models = _trained_models(summary)
    rows = []
    for m in models:
        ho = m.get("holdout_betting") or {}
        if ho.get("coverage") is None:
            continue
        rows.append({
            "label": f"{m['league'][:20]}/{m['target']}",
            "coverage": ho["coverage"],
            "roi": ho.get("roi") or 0,
            "n_bets": ho.get("n_bets", 0),
        })
    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Sin datos de holdout", ha="center", va="center")
        return _save(fig, "09_coverage_vs_roi.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    xs = [r["coverage"] for r in rows]
    ys = [r["roi"] for r in rows]
    sizes = [max(20, (r["n_bets"] or 0) * 2) for r in rows]
    colors = [_color_for_roi(r["roi"]) for r in rows]
    ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.7, edgecolors="black")
    for r, x, y in zip(rows, xs, ys):
        if (r["n_bets"] or 0) >= 20:
            ax.annotate(r["label"], (x, y), fontsize=7, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("coverage = # apuestas / # partidos")
    ax.set_ylabel(f"ROI holdout (odds {config.DEFAULT_ODDS})")
    ax.set_title("Trade-off coverage vs ROI por liga+target")
    ax.grid(True, alpha=0.3)
    return _save(fig, "09_coverage_vs_roi.png")


# ============================================================================
# Utilidad: reconstruir quarter_scores desde Sample
# ============================================================================

def _reconstruct_qs_from_sample(s: ds.Sample) -> dict[str, int]:
    f = s.features
    q1_total = int(f.get("q1_total", 0))
    q1_diff = int(f.get("q1_diff", 0))
    q1_home = (q1_total + q1_diff) // 2
    q1_away = q1_total - q1_home
    q2_total = int(f.get("q2_total", 0))
    q2_diff = int(f.get("q2_diff", 0))
    q2_home = (q2_total + q2_diff) // 2
    q2_away = q2_total - q2_home
    out = {
        "q1_home": q1_home, "q1_away": q1_away,
        "q2_home": q2_home, "q2_away": q2_away,
    }
    if s.target == "q4":
        q3_total = int(f.get("q3_total", 0))
        q3_diff = int(f.get("q3_diff", 0))
        q3_home = (q3_total + q3_diff) // 2
        q3_away = q3_total - q3_home
        out["q3_home"] = q3_home
        out["q3_away"] = q3_away
    return out


# ============================================================================
# Orquestador
# ============================================================================

def generate_all(
    use_cache: bool = True,
    include_calibration: bool = True,
    include_inference_based: bool = True,
) -> list[Path]:
    """Genera todas las graficas. Devuelve lista de paths."""
    summary = _load_summary()
    paths: list[Path] = []

    print("[v16/plots] 01 gap train-val...")
    paths.append(plot_train_val_gap(summary))

    print("[v16/plots] 07 feature importance...")
    paths.append(plot_feature_importance(summary))

    print("[v16/plots] 04 threshold curves...")
    paths.append(plot_threshold_curves(summary))

    print("[v16/plots] 03 ROI por liga...")
    paths.append(plot_roi_by_league(summary))

    print("[v16/plots] 09 coverage vs ROI...")
    paths.append(plot_coverage_vs_roi(summary))

    # Las siguientes requieren cargar muestras.
    print("[v16/plots] cargando muestras para plots 02/05/06/08...")
    samples, _ = ds.build_samples(use_cache=use_cache, verbose=False)
    splits = ds.split_temporal(samples)

    print("[v16/plots] 02 distribucion temporal...")
    paths.append(plot_temporal_distribution(splits))

    print("[v16/plots] 08 samples por liga...")
    paths.append(plot_samples_per_league(splits))

    if include_inference_based:
        print("[v16/plots] cargando engine + graph/pbp para 05/06...")
        engine = V15Engine.load(SUMMARY_PATH)
        # Preload graph/pbp para holdout solo
        holdout = [
            s for s in splits["holdout"]
            if (s.target == "q3" and s.snapshot_minute == config.Q3_GRAPH_CUTOFF)
            or (s.target == "q4" and s.snapshot_minute == config.Q4_GRAPH_CUTOFF)
        ]
        match_ids = sorted({s.match_id for s in holdout})
        conn = ds.get_db_connection()
        try:
            gp = ds.load_graph_points(conn, match_ids)
            pbp = ds.load_pbp_events(conn, match_ids)
        finally:
            conn.close()

        # Inferencia en holdout
        all_preds = []
        holdout_by_key: dict[tuple[str, str], list] = defaultdict(list)
        for s in holdout:
            holdout_by_key[(s.league, s.target)].append(s)
            quarter_scores = _reconstruct_qs_from_sample(s)
            pred = engine.predict(
                match_id=s.match_id, target=s.target, league=s.league,
                quarter_scores=quarter_scores,
                graph_points=gp.get(s.match_id, []),
                pbp_events=pbp.get(s.match_id, []),
            )
            all_preds.append({
                "signal": pred.signal,
                "probability_home": pred.probability,
                "true_winner": s.target_winner,
            })
        print("[v16/plots] 06 distribucion probabilidades...")
        paths.append(plot_probability_distribution(all_preds))

        if include_calibration:
            print("[v16/plots] 05 calibration curves...")
            paths.append(plot_calibration_curves(summary, engine, holdout_by_key, gp, pbp))

    print(f"[v16/plots] generadas {len(paths)} graficas en {PLOTS_DIR}")
    return paths


if __name__ == "__main__":
    generate_all()
