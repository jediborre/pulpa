"""
train.py - Orquestador de entrenamiento V17.

Pipeline:
  1. Build samples (cache disponible).
  2. Split temporal: train / val / cal / holdout.
  3. Pace thresholds (percentiles 33/66) a partir de TRAIN (no full).
  4. League stats walk-forward (nunca usa futuro).
  5. Para cada liga con >= MIN_LEAGUE_SAMPLES_TRAIN:
       - vectorizar features
       - entrenar clasificador Q3 y Q4
       - entrenar regresores (home / away / total)
       - calibrar con cal set
       - aprender threshold optimo de ROI en val (odds 1.40)
       - evaluar en holdout
  6. Persistir modelos (joblib) y training_summary.json con toda la metadata.

Salida: training/v17/model_outputs/
  - training_summary_v17.json       (config + metricas + thresholds por liga)
  - league_<slug>_q3_clf.joblib     (CalibratedClassifier con ensemble interno)
  - league_<slug>_q3_reg_total.joblib
  - league_<slug>_q3_vectorizer.joblib
  - ... (analogos para q4)
"""

from __future__ import annotations

import json
import time
import traceback
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer

from training.v17 import config, dataset as ds, features as feat, models as M


MODEL_DIR = Path(__file__).parent / "model_outputs"
SUMMARY_PATH = MODEL_DIR / "training_summary_v17.json"


# ============================================================================
# Utilidades
# ============================================================================

def _league_samples(samples: list[ds.Sample], league: str, target: str) -> list[ds.Sample]:
    return [s for s in samples if s.league == league and s.target == target]


def _vectorize(
    train_samples: list[ds.Sample],
    val_samples: list[ds.Sample],
    cal_samples: list[ds.Sample],
    holdout_samples: list[ds.Sample],
    graph_points: dict,
    pbp_events: dict,
    league_stats_map: dict[str, dict[str, float]],
    pace_thresholds: dict[str, float],
    tfm_cache: dict[tuple, dict[str, float]] | None = None,
) -> tuple[DictVectorizer, Any, Any, Any, Any]:
    def to_dicts(samples):
        return [
            feat.build_features_for_sample(
                s, graph_points, pbp_events,
                league_stats=league_stats_map.get(s.match_id),
                pace_thresholds=pace_thresholds,
                tfm_cache=tfm_cache,
            )
            for s in samples
        ]

    train_dicts = to_dicts(train_samples)
    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(train_dicts)
    X_val = vec.transform(to_dicts(val_samples)) if val_samples else np.zeros((0, X_train.shape[1]))
    X_cal = vec.transform(to_dicts(cal_samples)) if cal_samples else np.zeros((0, X_train.shape[1]))
    X_holdout = vec.transform(to_dicts(holdout_samples)) if holdout_samples else np.zeros((0, X_train.shape[1]))
    return vec, X_train, X_val, X_cal, X_holdout


def _targets(samples: list[ds.Sample]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_win = np.array([s.target_winner for s in samples], dtype=int)
    y_home = np.array([s.target_home_pts for s in samples], dtype=float)
    y_away = np.array([s.target_away_pts for s in samples], dtype=float)
    y_total = np.array([s.target_total_pts for s in samples], dtype=float)
    return y_win, y_home, y_away, y_total


# ============================================================================
# Aprendizaje de threshold optimo por liga (ROI en val con odds 1.40)
# ============================================================================

def _learn_optimal_threshold(
    probas: np.ndarray,
    y_true: np.ndarray,
    odds: float = config.DEFAULT_ODDS,
    grid: tuple[float, ...] = tuple(np.round(np.arange(0.55, 0.93, 0.01), 2)),
) -> dict[str, Any]:
    """
    Busca el threshold que maximiza ROI respetando un minimo de N apuestas.
    Devuelve metadata util para debug.
    """
    best = {"threshold": config.MIN_CONFIDENCE_BASE, "roi": -1.0, "n_bets": 0, "hit_rate": 0.0}
    if len(probas) == 0:
        return best
    # Confianza = max(p1, p0)
    conf = np.maximum(probas, 1 - probas)
    pred = (probas >= 0.5).astype(int)
    correct = (pred == y_true).astype(int)

    curve = []
    min_bets = max(30, int(0.05 * len(probas)))
    for t in grid:
        mask = conf >= t
        n_bets = int(mask.sum())
        if n_bets < min_bets:
            curve.append({"threshold": float(t), "n_bets": n_bets, "roi": None, "hit_rate": None})
            continue
        wins = int(correct[mask].sum())
        hit = wins / n_bets
        pnl = wins * (odds - 1) - (n_bets - wins)
        roi = pnl / n_bets
        curve.append({
            "threshold": float(t),
            "n_bets": n_bets,
            "hit_rate": float(hit),
            "roi": float(roi),
        })
        # Preferimos ROI positivo y hit rate >= MIN_ACCEPTABLE
        if hit >= config.MIN_ACCEPTABLE_HIT_RATE and roi > best["roi"]:
            best = {
                "threshold": float(t),
                "roi": float(roi),
                "n_bets": n_bets,
                "hit_rate": float(hit),
            }

    # Fallback: si no hubo ningun threshold con hit_rate >= MIN, usar el que
    # maximice ROI aunque sea negativo (senal de que esta liga no sirve).
    if best["roi"] < 0:
        best_curve = [c for c in curve if c.get("roi") is not None]
        if best_curve:
            best_entry = max(best_curve, key=lambda c: c["roi"])
            best = {
                "threshold": float(best_entry["threshold"]),
                "roi": float(best_entry["roi"]),
                "n_bets": int(best_entry["n_bets"]),
                "hit_rate": float(best_entry["hit_rate"]),
            }
    # Piso y techo
    best["threshold"] = max(
        min(best["threshold"], config.MAX_CONFIDENCE_CAP),
        config.MIN_CONFIDENCE_BASE,
    )
    best["curve"] = curve
    return best


# ============================================================================
# Entrenador por liga+target
# ============================================================================

def _train_league_target(
    league: str,
    target: str,
    all_samples: dict[str, list[ds.Sample]],
    graph_points: dict,
    pbp_events: dict,
    league_stats_map: dict[str, dict[str, float]],
    pace_thresholds: dict[str, float],
    tfm_cache: dict[tuple, dict[str, float]] | None = None,
) -> dict[str, Any] | None:
    train = _league_samples(all_samples["train"], league, target)
    val = _league_samples(all_samples["val"], league, target)
    cal = _league_samples(all_samples["cal"], league, target)
    holdout = _league_samples(all_samples["holdout"], league, target)

    n_train = len(train)
    if n_train < config.LEAGUE_MIN_SAMPLES_TRAIN:
        return {"skipped": True, "reason": "insufficient_train", "n_train": n_train}
    if len(val) < config.LEAGUE_MIN_SAMPLES_VAL:
        return {"skipped": True, "reason": "insufficient_val", "n_train": n_train, "n_val": len(val)}

    print(f"  [train] {league} / {target}: train={n_train} val={len(val)} cal={len(cal)} holdout={len(holdout)}")

    vec, X_train, X_val, X_cal, X_holdout = _vectorize(
        train, val, cal, holdout,
        graph_points, pbp_events, league_stats_map, pace_thresholds,
        tfm_cache=tfm_cache,
    )
    yw_tr, yh_tr, ya_tr, yt_tr = _targets(train)
    yw_vl, yh_vl, ya_vl, yt_vl = _targets(val)
    yw_cl, *_ = _targets(cal) if cal else (np.array([]), None, None, None)
    yw_ho, yh_ho, ya_ho, yt_ho = _targets(holdout) if holdout else (np.array([]),)*4

    # Clasificador
    clf_algos = M.select_clf_algos(n_train)
    clf = M.ClassifierEnsemble(algos=clf_algos)
    clf.fit(X_train, yw_tr, X_val=X_val, y_val=yw_vl)
    # Calibracion (requiere set cal separado)
    if len(cal) >= 30:
        calibrated = M.calibrate_classifier(clf, X_cal, yw_cl)
    else:
        calibrated = clf
    # Regresores (home, away, total)
    reg_algos = M.select_reg_algos(n_train)
    reg_home = M.RegressorEnsemble(algos=reg_algos).fit(X_train, yh_tr, X_val=X_val, y_val=yh_vl)
    reg_away = M.RegressorEnsemble(algos=reg_algos).fit(X_train, ya_tr, X_val=X_val, y_val=ya_vl)
    reg_total = M.RegressorEnsemble(algos=reg_algos).fit(X_train, yt_tr, X_val=X_val, y_val=yt_vl)

    # Metricas val / holdout
    val_metrics = M.eval_classifier(calibrated, X_val, yw_vl)
    train_metrics = M.eval_classifier(calibrated, X_train, yw_tr)
    holdout_metrics = M.eval_classifier(calibrated, X_holdout, yw_ho) if len(yw_ho) else None

    # Train-val gap: senal temprana de overfitting. Se preserva como campo
    # propio para que _summarize_prod.py y test_roi.py no tengan que recalcular.
    train_val_gap: float | None = None
    try:
        f1_tr = (train_metrics or {}).get("f1")
        f1_vl = (val_metrics or {}).get("f1")
        if f1_tr is not None and f1_vl is not None:
            train_val_gap = float(f1_tr) - float(f1_vl)
    except Exception:
        train_val_gap = None

    reg_metrics = {
        "home_val": M.eval_regressor(reg_home, X_val, yh_vl),
        "away_val": M.eval_regressor(reg_away, X_val, ya_vl),
        "total_val": M.eval_regressor(reg_total, X_val, yt_vl),
    }

    # Threshold optimo (usar val)
    val_probas = calibrated.predict_proba(X_val)[:, 1]
    threshold_meta = _learn_optimal_threshold(val_probas, yw_vl)

    # Evaluacion holdout con threshold aprendido
    holdout_betting = None
    if len(yw_ho):
        ho_probas = calibrated.predict_proba(X_holdout)[:, 1]
        ho_conf = np.maximum(ho_probas, 1 - ho_probas)
        mask = ho_conf >= threshold_meta["threshold"]
        if mask.any():
            preds = (ho_probas >= 0.5).astype(int)
            n = int(mask.sum())
            wins = int((preds[mask] == yw_ho[mask]).sum())
            hit = wins / n if n else 0.0
            roi = (wins * (config.DEFAULT_ODDS - 1) - (n - wins)) / n if n else 0.0
            holdout_betting = {
                "n_bets": n, "wins": wins, "hit_rate": hit, "roi": roi,
                "coverage": n / len(yw_ho),
            }

    # Importancia de features (si el primer ensemble tiene feature_importances_)
    feat_names = vec.get_feature_names_out().tolist()
    feat_importances = _best_effort_importances(clf, feat_names)

    # Persistir
    slug = ds.slugify_league(league)
    path_prefix = MODEL_DIR / f"league_{slug}_{target}"
    joblib.dump(vec, f"{path_prefix}_vectorizer.joblib")
    joblib.dump(calibrated, f"{path_prefix}_clf.joblib")
    joblib.dump(reg_home, f"{path_prefix}_reg_home.joblib")
    joblib.dump(reg_away, f"{path_prefix}_reg_away.joblib")
    joblib.dump(reg_total, f"{path_prefix}_reg_total.joblib")

    return {
        "league": league,
        "slug": slug,
        "target": target,
        "n_train": n_train, "n_val": len(val), "n_cal": len(cal), "n_holdout": len(holdout),
        "clf_algos": clf_algos,
        "reg_algos": reg_algos,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "train_val_gap": train_val_gap,
        "holdout_metrics": holdout_metrics,
        "regression_metrics": reg_metrics,
        "threshold": threshold_meta,
        "holdout_betting": holdout_betting,
        "top_features": feat_importances[: config.INCLUDE_TOP_FEATURES_IN_DEBUG],
        "model_files": {
            "clf": f"league_{slug}_{target}_clf.joblib",
            "vectorizer": f"league_{slug}_{target}_vectorizer.joblib",
            "reg_home": f"league_{slug}_{target}_reg_home.joblib",
            "reg_away": f"league_{slug}_{target}_reg_away.joblib",
            "reg_total": f"league_{slug}_{target}_reg_total.joblib",
        },
    }


def _precompute_tfm_cache(
    candidate_leagues: list[str],
    splits: dict[str, list[ds.Sample]],
    pbp_events: dict[str, list[dict]],
    verbose: bool = True,
) -> dict[tuple, dict[str, float]] | None:
    """
    Precomputa features TimesFM en BATCH para todos los samples relevantes.

    El batching reduce el coste por sample drasticamente (de ~50ms/sample a
    ~5ms/sample en CPU con batch=32). Si TimesFM no esta disponible o esta
    deshabilitado, devuelve None y `build_features_for_sample` cae al
    fallback Holt sin perder funcionalidad.
    """
    if not getattr(config, "TIMESFM_ENABLED", False):
        return None

    backend = getattr(config, "FORECAST_BACKEND", "timesfm")
    try:
        if backend == "chronos":
            from training.v17 import chronos_features as _fc
            is_available = _fc.is_chronos_available()
            diag_key = "chronos"
            pkg_label = "chronos"
        else:
            from training.v17 import timesfm_features as _fc
            is_available = _fc.is_timesfm_available()
            diag_key = "timesfm"
            pkg_label = "timesfm"
    except Exception as e:
        print(f"[v17] aviso: no pude importar forecast module ({e}). "
              f"build_features se servira sin cache.")
        return None

    if not is_available:
        if verbose:
            print(f"[v17/{diag_key}] paquete `{pkg_label}` no disponible; "
                  f"se usara fallback sin batch.")
        return None

    candidate_set = set(candidate_leagues)
    samples_to_compute: list = []
    for bucket_name in ("train", "val", "cal", "holdout"):
        for s in splits.get(bucket_name, []):
            if s.league in candidate_set:
                samples_to_compute.append(s)

    if not samples_to_compute:
        return {}

    if verbose:
        diag = _fc.diagnostic()
        print(f"[v17/{diag_key}] backend={diag['active_backend']} "
              f"cache_loaded={diag.get('cache_size', '?')} "
              f"samples_to_compute={len(samples_to_compute)}")

    # Pbp truncado al snapshot (mismo cutoff que features.py)
    pbp_inputs: list[list[dict]] = []
    for s in samples_to_compute:
        cutoff = float(s.snapshot_minute)
        pbp_all = pbp_events.get(s.match_id, [])
        pbp_before = feat.pbp_before_target(pbp_all, s.target, cutoff)
        pbp_inputs.append(pbp_before)

    feats_list = _fc.extract_forecast_features_batch(
        pbp_inputs,
        horizon=config.TIMESFM_FORECAST_HORIZON,
        use_timesfm=True,
        batch_size=32 if backend != "chronos" else 16,
        show_progress=verbose,
    )

    cache: dict[tuple, dict[str, float]] = {}
    for s, f in zip(samples_to_compute, feats_list):
        cache[(s.match_id, s.target, int(s.snapshot_minute))] = f
    if verbose:
        print(f"[v17/{diag_key}] cache poblado: {len(cache)} entries")
    return cache


def _best_effort_importances(clf: M.ClassifierEnsemble, names: list[str]) -> list[dict]:
    """Intenta extraer importancias desde el primer modelo del ensemble."""
    try:
        for m in clf.models_:
            if hasattr(m, "feature_importances_"):
                imps = m.feature_importances_
                pairs = sorted(zip(names, imps), key=lambda p: -p[1])
                return [{"feature": n, "importance": float(i)} for n, i in pairs]
        if hasattr(clf.models_[0], "coef_"):
            coefs = np.abs(clf.models_[0].coef_[0])
            pairs = sorted(zip(names, coefs), key=lambda p: -p[1])
            return [{"feature": n, "importance": float(i)} for n, i in pairs]
    except Exception:
        pass
    return []


# ============================================================================
# Main
# ============================================================================

def run_training(
    use_cache: bool = True,
    verbose: bool = True,
    train_days: int | None = None,
    val_days: int | None = None,
    cal_days: int | None = None,
    holdout_days: int | None = None,
    min_samples_train: int | None = None,
    min_samples_val: int | None = None,
    active_days: int | None = None,
    train_end_date: str | None = None,
    val_end_date: str | None = None,
    cal_end_date: str | None = None,
    use_league_activation: bool | None = None,
) -> dict[str, Any]:
    t_start = time.time()
    MODEL_DIR.mkdir(exist_ok=True, parents=True)

    eff_train_days = train_days if train_days is not None else config.TRAIN_DAYS
    eff_val_days = val_days if val_days is not None else config.VAL_DAYS
    eff_min_train = (
        min_samples_train if min_samples_train is not None
        else config.LEAGUE_MIN_SAMPLES_TRAIN
    )
    eff_min_val = (
        min_samples_val if min_samples_val is not None
        else config.LEAGUE_MIN_SAMPLES_VAL
    )

    # 1. Muestras
    print("[v17] building samples...")
    samples, _meta = ds.build_samples(use_cache=use_cache, verbose=verbose)

    # 1a. Selector inteligente de ligas (v17): scoring multi-factor en
    # dataset_analyzer. Solo se aceptan ligas con decision='activate'.
    if (getattr(config, "USE_LEAGUE_ACTIVATION", True)
            if use_league_activation is None else use_league_activation):
        try:
            from training.v17 import dataset_analyzer
            assessments = dataset_analyzer.assess_leagues()
            active_set = {a.league for a in assessments if a.decision == "activate"}
            dormant = [a for a in assessments if a.decision == "dormant"]
            before = len(samples)
            samples = [s for s in samples if s.league in active_set]
            print(f"[v17] selector inteligente de ligas: "
                  f"{before} -> {len(samples)} muestras "
                  f"({len(active_set)} activas / {len(dormant)} dormidas). "
                  f"Reporte en model_outputs/league_assessments_v17.json")
            dataset_analyzer.save_report(assessments)
        except Exception as e:
            print(f"[v17] aviso: dataset_analyzer fallo ({e}), "
                  f"continuo sin filtro de ligas inteligente.")

    # 1b. Filtro opcional "ligas activas": solo ligas con >=1 partido en los
    # ultimos `active_days` dias. Util en produccion cuando la temporada baja.
    if active_days and active_days > 0:
        from datetime import datetime, timedelta
        parsed = [(s, datetime.fromisoformat(s.date[:10])) for s in samples
                  if s.date]
        if parsed:
            newest = max(d for _, d in parsed)
            active_cutoff = newest - timedelta(days=active_days)
            active_leagues = {
                s.league for s, d in parsed if d >= active_cutoff
            }
            before = len(samples)
            samples = [s for s in samples if s.league in active_leagues]
            print(f"[v17] filtro ligas activas (ultimos {active_days}d): "
                  f"{before} -> {len(samples)} muestras  "
                  f"({len(active_leagues)} ligas activas)")

    # 2. Split
    print(f"[v17] split temporal "
          f"(train={eff_train_days}d val={eff_val_days}d "
          f"cal={cal_days}d holdout={holdout_days}d)...")
    splits = ds.split_temporal(
        samples,
        train_days=eff_train_days,
        val_days=eff_val_days,
        cal_days=cal_days,
        holdout_days=holdout_days,
        train_end_date=train_end_date,
        val_end_date=val_end_date,
        cal_end_date=cal_end_date,
    )
    for name, bucket in splits.items():
        print(f"  [{name}] {len(bucket)} muestras")

    # 3. Pace thresholds desde TRAIN
    pace_thresholds = ds.calculate_pace_thresholds(splits["train"])
    print(f"[v17] pace thresholds: {pace_thresholds}")

    # 4. League stats walk-forward sobre TODAS las muestras (cada match usa solo
    # historico previo, garantizando no leakage).
    print("[v17] computing walk-forward league stats...")
    conn = ds.get_db_connection()
    league_stats_map = ds.compute_league_stats_walkforward(samples, conn=conn)
    # Graph y PBP en memoria
    match_ids = sorted({s.match_id for s in samples})
    gp = ds.load_graph_points(conn, match_ids)
    pbp = ds.load_pbp_events(conn, match_ids)
    conn.close()

    # 5. Enumerar ligas candidatas (muestras en train >= threshold)
    # Aplicamos override local si se paso por parametro.
    _orig_min_train = config.LEAGUE_MIN_SAMPLES_TRAIN
    _orig_min_val = config.LEAGUE_MIN_SAMPLES_VAL
    config.LEAGUE_MIN_SAMPLES_TRAIN = eff_min_train
    config.LEAGUE_MIN_SAMPLES_VAL = eff_min_val

    leagues_train = defaultdict(int)
    for s in splits["train"]:
        leagues_train[s.league] += 1
    candidate_leagues = [
        l for l, n in sorted(leagues_train.items(), key=lambda kv: -kv[1])
        if n >= eff_min_train * 2  # x2 por targets Q3+Q4
    ]
    print(f"[v17] {len(candidate_leagues)} ligas candidatas "
          f"(de {len(leagues_train)} con al menos 1 muestra en train)")

    # 5b. Pre-computar features TimesFM en batch para TODOS los samples de
    # ligas candidatas. Acelera dramaticamente el training (decenas de miles
    # de llamadas al modelo se sirven con cache + batching de 32).
    tfm_cache = _precompute_tfm_cache(
        candidate_leagues, splits, pbp, verbose=verbose,
    )

    league_results: list[dict] = []
    skipped: list[dict] = []
    for league in candidate_leagues:
        for target in ("q3", "q4"):
            try:
                r = _train_league_target(
                    league, target, splits, gp, pbp, league_stats_map, pace_thresholds,
                    tfm_cache=tfm_cache,
                )
                if r is None or r.get("skipped"):
                    skipped.append({"league": league, "target": target, **(r or {})})
                    continue
                league_results.append(r)
            except Exception as e:
                traceback.print_exc()
                skipped.append({
                    "league": league, "target": target,
                    "skipped": True, "reason": f"exception:{e}",
                })

    # Persistir cache del backend activo para futuros re-entrenamientos
    _backend = getattr(config, "FORECAST_BACKEND", "timesfm")
    try:
        if _backend == "chronos":
            from training.v17 import chronos_features as _fc
        else:
            from training.v17 import timesfm_features as _fc
        _fc.save_cache()
    except Exception as e:
        print(f"[v17] aviso: no pude persistir {_backend} cache ({e})")

    # Restaurar config global
    config.LEAGUE_MIN_SAMPLES_TRAIN = _orig_min_train
    config.LEAGUE_MIN_SAMPLES_VAL = _orig_min_val

    # 6. Persistir summary
    summary = {
        "version": config.VERSION,
        "trained_at": int(time.time()),
        "training_seconds": time.time() - t_start,
        "config_snapshot": _config_snapshot(),
        "run_params": {
            "train_days": eff_train_days,
            "val_days": eff_val_days,
            "cal_days": cal_days,
            "holdout_days": holdout_days,
            "min_samples_train": eff_min_train,
            "min_samples_val": eff_min_val,
            "active_days": active_days,
            "train_end_date": train_end_date,
            "val_end_date": val_end_date,
            "cal_end_date": cal_end_date,
            "use_league_activation": use_league_activation,
        },
        "pace_thresholds": pace_thresholds,
        "splits": {name: len(b) for name, b in splits.items()},
        "n_leagues_trained": len(league_results),
        "n_leagues_skipped": len(skipped),
        "models": league_results,
        "skipped": skipped,
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_jsonable)
    print(f"[v17] summary -> {SUMMARY_PATH}")
    print(f"[v17] trained {len(league_results)} (league,target) pairs, "
          f"skipped {len(skipped)}")

    # Generar graficas de diagnostico
    try:
        from training.v17 import plots
        plots.generate_all(use_cache=True, include_inference_based=True)
    except Exception as e:
        print(f"[v17] aviso: generacion de plots fallo ({e}). "
              "Correr `python -m training.v17.cli plots` manualmente.")
    return summary


def _config_snapshot() -> dict[str, Any]:
    return {
        k: getattr(config, k)
        for k in dir(config)
        if k.isupper() and not k.startswith("_")
    }


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


if __name__ == "__main__":
    run_training()

