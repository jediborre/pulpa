"""
timesfm_features.py - Features derivadas del forecasting de la serie temporal
de diferencia de puntos (home - away) por evento PBP.

Doble implementacion:
  1. Si el paquete `timesfm` esta instalado y TIMESFM_ENABLED=True -> se usa
     el modelo pre-entrenado de Google Research (quantile forecasting).
  2. Si no -> fallback ligero con Holt exponential smoothing + bootstrap
     residual para estimar cuantiles. Las features son contract-compatibles.

Contract (features devueltas):
    tfm_winner_pick     : 1.0 si gana home, 0.0 si away, 0.5 si empate/abstencion
    tfm_margin          : margen proyectado al final del horizonte (puntos)
    tfm_uncertainty     : spread p90 - p10
    tfm_trend_slope     : pendiente promedio del forecast (puntos/evento)
    tfm_current_trend   : pendiente de los ultimos 5 eventos observados
    tfm_source          : "timesfm" | "holt_fallback" | "none" (no es feature
                          numerica, se guarda solo como metadata en debug)

Optimizaciones (v17+):
  - Cache en disco por hash de la serie. Acelera re-entrenamientos y
    re-evaluaciones cuando los samples no cambian.
  - Batch forecasting: extract_forecast_features_batch() procesa muchas
    series en una sola llamada al modelo (TimesFM acepta batches y eso
    reduce overhead de carga). Esencial para training (decenas de miles
    de muestras).

API:
    extract_forecast_features(pbp, horizon)         # singleton, util en inference
    extract_forecast_features_batch(pbps, horizon)  # bulk, util en training
    diagnostic()                                    # info backend activo
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Auto-detect timesfm
# ----------------------------------------------------------------------------

_TIMESFM_AVAILABLE = False
_timesfm_model = None   # lazy singleton
_model_lock = threading.Lock()

try:
    import timesfm as _tfm  # type: ignore
    _TIMESFM_AVAILABLE = True
except Exception:
    _tfm = None


def is_timesfm_available() -> bool:
    return _TIMESFM_AVAILABLE


# ----------------------------------------------------------------------------
# Cache en disco
# ----------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / "model_outputs"
_CACHE_PATH = _CACHE_DIR / "timesfm_features_cache.json"
_FEATURE_CACHE: dict[str, dict[str, float]] = {}
_CACHE_LOADED = False
_CACHE_DIRTY = False
_CACHE_LOCK = threading.Lock()


def _series_key(series: np.ndarray, horizon: int, backend: str) -> str:
    """Hash compacto + horizon + backend, para invalidar cache si cambia algo."""
    h = hashlib.blake2b(series.tobytes(), digest_size=12).hexdigest()
    return f"{backend}:h{horizon}:{h}"


def _load_cache() -> None:
    global _FEATURE_CACHE, _CACHE_LOADED
    if _CACHE_LOADED:
        return
    with _CACHE_LOCK:
        if _CACHE_LOADED:
            return
        if _CACHE_PATH.exists():
            try:
                _FEATURE_CACHE = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            except Exception:
                _FEATURE_CACHE = {}
        _CACHE_LOADED = True


def save_cache() -> None:
    """Persiste el cache en disco. Idempotente, llamada manual al final del training."""
    global _CACHE_DIRTY
    if not _CACHE_DIRTY:
        return
    with _CACHE_LOCK:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_FEATURE_CACHE), encoding="utf-8")
        tmp.replace(_CACHE_PATH)
        _CACHE_DIRTY = False


def clear_cache() -> None:
    global _FEATURE_CACHE, _CACHE_DIRTY
    with _CACHE_LOCK:
        _FEATURE_CACHE = {}
        _CACHE_DIRTY = True
        if _CACHE_PATH.exists():
            _CACHE_PATH.unlink()


# ----------------------------------------------------------------------------
# Loader singleton del modelo TimesFM real
# ----------------------------------------------------------------------------

def _get_timesfm_model(checkpoint: Optional[str] = None):
    """Carga el modelo timesfm una sola vez. Devuelve None si no esta disponible."""
    global _timesfm_model
    if _timesfm_model is not None:
        return _timesfm_model
    if not _TIMESFM_AVAILABLE:
        return None
    with _model_lock:
        if _timesfm_model is not None:
            return _timesfm_model
        try:
            _timesfm_model = _tfm.TimesFm(
                hparams=_tfm.TimesFmHparams(
                    backend="cpu",
                    per_core_batch_size=32,
                    horizon_len=32,
                    num_layers=20,
                    use_positional_embedding=False,
                    context_len=512,
                ),
                checkpoint=_tfm.TimesFmCheckpoint(
                    huggingface_repo_id=checkpoint or "google/timesfm-1.0-200m-pytorch"
                ),
            )
            return _timesfm_model
        except Exception:
            _timesfm_model = None
            return None


# ----------------------------------------------------------------------------
# Empty / extraccion de la serie diff
# ----------------------------------------------------------------------------

_MIN_SERIES_LEN = 10


def _empty_features() -> dict[str, float]:
    return {
        "tfm_winner_pick": 0.5,
        "tfm_margin": 0.0,
        "tfm_uncertainty": 99.0,
        "tfm_trend_slope": 0.0,
        "tfm_current_trend": 0.0,
    }


def _extract_diff_series(pbp: Iterable[dict]) -> np.ndarray:
    """Extrae home_score - away_score por evento, ordenado temporalmente."""
    events = [
        (e.get("minute"), e.get("home_score"), e.get("away_score"))
        for e in pbp
        if e.get("home_score") is not None and e.get("away_score") is not None
    ]
    events.sort(key=lambda x: x[0] if x[0] is not None else 0)
    diffs = [float(h - a) for _, h, a in events]
    return np.asarray(diffs, dtype=np.float32)


# ----------------------------------------------------------------------------
# Backend: TimesFM real (single + batch)
# ----------------------------------------------------------------------------

def _features_from_timesfm_output(
    series: np.ndarray, point_row: np.ndarray, quant_row: np.ndarray, horizon: int,
) -> dict[str, float]:
    """Convierte la salida de model.forecast (1 serie) en el dict de features."""
    p_end = float(point_row[-1])
    p10 = float(quant_row[-1, 0])
    p90 = float(quant_row[-1, -1])
    slope = float((point_row[-1] - point_row[0]) / max(horizon - 1, 1))
    recent_slope = _recent_slope(series)
    winner = 1.0 if p_end > 0.5 else (0.0 if p_end < -0.5 else 0.5)
    return {
        "tfm_winner_pick": winner,
        "tfm_margin": p_end,
        "tfm_uncertainty": max(p90 - p10, 0.0),
        "tfm_trend_slope": slope,
        "tfm_current_trend": recent_slope,
    }


def _forecast_timesfm(series: np.ndarray, horizon: int) -> dict[str, float]:
    """Usa el modelo TimesFM de Google Research (1 serie)."""
    model = _get_timesfm_model()
    if model is None:
        return _forecast_holt_fallback(series, horizon)
    try:
        point, quant = model.forecast(horizon=horizon, inputs=[series])
        return _features_from_timesfm_output(series, point[0], quant[0], horizon)
    except Exception:
        return _forecast_holt_fallback(series, horizon)


def _forecast_timesfm_batch(
    series_list: list[np.ndarray], horizon: int,
) -> list[dict[str, float]]:
    """Procesa N series en una sola llamada al modelo. Mucho mas rapido en CPU."""
    model = _get_timesfm_model()
    if model is None:
        return [_forecast_holt_fallback(s, horizon) for s in series_list]
    if not series_list:
        return []
    try:
        point, quant = model.forecast(horizon=horizon, inputs=series_list)
        return [
            _features_from_timesfm_output(s, point[i], quant[i], horizon)
            for i, s in enumerate(series_list)
        ]
    except Exception:
        return [_forecast_holt_fallback(s, horizon) for s in series_list]


# ----------------------------------------------------------------------------
# Backend: fallback Holt + bootstrap
# ----------------------------------------------------------------------------

def _holt_forecast(series: np.ndarray, horizon: int,
                   alpha: float = 0.5, beta: float = 0.2) -> np.ndarray:
    """Holt double exponential smoothing. Devuelve array (horizon,)."""
    if len(series) == 0:
        return np.zeros(horizon, dtype=np.float32)
    level = float(series[0])
    trend = float(series[1] - series[0]) if len(series) > 1 else 0.0
    for t in range(1, len(series)):
        y = float(series[t])
        new_level = alpha * y + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level, trend = new_level, new_trend
    out = np.asarray(
        [level + (h + 1) * trend for h in range(horizon)],
        dtype=np.float32,
    )
    return out


def _bootstrap_quantiles(
    series: np.ndarray, horizon: int, forecast: np.ndarray,
    n_bootstrap: int = 500, seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap residual para p10 y p90 al final del horizon."""
    if len(series) < 3:
        return (forecast[-1] - 5.0, forecast[-1] + 5.0)
    fit_in_sample = []
    level = float(series[0])
    trend = float(series[1] - series[0]) if len(series) > 1 else 0.0
    for t in range(1, len(series)):
        y = float(series[t])
        pred = level + trend
        fit_in_sample.append(pred)
        level = 0.5 * y + 0.5 * (level + trend)
        trend = 0.2 * (level - pred) + 0.8 * trend
    residuals = np.asarray(
        [series[t + 1] - fit_in_sample[t] for t in range(len(fit_in_sample))],
        dtype=np.float32,
    )
    if len(residuals) == 0 or residuals.std() < 1e-6:
        return (forecast[-1] - 3.0, forecast[-1] + 3.0)
    rng = np.random.default_rng(seed)
    end_samples = np.empty(n_bootstrap, dtype=np.float32)
    for b in range(n_bootstrap):
        draws = rng.choice(residuals, size=horizon, replace=True)
        end_samples[b] = forecast[-1] + draws.sum()
    p10 = float(np.percentile(end_samples, 10))
    p90 = float(np.percentile(end_samples, 90))
    return (p10, p90)


def _recent_slope(series: np.ndarray, window: int = 5) -> float:
    """Slope lineal de los ultimos `window` valores."""
    if len(series) < 2:
        return 0.0
    tail = series[-window:]
    if len(tail) < 2:
        return 0.0
    xs = np.arange(len(tail), dtype=np.float32)
    ys = tail
    x_mean = xs.mean()
    y_mean = ys.mean()
    num = float(((xs - x_mean) * (ys - y_mean)).sum())
    den = float(((xs - x_mean) ** 2).sum())
    return num / den if den > 0 else 0.0


_MARGIN_CLIP = 40.0   # basket diff realista


def _forecast_holt_fallback(series: np.ndarray, horizon: int) -> dict[str, float]:
    forecast = _holt_forecast(series, horizon)
    forecast = np.clip(forecast, -_MARGIN_CLIP, _MARGIN_CLIP)
    p10, p90 = _bootstrap_quantiles(series, horizon, forecast)
    p10 = max(-_MARGIN_CLIP, min(_MARGIN_CLIP, p10))
    p90 = max(-_MARGIN_CLIP, min(_MARGIN_CLIP, p90))
    p_end = float(forecast[-1])
    slope = float((forecast[-1] - forecast[0]) / max(horizon - 1, 1))
    recent = _recent_slope(series)
    winner = 1.0 if p_end > 0.5 else (0.0 if p_end < -0.5 else 0.5)
    return {
        "tfm_winner_pick": winner,
        "tfm_margin": p_end,
        "tfm_uncertainty": max(p90 - p10, 0.0),
        "tfm_trend_slope": slope,
        "tfm_current_trend": recent,
    }


# ----------------------------------------------------------------------------
# API publica
# ----------------------------------------------------------------------------

def extract_forecast_features(
    pbp: Iterable[dict],
    horizon: int = 20,
    use_timesfm: bool = True,
) -> dict[str, float]:
    """
    Extrae features de forecasting sobre la serie diff del partido (1 muestra).
    Util en INFERENCE live (donde solo hay una prediccion a la vez).
    """
    global _CACHE_DIRTY
    series = _extract_diff_series(pbp)
    if len(series) < _MIN_SERIES_LEN:
        return _empty_features()
    backend = "timesfm" if (use_timesfm and _TIMESFM_AVAILABLE) else "holt"
    _load_cache()
    key = _series_key(series, horizon, backend)
    if key in _FEATURE_CACHE:
        return dict(_FEATURE_CACHE[key])
    if backend == "timesfm":
        out = _forecast_timesfm(series, horizon)
    else:
        out = _forecast_holt_fallback(series, horizon)
    _FEATURE_CACHE[key] = dict(out)
    _CACHE_DIRTY = True
    return out


def extract_forecast_features_batch(
    pbps: list[Iterable[dict]],
    horizon: int = 20,
    use_timesfm: bool = True,
    batch_size: int = 32,
    show_progress: bool = False,
) -> list[dict[str, float]]:
    """
    Extrae features de forecasting para una lista de pbp (1 por muestra).

    El procesamiento batchea (batch_size) llamadas al modelo, lo que reduce
    drasticamente el tiempo total cuando hay miles de muestras (training).
    Resultados que estan en cache se sirven sin llamar al modelo.

    Devuelve una lista de dicts en el mismo orden que `pbps`.
    """
    global _CACHE_DIRTY
    _load_cache()
    backend = "timesfm" if (use_timesfm and _TIMESFM_AVAILABLE) else "holt"

    results: list[dict[str, float] | None] = [None] * len(pbps)
    pending: list[tuple[int, np.ndarray, str]] = []

    for idx, pbp in enumerate(pbps):
        series = _extract_diff_series(pbp)
        if len(series) < _MIN_SERIES_LEN:
            results[idx] = _empty_features()
            continue
        key = _series_key(series, horizon, backend)
        if key in _FEATURE_CACHE:
            results[idx] = dict(_FEATURE_CACHE[key])
            continue
        pending.append((idx, series, key))

    if not pending:
        return [r if r is not None else _empty_features() for r in results]

    if backend != "timesfm":
        for idx, series, key in pending:
            out = _forecast_holt_fallback(series, horizon)
            _FEATURE_CACHE[key] = dict(out)
            results[idx] = out
        _CACHE_DIRTY = True
        return [r if r is not None else _empty_features() for r in results]

    iterator = range(0, len(pending), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc=f"timesfm batch ({len(pending)} series)")
        except Exception:
            pass

    for start in iterator:
        chunk = pending[start: start + batch_size]
        series_list = [s for _, s, _ in chunk]
        outs = _forecast_timesfm_batch(series_list, horizon)
        for (idx, _series, key), out in zip(chunk, outs):
            _FEATURE_CACHE[key] = dict(out)
            results[idx] = out

    _CACHE_DIRTY = True
    return [r if r is not None else _empty_features() for r in results]


def diagnostic() -> dict[str, str]:
    """Util para logging: dice que backend se esta usando."""
    return {
        "timesfm_available": str(_TIMESFM_AVAILABLE),
        "active_backend": "timesfm" if _TIMESFM_AVAILABLE else "holt_fallback",
        "cache_path": str(_CACHE_PATH),
        "cache_size": str(len(_FEATURE_CACHE)),
    }

