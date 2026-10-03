"""
chronos_features.py - Backend de forecasting basado en Amazon Chronos/ChronosBolt.

Contract-compatible con timesfm_features (mismas claves tfm_*, misma API).
Implementa cache en disco por hash de serie y batch forecasting nativo.

Activacion:
  1. pip install chronos-forecasting
  2. config.FORECAST_BACKEND = "chronos"
  3. config.TIMESFM_ENABLED = True  (o False, el flag solo habilita grupo G8)

Modelos disponibles (CHRONOS_MODEL_NAME en config.py):
  amazon/chronos-bolt-tiny    (~8M  params, ~100 ms/32-batch CPU)  <- DEFAULT
  amazon/chronos-bolt-small   (~46M params)
  amazon/chronos-t5-small     (~46M params, sampling, mas lento)
  amazon/chronos-t5-base      (~200M params)

Cache en disco: model_outputs/chronos_features_cache.json
La clave incluye nombre del modelo + horizon para invalidar automaticamente.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Auto-detect chronos
# ---------------------------------------------------------------------------

_CHRONOS_AVAILABLE = False
_chronos_model = None
_model_lock = threading.Lock()
_is_bolt = False   # True si el modelo activo es ChronosBolt (quantile direct)

try:
    from chronos import ChronosBoltPipeline as _ChronosBoltPipeline  # type: ignore
    from chronos import ChronosPipeline as _ChronosPipeline           # type: ignore
    _CHRONOS_AVAILABLE = True
except Exception:
    _ChronosBoltPipeline = None
    _ChronosPipeline = None


def is_chronos_available() -> bool:
    return _CHRONOS_AVAILABLE


# Default: Chronos Bolt tiny (~8M params).
_DEFAULT_MODEL = "amazon/chronos-bolt-tiny"


def _active_model_name() -> str:
    try:
        from training.v16 import config as _cfg
        return getattr(_cfg, "CHRONOS_MODEL_NAME", _DEFAULT_MODEL)
    except Exception:
        return _DEFAULT_MODEL


def _get_chronos_model(model_name: Optional[str] = None):
    global _chronos_model, _is_bolt
    if _chronos_model is not None:
        return _chronos_model
    if not _CHRONOS_AVAILABLE:
        return None
    name = model_name or _active_model_name() or _DEFAULT_MODEL
    with _model_lock:
        if _chronos_model is not None:
            return _chronos_model
        try:
            import torch
            if "bolt" in name.lower():
                _chronos_model = _ChronosBoltPipeline.from_pretrained(
                    name, device_map="cpu", torch_dtype=torch.float32,
                )
                _is_bolt = True
            else:
                _chronos_model = _ChronosPipeline.from_pretrained(
                    name, device_map="cpu", torch_dtype=torch.float32,
                )
                _is_bolt = False
            return _chronos_model
        except Exception as e:
            print(f"[v16/chronos] fallo al cargar el modelo {name}: {e}")
            _chronos_model = None
            return None


# ---------------------------------------------------------------------------
# Cache en disco
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).resolve().parent / "model_outputs"
_CACHE_PATH = _CACHE_DIR / "chronos_features_cache.json"
_FEATURE_CACHE: dict[str, dict[str, float]] = {}
_CACHE_LOADED = False
_CACHE_DIRTY = False
_CACHE_LOCK = threading.Lock()


def _series_key(series: np.ndarray, horizon: int) -> str:
    h = hashlib.blake2b(series.tobytes(), digest_size=12).hexdigest()
    return f"chronos:{_active_model_name()}:h{horizon}:{h}"


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MIN_SERIES_LEN = 10
_MARGIN_CLIP = 40.0


def _empty_features() -> dict[str, float]:
    return {
        "tfm_winner_pick": 0.5,
        "tfm_margin": 0.0,
        "tfm_uncertainty": 99.0,
        "tfm_trend_slope": 0.0,
        "tfm_current_trend": 0.0,
    }


def _extract_diff_series(pbp: Iterable[dict]) -> np.ndarray:
    events = [
        (e.get("minute"), e.get("home_score"), e.get("away_score"))
        for e in pbp
        if e.get("home_score") is not None and e.get("away_score") is not None
    ]
    events.sort(key=lambda x: x[0] if x[0] is not None else 0)
    return np.asarray([float(h - a) for _, h, a in events], dtype=np.float32)


def _recent_slope(series: np.ndarray, window: int = 5) -> float:
    if len(series) < 2:
        return 0.0
    tail = series[-window:]
    if len(tail) < 2:
        return 0.0
    xs = np.arange(len(tail), dtype=np.float32)
    x_mean, y_mean = xs.mean(), tail.mean()
    num = float(((xs - x_mean) * (tail - y_mean)).sum())
    den = float(((xs - x_mean) ** 2).sum())
    return num / den if den > 0 else 0.0


def _features_from_samples(
    series: np.ndarray, samples: np.ndarray, horizon: int,
) -> dict[str, float]:
    """samples shape: (num_samples, horizon) -> 5 features tfm_*."""
    point = samples.mean(axis=0)
    p10 = np.percentile(samples, 10, axis=0)
    p90 = np.percentile(samples, 90, axis=0)
    p_end = float(np.clip(point[-1], -_MARGIN_CLIP, _MARGIN_CLIP))
    winner = 1.0 if p_end > 0.5 else (0.0 if p_end < -0.5 else 0.5)
    return {
        "tfm_winner_pick": winner,
        "tfm_margin": p_end,
        "tfm_uncertainty": float(max(p90[-1] - p10[-1], 0.0)),
        "tfm_trend_slope": float((point[-1] - point[0]) / max(horizon - 1, 1)),
        "tfm_current_trend": _recent_slope(series),
    }


def _features_from_quantiles(
    series: np.ndarray, quantiles: np.ndarray, horizon: int,
) -> dict[str, float]:
    """
    ChronosBolt devuelve 9 quantiles [0.1..0.9] shape (9, horizon).
    q10=idx0, q50=idx4, q90=idx8.
    """
    q10 = quantiles[0]
    q50 = quantiles[4]
    q90 = quantiles[8]
    p_end = float(np.clip(q50[-1], -_MARGIN_CLIP, _MARGIN_CLIP))
    winner = 1.0 if p_end > 0.5 else (0.0 if p_end < -0.5 else 0.5)
    return {
        "tfm_winner_pick": winner,
        "tfm_margin": p_end,
        "tfm_uncertainty": float(max(q90[-1] - q10[-1], 0.0)),
        "tfm_trend_slope": float((q50[-1] - q50[0]) / max(horizon - 1, 1)),
        "tfm_current_trend": _recent_slope(series),
    }


# ---------------------------------------------------------------------------
# Forecast singleton + batch nativo
# ---------------------------------------------------------------------------

def _forecast_one(series: np.ndarray, horizon: int, num_samples: int = 20) -> dict[str, float]:
    pipeline = _get_chronos_model()
    if pipeline is None:
        return _empty_features()
    try:
        import torch
        inp = torch.tensor(series, dtype=torch.float32)
        if _is_bolt:
            fc = pipeline.predict(inputs=inp, prediction_length=horizon,
                                  limit_prediction_length=False)
        else:
            fc = pipeline.predict(inputs=inp, prediction_length=horizon,
                                  num_samples=num_samples,
                                  limit_prediction_length=False)
        arr0 = fc[0].cpu().numpy() if hasattr(fc[0], "cpu") else np.asarray(fc[0])
        if _is_bolt:
            return _features_from_quantiles(series, arr0, horizon)
        return _features_from_samples(series, arr0, horizon)
    except Exception as e:
        print(f"[v16/chronos] fallo singleton: {e}")
        return _empty_features()


def _forecast_batch(
    series_list: list[np.ndarray], horizon: int, num_samples: int = 20,
) -> list[dict[str, float]]:
    pipeline = _get_chronos_model()
    if pipeline is None:
        return [_empty_features() for _ in series_list]
    try:
        import torch
        inputs = [torch.tensor(s, dtype=torch.float32) for s in series_list]
        if _is_bolt:
            fc = pipeline.predict(inputs=inputs, prediction_length=horizon,
                                  limit_prediction_length=False)
        else:
            fc = pipeline.predict(inputs=inputs, prediction_length=horizon,
                                  num_samples=num_samples,
                                  limit_prediction_length=False)
        arr = fc.cpu().numpy() if hasattr(fc, "cpu") else np.asarray(fc)
        out = []
        for i, s in enumerate(series_list):
            if _is_bolt:
                out.append(_features_from_quantiles(s, arr[i], horizon))
            else:
                out.append(_features_from_samples(s, arr[i], horizon))
        return out
    except Exception as e:
        print(f"[v16/chronos] fallo batch ({e}), caigo a iteracion")
        return [_forecast_one(s, horizon, num_samples) for s in series_list]


# ---------------------------------------------------------------------------
# API publica (mismo contrato que timesfm_features)
# ---------------------------------------------------------------------------

def extract_forecast_features(
    pbp: Iterable[dict],
    horizon: int = 20,
    use_timesfm: bool = True,
) -> dict[str, float]:
    global _CACHE_DIRTY
    series = _extract_diff_series(pbp)
    if len(series) < _MIN_SERIES_LEN:
        return _empty_features()
    _load_cache()
    key = _series_key(series, horizon)
    if key in _FEATURE_CACHE:
        return dict(_FEATURE_CACHE[key])
    out = _forecast_one(series, horizon)
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
    global _CACHE_DIRTY
    _load_cache()

    results: list[dict[str, float] | None] = [None] * len(pbps)
    pending: list[tuple[int, np.ndarray, str]] = []

    for idx, pbp in enumerate(pbps):
        series = _extract_diff_series(pbp)
        if len(series) < _MIN_SERIES_LEN:
            results[idx] = _empty_features()
            continue
        key = _series_key(series, horizon)
        if key in _FEATURE_CACHE:
            results[idx] = dict(_FEATURE_CACHE[key])
            continue
        pending.append((idx, series, key))

    if not pending:
        return [r if r is not None else _empty_features() for r in results]

    iterator: Iterable[int] = range(0, len(pending), batch_size)
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iterator), desc=f"chronos batch ({len(pending)} series)")
        except Exception:
            pass

    for start in iterator:
        chunk = pending[start: start + batch_size]
        series_list = [s for _, s, _ in chunk]
        outs = _forecast_batch(series_list, horizon)
        for (idx, _s, key), out in zip(chunk, outs):
            _FEATURE_CACHE[key] = dict(out)
            results[idx] = out

    _CACHE_DIRTY = True
    return [r if r is not None else _empty_features() for r in results]


def diagnostic() -> dict[str, str]:
    _load_cache()
    return {
        "chronos_available": str(_CHRONOS_AVAILABLE),
        "active_backend": "chronos_bolt" if (_CHRONOS_AVAILABLE and _is_bolt)
                          else ("chronos" if _CHRONOS_AVAILABLE else "unavailable"),
        "default_model": _active_model_name(),
        "cache_size": str(len(_FEATURE_CACHE)),
    }
