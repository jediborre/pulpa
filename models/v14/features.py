"""
features.py — Feature engineering para V14.

Diferencias respecto a V13:
1. _trajectory_features(): features directas del PBP (score real, no graph_points)
   - lead_changes, times_tied, largest_lead, current_run, last_5_events
2. _timesfm_features(): integración TimesFM (Fase 3 — deshabilitado hasta activar)
3. SampleLike Protocol: contrato explícito entre dataset.py e infer_match_v14.py
   → elimina el bug 'dict object has no attribute match_id' de V13

REGLA de anti-leakage:
  Solo usar datos disponibles ANTES de cutoff_minute.
  Para Q3: PBP de Q1+Q2 únicamente.
  Para Q4: PBP de Q1+Q2+Q3 únicamente.
  NUNCA usar scores del cuarto que se está prediciendo.
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any
from dataclasses import dataclass

if TYPE_CHECKING:
    # Solo para type checking, no importa en runtime
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Protocolo de sample (contrato entre dataset.py e infer_match_v14.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveSample:
    """
    Adaptador para inferencia live. Cumple el mismo contrato que TrainingSample
    de dataset.py sin necesitar importar esa clase.

    Ejemplo de uso en infer_match_v14.py:
        sample = LiveSample(
            match_id=match_id,
            target='q3',
            snapshot_minute=22,
            features={'halftime_diff': 5, 'halftime_total': 78},
        )
        feats = build_features_for_sample(sample, gp_dict, pbp_dict)
    """
    match_id: str
    target: str           # 'q3' | 'q4'
    snapshot_minute: int
    features: dict        # features pre-calculadas (halftime_diff, etc.)


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def build_features_for_sample(
    sample: LiveSample,
    graph_points: dict[str, list],
    pbp_events: dict[str, list],
) -> dict[str, Any]:
    """
    Construye feature dict para una muestra.

    Args:
        sample: objeto con .match_id, .target, .snapshot_minute, .features
        graph_points: dict {match_id: [{"minute": int, "value": int}, ...]}
        pbp_events: dict {match_id: [{"quarter": str, "home_score": int,
                                       "away_score": int, "minute": float,
                                       "points": int, "team": str}, ...]}

    Returns:
        dict de features listo para pasar al vectorizador
    """
    mid = sample.match_id
    target = sample.target
    cutoff = sample.snapshot_minute

    features: dict[str, Any] = {}

    # Metadata
    features['target'] = target
    features['snapshot_minute'] = cutoff

    # Graph point features (igual que V13 — proxy de momentum de Sofascore)
    gp = graph_points.get(mid, [])
    gp_upto_cutoff = [p for p in gp if p['minute'] <= cutoff]
    features.update(_graph_features(gp_upto_cutoff, cutoff))

    # PBP: solo cuartos anteriores al target (anti-leakage)
    pbp = pbp_events.get(mid, [])
    pbp_upto = _pbp_upto_quarter(pbp, target)
    features.update(_pbp_features(pbp_upto))

    # Trajectory features — NUEVO en V14
    # Usa home_score/away_score real del PBP, más informativo que graph_points
    features.update(_trajectory_features(pbp_upto, cutoff))

    # Quarter score features
    features.update(_score_features(sample))

    # TimesFM features — FASE 3: deshabilitado hasta activar config.TIMESFM_ENABLED
    # features.update(_timesfm_features(pbp_upto, target, cutoff))

    return features


# ─────────────────────────────────────────────────────────────────────────────
# Features heredadas de V13 (sin cambios)
# ─────────────────────────────────────────────────────────────────────────────

def _graph_features(gp: list[dict], cutoff: int) -> dict[str, Any]:
    """Features del graph_points de Sofascore (curva de presión/momentum)."""
    features: dict[str, Any] = {}
    features['gp_count'] = len(gp)

    if not gp:
        features['gp_diff'] = 0
        features['gp_slope_3m'] = 0
        features['gp_slope_5m'] = 0
        features['gp_acceleration'] = 0
        features['gp_peak'] = 0
        features['gp_valley'] = 0
        features['gp_amplitude'] = 0
        features['gp_swings'] = 0
        return features

    latest = gp[-1]
    features['gp_diff'] = latest['value']

    if len(gp) >= 3:
        recent_3 = gp[-3:]
        features['gp_slope_3m'] = (recent_3[-1]['value'] - recent_3[0]['value']) / 3.0
    else:
        features['gp_slope_3m'] = 0

    if len(gp) >= 5:
        recent_5 = gp[-5:]
        features['gp_slope_5m'] = (recent_5[-1]['value'] - recent_5[0]['value']) / 5.0
    else:
        features['gp_slope_5m'] = features['gp_slope_3m']

    if len(gp) >= 6:
        slope_1 = (gp[-3]['value'] - gp[-6]['value']) / 3.0
        slope_2 = (gp[-1]['value'] - gp[-3]['value']) / 3.0
        features['gp_acceleration'] = slope_2 - slope_1
    else:
        features['gp_acceleration'] = 0

    values = [p['value'] for p in gp]
    features['gp_peak'] = max(values)
    features['gp_valley'] = min(values)
    features['gp_amplitude'] = features['gp_peak'] - features['gp_valley']

    swings = sum(
        1 for i in range(2, len(values))
        if (values[i] - values[i-1]) * (values[i-1] - values[i-2]) < 0
    )
    features['gp_swings'] = swings

    return features


def _pbp_upto_quarter(pbp: list[dict], target: str) -> list[dict]:
    """Filtra PBP a los cuartos anteriores al target (anti-leakage)."""
    if target == 'q3':
        return [e for e in pbp if e.get('quarter') in ('Q1', 'Q2')]
    else:  # q4
        return [e for e in pbp if e.get('quarter') in ('Q1', 'Q2', 'Q3')]


def _pbp_features(pbp: list[dict]) -> dict[str, Any]:
    """Features resumen de PBP (igual que V13)."""
    features: dict[str, Any] = {}
    features['pbp_count'] = len(pbp)

    if not pbp:
        features['pbp_pts_per_event'] = 0
        features['pbp_home_pts'] = 0
        features['pbp_away_pts'] = 0
        features['pbp_home_3pt'] = 0
        features['pbp_away_3pt'] = 0
        return features

    total_pts = sum(e.get('points', 0) for e in pbp if (e.get('points') or 0) > 0)
    features['pbp_pts_per_event'] = total_pts / len(pbp)

    home_pts = sum(e.get('points', 0) for e in pbp if e.get('team') == 'home' and (e.get('points') or 0) > 0)
    away_pts = sum(e.get('points', 0) for e in pbp if e.get('team') == 'away' and (e.get('points') or 0) > 0)
    features['pbp_home_pts'] = home_pts
    features['pbp_away_pts'] = away_pts

    home_3pt = sum(1 for e in pbp if e.get('team') == 'home' and e.get('points') == 3)
    away_3pt = sum(1 for e in pbp if e.get('team') == 'away' and e.get('points') == 3)
    features['pbp_home_3pt'] = home_3pt
    features['pbp_away_3pt'] = away_3pt

    return features


def _score_features(sample: LiveSample) -> dict[str, Any]:
    """Features de scores de cuartos anteriores (sin leakage)."""
    features: dict[str, Any] = {}
    base = sample.features or {}
    features['halftime_diff'] = base.get('halftime_diff', 0)
    features['halftime_total'] = base.get('halftime_total', 0)
    features['q1_diff'] = base.get('q1_diff', 0)
    features['q2_diff'] = base.get('q2_diff', 0)
    return features


# ─────────────────────────────────────────────────────────────────────────────
# NUEVO en V14: trajectory features del score real
# ─────────────────────────────────────────────────────────────────────────────

def _trajectory_features(pbp: list[dict], cutoff_minute: float) -> dict[str, Any]:
    """
    Features de trayectoria del score real, directas del PBP.

    Por qué es mejor que solo usar graph_points:
    - Usa el score marcador real (home_score, away_score), no el índice de
      presión/momentum de Sofascore
    - Captura eventos discretos importantes: lead_changes, times_tied,
      comeback, racha activa
    - Un equipo que va -10 y remonta a 0 tiene el mismo gp_slope que uno
      que va +10 y baja a 0, pero su probabilidad de ganar Q3 es muy distinta

    Todos los campos se rellenan con 0 si no hay datos (nunca NaN).
    """
    features: dict[str, Any] = {}

    # Filtrar solo eventos con score válido
    scored_events = [
        e for e in pbp
        if e.get('home_score') is not None and e.get('away_score') is not None
    ]

    if not scored_events:
        features['traj_lead_changes'] = 0
        features['traj_times_tied'] = 0
        features['traj_largest_lead_home'] = 0
        features['traj_largest_lead_away'] = 0
        features['traj_current_run_home'] = 0
        features['traj_current_run_away'] = 0
        features['traj_last5_home_pts'] = 0
        features['traj_last5_away_pts'] = 0
        features['traj_last5_diff'] = 0
        features['traj_score_diff_end'] = 0
        features['traj_total_pts_scored'] = 0
        features['traj_comeback_flag'] = 0
        return features

    diffs = [e['home_score'] - e['away_score'] for e in scored_events]

    # Cambios de lider
    lead_changes = sum(
        1 for i in range(1, len(diffs))
        if (diffs[i] > 0) != (diffs[i-1] > 0) and diffs[i-1] != 0
    )
    features['traj_lead_changes'] = lead_changes

    # Veces empatados
    times_tied = sum(1 for d in diffs if d == 0)
    features['traj_times_tied'] = times_tied

    # Mayor ventaja de cada equipo
    features['traj_largest_lead_home'] = max(max(diffs), 0)
    features['traj_largest_lead_away'] = abs(min(min(diffs), 0))

    # Diferencia actual
    features['traj_score_diff_end'] = diffs[-1]

    # Total de puntos anotados
    last = scored_events[-1]
    features['traj_total_pts_scored'] = (last['home_score'] or 0) + (last['away_score'] or 0)

    # Racha activa al final: cuántos puntos consecutivos anotó un equipo sin que
    # el otro respondiera
    current_run_home = 0
    current_run_away = 0
    for e in reversed(scored_events):
        pts = e.get('points', 0) or 0
        team = e.get('team', '')
        if pts == 0:
            continue
        if team == 'home':
            if current_run_away > 0:
                break
            current_run_home += pts
        elif team == 'away':
            if current_run_home > 0:
                break
            current_run_away += pts
    features['traj_current_run_home'] = current_run_home
    features['traj_current_run_away'] = current_run_away

    # Puntos de los últimos 5 eventos scored
    last5 = scored_events[-5:]
    last5_home = sum(e.get('points', 0) or 0 for e in last5 if e.get('team') == 'home')
    last5_away = sum(e.get('points', 0) or 0 for e in last5 if e.get('team') == 'away')
    features['traj_last5_home_pts'] = last5_home
    features['traj_last5_away_pts'] = last5_away
    features['traj_last5_diff'] = last5_home - last5_away

    # Flag de comeback: equipo que estaba perdiendo en la primera mitad del
    # PBP pero ahora va ganando (o empatado).
    # Detecta el patrón "comeback" vs. "pérdida de ventaja"
    if len(diffs) >= 4:
        mid_diff = diffs[len(diffs) // 2]
        end_diff = diffs[-1]
        comeback = int(
            (mid_diff < -3 and end_diff >= 0) or  # home comeback
            (mid_diff > 3 and end_diff <= 0)       # away comeback
        )
    else:
        comeback = 0
    features['traj_comeback_flag'] = comeback

    return features


# ─────────────────────────────────────────────────────────────────────────────
# FASE 3 — TimesFM features (esqueleto, no implementar hasta Fase 3)
# ─────────────────────────────────────────────────────────────────────────────

def _timesfm_features(
    pbp: list[dict],
    target: str,
    cutoff_minute: float,
    timesfm_model=None,  # instancia de TimesFM cargada por el caller
) -> dict[str, Any]:
    """
    Genera 3 features usando TimesFM como regresor de la serie de scores.

    Flujo:
    1. Extraer serie home_score y away_score del PBP hasta cutoff
    2. Pasar al modelo: model.forecast(horizon=H, inputs=[home_series, away_series])
    3. Derivar: winner_pick, margin, uncertainty desde el forecast

    Args:
        pbp: eventos PBP filtrados (solo cuartos anteriores al target)
        target: 'q3' | 'q4'
        cutoff_minute: minuto de corte
        timesfm_model: instancia del modelo ya cargada (None desactiva TimesFM)

    Returns:
        dict con tfm_winner_pick, tfm_margin, tfm_uncertainty (todos float/int)
        Si timesfm_model es None → devuelve 0.0 para todas (fallback seguro)

    NOTA DE IMPLEMENTACIÓN (Fase 3):
    - El model se carga UNA vez al inicio de V14Engine.__init__()
    - Horizon: ~15-20 eventos (no en minutos; los eventos son más regulares)
    - Usar `use_continuous_quantile_head=True` para obtener intervalos p10-p90
    - Serie: `home_score - away_score` por evento (la diff, no scores absolutos)
      → más estable que 2 series separadas para el foundation model
    - Si len(serie) < 10: no usar TimesFM (insuficiente contexto) → retornar 0.0
    """
    # Fallback: TimesFM deshabilitado o no disponible
    if timesfm_model is None:
        return {
            'tfm_winner_pick': 0.0,  # 0 = sin señal (ni 0=away ni 1=home)
            'tfm_margin': 0.0,
            'tfm_uncertainty': 99.0,  # alta incertidumbre → gate lo ignora
        }

    # TODO (Fase 3): implementar extracción de serie y llamada al modelo
    # Esqueleto:
    #
    # scored_events = [e for e in pbp if e.get('home_score') is not None]
    # if len(scored_events) < 10:
    #     return _timesfm_fallback()
    #
    # diff_series = [e['home_score'] - e['away_score'] for e in scored_events]
    # horizon = 20  # eventos esperados hasta fin del cuarto
    #
    # point_forecast, quantile_forecast = timesfm_model.forecast(
    #     horizon=horizon,
    #     inputs=[np.array(diff_series, dtype=np.float32)],
    # )
    # pred_diff_end = float(point_forecast[0, -1])
    # p10 = float(quantile_forecast[0, -1, 0])
    # p90 = float(quantile_forecast[0, -1, -1])
    # uncertainty = p90 - p10
    #
    # return {
    #     'tfm_winner_pick': 1.0 if pred_diff_end > 0 else 0.0,
    #     'tfm_margin': pred_diff_end,
    #     'tfm_uncertainty': uncertainty,
    # }
    raise NotImplementedError("TimesFM features no implementadas (Fase 3)")
