"""
features.py - Feature engineering de V17.

DiseÃ±o:
- Contrato unico build_features_for_sample(sample, graph_points, pbp_events,
    league_stats, pace_thresholds) -> dict[str, Any]
- Todas las features son deterministas y respetan cutoff (anti-leakage).
- Cada modulo de features es una funcion pequena que devuelve un dict,
  facil de habilitar/deshabilitar para ablation studies.
- Las features que vimos que son ruidosas o redundantes en v13 fueron
  eliminadas. La lista de descartadas esta al final del archivo.

Grupos de features que se generan:
  G1  - snapshot / score de cuartos anteriores (score.py equivalente)
  G2  - graph_points resumen (slope, aceleracion, amplitud, swings)
  G3  - trayectoria real del score (lead changes, rachas, comebacks)
  G4  - play-by-play resumen (volumen, 3pt rate, pts_per_event)
  G5  - pace y bucket (como feature, no como segmentacion)
  G6  - stats de liga walk-forward (sample size, totales promedio)
  G7  - minuto y meta (snapshot_minute, tiempo restante)
  G8  - fusion legacy (pressure/comeback, clutch, Monte Carlo simple)
  G9  - forecast opcional (apagado por defecto)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Protocol

import numpy as np

from training.v17 import config, dataset as ds


def _get_forecast_extract_fn():
    """
    Dispatch por FORECAST_BACKEND: devuelve la funcion extract_forecast_features
    del modulo apropiado. Import lazy para evitar cargar chronos/timesfm cuando
    no se usan.
    """
    backend = getattr(config, "FORECAST_BACKEND", "timesfm")
    if backend == "chronos":
        from training.v17.chronos_features import extract_forecast_features as _fn
        return _fn
    from training.v17.timesfm_features import extract_forecast_features as _fn
    return _fn


# ============================================================================
# Contrato para inferencia live
# ============================================================================

@dataclass
class LiveSample:
    """Adaptador para inferencia; cumple interfaz Sample sin importar su modulo."""
    match_id: str
    target: str
    snapshot_minute: int
    date: str
    league: str
    gender: str
    features: dict[str, Any]
    pace_total_prior: float = 0.0
    target_winner: int | None = None
    target_home_pts: int | None = None
    target_away_pts: int | None = None
    target_total_pts: int | None = None


class _SampleLike(Protocol):
    match_id: str
    target: str
    snapshot_minute: int
    league: str
    features: dict[str, Any]
    pace_total_prior: float


# ============================================================================
# Entrypoint
# ============================================================================

def build_features_for_sample(
    sample: _SampleLike,
    graph_points: dict[str, list[dict]],
    pbp_events: dict[str, list[dict]],
    league_stats: dict[str, float] | None = None,
    pace_thresholds: dict[str, float] | None = None,
    tfm_cache: dict[tuple, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """
    Construye el dict completo de features. Orden fijo: asi el DictVectorizer
    mantiene consistencia entre training e inference.
    """
    cutoff = float(sample.snapshot_minute)

    feats: dict[str, Any] = {}
    feats["meta_snapshot_minute"] = cutoff
    feats["meta_target_is_q4"] = 1 if sample.target == "q4" else 0
    feats["meta_minutes_to_quarter_end"] = _minutes_to_quarter_end(sample.target, cutoff)

    # G1: score de cuartos anteriores (ya vienen pre-computados en sample.features)
    feats.update(_score_features(sample))

    # G2+G3: graph_points + trayectoria PBP
    gp = [p for p in graph_points.get(sample.match_id, []) if p.get("minute", 0) <= cutoff]
    feats.update(_graph_features(gp))

    pbp_all = pbp_events.get(sample.match_id, [])
    pbp_before = _pbp_before_target(pbp_all, sample.target, cutoff)

    feats.update(_pbp_summary_features(pbp_before))
    feats.update(_trajectory_features(pbp_before))
    feats.update(_legacy_hybrid_features(sample, pbp_before))

    # G5: pace como feature numerica y bucket como one-hot
    feats.update(_pace_features(sample, pace_thresholds))

    # G6: liga
    feats.update(_league_features(league_stats))

    # G9 (v17): forecast opcional. Se deja apagado por defecto para que el
    # modelo priorice senales live mas robustas y menos fragiles.
    if getattr(config, "TIMESFM_ENABLED", False):
        tfm_feats = None
        if tfm_cache is not None:
            tfm_key = (sample.match_id, sample.target, int(sample.snapshot_minute))
            tfm_feats = tfm_cache.get(tfm_key)
        if tfm_feats is None:
            _extract = _get_forecast_extract_fn()
            tfm_feats = _extract(
                pbp_before,
                horizon=config.TIMESFM_FORECAST_HORIZON,
                use_timesfm=True,
            )
        feats.update(tfm_feats)

    # Feature pruning (mejora #1 del ROADMAP). Se eliminan al final para
    # preservar intermedios que alimentan otras features (ej. score_halftime_*
    # participa del calculo derivado y tiene que estar hasta aqui).
    if getattr(config, "FEATURE_PRUNING_ENABLED", False):
        for _dead in getattr(config, "FEATURE_BLACKLIST", ()):
            feats.pop(_dead, None)

    return feats


# ============================================================================
# G1: score features (cuartos anteriores)
# ============================================================================

def _score_features(s: _SampleLike) -> dict[str, float]:
    base = s.features or {}
    out = {
        "score_halftime_diff": float(base.get("halftime_diff", 0)),
        "score_halftime_total": float(base.get("halftime_total", 0)),
        "score_q1_diff": float(base.get("q1_diff", 0)),
        "score_q2_diff": float(base.get("q2_diff", 0)),
        "score_q1_total": float(base.get("q1_total", 0)),
        "score_q2_total": float(base.get("q2_total", 0)),
    }
    # Ratios y diferenciales derivados
    ht = out["score_halftime_total"] or 1e-6
    out["score_q1_share"] = out["score_q1_total"] / ht
    out["score_halftime_diff_ratio"] = out["score_halftime_diff"] / ht
    # Solo para Q4: q3 end
    if s.target == "q4":
        out["score_q3_diff"] = float(base.get("q3_diff", 0))
        out["score_q3_total"] = float(base.get("q3_total", 0))
        full_total = out["score_halftime_total"] + out["score_q3_total"]
        out["score_cumulative_total"] = full_total
        out["score_cumulative_diff"] = out["score_halftime_diff"] + out["score_q3_diff"]
        # Momentum entre mitad y Q3: si Q3 dif subio vs halftime dif
        out["score_q3_vs_ht_momentum"] = out["score_q3_diff"] - out["score_halftime_diff"]
    else:
        # placeholders 0 para mantener shape
        out["score_q3_diff"] = 0.0
        out["score_q3_total"] = 0.0
        out["score_cumulative_total"] = out["score_halftime_total"]
        out["score_cumulative_diff"] = out["score_halftime_diff"]
        out["score_q3_vs_ht_momentum"] = 0.0
    return out


# ============================================================================
# G2: graph_points features
# ============================================================================

def _graph_features(gp: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {"gp_count": float(len(gp))}
    if not gp:
        for k in (
            "gp_latest_diff", "gp_slope_3m", "gp_slope_5m", "gp_acceleration",
            "gp_peak", "gp_valley", "gp_amplitude", "gp_swings",
            "gp_sign_changes", "gp_stddev", "gp_last_sign",
        ):
            out[k] = 0.0
        return out
    vals = [float(p["value"]) for p in gp]
    out["gp_latest_diff"] = vals[-1]
    out["gp_slope_3m"] = (vals[-1] - vals[-3]) / 3.0 if len(vals) >= 3 else 0.0
    out["gp_slope_5m"] = (vals[-1] - vals[-5]) / 5.0 if len(vals) >= 5 else out["gp_slope_3m"]
    if len(vals) >= 6:
        s1 = (vals[-3] - vals[-6]) / 3.0
        s2 = (vals[-1] - vals[-3]) / 3.0
        out["gp_acceleration"] = s2 - s1
    else:
        out["gp_acceleration"] = 0.0
    out["gp_peak"] = float(max(vals))
    out["gp_valley"] = float(min(vals))
    out["gp_amplitude"] = out["gp_peak"] - out["gp_valley"]
    # Swings: cambios de direccion
    swings = sum(
        1 for i in range(2, len(vals))
        if (vals[i] - vals[i - 1]) * (vals[i - 1] - vals[i - 2]) < 0
    )
    out["gp_swings"] = float(swings)
    # Cambios de signo (cambio de lider)
    sign_changes = sum(
        1 for i in range(1, len(vals))
        if vals[i - 1] != 0 and (vals[i] > 0) != (vals[i - 1] > 0)
    )
    out["gp_sign_changes"] = float(sign_changes)
    out["gp_stddev"] = float(np.std(vals)) if len(vals) > 1 else 0.0
    out["gp_last_sign"] = float(np.sign(vals[-1]))
    return out


# ============================================================================
# G3: trayectoria real del score (from PBP)
# ============================================================================

def pbp_before_target(pbp: list[dict], target: str, cutoff_minute: float) -> list[dict]:
    """Filtra PBP respetando anti-leakage: solo eventos anteriores al cuarto
    objetivo Y con minuto estimado <= cutoff_minute. Publico para que train.py
    pueda re-usarlo al precomputar features TimesFM en batch."""
    allowed_quarters = ("Q1", "Q2") if target == "q3" else ("Q1", "Q2", "Q3")
    return [
        e for e in pbp
        if e.get("quarter") in allowed_quarters and e.get("minute", 0) <= cutoff_minute
    ]


# Alias privado para mantener compat con codigo viejo que importaba el nombre _pbp_*
_pbp_before_target = pbp_before_target


def _trajectory_features(pbp: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    scored = [
        e for e in pbp
        if e.get("home_score") is not None and e.get("away_score") is not None
    ]
    if not scored:
        for k in (
            "traj_lead_changes", "traj_times_tied", "traj_largest_lead_home",
            "traj_largest_lead_away", "traj_score_diff_end",
            "traj_current_run_home", "traj_current_run_away",
            "traj_last5_home_pts", "traj_last5_away_pts", "traj_last5_diff",
            "traj_last10_diff", "traj_comeback_flag", "traj_momentum_idx",
        ):
            out[k] = 0.0
        return out
    diffs = [e["home_score"] - e["away_score"] for e in scored]
    out["traj_lead_changes"] = float(sum(
        1 for i in range(1, len(diffs))
        if (diffs[i] > 0) != (diffs[i - 1] > 0) and diffs[i - 1] != 0
    ))
    out["traj_times_tied"] = float(sum(1 for d in diffs if d == 0))
    out["traj_largest_lead_home"] = float(max(max(diffs), 0))
    out["traj_largest_lead_away"] = float(abs(min(min(diffs), 0)))
    out["traj_score_diff_end"] = float(diffs[-1])

    run_home = 0
    run_away = 0
    for e in reversed(scored):
        pts = e.get("points", 0) or 0
        if pts == 0:
            continue
        team = e.get("team", "")
        if team == "home":
            if run_away > 0:
                break
            run_home += pts
        elif team == "away":
            if run_home > 0:
                break
            run_away += pts
    out["traj_current_run_home"] = float(run_home)
    out["traj_current_run_away"] = float(run_away)

    last5 = scored[-5:]
    last5_home = sum((e.get("points", 0) or 0) for e in last5 if e.get("team") == "home")
    last5_away = sum((e.get("points", 0) or 0) for e in last5 if e.get("team") == "away")
    out["traj_last5_home_pts"] = float(last5_home)
    out["traj_last5_away_pts"] = float(last5_away)
    out["traj_last5_diff"] = float(last5_home - last5_away)

    # Diferencia de diffs entre los ultimos 10 eventos y los 10 anteriores
    if len(diffs) >= 20:
        out["traj_last10_diff"] = float(np.mean(diffs[-10:]) - np.mean(diffs[-20:-10]))
    else:
        out["traj_last10_diff"] = 0.0

    # Comeback detection sin ruido del inicio
    if len(diffs) >= 6:
        mid = diffs[len(diffs) // 2]
        end = diffs[-1]
        out["traj_comeback_flag"] = float(
            (mid < -4 and end >= 0) or (mid > 4 and end <= 0)
        )
    else:
        out["traj_comeback_flag"] = 0.0

    # Indice de momentum simple: proporcion de eventos recientes anotados por el
    # lider actual.
    n_recent = min(10, len(scored))
    recent = scored[-n_recent:]
    current_leader = "home" if diffs[-1] > 0 else ("away" if diffs[-1] < 0 else None)
    if current_leader is None:
        out["traj_momentum_idx"] = 0.0
    else:
        leader_pts = sum(
            (e.get("points", 0) or 0) for e in recent if e.get("team") == current_leader
        )
        other_pts = sum(
            (e.get("points", 0) or 0) for e in recent
            if e.get("team") in ("home", "away") and e.get("team") != current_leader
        )
        total = leader_pts + other_pts
        out["traj_momentum_idx"] = leader_pts / total if total > 0 else 0.0
    return out


# ============================================================================
# G4: PBP summary features
# ============================================================================

def _pbp_summary_features(pbp: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {"pbp_count": float(len(pbp))}
    if not pbp:
        for k in (
            "pbp_pts_per_event", "pbp_home_pts", "pbp_away_pts",
            "pbp_home_3pt_rate", "pbp_away_3pt_rate",
            "pbp_scoring_density",
        ):
            out[k] = 0.0
        return out
    scoring = [e for e in pbp if (e.get("points") or 0) > 0]
    total_pts = sum((e.get("points", 0) or 0) for e in scoring)
    out["pbp_pts_per_event"] = total_pts / max(len(pbp), 1)
    home_pts = sum((e.get("points", 0) or 0) for e in scoring if e.get("team") == "home")
    away_pts = sum((e.get("points", 0) or 0) for e in scoring if e.get("team") == "away")
    out["pbp_home_pts"] = float(home_pts)
    out["pbp_away_pts"] = float(away_pts)
    # 3pt rate: proporcion de anotaciones que son triples
    home_scored = [e for e in scoring if e.get("team") == "home"]
    away_scored = [e for e in scoring if e.get("team") == "away"]
    h3 = sum(1 for e in home_scored if (e.get("points") or 0) == 3)
    a3 = sum(1 for e in away_scored if (e.get("points") or 0) == 3)
    out["pbp_home_3pt_rate"] = h3 / len(home_scored) if home_scored else 0.0
    out["pbp_away_3pt_rate"] = a3 / len(away_scored) if away_scored else 0.0
    out["pbp_scoring_density"] = len(scoring) / max(len(pbp), 1)
    return out


def _legacy_hybrid_features(s: _SampleLike, pbp: list[dict]) -> dict[str, float]:
    base = s.features or {}
    if s.target == "q3":
        halftime_total = float(base.get("halftime_total", 0))
        halftime_diff = float(base.get("halftime_diff", 0))
        score_home = (halftime_total + halftime_diff) / 2.0
        score_away = (halftime_total - halftime_diff) / 2.0
        elapsed_minutes = 24.0
    else:
        halftime_total = float(base.get("halftime_total", 0))
        halftime_diff = float(base.get("halftime_diff", 0))
        q3_total = float(base.get("q3_total", 0))
        q3_diff = float(base.get("q3_diff", 0))
        score_home = (halftime_total + halftime_diff + q3_total + q3_diff) / 2.0
        score_away = (halftime_total - halftime_diff + q3_total - q3_diff) / 2.0
        elapsed_minutes = 36.0

    home_plays = 0
    away_plays = 0
    home_pts = 0.0
    away_pts = 0.0
    home_3pt = 0
    away_3pt = 0
    scoring_events: list[tuple[str, float]] = []
    for e in pbp:
        pts = float(e.get("points", 0) or 0)
        team = e.get("team")
        if team == "home":
            home_plays += 1
            home_pts += pts
            if pts == 3:
                home_3pt += 1
        elif team == "away":
            away_plays += 1
            away_pts += pts
            if pts == 3:
                away_3pt += 1
        if team in ("home", "away") and pts > 0:
            scoring_events.append((team, pts))

    out: dict[str, float] = {}
    if getattr(config, "ENABLE_LEGACY_PRESSURE_FEATURES", False):
        out.update(_score_pressure_features(
            score_home=score_home,
            score_away=score_away,
            pbp_home_plays=home_plays,
            pbp_away_plays=away_plays,
            pbp_home_3pt=home_3pt,
            pbp_away_3pt=away_3pt,
            elapsed_minutes=elapsed_minutes,
            minutes_left=12.0,
        ))
    if getattr(config, "ENABLE_LEGACY_CLUTCH_FEATURES", False):
        last10 = scoring_events[-10:]
        out["legacy_last10_home_pts"] = float(sum(pts for team, pts in last10 if team == "home"))
        out["legacy_last10_away_pts"] = float(sum(pts for team, pts in last10 if team == "away"))
        out["legacy_last10_diff"] = out["legacy_last10_home_pts"] - out["legacy_last10_away_pts"]
        out["legacy_current_run_home"] = float(_current_scoring_run(scoring_events, "home"))
        out["legacy_current_run_away"] = float(_current_scoring_run(scoring_events, "away"))
        out["legacy_max_run_home"] = float(_max_scoring_run(scoring_events, "home"))
        out["legacy_max_run_away"] = float(_max_scoring_run(scoring_events, "away"))
        out["legacy_run_diff"] = out["legacy_max_run_home"] - out["legacy_max_run_away"]
    if getattr(config, "ENABLE_LEGACY_MONTE_CARLO_FEATURES", False):
        out.update(_monte_carlo_win_prob(
            score_home=score_home,
            score_away=score_away,
            pbp_home_plays=home_plays,
            pbp_away_plays=away_plays,
            pbp_home_pts=home_pts,
            pbp_away_pts=away_pts,
            elapsed_minutes=elapsed_minutes,
            minutes_left=12.0,
            num_sims=getattr(config, "LEGACY_MONTE_CARLO_SIMS", 750),
        ))
    return out


# ============================================================================
# G5: pace features
# ============================================================================

def _pace_features(
    sample: _SampleLike,
    pace_thresholds: dict[str, float] | None,
) -> dict[str, float]:
    thresholds = pace_thresholds or {}
    pace = float(sample.pace_total_prior)
    out = {
        "pace_total_prior": pace,
        "pace_ratio_vs_median": pace / (
            thresholds.get("q3_high_lower" if sample.target == "q3" else "q4_high_lower", 85) or 1
        ),
    }
    bucket = ds.pace_bucket_for(sample.target, pace, thresholds) if thresholds else "medium"
    out["pace_bucket_low"] = 1.0 if bucket == "low" else 0.0
    out["pace_bucket_medium"] = 1.0 if bucket == "medium" else 0.0
    out["pace_bucket_high"] = 1.0 if bucket == "high" else 0.0
    return out


# ============================================================================
# G6: league walk-forward features
# ============================================================================

def _league_features(stats: dict[str, float] | None) -> dict[str, float]:
    s = stats or {}
    return {
        "league_samples": float(s.get("league_samples", 0)),
        "league_ht_total_mean": float(s.get("league_ht_total_mean", 0)),
        "league_ht_total_std": float(s.get("league_ht_total_std", 0)),
        "league_home_advantage_mean": float(s.get("league_home_advantage_mean", 0)),
        "league_q3_total_mean": float(s.get("league_q3_total_mean", 0)),
        "league_q4_total_mean": float(s.get("league_q4_total_mean", 0)),
    }


# ============================================================================
# Utilidades
# ============================================================================

def _minutes_to_quarter_end(target: str, cutoff: float) -> float:
    end = 20.0 if target == "q3" else 30.0
    return max(end - cutoff, 0.0)


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _score_pressure_features(
    *,
    score_home: float,
    score_away: float,
    pbp_home_plays: int,
    pbp_away_plays: int,
    pbp_home_3pt: int,
    pbp_away_3pt: int,
    elapsed_minutes: float,
    minutes_left: float,
) -> dict[str, float]:
    diff = score_home - score_away
    abs_diff = abs(diff)
    if diff > 0:
        trailing_score = score_away
        trailing_plays = pbp_away_plays
        trailing_3pt = pbp_away_3pt
        leading_score = score_home
        trailing_is_home = 0.0
        trailing_is_away = 1.0
    elif diff < 0:
        trailing_score = score_home
        trailing_plays = pbp_home_plays
        trailing_3pt = pbp_home_3pt
        leading_score = score_away
        trailing_is_home = 1.0
        trailing_is_away = 0.0
    else:
        trailing_score = max(score_home, score_away)
        trailing_plays = max(pbp_home_plays, pbp_away_plays)
        trailing_3pt = max(pbp_home_3pt, pbp_away_3pt)
        leading_score = trailing_score
        trailing_is_home = 0.0
        trailing_is_away = 0.0
    total_plays = pbp_home_plays + pbp_away_plays
    trailing_points_per_min = _safe_rate(trailing_score, elapsed_minutes)
    leading_points_per_min = _safe_rate(leading_score, elapsed_minutes)
    points_to_tie = abs_diff
    points_to_lead = abs_diff + (0.0 if diff == 0 else 1.0)
    return {
        "legacy_global_diff": float(diff),
        "legacy_global_abs_diff": float(abs_diff),
        "legacy_trailing_is_home": trailing_is_home,
        "legacy_trailing_is_away": trailing_is_away,
        "legacy_required_ppm_tie": _safe_rate(points_to_tie, minutes_left),
        "legacy_required_ppm_lead": _safe_rate(points_to_lead, minutes_left),
        "legacy_trailing_points_per_min": trailing_points_per_min,
        "legacy_leading_points_per_min": leading_points_per_min,
        "legacy_trailing_play_share": _safe_rate(trailing_plays, total_plays),
        "legacy_trailing_plays_per_min": _safe_rate(trailing_plays, elapsed_minutes),
        "legacy_pressure_ratio_tie": _safe_rate(_safe_rate(points_to_tie, minutes_left), trailing_points_per_min),
        "legacy_pressure_ratio_lead": _safe_rate(_safe_rate(points_to_lead, minutes_left), trailing_points_per_min),
        "legacy_scoring_gap_per_min": trailing_points_per_min - leading_points_per_min,
        "legacy_urgency_index": _safe_rate(points_to_lead, minutes_left) * (1.0 + max(0.0, -(trailing_points_per_min - leading_points_per_min))),
        "legacy_trailing_3pt_rate": _safe_rate(trailing_3pt, trailing_plays),
    }


def _monte_carlo_win_prob(
    *,
    score_home: float,
    score_away: float,
    pbp_home_plays: int,
    pbp_away_plays: int,
    pbp_home_pts: float,
    pbp_away_pts: float,
    elapsed_minutes: float,
    minutes_left: float,
    num_sims: int,
) -> dict[str, float]:
    home_ppm = _safe_rate(pbp_home_pts, elapsed_minutes)
    away_ppm = _safe_rate(pbp_away_pts, elapsed_minutes)
    if home_ppm <= 0 and away_ppm <= 0:
        if score_home == score_away:
            return {"legacy_mc_home_win_prob": 0.5}
        return {"legacy_mc_home_win_prob": 1.0 if score_home > score_away else 0.0}
    var_home = max(0.01, home_ppm * 1.3)
    var_away = max(0.01, away_ppm * 1.3)
    sim_home = np.random.normal(
        score_home + home_ppm * minutes_left,
        np.sqrt(var_home * minutes_left),
        num_sims,
    )
    sim_away = np.random.normal(
        score_away + away_ppm * minutes_left,
        np.sqrt(var_away * minutes_left),
        num_sims,
    )
    home_wins = np.sum(sim_home > sim_away)
    ties = np.sum(np.abs(sim_home - sim_away) < 0.5)
    return {"legacy_mc_home_win_prob": float(home_wins + 0.5 * ties) / float(num_sims)}


def _current_scoring_run(events: list[tuple[str, float]], team_name: str) -> float:
    run = 0.0
    for team, pts in reversed(events):
        if team == team_name and pts > 0:
            run += pts
        else:
            break
    return run


def _max_scoring_run(events: list[tuple[str, float]], team_name: str) -> float:
    best = 0.0
    run = 0.0
    for team, pts in events:
        if team == team_name and pts > 0:
            run += pts
            if run > best:
                best = run
        else:
            run = 0.0
    return best


# ============================================================================
# Features DESCARTADAS de v13/v14 (documentacion)
# ============================================================================
# Se retiraron porque aportan ruido o son redundantes:
# - gp_diff (equivalente a gp_latest_diff, se unifico).
# - pbp_home_pts / pbp_away_pts COMO FEATURES CRUDAS: correlaciones
#   demasiado altas con score_halftime_total. Se mantienen solo como rate.
#   (En esta version se mantienen como ctx pero se evitan al hacer
#   selection automatica si generan multicolinearidad severa.)
# - features binarias de "is_nba", "is_college": reemplazadas por modelos
#   por liga (el modelo aprende la liga por construccion, no como flag).
# - Top-level pace_bucket como MODEL SEGMENTATION: se reemplaza por
#   pace como feature dentro de un unico modelo por liga.

