"""
inference.py - Motor de inferencia V16 (carga una vez, inferencia multiple).

Uso tipico:

    engine = V15Engine.load()   # carga todos los modelos a memoria
    result = engine.predict(
        match_id="abc",
        target="q3",
        graph_points=[...],      # lista de {minute, value}
        pbp_events=[...],        # eventos PBP actualizados
        quarter_scores={"q1_home": 18, "q1_away": 20, "q2_home": ..., ...},
        league="NBA",
    )
    print(result.signal)         # 'BET_HOME' | 'BET_AWAY' | 'NO_BET'
    print(result.debug.to_dict())  # todo el contexto para UI grafica / API

Diseno:
- Nunca hace fallback a modelo global. Si no hay modelo para (liga, target)
  la signal es NO_BET con reason='no_model_for_league'.
- Debug incluye: modelo usado, threshold, probabilidades cruda y calibrada,
  features, gates, regresion, quality de datos, stats de liga, etc.
- Resultado serializable a JSON para alimentar APIs / simulaciones.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from training.v16 import config, dataset as ds, features as feat, gates, league_overrides


MODEL_DIR = Path(__file__).parent / "model_outputs"
SUMMARY_PATH = MODEL_DIR / "training_summary_v16.json"


# ============================================================================
# Resultados
# ============================================================================

@dataclass
class GateReport:
    name: str
    passed: bool
    reason: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class Debug:
    match_id: str
    target: str
    league: str
    league_slug: str
    model_found: bool
    model_files: dict = field(default_factory=dict)

    # Threshold & calibration
    threshold_used: float = 0.0
    threshold_base: float = 0.0
    league_override: dict = field(default_factory=dict)

    # Probabilities
    raw_probability_home: float | None = None
    calibrated_probability_home: float | None = None
    confidence: float | None = None

    # Regression
    pred_home: float | None = None
    pred_away: float | None = None
    pred_total: float | None = None
    reg_mae_total: float | None = None

    # Data quality
    gp_count: int = 0
    pbp_count: int = 0
    scored_events_count: int = 0

    # League context
    league_samples_train: int = 0
    league_samples_history: int = 0
    league_val_hit_rate: float | None = None
    league_val_roi: float | None = None
    league_holdout_hit_rate: float | None = None
    league_holdout_roi: float | None = None
    league_holdout_n_bets: int = 0

    # Top features con valor
    top_features: list[dict] = field(default_factory=list)

    # Gates detallados
    gates: list[GateReport] = field(default_factory=list)

    # Pace
    pace_total_prior: float = 0.0
    pace_bucket: str = ""

    # Timing
    snapshot_minute: float = 0.0

    # Version / trained_at
    model_version: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prediction:
    signal: str                     # 'BET_HOME' | 'BET_AWAY' | 'NO_BET'
    reason: str
    probability: float | None
    confidence: float | None
    threshold: float
    debug: Debug

    def to_dict(self) -> dict:
        return {
            "signal": self.signal,
            "reason": self.reason,
            "probability": self.probability,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "debug": self.debug.to_dict(),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=_jsonable)


# ============================================================================
# V15Engine
# ============================================================================

@dataclass
class _LeagueModelBundle:
    slug: str
    clf: Any
    vec: Any
    reg_home: Any
    reg_away: Any
    reg_total: Any
    threshold: float
    metrics: dict


class V15Engine:
    def __init__(
        self,
        summary: dict,
        models: dict[tuple[str, str], _LeagueModelBundle],
        league_stats_current: dict[str, dict[str, float]],
        league_history_counts: dict[str, int],
        pace_thresholds: dict[str, float],
    ):
        self.summary = summary
        self.models = models                            # (league, target) -> bundle
        self.league_stats_current = league_stats_current
        self.league_history_counts = league_history_counts
        self.pace_thresholds = pace_thresholds

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, summary_path: Path | str = SUMMARY_PATH) -> "V15Engine":
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        pace_thresholds = summary.get("pace_thresholds", {})

        models: dict[tuple[str, str], _LeagueModelBundle] = {}
        for m in summary.get("models", []):
            if m.get("skipped"):
                continue
            slug = m["slug"]
            target = m["target"]
            league = m["league"]
            files = m["model_files"]
            try:
                bundle = _LeagueModelBundle(
                    slug=slug,
                    clf=joblib.load(MODEL_DIR / files["clf"]),
                    vec=joblib.load(MODEL_DIR / files["vectorizer"]),
                    reg_home=joblib.load(MODEL_DIR / files["reg_home"]),
                    reg_away=joblib.load(MODEL_DIR / files["reg_away"]),
                    reg_total=joblib.load(MODEL_DIR / files["reg_total"]),
                    threshold=float(m["threshold"]["threshold"]),
                    metrics=m,
                )
            except Exception as e:
                print(f"[v16] error cargando modelo {league}/{target}: {e}")
                continue
            models[(league, target)] = bundle

        # Stats de liga actuales (no walk-forward, usado para features en live
        # cuando llega un nuevo partido; es aceptable porque el partido en curso
        # NO esta todavia en la DB).
        conn = ds.get_db_connection()
        try:
            league_stats = _compute_current_league_stats(conn)
            league_history = _count_league_history(conn)
        finally:
            conn.close()

        return cls(summary, models, league_stats, league_history, pace_thresholds)

    # --------------------------------------------------------------- predict
    def predict(
        self,
        match_id: str,
        target: str,
        league: str,
        quarter_scores: dict[str, int],
        graph_points: list[dict],
        pbp_events: list[dict],
    ) -> Prediction:
        """
        Predice para un partido en vivo.

        Args:
            quarter_scores: {"q1_home", "q1_away", "q2_home", "q2_away",
                             "q3_home"(opcional), "q3_away"(opcional)}
            graph_points: lista de {minute, value} tal como viene de Sofascore
            pbp_events: lista de eventos (ver dataset._estimate_minute)
        """
        debug = Debug(
            match_id=match_id, target=target, league=league,
            league_slug=ds.slugify_league(league),
            model_found=False,
            model_version=self.summary.get("version", "v16"),
        )

        override = league_overrides.get_league_override(league)
        debug.league_override = dict(override) if override else {}

        cutoff = config.Q3_GRAPH_CUTOFF if target == "q3" else config.Q4_GRAPH_CUTOFF
        debug.snapshot_minute = float(cutoff)

        # Construimos la muestra live (reutilizamos Sample-like via features.LiveSample)
        halftime_total = sum(
            quarter_scores.get(k, 0) for k in ("q1_home", "q1_away", "q2_home", "q2_away")
        )
        halftime_diff = (
            quarter_scores.get("q1_home", 0) + quarter_scores.get("q2_home", 0)
            - quarter_scores.get("q1_away", 0) - quarter_scores.get("q2_away", 0)
        )
        prev_features = {
            "halftime_total": halftime_total,
            "halftime_diff": halftime_diff,
            "q1_diff": quarter_scores.get("q1_home", 0) - quarter_scores.get("q1_away", 0),
            "q2_diff": quarter_scores.get("q2_home", 0) - quarter_scores.get("q2_away", 0),
            "q1_total": quarter_scores.get("q1_home", 0) + quarter_scores.get("q1_away", 0),
            "q2_total": quarter_scores.get("q2_home", 0) + quarter_scores.get("q2_away", 0),
        }
        pace_total_prior = halftime_total
        if target == "q4":
            q3_home = quarter_scores.get("q3_home", 0)
            q3_away = quarter_scores.get("q3_away", 0)
            prev_features["q3_diff"] = q3_home - q3_away
            prev_features["q3_total"] = q3_home + q3_away
            pace_total_prior += q3_home + q3_away
        debug.pace_total_prior = pace_total_prior
        debug.pace_bucket = ds.pace_bucket_for(target, pace_total_prior, self.pace_thresholds)

        live_sample = feat.LiveSample(
            match_id=match_id, target=target, snapshot_minute=cutoff,
            date="", league=league, gender="men",
            features=prev_features, pace_total_prior=pace_total_prior,
        )

        # Features para liga (stats walk-forward disponibles)
        league_stats = self.league_stats_current.get(league, {})
        debug.league_samples_history = int(league_stats.get("league_samples", 0))

        gp_dict = {match_id: graph_points or []}
        pbp_dict = {match_id: [_normalize_pbp(e) for e in (pbp_events or [])]}

        features = feat.build_features_for_sample(
            live_sample, gp_dict, pbp_dict,
            league_stats=league_stats,
            pace_thresholds=self.pace_thresholds,
        )
        debug.gp_count = int(features.get("gp_count", 0))
        debug.pbp_count = int(features.get("pbp_count", 0))
        debug.scored_events_count = sum(
            1 for e in pbp_dict[match_id]
            if e.get("home_score") is not None and e.get("away_score") is not None
        )

        # Buscar modelo
        bundle = self.models.get((league, target))
        model_found = bundle is not None
        debug.model_found = model_found
        debug.threshold_base = config.MIN_CONFIDENCE_BASE
        if bundle:
            debug.league_slug = bundle.slug
            debug.model_files = bundle.metrics.get("model_files", {})
            debug.threshold_used = bundle.threshold
            debug.league_samples_train = int(bundle.metrics.get("n_train", 0))
            debug.league_val_hit_rate = _safe_get(bundle.metrics, "threshold.hit_rate")
            debug.league_val_roi = _safe_get(bundle.metrics, "threshold.roi")
            hb = bundle.metrics.get("holdout_betting") or {}
            debug.league_holdout_hit_rate = hb.get("hit_rate")
            debug.league_holdout_roi = hb.get("roi")
            debug.league_holdout_n_bets = int(hb.get("n_bets", 0))
            debug.reg_mae_total = _safe_get(bundle.metrics, "regression_metrics.total_val.mae")

            X = bundle.vec.transform([features])
            proba_home = float(bundle.clf.predict_proba(X)[0, 1])
            debug.raw_probability_home = proba_home
            debug.calibrated_probability_home = proba_home
            debug.confidence = float(max(proba_home, 1 - proba_home))

            debug.pred_home = float(bundle.reg_home.predict(X)[0])
            debug.pred_away = float(bundle.reg_away.predict(X)[0])
            debug.pred_total = float(bundle.reg_total.predict(X)[0])

            # Top features con valores
            debug.top_features = _build_top_features(
                bundle.metrics.get("top_features", []), features,
            )
        else:
            proba_home = None

        # Gates
        gate_ctx = gates.GateContext(
            target=target, league=league, model_available=model_found,
            league_overrides=dict(override) if override else {},
            proba_home=proba_home if proba_home is not None else 0.5,
            threshold=(bundle.threshold if bundle else config.MIN_CONFIDENCE_BASE),
            gp_count=debug.gp_count,
            pbp_count=debug.pbp_count,
            match_minute=float(cutoff),
            pred_total=debug.pred_total,
            pred_home=debug.pred_home,
            pred_away=debug.pred_away,
            reg_mae_total=debug.reg_mae_total,
            current_run_pts=max(
                features.get("traj_current_run_home", 0),
                features.get("traj_current_run_away", 0),
            ),
            volatility_swings=int(features.get("gp_swings", 0)),
            league_samples_history=debug.league_samples_history,
        )
        passed, gate_results = gates.run_gates(gate_ctx)
        debug.gates = [GateReport(r.name, r.passed, r.reason, r.details) for r in gate_results]

        if not passed or proba_home is None:
            reason = next((g.reason for g in gate_results if not g.passed), "no_model")
            return Prediction(
                signal="NO_BET", reason=reason,
                probability=proba_home, confidence=debug.confidence,
                threshold=debug.threshold_used, debug=debug,
            )

        signal = "BET_HOME" if proba_home >= 0.5 else "BET_AWAY"
        return Prediction(
            signal=signal,
            reason="all_gates_passed",
            probability=proba_home,
            confidence=debug.confidence,
            threshold=debug.threshold_used,
            debug=debug,
        )


# ============================================================================
# Utilidades
# ============================================================================

def _normalize_pbp(e: dict) -> dict:
    """Adapta un evento PBP recibido en tiempo real al shape esperado."""
    out = dict(e)
    if "minute" not in out:
        out["minute"] = ds._estimate_minute(out.get("quarter", "Q1"), out.get("time"))
    return out


def _safe_get(d: dict, path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            return None
        if cur is None:
            return None
    return cur


def _build_top_features(metadata_top: list[dict], current_features: dict) -> list[dict]:
    out: list[dict] = []
    for entry in metadata_top:
        name = entry.get("feature")
        if name is None:
            continue
        out.append({
            "feature": name,
            "importance": entry.get("importance"),
            "value": current_features.get(name),
        })
    return out


def _compute_current_league_stats(conn) -> dict[str, dict[str, float]]:
    """Snapshot de stats por liga usando toda la DB (solo para features de
    inferencia; no usar en training)."""
    cur = conn.cursor()
    cur.execute(
        "SELECT m.league, qs1.home as q1h, qs1.away as q1a, "
        "qs2.home as q2h, qs2.away as q2a, "
        "qs3.home as q3h, qs3.away as q3a, qs4.home as q4h, qs4.away as q4a "
        "FROM matches m "
        "JOIN quarter_scores qs1 ON m.match_id=qs1.match_id AND qs1.quarter='Q1' "
        "JOIN quarter_scores qs2 ON m.match_id=qs2.match_id AND qs2.quarter='Q2' "
        "LEFT JOIN quarter_scores qs3 ON m.match_id=qs3.match_id AND qs3.quarter='Q3' "
        "LEFT JOIN quarter_scores qs4 ON m.match_id=qs4.match_id AND qs4.quarter='Q4'"
    )
    agg: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"ht": [], "ha": [], "q3": [], "q4": []}
    )
    for row in cur.fetchall():
        league = row["league"] or "Unknown"
        ht = (row["q1h"] or 0) + (row["q1a"] or 0) + (row["q2h"] or 0) + (row["q2a"] or 0)
        ha = (row["q1h"] or 0) + (row["q2h"] or 0) - (row["q1a"] or 0) - (row["q2a"] or 0)
        agg[league]["ht"].append(ht)
        agg[league]["ha"].append(ha)
        if row["q3h"] is not None:
            agg[league]["q3"].append((row["q3h"] or 0) + (row["q3a"] or 0))
        if row["q4h"] is not None:
            agg[league]["q4"].append((row["q4h"] or 0) + (row["q4a"] or 0))
    out: dict[str, dict[str, float]] = {}
    for league, d in agg.items():
        out[league] = {
            "league_samples": float(len(d["ht"])),
            "league_ht_total_mean": float(np.mean(d["ht"])) if d["ht"] else 0.0,
            "league_ht_total_std": float(np.std(d["ht"])) if len(d["ht"]) > 1 else 0.0,
            "league_home_advantage_mean": float(np.mean(d["ha"])) if d["ha"] else 0.0,
            "league_q3_total_mean": float(np.mean(d["q3"])) if d["q3"] else 0.0,
            "league_q4_total_mean": float(np.mean(d["q4"])) if d["q4"] else 0.0,
        }
    return out


def _count_league_history(conn) -> dict[str, int]:
    cur = conn.cursor()
    cur.execute("SELECT league, COUNT(*) as n FROM matches GROUP BY league")
    return {row["league"]: row["n"] for row in cur.fetchall()}


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)
