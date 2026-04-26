"""
dataset.py - Carga DB, construye muestras y particiona temporalmente.

Responsabilidad:
- Conectar a la DB de partidos (matches, quarter_scores, graph_points, play_by_play)
- Filtrar mujeres (configurable via config.ALLOWED_GENDERS)
- Construir muestras para targets Q3 y Q4 con cutoffs dinamicos
- Split temporal: train -> val -> calibration -> holdout
- Util: walk-forward de estadisticas de liga (historico real al momento del partido)

No calcula features sofisticadas (eso es features.py). Aqui solo se arman
los scaffolds y se guardan los crudos de cada partido.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

from training.v17 import config


DB_PATH = Path(__file__).parents[2] / "matches.db"
MODEL_OUTPUTS_DIR = Path(__file__).parent / "model_outputs"
CACHE_PATH = MODEL_OUTPUTS_DIR / "samples_cache_v17.json"


# ============================================================================
# Dataclasses (contrato estable entre dataset.py, features.py y inferencia)
# ============================================================================

@dataclass
class MatchData:
    match_id: str
    date: str
    league: str
    gender: str
    q1_home: int
    q1_away: int
    q2_home: int
    q2_away: int
    q3_home: int | None
    q3_away: int | None
    q4_home: int | None
    q4_away: int | None


@dataclass
class Sample:
    """
    Muestra utilizable por features.build_features_for_sample().
    Contrato compartido con LiveSample en inferencia.
    """
    match_id: str
    target: str              # 'q3' | 'q4'
    snapshot_minute: int
    date: str
    league: str
    gender: str
    # Contexto del marcador previo al cuarto objetivo.
    features: dict[str, Any] = field(default_factory=dict)
    # Targets
    target_winner: int | None = None
    target_home_pts: int | None = None
    target_away_pts: int | None = None
    target_total_pts: int | None = None
    # Contexto adicional (pace_total del cuarto previo) - util como feature.
    pace_total_prior: float = 0.0


# ============================================================================
# Utilidades de carga
# ============================================================================

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def is_women_league(league_name: str) -> bool:
    """Detecta si el nombre de liga pertenece a categoria femenina."""
    if not league_name:
        return False
    low = league_name.lower()
    return any(kw in low for kw in config.WOMEN_KEYWORDS)


def slugify_league(league: str) -> str:
    """Slug estable del nombre de liga para filenames de modelo."""
    s = league.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s or "unknown"


def _load_quarter_scores(conn) -> dict[str, dict[str, int]]:
    cur = conn.cursor()
    cur.execute(
        "SELECT match_id, quarter, home, away FROM quarter_scores "
        "WHERE quarter IN ('Q1','Q2','Q3','Q4')"
    )
    pivot: dict[str, dict[str, int]] = {}
    for row in cur.fetchall():
        mid = row["match_id"]
        q = row["quarter"].lower()
        if mid not in pivot:
            pivot[mid] = {}
        pivot[mid][f"{q}_home"] = row["home"]
        pivot[mid][f"{q}_away"] = row["away"]
    return pivot


def _load_metadata(conn) -> dict[str, dict]:
    cur = conn.cursor()
    cur.execute("SELECT match_id, date, league FROM matches")
    out = {}
    for row in cur.fetchall():
        league = row["league"] or "Unknown"
        out[row["match_id"]] = {
            "date": row["date"] or "",
            "league": league,
            "gender": "women" if is_women_league(league) else "men",
        }
    return out


def load_graph_points(conn, match_ids: list[str]) -> dict[str, list[dict]]:
    cur = conn.cursor()
    out: dict[str, list[dict]] = {}
    for i in tqdm(range(0, len(match_ids), 500), desc="graph_points"):
        batch = match_ids[i:i + 500]
        placeholders = ",".join("?" for _ in batch)
        cur.execute(
            f"SELECT match_id, minute, value FROM graph_points "
            f"WHERE match_id IN ({placeholders}) ORDER BY match_id, minute",
            batch,
        )
        for row in cur.fetchall():
            out.setdefault(row["match_id"], []).append(
                {"minute": row["minute"], "value": row["value"]}
            )
    return out


def load_pbp_events(conn, match_ids: list[str]) -> dict[str, list[dict]]:
    cur = conn.cursor()
    out: dict[str, list[dict]] = {}
    for i in tqdm(range(0, len(match_ids), 500), desc="pbp_events"):
        batch = match_ids[i:i + 500]
        placeholders = ",".join("?" for _ in batch)
        cur.execute(
            f"SELECT match_id, quarter, seq, time, points, team, "
            f"       home_score, away_score "
            f"FROM play_by_play WHERE match_id IN ({placeholders}) "
            f"ORDER BY match_id, quarter, seq",
            batch,
        )
        for row in cur.fetchall():
            out.setdefault(row["match_id"], []).append({
                "quarter": row["quarter"],
                "seq": row["seq"],
                "time": row["time"],
                "points": row["points"],
                "team": row["team"],
                "home_score": row["home_score"],
                "away_score": row["away_score"],
                # Estimacion de minuto-de-partido a partir del cuarto.
                # time en PBP suele venir como "MM:SS" del reloj del cuarto.
                "minute": _estimate_minute(row["quarter"], row["time"]),
            })
    return out


def _estimate_minute(quarter: str, time_str: str | None) -> float:
    """Convierte tiempo dentro de cuarto a minuto acumulado del partido."""
    try:
        q_start = {"Q1": 0, "Q2": 10, "Q3": 20, "Q4": 30}.get(quarter, 0)
        if not time_str:
            return float(q_start)
        parts = time_str.split(":")
        if len(parts) >= 2:
            # En basket el reloj suele descontar, asumimos 10 min por cuarto.
            mins = int(parts[0])
            secs = int(parts[1])
            elapsed = (10 - mins) - (secs / 60.0)
            if elapsed < 0:
                elapsed = 0
            return float(q_start + elapsed)
    except Exception:
        pass
    return float({"Q1": 5, "Q2": 15, "Q3": 25, "Q4": 35}.get(quarter, 0))


# ============================================================================
# Construccion de muestras
# ============================================================================

def _halftime_features(match: dict) -> dict[str, float]:
    q1h, q1a = match["q1_home"], match["q1_away"]
    q2h, q2a = match["q2_home"], match["q2_away"]
    ht_total = q1h + q1a + q2h + q2a
    ht_diff = (q1h + q2h) - (q1a + q2a)
    return {
        "halftime_total": float(ht_total),
        "halftime_diff": float(ht_diff),
        "q1_diff": float(q1h - q1a),
        "q2_diff": float(q2h - q2a),
        "q1_total": float(q1h + q1a),
        "q2_total": float(q2h + q2a),
    }


def _q3_end_features(match: dict) -> dict[str, float]:
    q3_diff = match["q3_home"] - match["q3_away"]
    q3_total = match["q3_home"] + match["q3_away"]
    return {
        "q3_diff": float(q3_diff),
        "q3_total": float(q3_total),
    }


def build_samples(
    allowed_genders: tuple[str, ...] = config.ALLOWED_GENDERS,
    use_cache: bool = True,
    verbose: bool = True,
) -> tuple[list[Sample], dict]:
    """
    Construye muestras desde la DB con cutoffs dinamicos.

    Filtros:
    - Solo generos en allowed_genders (por defecto: hombres).
    - Solo partidos con Q1, Q2 y Q3 completos (Q4 opcional para target Q4).
    """
    MODEL_OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    if use_cache and CACHE_PATH.exists():
        if verbose:
            print(f"[v17] usando cache {CACHE_PATH}")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        samples = [Sample(**s) for s in data["samples"]]
        return samples, data["metadata"]

    conn = get_db_connection()
    try:
        if verbose:
            print("[v17] cargando quarter scores...")
        qs = _load_quarter_scores(conn)
        if verbose:
            print("[v17] cargando metadata...")
        meta = _load_metadata(conn)

        matches: list[dict] = []
        for mid, s in qs.items():
            need = [f"q{i}_{side}" for i in (1, 2, 3) for side in ("home", "away")]
            if not all(k in s for k in need):
                continue
            m = meta.get(mid)
            if not m:
                continue
            if allowed_genders and m["gender"] not in allowed_genders:
                continue
            matches.append({
                "match_id": mid,
                "date": m["date"],
                "league": m["league"],
                "gender": m["gender"],
                **s,
            })
        if verbose:
            print(f"[v17] {len(matches)} partidos completos (genero filtrado)")

        samples: list[Sample] = []
        for match in tqdm(matches, desc="build_samples", disable=not verbose):
            ht = _halftime_features(match)
            pace_total_q3 = ht["halftime_total"]
            q3_winner = 1 if match["q3_home"] > match["q3_away"] else 0

            for snap in config.Q3_TRAIN_SNAPSHOTS:
                samples.append(Sample(
                    match_id=match["match_id"],
                    target="q3",
                    snapshot_minute=snap,
                    date=match["date"],
                    league=match["league"],
                    gender=match["gender"],
                    features=dict(ht),
                    target_winner=q3_winner,
                    target_home_pts=match["q3_home"],
                    target_away_pts=match["q3_away"],
                    target_total_pts=match["q3_home"] + match["q3_away"],
                    pace_total_prior=pace_total_q3,
                ))

            # Q4 solo si hay datos
            if match.get("q4_home") is None or match.get("q4_away") is None:
                continue
            q3_end = _q3_end_features(match)
            pace_total_q4 = pace_total_q3 + q3_end["q3_total"]
            q4_winner = 1 if match["q4_home"] > match["q4_away"] else 0
            feats_q4 = {**ht, **q3_end}
            for snap in config.Q4_TRAIN_SNAPSHOTS:
                samples.append(Sample(
                    match_id=match["match_id"],
                    target="q4",
                    snapshot_minute=snap,
                    date=match["date"],
                    league=match["league"],
                    gender=match["gender"],
                    features=dict(feats_q4),
                    target_winner=q4_winner,
                    target_home_pts=match["q4_home"],
                    target_away_pts=match["q4_away"],
                    target_total_pts=match["q4_home"] + match["q4_away"],
                    pace_total_prior=pace_total_q4,
                ))

        metadata = {
            "n_samples": len(samples),
            "n_matches": len(matches),
            "genders": list(allowed_genders),
            "q3_cutoffs": list(config.Q3_TRAIN_SNAPSHOTS),
            "q4_cutoffs": list(config.Q4_TRAIN_SNAPSHOTS),
        }
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {"samples": [asdict(s) for s in samples], "metadata": metadata},
                f,
            )
        if verbose:
            print(f"[v17] {len(samples)} muestras cacheadas en {CACHE_PATH}")
        return samples, metadata
    finally:
        conn.close()


# ============================================================================
# Particion temporal
# ============================================================================

def _parse_date(d: str) -> datetime | None:
    try:
        return datetime.fromisoformat(d[:10])
    except Exception:
        return None


def split_temporal(
    samples: list[Sample],
    train_days: int = config.TRAIN_DAYS,
    val_days: int = config.VAL_DAYS,
    cal_days: int | None = None,
    holdout_days: int | None = None,
    train_end_date: str | None = None,
    val_end_date: str | None = None,
    cal_end_date: str | None = None,
) -> dict[str, list[Sample]]:
    # Defaults de config si no se pasan explicitos
    if cal_days is None:
        cal_days = getattr(config, "CAL_DAYS", None)
    if holdout_days is None:
        holdout_days = getattr(config, "HOLDOUT_DAYS", None)
    """
    Splits por dias desde la fecha mas antigua.

    - Si cal_days/holdout_days se pasan explicitos, se usan tal cual.
    - Si no, el resto del dataset (despues de train+val) se divide 50/50.
    - holdout_days=0 => produccion pura: todo va a train+val+cal.
    """
    dated = [(s, _parse_date(s.date)) for s in samples]
    dated = [(s, d) for s, d in dated if d is not None]
    if not dated:
        raise RuntimeError("No se pudieron parsear fechas de las muestras")

    if train_end_date:
        train_end = _parse_date(train_end_date)
        val_end = _parse_date(val_end_date) if val_end_date else train_end
        cal_end = _parse_date(cal_end_date) if cal_end_date else val_end
        if not train_end or not val_end or not cal_end:
            raise RuntimeError("No se pudieron parsear train_end_date/val_end_date/cal_end_date")
        buckets = {"train": [], "val": [], "cal": [], "holdout": []}
        for s, d in dated:
            if d < train_end:
                buckets["train"].append(s)
            elif d < val_end:
                buckets["val"].append(s)
            elif d < cal_end:
                buckets["cal"].append(s)
            else:
                buckets["holdout"].append(s)
        return buckets

    oldest = min(d for _, d in dated)
    newest = max(d for _, d in dated)

    train_end = oldest + timedelta(days=train_days)
    val_end = train_end + timedelta(days=val_days)

    if cal_days is None and holdout_days is None:
        remaining = (newest - val_end).days
        cal_end = val_end + timedelta(days=max(remaining // 2, 7))
    else:
        cal_days = cal_days if cal_days is not None else 0
        cal_end = val_end + timedelta(days=cal_days)

    buckets = {"train": [], "val": [], "cal": [], "holdout": []}
    for s, d in dated:
        if d < train_end:
            buckets["train"].append(s)
        elif d < val_end:
            buckets["val"].append(s)
        elif d < cal_end:
            buckets["cal"].append(s)
        else:
            buckets["holdout"].append(s)

    return buckets


# ============================================================================
# Estadisticas de liga walk-forward (para usar como feature sin leakage)
# ============================================================================

@dataclass
class LeagueStatsAccumulator:
    """Rolling stats por liga para walk-forward.

    En training se calcula offline; en inferencia se puede reconstruir desde
    la DB en O(N) una sola vez al inicializar el engine.
    """
    league_samples_count: int = 0
    ht_totals: list[float] = field(default_factory=list)
    home_advantages: list[float] = field(default_factory=list)
    q3_totals: list[float] = field(default_factory=list)
    q4_totals: list[float] = field(default_factory=list)

    def add(self, match: dict) -> None:
        ht = _halftime_features(match)
        self.ht_totals.append(ht["halftime_total"])
        self.home_advantages.append(ht["halftime_diff"])
        if match.get("q3_home") is not None:
            self.q3_totals.append(match["q3_home"] + match["q3_away"])
        if match.get("q4_home") is not None:
            self.q4_totals.append(match["q4_home"] + match["q4_away"])
        self.league_samples_count += 1

    def snapshot(self) -> dict[str, float]:
        def _m(xs: list[float]) -> float:
            return float(np.mean(xs)) if xs else 0.0

        def _sd(xs: list[float]) -> float:
            return float(np.std(xs)) if len(xs) > 1 else 0.0

        return {
            "league_samples": float(self.league_samples_count),
            "league_ht_total_mean": _m(self.ht_totals),
            "league_ht_total_std": _sd(self.ht_totals),
            "league_home_advantage_mean": _m(self.home_advantages),
            "league_q3_total_mean": _m(self.q3_totals),
            "league_q4_total_mean": _m(self.q4_totals),
        }


def compute_league_stats_walkforward(
    samples: list[Sample],
    conn: sqlite3.Connection | None = None,
) -> dict[str, dict[str, float]]:
    """
    Para cada muestra devuelve un snapshot de stats de su liga con solo los
    partidos que ocurrieron ANTES de su fecha.

    Retorna dict keyed por match_id con stats. Se usa en features.league.
    Para evitar O(N^2) se recorren muestras ordenadas por fecha manteniendo
    acumuladores incrementales.
    """
    if conn is None:
        conn = get_db_connection()
        owns = True
    else:
        owns = False
    try:
        # Tomamos un partido unico por match_id para alimentar acumuladores.
        seen: set[str] = set()
        match_rows: list[dict] = []
        for s in samples:
            if s.match_id in seen:
                continue
            seen.add(s.match_id)
            match_rows.append({
                "match_id": s.match_id,
                "date": s.date,
                "league": s.league,
                **s.features,  # ht features ya calculadas
                # Reconstruimos q3/q4 si targets estan presentes
            })

        # Ordenamos por fecha ascendente.
        match_rows.sort(key=lambda r: r["date"])

        accs: dict[str, LeagueStatsAccumulator] = defaultdict(LeagueStatsAccumulator)
        out: dict[str, dict[str, float]] = {}

        # Recuperamos q3_home/away para cada match (una query por batch)
        cur = conn.cursor()
        qs_for_match = {}
        ids = [m["match_id"] for m in match_rows]
        for i in range(0, len(ids), 800):
            batch = ids[i:i + 800]
            placeholders = ",".join("?" for _ in batch)
            cur.execute(
                f"SELECT match_id, quarter, home, away FROM quarter_scores "
                f"WHERE match_id IN ({placeholders}) "
                f"AND quarter IN ('Q3','Q4')",
                batch,
            )
            for row in cur.fetchall():
                mid = row["match_id"]
                qs_for_match.setdefault(mid, {})[row["quarter"]] = (row["home"], row["away"])

        for m in match_rows:
            mid = m["match_id"]
            league = m["league"]
            acc = accs[league]
            # Snapshot ANTES de agregar este partido
            out[mid] = acc.snapshot()
            # Agregar este partido al acumulador (usado por los siguientes)
            extras = {
                "q1_home": 0, "q1_away": 0, "q2_home": 0, "q2_away": 0,
                "q3_home": None, "q3_away": None,
                "q4_home": None, "q4_away": None,
                "halftime_total": m.get("halftime_total", 0),
                "halftime_diff": m.get("halftime_diff", 0),
            }
            if "Q3" in qs_for_match.get(mid, {}):
                extras["q3_home"], extras["q3_away"] = qs_for_match[mid]["Q3"]
            if "Q4" in qs_for_match.get(mid, {}):
                extras["q4_home"], extras["q4_away"] = qs_for_match[mid]["Q4"]
            # Reconstruimos q1+q2 a partir de halftime_total? No es critico; solo
            # necesitamos las totales agregadas; usamos shortcuts.
            match_like = {
                "q1_home": 0, "q1_away": 0,
                "q2_home": int(extras["halftime_total"]) if extras["halftime_total"] else 0,
                "q2_away": 0,
                "q3_home": extras["q3_home"],
                "q3_away": extras["q3_away"],
                "q4_home": extras["q4_home"],
                "q4_away": extras["q4_away"],
            }
            # Ajuste: no usamos valores inventados para home-advantage walk-forward.
            # Pasamos halftime_* directamente al accumulator para evitar ruido.
            acc.ht_totals.append(float(m.get("halftime_total", 0)))
            acc.home_advantages.append(float(m.get("halftime_diff", 0)))
            if extras["q3_home"] is not None:
                acc.q3_totals.append(float(extras["q3_home"] + extras["q3_away"]))
            if extras["q4_home"] is not None:
                acc.q4_totals.append(float(extras["q4_home"] + extras["q4_away"]))
            acc.league_samples_count += 1
        return out
    finally:
        if owns:
            conn.close()


# ============================================================================
# Pace buckets como FEATURE (no como segmentacion de modelo)
# ============================================================================

def pace_bucket_for(
    target: str,
    pace_total: float,
    thresholds: dict[str, float],
) -> str:
    """Clasifica en 'low'/'medium'/'high' segun percentiles aprendidos."""
    if target == "q3":
        low = thresholds.get("q3_low_upper", 72)
        high = thresholds.get("q3_high_lower", 85)
    else:
        low = thresholds.get("q4_low_upper", 108)
        high = thresholds.get("q4_high_lower", 126)
    if pace_total <= low:
        return "low"
    if pace_total >= high:
        return "high"
    return "medium"


def calculate_pace_thresholds(samples: list[Sample]) -> dict[str, float]:
    q3 = [s.pace_total_prior for s in samples if s.target == "q3"]
    q4 = [s.pace_total_prior for s in samples if s.target == "q4"]
    if not q3 or not q4:
        return {
            "q3_low_upper": 72.0, "q3_high_lower": 85.0,
            "q4_low_upper": 108.0, "q4_high_lower": 126.0,
        }
    return {
        "q3_low_upper": float(np.percentile(q3, 33)),
        "q3_high_lower": float(np.percentile(q3, 66)),
        "q4_low_upper": float(np.percentile(q4, 33)),
        "q4_high_lower": float(np.percentile(q4, 66)),
    }

