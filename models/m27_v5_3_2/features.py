"""
Extractor UNICO de features de m27_v5_3_2 (entrenamiento, evaluacion offline y monitor_v3 en vivo).

Diseño (ver docs/findings_m27_v5_3_2.md):
  * Una sola funcion `snapshot_features(ctx, state)` produce el vector; el entrenamiento arma `ctx` desde la BD y el
    monitor lo arma desde el payload de SofaScore con `ctx_from_payload`. Cero formulas duplicadas => cero skew.
  * Resiliente al minuto: el vector depende del minuto real `t` (25.5..30). Se entrena con snapshots en todo ese rango.
    Si el partido ya esta en Q4 (o hay retraso) se usa el cierre de Q3 (t=30) calculado desde Q1+Q2+Q3.
  * Los priors (ELO, forma, H2H, liga) salen de `PriorState`, reproducible incrementalmente desde matches.db.
  * Faltantes = NaN (no 0): los arboles aprenden la ausencia; el LR usa imputacion + indicadores.
Alcance: baloncesto de cuartos de 10 min (FIBA hombres/mujeres). 12m (NBA/CBA/PBA) no esta soportado.
"""

from __future__ import annotations

import json
import math
import pickle
import sqlite3
from collections import deque
from datetime import date as _date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from models.common.competition_stage import classify_stage

MODEL_DIR = Path(__file__).resolve().parent / "model_outputs"
STATE_PATH = MODEL_DIR / "v5_3_state.pkl"
NAN = float("nan")

# (nombre, regla de espejo): 'n' negar, 's' igual, 'o' 1-x, 'h' bandera de orientacion local
FEATURE_SPEC: list[tuple[str, str]] = [
    # A. Estado de juego al minuto t
    ("minute", "s"),
    ("cur_score_diff", "n"),
    ("abs_cur_score_diff", "s"),
    ("ht_diff", "n"),
    ("q1_diff", "n"),
    ("q2_diff", "n"),
    ("q3_partial_diff", "n"),
    ("q3_pace_diff", "n"),
    ("q3_vs_ht_momentum", "n"),
    ("recent_pts_diff_2m", "n"),
    ("recent_pts_diff_5m", "n"),
    ("recent_pace_ratio", "s"),
    ("pace_pts_per_min", "s"),
    ("pace_vs_league_prior", "s"),
    ("gp_val", "n"),
    ("gp_slope_3m", "n"),
    ("lead_z", "n"),
    ("lead_vs_lg_vol", "n"),
    # B. Fuerza pre-partido (as-of, PriorState)
    ("d_elo_ft", "n"),
    ("d_elo_q4", "n"),
    ("d_wr15", "n"),
    ("d_q4_bayes", "n"),
    ("d_ft_margin15", "n"),
    ("d_q4_net12", "n"),
    ("d_q3_net12", "n"),
    ("d_season_wr", "n"),
    ("h2h_ft_margin", "n"),
    ("h2h_q4_margin", "n"),
    ("elo_reliability", "s"),
    # C. Mercado pre-partido
    ("has_ml", "s"),
    ("odds_log_ratio", "n"),
    ("mkt_p_home", "o"),
    ("expected_ft_margin", "n"),
    ("mkt_margin_spread", "n"),
    ("total_line", "s"),
    ("cur_vs_expected", "n"),
    # D. Liga
    ("lg_avg_ft", "s"),
    ("lg_avg_q4", "s"),
    ("lg_q4_vol", "s"),
    ("lg_home_q4_rate", "o"),
    ("is_women", "s"),
    # E. Orientacion (1 = vista original local; 0 = vista espejo)
    ("is_home_view", "h"),
]
FEATURE_NAMES = [n for n, _ in FEATURE_SPEC]
N_FEATURES = len(FEATURE_NAMES)
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
_NEG = np.array([r == "n" for _, r in FEATURE_SPEC])
_ONE_MINUS = np.array([r == "o" for _, r in FEATURE_SPEC])
_HOME_FLAG = IDX["is_home_view"]

SNAPSHOT_MIN, SNAPSHOT_MAX = 25.5, 30.0

# Patrones literales de ligas toxicas. Solo se listan los que CASAN con nombres
# reales de SofaScore: los que no coinciden con ninguna liga de `matches` no
# podan nada del train ni bloquean nada en live (falsa sensacion de cobertura).
# La lista se audito contra las 3.761 ligas del universo de entrenamiento y
# las 434 vistas en vivo (ver docs/modelos/ANALISIS_LIVE_M27_V5_3.md).
TOXIC_LEAGUE_PATTERNS = [
    "korean basketball league", "kbl", "lmb, apertura", "tipos sbl",
    "korisliiga, women", "betsafe-lkl", "nlb", "1st dbbl",
    "serie a2, women, group b", "serie b, group b", "israeli basketball super league",
]

# Equilibrio a la cuota que se usa como referencia operativa (1.40).
BREAKEVEN_140 = 1 / 1.40
_CALIBRATION_PATH = MODEL_DIR / "league_calibration.json"
_MEASURED_TOXIC: dict[str, tuple[float, list[float]]] | None = None

# Categorias de FASE DE TORNEO que se descartan del entrenamiento y se
# bloquean en inferencia. Decision del 2026-10-20 a partir de la premisa de
# docs/ligas/COMPORTAMIENTO_LIGAS_Y_GUIA_ENTRENAMIENTO.md y del backtest por fase
# (docs/modelos/ANALISIS_LIVE_M27_V5_3.md).
#   - ELIMINATORIA y COPA NO se descartan: el backtest medido en el holdout C
#     da 84.3 % y 82.8 % de acierto, por encima del 77.0 % de la temporada
#     regular, asi que no son ruido para este objetivo (ganador de Q4).
#   - AMISTOSOS y el bloque de formatos no estandar SI se descartan: el
#     formato (ritmo, localia, numero de equipos) no es el de una liga.
BLOCKED_STAGE_CATEGORIES = {
    "AMISTOSOS": "torneo amistoso o invitacion (sin formato de liga)",
    "SEGUNDA_FASE": "segunda fase o ronda de campeonato con grupos (formato distinto)",
    "PROMOCION": "fase de promotion/relegation (formato de supervivencia)",
    "PRELIMINAR": "fase preliminar o clasificatoria",
    "PUESTO": "fase de colocacion (partidos por puesto, no por titulo)",
}


def is_toxic_league(name: str) -> bool:
    """Bloqueo por patron literal. Es la regla que se usa al PODAR el train."""
    s = str(name or "").lower()
    return any(p in s for p in TOXIC_LEAGUE_PATTERNS)


def measured_toxic_leagues() -> dict[str, tuple[float, list[float]]]:
    """Ligas que el PROPIO modelo midio como toxicas.

    Criterio: el IC95% SUPERIOR de su WR en Q4 queda por debajo del equilibrio
    a cuota 1.40 (71.4 %). Es decir, ni en el extremo optimista del intervalo
    la liga llega al punto de equilibrio. Sale de `league_calibration.json`,
    que el pipeline regenera en cada entrenamiento, asi que la lista se
    mantiene sola con evidencia y no a mano.

    Devuelve {liga: (wr, ic95)} para poder explicar el motivo del descarte.
    """
    global _MEASURED_TOXIC
    if _MEASURED_TOXIC is not None:
        return _MEASURED_TOXIC
    out: dict[str, tuple[float, list[float]]] = {}
    try:
        if _CALIBRATION_PATH.exists():
            for row in json.loads(_CALIBRATION_PATH.read_text(encoding="utf-8")):
                ci = row.get("wr_ci95") or []
                if len(ci) == 2 and float(ci[1]) < BREAKEVEN_140 * 100:
                    name = str(row.get("league") or "").strip().lower()
                    if name:
                        out[name] = (float(row.get("wr") or 0.0), [float(ci[0]), float(ci[1])])
    except Exception:
        out = {}
    _MEASURED_TOXIC = out
    return out


def _matches_measured(name: str) -> tuple[float, list[float]] | None:
    s = str(name).strip().lower()
    for toxic, stats in measured_toxic_leagues().items():
        if s == toxic or s.startswith(toxic + ",") or s.startswith(toxic + " "):
            return stats
    return None


def block_reason(name: str) -> str | None:
    """MOTIVO por el que una liga se descarta, o None si no se descarta.

    El modelo lo reporta en el log para que se sepa POR QUE se paso un
    partido, no solo que se paso.

    Orden de precedencia:
      1. Lista negra de composicion (patrones literales).
      2. Medicion propia del modelo (WR con IC95 por debajo del equilibrio).
      3. Categoria de fase de torneo descartada por diseno.
    """
    s = str(name or "").strip()
    if not s:
        return None
    if is_toxic_league(s):
        return "descartada: liga en la lista negra de composicion"
    med = _matches_measured(s)
    if med:
        wr, ci = med
        return (f"descartada: liga medida como toxica por el modelo "
                f"(WR {wr:.1f}%, IC95 [{ci[0]:.1f}, {ci[1]:.1f}] < equilibrio "
                f"{BREAKEVEN_140 * 100:.1f}% a cuota 1.40)")
    cat, motivo = classify_stage(s)
    if cat in BLOCKED_STAGE_CATEGORIES:
        return f"descartada: {motivo} (categoria {cat})"
    return None


def is_blocked_league(name: str) -> bool:
    """Bloqueo en INFERENCIA: composicion + medicion propia + fase descartada.

    Es un superconjunto de `is_toxic_league` (nunca menos), de modo que
    bloquear en vivo no puede dejar pasar una liga que el train ya podo.
    """
    return block_reason(name) is not None


def is_excluded_stage(league: str) -> bool:
    """True si la FASE de la liga esta en la lista de formatos descartados.

    Es la MISMA regla que usa el bloqueo en inferencia, para que el conjunto
    de entrenamiento y el de operacion no diverjan. La aplica
    `build_dataset.py` al construir el dataset.
    """
    cat, _ = classify_stage(league)
    return cat in BLOCKED_STAGE_CATEGORIES


def mirror(X: np.ndarray) -> np.ndarray:
    """Vista espejo (local<->visitante) de una matriz de features (NaN se conserva)."""
    M = np.array(X, dtype=np.float32, copy=True)
    M[..., _NEG] = -M[..., _NEG]
    M[..., _ONE_MINUS] = 1.0 - M[..., _ONE_MINUS]
    M[..., _HOME_FLAG] = 0.0
    return M


# ---------------------------------------------------------------------------
# Estado de priors (ELO, forma, H2H, liga)
# ---------------------------------------------------------------------------

def date_ord(s: Any) -> int:
    """'YYYY-MM-DD...' -> ordinal; si no se puede parsear usa hoy."""
    try:
        return datetime.strptime(str(s).split(" ")[0].split("T")[0], "%Y-%m-%d").toordinal()
    except Exception:
        return _date.today().toordinal()


class PriorState:
    """Priors historicos as-of. Misma clase para replay de entrenamiento y para produccion."""

    def __init__(self) -> None:
        self.elo_ft: dict[str, float] = {}
        self.elo_q4: dict[str, float] = {}
        self.last_date: dict[str, int] = {}
        self.ft_hist: dict[str, deque] = {}
        self.q4_hist: dict[str, deque] = {}
        self.q3_hist: dict[str, deque] = {}
        self.q4net_hist: dict[str, deque] = {}
        self.wl: dict[str, list[int]] = {}
        self.h2h: dict[tuple[str, str], deque] = {}
        self.lg: dict[str, list[float]] = {}  # liga -> [n, sum|q4|, sum home_q4_win]
        self.lc_map: dict[str, dict[str, Any]] = {}
        self.built_through: tuple[str, str] = ("", "")

    # -- helpers --------------------------------------------------------
    def _view(self, team: str, dord: int) -> dict[str, Any]:
        e_ft = self.elo_ft.get(team, 1500.0)
        e_q4 = self.elo_q4.get(team, 1500.0)
        ft = list(self.ft_hist.get(team, ()))
        q4 = list(self.q4_hist.get(team, ()))
        q3 = list(self.q3_hist.get(team, ()))
        q4n = list(self.q4net_hist.get(team, ()))
        wl = list(self.wl.get(team, (0, 0)))
        ld = self.last_date.get(team)
        if ld is not None:
            gap = dord - ld
            if gap > 300:
                e_ft = 1500.0 + 0.25 * (e_ft - 1500.0)
                e_q4 = 1500.0 + 0.20 * (e_q4 - 1500.0)
                ft, q4, q3, q4n = [], [], [], []
            elif gap > 60:
                e_ft = 1500.0 + 0.65 * (e_ft - 1500.0)
                e_q4 = 1500.0 + 0.50 * (e_q4 - 1500.0)
                ft, q4, q3, q4n = ft[-5:], q4[-5:], q3[-5:], q4n[-5:]
            if gap > 150:
                wl = [0, 0]
        return {"elo_ft": e_ft, "elo_q4": e_q4, "ft": ft[-15:], "q4": q4[-15:], "q3": q3[-12:], "q4n": q4n[-12:], "wl": wl}

    def league_info(self, league: str, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
        info = self.lc_map.get(league)
        if info:
            return info
        return fallback or {"q_dur": 10, "is_w": 0, "lg_avg_ft": 158.0, "lg_avg_q4": 39.5}

    # -- features pre-partido -------------------------------------------
    def prematch(self, ht: str, at: str, dord: int, league: str, info: dict[str, Any]) -> dict[str, float]:
        h, a = self._view(ht, dord), self._view(at, dord)

        def wr(v):
            return (sum(1.0 for m in v["ft"] if m > 0) + 3.0) / (len(v["ft"]) + 6.0)

        def shr(lst, k=6.0):
            return sum(lst) / (len(lst) + k)

        pair = tuple(sorted((ht, at)))
        past = list(self.h2h.get(pair, ()))[-8:]
        if past:
            ft_m = [g[1] if g[0] == ht else -g[1] for g in past]
            q4_m = [g[2] if g[0] == ht else -g[2] for g in past]
            sh = len(past) / (len(past) + 4.0)
            h2h_ft, h2h_q4 = float(np.mean(ft_m)) * sh, float(np.mean(q4_m)) * sh
        else:
            h2h_ft = h2h_q4 = 0.0

        n_lg, s_abs, s_home = self.lg.get(league, (0.0, 0.0, 0.0))
        return {
            "d_elo_ft": h["elo_ft"] - a["elo_ft"],
            "d_elo_q4": h["elo_q4"] - a["elo_q4"],
            "d_wr15": wr(h) - wr(a),
            "d_q4_bayes": shr(h["q4"]) - shr(a["q4"]),
            "d_ft_margin15": shr(h["ft"]) - shr(a["ft"]),
            "d_q4_net12": shr(h["q4n"], 4.0) - shr(a["q4n"], 4.0),
            "d_q3_net12": shr(h["q3"], 4.0) - shr(a["q3"], 4.0),
            "d_season_wr": (h["wl"][0] + 3.0) / (sum(h["wl"]) + 6.0) - (a["wl"][0] + 3.0) / (sum(a["wl"]) + 6.0),
            "h2h_ft_margin": h2h_ft,
            "h2h_q4_margin": h2h_q4,
            "elo_reliability": min(1.0, min(len(h["ft"]), len(a["ft"])) / 12.0),
            "lg_avg_ft": float(info.get("lg_avg_ft", 158.0)),
            "lg_avg_q4": float(info.get("lg_avg_q4", 39.5)),
            "lg_q4_vol": (s_abs + 40.0 * 6.0) / (n_lg + 40.0),
            "lg_home_q4_rate": (s_home + 40.0 * 0.5) / (n_lg + 40.0),
            "is_women": float(info.get("is_w", 0)),
        }

    # -- actualizacion post-partido -------------------------------------
    def update(self, ht: str, at: str, dord: int, league: str, q1: int, q2: int, q3: int, q4: int) -> None:
        """q1..q4 = diferenciales local-visitante por cuarto. Incluye empates de Q4 en el estado."""
        ft = q1 + q2 + q3 + q4
        for t in (ht, at):  # materializa el decaimiento por inactividad
            v = self._view(t, dord)
            self.elo_ft[t], self.elo_q4[t] = v["elo_ft"], v["elo_q4"]
            for store, key, ml in ((self.ft_hist, "ft", 15), (self.q4_hist, "q4", 15), (self.q3_hist, "q3", 12), (self.q4net_hist, "q4n", 12)):
                store[t] = deque(v[key], maxlen=ml)
            self.wl[t] = v["wl"]

        a_ft = 1.0 if ft > 0 else (0.0 if ft < 0 else 0.5)
        e_ft = 1.0 / (1.0 + 10.0 ** ((self.elo_ft[at] - self.elo_ft[ht]) / 400.0))
        self.elo_ft[ht] += 20.0 * (a_ft - e_ft)
        self.elo_ft[at] -= 20.0 * (a_ft - e_ft)
        a_q4 = 1.0 if q4 > 0 else (0.0 if q4 < 0 else 0.5)
        e_q4 = 1.0 / (1.0 + 10.0 ** ((self.elo_q4[at] - self.elo_q4[ht]) / 400.0))
        self.elo_q4[ht] += 15.0 * (a_q4 - e_q4)
        self.elo_q4[at] -= 15.0 * (a_q4 - e_q4)

        for t, sgn in ((ht, 1), (at, -1)):
            self.ft_hist[t].append(sgn * ft)
            self.q4_hist[t].append(sgn * q4)
            self.q3_hist[t].append(sgn * q3)
            self.q4net_hist[t].append(sgn * q4)
            self.last_date[t] = dord
        if ft != 0:
            self.wl[ht][0 if ft > 0 else 1] += 1
            self.wl[at][1 if ft > 0 else 0] += 1
        pair = tuple(sorted((ht, at)))
        self.h2h.setdefault(pair, deque(maxlen=8)).append((ht, ft, q4))
        if q4 != 0:
            st = self.lg.setdefault(league, [0.0, 0.0, 0.0])
            st[0] += 1
            st[1] += abs(q4)
            st[2] += 1.0 if q4 > 0 else 0.0

    # -- persistencia ---------------------------------------------------
    def save(self, path: Path = STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def refresh_from_db(self, conn: sqlite3.Connection) -> int:
        """Incorpora al estado los partidos terminados posteriores a `built_through` (orden cronologico)."""
        d0, m0 = self.built_through
        rows = conn.execute(
            "SELECT m.match_id, m.date, m.league, m.home_team, m.away_team, "
            "MAX(CASE WHEN q.quarter='Q1' THEN q.home END), MAX(CASE WHEN q.quarter='Q1' THEN q.away END), "
            "MAX(CASE WHEN q.quarter='Q2' THEN q.home END), MAX(CASE WHEN q.quarter='Q2' THEN q.away END), "
            "MAX(CASE WHEN q.quarter='Q3' THEN q.home END), MAX(CASE WHEN q.quarter='Q3' THEN q.away END), "
            "MAX(CASE WHEN q.quarter='Q4' THEN q.home END), MAX(CASE WHEN q.quarter='Q4' THEN q.away END) "
            "FROM matches m JOIN quarter_scores q ON q.match_id = m.match_id "
            "WHERE m.status_type='finished' AND m.date IS NOT NULL AND (m.date > ? OR (m.date = ? AND m.match_id > ?)) "
            "GROUP BY m.match_id ORDER BY m.date ASC, m.match_id ASC",
            (d0, d0, m0),
        ).fetchall()
        n = 0
        for r in rows:
            if None in r[5:13] or not r[3] or not r[4]:
                continue
            self.update(r[3], r[4], date_ord(r[1]), r[2], r[5] - r[6], r[7] - r[8], r[9] - r[10], r[11] - r[12])
            self.built_through = (str(r[1]), str(r[0]))
            n += 1
        return n


_STATE_CACHE: PriorState | None = None


def get_state(reload: bool = False) -> PriorState:
    global _STATE_CACHE
    if _STATE_CACHE is None or reload:
        if STATE_PATH.exists():
            with open(STATE_PATH, "rb") as f:
                _STATE_CACHE = pickle.load(f)
        else:
            _STATE_CACHE = PriorState()
    return _STATE_CACHE


def refresh_state(db_path: str | Path, save: bool = False) -> int:
    """Incorpora al estado en memoria los partidos terminados desde el ultimo corte. Copia-y-reemplaza para no
    interferir con inferencias concurrentes. Devuelve cuantos partidos se agregaron."""
    import copy

    global _STATE_CACHE
    st = copy.deepcopy(get_state())
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=15)
    try:
        n = st.refresh_from_db(conn)
    finally:
        conn.close()
    if n:
        _STATE_CACHE = st
        if save:
            st.save()
    return n


# ---------------------------------------------------------------------------
# Mercado
# ---------------------------------------------------------------------------

def extract_market(rows: Any) -> tuple[float, float, float, float]:
    """rows (esquema match_odds / _parse_odds) -> (ml_home, ml_away, spread_line_home, total_line); NaN si no hay.

    Solo se aceptan cuotas pre-partido: _parse_odds ya descarta los mercados in-play (isLive).
    """
    ml_h = ml_a = sp = tl = NAN
    if not isinstance(rows, (list, tuple)):
        return ml_h, ml_a, sp, tl
    for r in rows:
        if not isinstance(r, dict):
            continue
        otype = str(r.get("odds_type") or "")
        per = str(r.get("market_period") or "")
        if otype == "Home/Away" and per == "Match" and math.isnan(ml_h):
            try:
                h, a = float(r.get("home_value") or 0), float(r.get("away_value") or 0)
            except (TypeError, ValueError):
                continue
            if h > 1.0 and a > 1.0:
                ml_h, ml_a = h, a
        elif otype == "Point spread" and per == "Match" and math.isnan(sp) and r.get("line") is not None:
            sp = float(r["line"])
        elif otype == "Over/Under" and per == "Match" and math.isnan(tl) and r.get("line") is not None:
            tl = float(r["line"])
    return ml_h, ml_a, sp, tl


# ---------------------------------------------------------------------------
# Eventos y score al minuto t
# ---------------------------------------------------------------------------

EventArrays = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # t_sec, signed_pts, home_score, away_score


def events_to_arrays(rows: list[dict[str, Any]]) -> EventArrays | None:
    """Eventos de gol (dict con time_seconds, team, points, home_score, away_score) -> arrays ordenados por tiempo."""
    ts, sg, hs, as_ = [], [], [], []
    for e in rows:
        pts = e.get("points")
        t = e.get("time_seconds")
        h, a = e.get("home_score"), e.get("away_score")
        team = e.get("team")
        if not pts or t is None or h is None or a is None or team not in ("home", "away"):
            continue
        ts.append(float(t))
        sg.append(float(pts) if team == "home" else -float(pts))
        hs.append(float(h))
        as_.append(float(a))
    if not ts:
        return None
    return sort_event_arrays(np.array(ts), np.array(sg), np.array(hs), np.array(as_))


def sort_event_arrays(ts: np.ndarray, sg: np.ndarray, hs: np.ndarray, as_: np.ndarray) -> EventArrays:
    o = np.lexsort((hs + as_, ts))
    return ts[o], sg[o], hs[o], as_[o]


def score_at(ev: EventArrays, t_sec: float) -> tuple[float, float]:
    idx = int(np.searchsorted(ev[0], t_sec, side="right"))
    return (0.0, 0.0) if idx == 0 else (float(ev[2][idx - 1]), float(ev[3][idx - 1]))


def _window_diff(ev: EventArrays, t_sec: float, minutes: float) -> tuple[float, float]:
    m = (ev[0] > t_sec - minutes * 60.0) & (ev[0] <= t_sec)
    return float(ev[1][m].sum()), float(np.abs(ev[1][m]).sum())


# ---------------------------------------------------------------------------
# Vector de features
# ---------------------------------------------------------------------------

def snapshot_features(ctx: dict[str, Any], state: PriorState) -> np.ndarray:
    """
    ctx: ht, at, league, dord, minute, q1/q2 (h,a), cur (h,a), ev (EventArrays|None), gp (list[(min,val)]|None),
         ml (home,away)|None, spread_line, total_line, info (liga).
    """
    t = float(min(max(ctx["minute"], 20.0), SNAPSHOT_MAX))
    q1h, q1a = ctx["q1"]
    q2h, q2a = ctx["q2"]
    ch, ca = ctx["cur"]
    info = ctx["info"]
    pre = state.prematch(ctx["ht"], ctx["at"], ctx["dord"], ctx["league"], info)

    cur = ch - ca
    total = ch + ca
    q1_diff, q2_diff = q1h - q1a, q2h - q2a
    ht_diff = q1_diff + q2_diff
    q3p = cur - ht_diff
    q3_pace = q3p * 10.0 / max(t - 20.0, 3.0)
    pace = total / max(t, 1.0)
    lg_ppm = info.get("lg_avg_ft", 158.0) * 0.75 / 30.0

    r2 = r5 = rpace = NAN
    ev = ctx.get("ev")
    if ev is not None:
        d2, _ = _window_diff(ev, t * 60.0, 2.0)
        d5, _ = _window_diff(ev, t * 60.0, 5.0)
        _, tot4 = _window_diff(ev, t * 60.0, 4.0)
        r2, r5 = d2, d5
        rpace = tot4 / max(pace * 4.0, 1.0)

    gp_val = gp_slope = NAN
    gp = ctx.get("gp")
    if gp:
        vals = [float(v) for m, v in gp if m is not None and v is not None and float(m) <= int(t)]
        if vals:
            gp_val = vals[-1]
            gp_slope = vals[-1] - vals[-4] if len(vals) >= 4 else vals[-1] - vals[0]

    ml = ctx.get("ml")
    ml_h, ml_a = (ml if ml else (NAN, NAN))
    has_ml = float(not math.isnan(ml_h))
    olr = mkt_p = NAN
    if has_ml:
        olr = float(np.clip(math.log(ml_a / ml_h), -2.5, 2.5))
        ih, ia = 1.0 / ml_h, 1.0 / ml_a
        mkt_p = ih / (ih + ia)
    efm = (6.5 * olr + 0.5 * pre["d_elo_ft"] / 28.0) if has_ml else pre["d_elo_ft"] / 28.0
    sp = ctx.get("spread_line", NAN)
    tl = ctx.get("total_line", NAN)

    vol = max(pre["lg_q4_vol"], 1.0)
    f = {
        "minute": t,
        "cur_score_diff": cur,
        "abs_cur_score_diff": abs(cur),
        "ht_diff": ht_diff,
        "q1_diff": q1_diff,
        "q2_diff": q2_diff,
        "q3_partial_diff": q3p,
        "q3_pace_diff": q3_pace,
        "q3_vs_ht_momentum": q3_pace - 0.5 * ht_diff,
        "recent_pts_diff_2m": r2,
        "recent_pts_diff_5m": r5,
        "recent_pace_ratio": rpace,
        "pace_pts_per_min": pace,
        "pace_vs_league_prior": pace - lg_ppm,
        "gp_val": gp_val,
        "gp_slope_3m": gp_slope,
        "lead_z": cur / math.sqrt(max(total, 20.0)),
        "lead_vs_lg_vol": cur / vol,
        **{k: pre[k] for k in ("d_elo_ft", "d_elo_q4", "d_wr15", "d_q4_bayes", "d_ft_margin15", "d_q4_net12", "d_q3_net12",
                               "d_season_wr", "h2h_ft_margin", "h2h_q4_margin", "elo_reliability")},
        "has_ml": has_ml,
        "odds_log_ratio": olr,
        "mkt_p_home": mkt_p,
        "expected_ft_margin": efm,
        "mkt_margin_spread": -sp if not math.isnan(sp) else NAN,
        "total_line": tl,
        "cur_vs_expected": cur - 0.75 * efm,
        **{k: pre[k] for k in ("lg_avg_ft", "lg_avg_q4", "lg_q4_vol", "lg_home_q4_rate", "is_women")},
        "is_home_view": 1.0,
    }
    return np.array([f[n] for n in FEATURE_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# Contexto desde payload (live / offline load_match)
# ---------------------------------------------------------------------------

def _q(match_data: dict[str, Any], q: str) -> tuple[int, int] | None:
    qs = (match_data.get("score") or {}).get("quarters") or {}
    v = qs.get(q)
    if not v or v.get("home") is None or v.get("away") is None:
        return None
    return int(v["home"]), int(v["away"])


def _iter_events(match_data: dict[str, Any]) -> list[dict[str, Any]]:
    src = match_data.get("events") or match_data.get("play_by_play") or {}
    out: list[dict[str, Any]] = []
    if isinstance(src, dict):
        for plays in src.values():
            out.extend(p for p in (plays or []) if isinstance(p, dict))
    elif isinstance(src, list):
        out = [p for p in src if isinstance(p, dict)]
    return out


def fallback_info(league: str, home: str, away: str, q_dur: int = 10) -> dict[str, Any]:
    """Priors de liga para ligas ausentes de leagues_classification (misma regla en train y live)."""
    from models.common.pbp_utils import infer_gender

    women = infer_gender(league, home, away) == "women"
    return {"q_dur": q_dur, "is_w": int(women), "lg_avg_ft": 140.0 if women else 158.0, "lg_avg_q4": 34.5 if women else 39.5}


def resolve_league_info(state: PriorState, match_data: dict[str, Any]) -> dict[str, Any]:
    from models.common.pbp_utils import infer_regulation_quarter_minutes

    m = match_data.get("match") or {}
    league = str(m.get("league") or match_data.get("league") or "")
    info = state.lc_map.get(league)
    if info:
        return info
    q_dur = int(round(infer_regulation_quarter_minutes(match_data)))
    return fallback_info(league, str(m.get("home_team") or ""), str(m.get("away_team") or ""), q_dur)


def ctx_from_payload(match_data: dict[str, Any], snapshot_minute: float | None, state: PriorState) -> dict[str, Any] | None:
    """Construye ctx a partir del payload canonico. Devuelve None si faltan datos esenciales."""
    m = match_data.get("match") or {}
    ht = str(m.get("home_team") or match_data.get("home_team") or "")
    at = str(m.get("away_team") or match_data.get("away_team") or "")
    league = str(m.get("league") or match_data.get("league") or "")
    if not ht or not at:
        return None
    info = resolve_league_info(state, match_data)

    status = str(m.get("status_type") or "").lower()
    in_play = status in ("inprogress", "live")
    gsp = m.get("game_seconds_played")
    minute = None
    if in_play and gsp is not None:
        minute = float(gsp) / 60.0
    if minute is None:
        minute = float(snapshot_minute) if snapshot_minute is not None else SNAPSHOT_MAX

    ev = events_to_arrays(_iter_events(match_data))

    q1, q2, q3 = _q(match_data, "Q1"), _q(match_data, "Q2"), _q(match_data, "Q3")
    if (q1 is None or q2 is None) and ev is not None:
        q1 = tuple(int(x) for x in score_at(ev, 600.0))
        s20 = score_at(ev, 1200.0)
        q2 = (int(s20[0]) - q1[0], int(s20[1]) - q1[1])
    if q1 is None or q2 is None:
        return None

    t_eff = min(minute, SNAPSHOT_MAX)
    in_q4 = minute > SNAPSHOT_MAX or "4th" in str(m.get("status_description") or "").lower()
    if (in_q4 or t_eff >= SNAPSHOT_MAX - 0.01) and q3 is not None:
        # Q3 cerrado: el marcador exacto sale de los cuartos (misma fuente que la fila ANCHOR del entrenamiento)
        t_eff = SNAPSHOT_MAX
        cur = (q1[0] + q2[0] + q3[0], q1[1] + q2[1] + q3[1])
    elif in_play and (match_data.get("score") or {}).get("home") is not None:
        sc = match_data["score"]
        cur = (int(sc["home"]), int(sc["away"]))
    elif ev is not None:
        cur = tuple(int(x) for x in score_at(ev, t_eff * 60.0))
    elif q3 is not None:
        t_eff = SNAPSHOT_MAX
        cur = (q1[0] + q2[0] + q3[0], q1[1] + q2[1] + q3[1])
    else:
        return None

    gp_raw = match_data.get("graph_points") or []
    gp = [(p.get("minute"), p.get("value")) for p in gp_raw if isinstance(p, dict)] if isinstance(gp_raw, list) else None

    ml_h, ml_a, sp, tl = extract_market(match_data.get("odds"))
    return {
        "ht": ht, "at": at, "league": league, "dord": date_ord(m.get("date")),
        "minute": t_eff, "minute_raw": float(minute),
        "q1": q1, "q2": q2, "cur": cur, "ev": ev, "gp": gp,
        "ml": None if math.isnan(ml_h) else (ml_h, ml_a), "spread_line": sp, "total_line": tl, "info": info,
    }


def extract_features_v5_3(match_data: dict[str, Any], snapshot_minute: float | None = None) -> np.ndarray | None:
    state = get_state()
    ctx = ctx_from_payload(match_data, snapshot_minute, state)
    return None if ctx is None else snapshot_features(ctx, state)
