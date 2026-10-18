"""Extractor de los 36 features limpios y resilientes para el modelo M27 V4 (Tríada Canónica)."""

from __future__ import annotations

import math
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np

from models.common.pbp_utils import (
    graph_stats_upto,
    infer_gender,
    infer_minute_from_pbp,
    infer_regulation_quarter_minutes,
    pbp_events_upto,
    quarter_points,
    recent_window_features,
    score_upto,
)

MODEL_DIR = Path(__file__).resolve().parent / "model_outputs"
_STATE_CACHE: dict[str, Any] | None = None

FEATURE_NAMES_CLEAN = [
    "cur_score_diff",
    "abs_cur_score_diff",
    "ht_diff",
    "q1_diff",
    "q2_diff",
    "q3_eff_diff",
    "q3_vs_ht_momentum",
    "recent_2m_run_diff",
    "pace_pts_per_min",
    "pace_vs_league_prior",
    "gp_val",
    "gp_slope_3m",
    "fraction_q3_completed",
    "d_elo_ft",
    "d_elo_q4",
    "d_elo_q1",
    "elo_reliability",
    "d_q4_bayes",
    "d_wr_bayes",
    "has_odds",
    "odds_log_ratio",
    "expected_ft_margin",
    "h2h_bayes_ft_margin",
    "h2h_bayes_q4_margin",
    "ts_delta_wp_bayes",
    "ts_delta_form",
    "pbp_2pt_diff_q1_q3",
    "pbp_3pt_diff_q1_q3",
    "pbp_ft_diff_q1_q3",
    "d_rol_plus_minus",
    "d_rol_bench_rating",
    "rubber_band_gap",
    "white_flag_risk_zone",
    "blowout_garbage_time",
    "in_q4_partial_diff",
    "lg_avg_q4_prior",
]


def get_priors_state() -> dict[str, Any]:
    """Carga en caché el estado de ELO, jugadores y clasificación de ligas en O(1)."""
    global _STATE_CACHE
    if _STATE_CACHE is None:
        state_path = MODEL_DIR / "team_priors_state.pkl"
        if state_path.exists():
            with open(state_path, "rb") as f:
                _STATE_CACHE = pickle.load(f)
        else:
            _STATE_CACHE = {"team_priors": {}, "lc_map": {}, "player_priors": {}}
    return _STATE_CACHE


def resolve_archetype(match_data: dict[str, Any]) -> tuple[str, int, float, float]:
    """Determina el arquetipo ('fiba_men_10m', 'fiba_women_10m', 'nba_12m') y priors de liga."""
    state = get_priors_state()
    lc_map = state.get("lc_map", {})
    m_info = match_data.get("match") or {}
    league = str(m_info.get("league") or match_data.get("league") or "").strip()
    ht = str(m_info.get("home_team") or "")
    at = str(m_info.get("away_team") or "")

    if league in lc_map:
        info = lc_map[league]
        return info["arch"], int(info["q_dur"]), float(info["lg_avg_ft"]), float(info["lg_avg_q4"])

    q_dur = int(round(infer_regulation_quarter_minutes(match_data)))
    is_w = infer_gender(league, ht, at) == "women"
    if q_dur == 12:
        return "nba_12m", 12, 218.0, 53.5
    if is_w:
        return "fiba_women_10m", 10, 138.0, 34.5
    return "fiba_men_10m", 10, 159.0, 39.8


def _extract_odds_from_payload(raw_odds: Any) -> tuple[float, float]:
    """Extrae (has_odds, odds_log_ratio) desde match_data['odds'] (lista de mercados o dict)."""
    if isinstance(raw_odds, dict):
        ho = raw_odds.get("home_odd") or raw_odds.get("home_value")
        ao = raw_odds.get("away_odd") or raw_odds.get("away_value")
        if ho and ao:
            try:
                fh, fa = float(ho), float(ao)
                if fh > 1.0 and fa > 1.0:
                    return 1.0, math.log(fa / fh)
            except (ValueError, TypeError):
                pass
    elif isinstance(raw_odds, list):
        # Priorizar mercado Full time Home/Away (period Match)
        best_pair = None
        for item in raw_odds:
            if not isinstance(item, dict):
                continue
            ho = item.get("home_value") or item.get("home_odd")
            ao = item.get("away_value") or item.get("away_odd")
            if not ho or not ao:
                continue
            try:
                fh, fa = float(ho), float(ao)
            except (ValueError, TypeError):
                continue
            if fh <= 1.0 or fa <= 1.0:
                continue
            otype = str(item.get("odds_type") or "").lower()
            mper = str(item.get("market_period") or "").lower()
            mname = str(item.get("market_name") or "").lower()
            if ("home/away" in otype or "full" in mname) and ("match" in mper or "full" in mper or not mper):
                return 1.0, math.log(fa / fh)
            if best_pair is None and "spread" not in otype and "total" not in otype:
                best_pair = (fh, fa)
        if best_pair is not None:
            return 1.0, math.log(best_pair[1] / best_pair[0])
    return 0.0, 0.0


def _compute_roster_rolling(
    lineups: list[dict[str, Any]],
    player_priors: dict[str, tuple[float, float, int]],
) -> tuple[float, float]:
    """Calcula (d_rol_plus_minus, d_rol_bench_rating) en O(1) replicando exactamente el entrenamiento."""
    if not lineups or not player_priors:
        return 0.0, 0.0
    h_pm, a_pm = [], []
    h_bn_rat, a_bn_rat = [], []
    for row in lineups:
        if not isinstance(row, dict):
            continue
        pid = str(row.get("player_id") or "")
        if not pid or pid not in player_priors:
            continue
        r_rat, r_pm, n_g = player_priors[pid]
        if n_g < 2:
            continue
        side = str(row.get("team") or "").lower()
        is_st = int(row.get("is_starter") or 0)
        if side == "home":
            h_pm.append(float(r_pm))
            if not is_st:
                h_bn_rat.append(float(r_rat))
        elif side == "away":
            a_pm.append(float(r_pm))
            if not is_st:
                a_bn_rat.append(float(r_rat))
    if len(h_pm) >= 3 and len(a_pm) >= 3:
        d_rol_pm = float(np.mean(h_pm) - np.mean(a_pm)) * 0.5
        h_bn = float(np.mean(h_bn_rat)) if h_bn_rat else 6.4
        a_bn = float(np.mean(a_bn_rat)) if a_bn_rat else 6.4
        d_rol_bn = h_bn - a_bn
        return d_rol_pm, d_rol_bn
    return 0.0, 0.0


def extract_m27_v4_features(
    match_data: dict[str, Any],
    conn: sqlite3.Connection | None = None,
    snapshot_minute: float | None = None,
    match_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extrae el vector de 36 features para inferencia en vivo o histórica sin fugas ni NaNs."""
    arch, q_dur, lg_avg_ft, lg_avg_q4 = resolve_archetype(match_data)
    m_info = match_data.get("match") or {}
    mid = str(match_id or m_info.get("match_id") or match_data.get("match_id") or "")
    league = str(m_info.get("league") or match_data.get("league") or "").strip()
    country = str(m_info.get("country") or "").strip()
    ht = str(m_info.get("home_team") or "")
    at = str(m_info.get("away_team") or "")
    dt_str = str(m_info.get("date") or "").split(" ")[0]
    if not dt_str:
        dt_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    q1h, q1a = quarter_points(match_data, "Q1")
    q2h, q2a = quarter_points(match_data, "Q2")
    q3h, q3a = quarter_points(match_data, "Q3")
    q1h, q1a = int(q1h or 0), int(q1a or 0)
    q2h, q2a = int(q2h or 0), int(q2a or 0)
    q1_d = float(q1h - q1a)
    q2_d = float(q2h - q2a)
    ht_d = q1_d + q2_d

    default_snap = 33.0 if q_dur == 12 else 27.0
    max_live_snap = 38.0 if q_dur == 12 else 32.0
    min_live_snap = 25.0 if q_dur == 12 else 21.0
    status_type = str(m_info.get("status_type") or "").lower()
    is_live_match = status_type in ("inprogress", "live")
    q4h_check, q4a_check = quarter_points(match_data, "Q4")

    if snapshot_minute is None:
        if status_type == "finished" or (q4h_check is not None and q4a_check is not None and not is_live_match):
            snapshot_minute = default_snap
        else:
            played_secs = m_info.get("game_seconds_played")
            if played_secs is not None:
                inferred = int(played_secs) // 60
            else:
                inferred = infer_minute_from_pbp(match_data)
            if inferred is not None and min_live_snap <= float(inferred) <= max_live_snap:
                snapshot_minute = float(inferred)
            else:
                snapshot_minute = default_snap

    q3_end_min = 36.0 if q_dur == 12 else 30.0
    q2_end_min = 24.0 if q_dur == 12 else 20.0
    frac_q3 = max(0.0, min(1.0, (float(snapshot_minute) - q2_end_min) / float(q_dur)))

    est_h, est_a = score_upto(match_data, int(round(snapshot_minute)))

    # Si el partido está EN VIVO y evaluamos en el minuto actual, respaldar el marcador con score.home/away
    # si el feed de incidentes/PBP viene vacío o retrasado respecto al tablero oficial en vivo.
    score_obj = match_data.get("score") or {}
    live_h = int(score_obj.get("home") or 0)
    live_a = int(score_obj.get("away") or 0)

    if is_live_match and float(snapshot_minute) <= q3_end_min and (live_h + live_a) > (est_h + est_a):
        est_h, est_a = live_h, live_a
    elif (est_h + est_a) == 0 and q3h is not None and q3a is not None:
        if float(snapshot_minute) < q3_end_min:
            est_h = int(round(q1h + q2h + frac_q3 * int(q3h)))
            est_a = int(round(q1a + q2a + frac_q3 * int(q3a)))
        else:
            est_h = q1h + q2h + int(q3h)
            est_a = q1a + q2a + int(q3a)

    cur_d = float(est_h - est_a)
    cur_t = float(est_h + est_a)
    if q3h is not None and q3a is not None and float(snapshot_minute) >= q3_end_min:
        q3_end_d = ht_d + float(int(q3h) - int(q3a))
    else:
        q3_end_d = cur_d

    q3_eff_d = (q3_end_d - ht_d) if float(snapshot_minute) >= q3_end_min else (cur_d - ht_d)
    in_q4_d = (cur_d - q3_end_d) if float(snapshot_minute) > q3_end_min else 0.0

    rec_2m = recent_window_features(match_data, float(snapshot_minute), 2.0, "recent_2m")
    cur_r = float(rec_2m.get("recent_2m_points_diff", 0.0))

    pace_ppm = cur_t / max(1.0, float(snapshot_minute))
    lg_expected_ppm = lg_avg_ft / (48.0 if q_dur == 12 else 40.0)

    gp_stats = graph_stats_upto(match_data.get("graph_points") or [], int(round(snapshot_minute)))
    gp_v = float(gp_stats.get("gp_last", 0.0)) if gp_stats.get("gp_count", 0) > 0 else cur_d * 0.8
    gp_slp = float(gp_stats.get("gp_slope_3m", 0.0))

    # Desglose de canastas 2PT, 3PT y FT hasta el corte del Q3
    evs_q3 = pbp_events_upto(match_data, min(float(snapshot_minute), q3_end_min))
    if evs_q3:
        h2 = sum(1 for e in evs_q3 if e.get("team") == "home" and int(e.get("points") or 0) == 2)
        a2 = sum(1 for e in evs_q3 if e.get("team") == "away" and int(e.get("points") or 0) == 2)
        h3 = sum(1 for e in evs_q3 if e.get("team") == "home" and int(e.get("points") or 0) == 3)
        a3 = sum(1 for e in evs_q3 if e.get("team") == "away" and int(e.get("points") or 0) == 3)
        h1 = sum(1 for e in evs_q3 if e.get("team") == "home" and int(e.get("points") or 0) == 1)
        a1 = sum(1 for e in evs_q3 if e.get("team") == "away" and int(e.get("points") or 0) == 1)
        p2pt, p3pt, pft = float(h2 - a2), float(h3 - a3), float(h1 - a1)
    else:
        p2pt, p3pt, pft = cur_d * 0.30, cur_d * 0.08, cur_d * 0.15

    # ELO & Priors desde caché O(1)
    state = get_priors_state()
    t_priors = state.get("team_priors", {})
    player_priors = state.get("player_priors", {})
    hp = t_priors.get(ht, (1500.0, 1500.0, 1500.0, 0.0, 0.5, 0))
    ap = t_priors.get(at, (1500.0, 1500.0, 1500.0, 0.0, 0.5, 0))

    d_elo_ft = float(hp[0] - ap[0])
    d_elo_q4 = float(hp[1] - ap[1])
    d_elo_q1 = float(hp[2] - ap[2])
    elo_rel = min(1.0, min(int(hp[5]), int(ap[5])) / 12.0)
    d_q4_bayes = float(hp[3] - ap[3])
    d_wr_bayes = float(hp[4] - ap[4])

    # Cuotas pre-partido (desde payload en vivo o desde SQLite)
    has_odds, odds_logr = _extract_odds_from_payload(match_data.get("odds"))

    # Rolling de plantilla (Features #29 y #30) desde lineups en memoria + player_priors O(1)
    lineups_list = match_data.get("lineups") or []
    d_rol_pm, d_rol_bn = _compute_roster_rolling(lineups_list, player_priors)

    # Standings / Team Strength desde payload en memoria
    ts_d_wp_bayes, ts_d_form = 0.0, 0.0
    ts_payload = match_data.get("team_strength") or []
    if isinstance(ts_payload, list) and len(ts_payload) >= 2:
        ts_d = {}
        for row_ts in ts_payload:
            if not isinstance(row_ts, dict):
                continue
            tname = str(row_ts.get("team_name") or "")
            w = int(row_ts.get("wins") or 0)
            l = int(row_ts.get("losses") or 0)
            fm = str(row_ts.get("form") or "")
            wp_b = (w + 3.0) / (w + l + 6.0)
            fv = 0.5
            if fm:
                wc, lc_ = fm.count("W"), fm.count("L")
                if wc + lc_ > 0:
                    fv = (wc + 1.5) / (wc + lc_ + 3.0)
            ts_d[tname] = (wp_b, fv)
        if ht in ts_d and at in ts_d:
            ts_d_wp_bayes = ts_d[ht][0] - ts_d[at][0]
            ts_d_form = ts_d[ht][1] - ts_d[at][1]

    h2h_bayes_ft_m, h2h_bayes_q4_m = 0.0, 0.0
    ft_l: list[float] = []
    q4_l: list[float] = []

    if conn is not None:
        try:
            if has_odds == 0.0 and mid:
                row_o = conn.execute(
                    "SELECT home_value, away_value FROM match_odds WHERE match_id = ? AND odds_type = 'Home/Away' AND market_period = 'Match' LIMIT 1",
                    (mid,),
                ).fetchone()
                if row_o and row_o[0] and row_o[1] and float(row_o[0]) > 1.0 and float(row_o[1]) > 1.0:
                    has_odds = 1.0
                    odds_logr = math.log(float(row_o[1]) / float(row_o[0]))

            if ts_d_wp_bayes == 0.0 and ts_d_form == 0.0 and mid:
                rows_ts = conn.execute(
                    "SELECT team_name, wins, losses, form FROM team_strength WHERE match_id = ?",
                    (mid,),
                ).fetchall()
                if len(rows_ts) >= 2:
                    ts_d = {}
                    for tname, w, l, fm in rows_ts:
                        w, l = int(w or 0), int(l or 0)
                        wp_b = (w + 3.0) / (w + l + 6.0)
                        fv = 0.5
                        if fm:
                            wc, lc_ = str(fm).count("W"), str(fm).count("L")
                            if wc + lc_ > 0:
                                fv = (wc + 1.5) / (wc + lc_ + 3.0)
                        ts_d[tname] = (wp_b, fv)
                    if ht in ts_d and at in ts_d:
                        ts_d_wp_bayes = ts_d[ht][0] - ts_d[at][0]
                        ts_d_form = ts_d[ht][1] - ts_d[at][1]

            if d_rol_pm == 0.0 and d_rol_bn == 0.0 and mid and player_priors:
                db_lineups = [
                    {"team": r[0], "player_id": str(r[1]), "is_starter": int(r[2] or 0)}
                    for r in conn.execute(
                        "SELECT team, player_id, is_starter FROM lineups WHERE match_id = ? AND player_id IS NOT NULL",
                        (mid,),
                    ).fetchall()
                ]
                if db_lineups:
                    d_rol_pm, d_rol_bn = _compute_roster_rolling(db_lineups, player_priors)

            # H2H estrictamente anterior a dt_str (< dt_str)
            if ht and at and dt_str:
                rows_h2h = conn.execute(
                    """
                    SELECT m.home_team,
                           (q1.home+q2.home+q3.home+q4.home) - (q1.away+q2.away+q3.away+q4.away) as ftm,
                           (q4.home - q4.away) as q4m
                    FROM matches m
                    JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
                    JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
                    JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
                    JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
                    WHERE ((m.home_team = ? AND m.away_team = ?) OR (m.home_team = ? AND m.away_team = ?))
                      AND SUBSTR(m.date, 1, 10) < ?
                    ORDER BY m.date DESC LIMIT 8
                    """,
                    (ht, at, at, ht, dt_str),
                ).fetchall()
                for g_ht, ftm, q4m in rows_h2h:
                    if ftm is not None and q4m is not None:
                        sgn = 1.0 if g_ht == ht else -1.0
                        ft_l.append(float(ftm) * sgn)
                        q4_l.append(float(q4m) * sgn)
        except Exception:
            pass

    # Suplementar H2H desde el payload en vivo si la BD no tenía enfrentamientos
    if not ft_l:
        raw_h2h = match_data.get("h2h") or []
        if isinstance(raw_h2h, list):
            for entry in raw_h2h:
                if not isinstance(entry, dict):
                    continue
                e_dt = str(entry.get("date") or "").split(" ")[0]
                if not e_dt or e_dt >= dt_str:
                    continue
                q4_eh, q4_ea = entry.get("q4_home"), entry.get("q4_away")
                ft_eh, ft_ea = entry.get("home_score"), entry.get("away_score")
                e_ht = str(entry.get("home_team") or "")
                if q4_eh is not None and q4_ea is not None and ft_eh is not None and ft_ea is not None and e_ht in (ht, at):
                    sgn = 1.0 if e_ht == ht else -1.0
                    ft_l.append(float(int(ft_eh) - int(ft_ea)) * sgn)
                    q4_l.append(float(int(q4_eh) - int(q4_ea)) * sgn)
                    if len(ft_l) >= 8:
                        break

    if ft_l:
        shrink = len(ft_l) / (len(ft_l) + 4.0)
        h2h_bayes_ft_m = float(np.mean(ft_l)) * shrink
        h2h_bayes_q4_m = float(np.mean(q4_l)) * shrink

    # Respaldo O(1) desde ts_priors para clasificación/forma si no estuvo en el payload ni en SQLite
    if ts_d_wp_bayes == 0.0 and ts_d_form == 0.0:
        ts_priors = state.get("ts_priors", {})
        tsh = ts_priors.get(ht, (0.5, 0.5))
        tsa = ts_priors.get(at, (0.5, 0.5))
        ts_d_wp_bayes = float(tsh[0] - tsa[0])
        ts_d_form = float(tsh[1] - tsa[1])

    elo_exp = d_elo_ft * 0.035
    exp_ft_m = (0.40 * elo_exp + 0.60 * (odds_logr * 5.2)) if has_odds > 0.5 else elo_exp
    rubber_band = exp_ft_m - cur_d

    fav_is_home = exp_ft_m > 3.0
    fav_is_away = exp_ft_m < -3.0
    white_flag = 0.0
    if fav_is_home and (-26.0 <= cur_d <= -14.0):
        white_flag = -1.0
    elif fav_is_away and (14.0 <= cur_d <= 26.0):
        white_flag = 1.0
    blowout_gt = 1.0 if abs(cur_d) >= 22.0 else 0.0

    X = np.array([[
        cur_d, abs(cur_d), ht_d, q1_d, q2_d, q3_eff_d, q3_eff_d - 0.5 * ht_d, cur_r,
        pace_ppm, pace_ppm - lg_expected_ppm, gp_v, gp_slp, frac_q3,
        d_elo_ft, d_elo_q4, d_elo_q1, elo_rel, d_q4_bayes, d_wr_bayes,
        has_odds, odds_logr, exp_ft_m, h2h_bayes_ft_m, h2h_bayes_q4_m, ts_d_wp_bayes, ts_d_form,
        p2pt, p3pt, pft, d_rol_pm, d_rol_bn,
        rubber_band, white_flag, blowout_gt, in_q4_d, lg_avg_q4,
    ]], dtype=np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    meta = {
        "match_id": mid,
        "league": league,
        "country": country,
        "archetype": arch,
        "quarter_duration": q_dur,
        "snapshot_minute": float(snapshot_minute),
        "cur_score_diff": cur_d,
        "expected_ft_margin": round(exp_ft_m, 2),
        "rubber_band_gap": round(rubber_band, 2),
        "d_elo_ft": round(d_elo_ft, 1),
        "d_elo_q4": round(d_elo_q4, 1),
        "d_rol_plus_minus": round(d_rol_pm, 2),
        "has_odds": int(has_odds),
        "h2h_available": bool(ft_l),
        "lg_avg_q4": round(lg_avg_q4, 1),
    }
    return X, meta
