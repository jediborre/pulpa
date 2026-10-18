"""
EDA Profundo de Estrategias de Dosificación por Cuarto (Q1, Q2, Q3, Q4),
Perfiles de Agresividad de Equipos, Uso de Estrellas por Cuarto y Roster ELO.
Optimizado con los nuevos índices de cobertura de SQLite y acumuladores O(1) en Python.

Aborda las dos consultas del usuario:
1. Contexto de cómo juega cada ELO los cuartos (Q1, Q2, Q3, Q4) y el problema de que un mismo
   equipo histórico cambia de jugadores entre temporadas (Roster ELO vs Team Logo ELO, Missing Stars,
   profundidad de banca vs titulares, descanso/fatiga de estrellas en Q2/Q3).
2. Estrategia de dosificación de estrellas y agresividad por cuarto de cada equipo:
   - Estrellas que tienen baja participación al inicio (Q1/Q2 o Q3) por estrategia para no cansarse
     y que "le echan toda la galleta" en el Q4 (Clutch Star Share / Energy Saved).
   - Perfiles estratégicos de agresividad por cuarto de cada equipo (Sprinters Q1/Q2, Ajustadores Q3,
     Cerradores Diesel Q4) y el choque táctico (Counter-Strategy) cuando el equipo contrario neutraliza
     el cuarto fuerte del rival.
"""
import sys
import time
import sqlite3
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
from scipy.stats import pearsonr

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"


def parse_mins(m_val):
    if m_val is None:
        return 0.0
    s = str(m_val).strip()
    if not s:
        return 0.0
    if ":" in s:
        parts = s.split(":")
        try:
            return float(parts[0]) + float(parts[1]) / 60.0
        except Exception:
            return 0.0
    try:
        return float(s)
    except Exception:
        return 0.0


def fast_mean(dq, default=0.0):
    n = len(dq)
    return (sum(dq) / n) if n else default


def main():
    t0 = time.time()
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-256000;")
    cur = conn.cursor()

    print("1. Cargando partidos y marcadores Q1..Q4 (usando idx_qs_covering e idx_matches_date_id)...", flush=True)
    q_rows = cur.execute(
        "SELECT match_id, quarter, home, away FROM quarter_scores INDEXED BY idx_qs_covering WHERE quarter IN ('Q1','Q2','Q3','Q4')"
    ).fetchall()
    q_map = defaultdict(dict)
    for mid, q, h, a in q_rows:
        if h is not None and a is not None:
            q_map[mid][q] = (int(h), int(a))

    m_rows = cur.execute(
        "SELECT match_id, date, league, home_team, away_team, home_team_id, away_team_id FROM matches INDEXED BY idx_matches_date_id WHERE date IS NOT NULL ORDER BY date ASC, match_id ASC"
    ).fetchall()

    valid_matches = []
    valid_mids = set()
    for mid, dt, lg, ht, at, htid, atid in m_rows:
        qs = q_map.get(mid)
        if not qs or len(qs) < 4:
            continue
        if qs["Q4"][0] == qs["Q4"][1]:
            continue
        valid_matches.append((mid, dt, lg, ht, at, htid or ht, atid or at, qs))
        valid_mids.add(mid)

    print(f"   -> {len(valid_matches):,} partidos válidos cargados ({time.time()-t0:.1f}s)", flush=True)

    print("2. Cargando `player_stats`...", flush=True)
    ps_by_match = defaultdict(list)
    for row in cur.execute(
        "SELECT match_id, team, player_id, player_name, sofascore_rating, minutes_played, points, fouls, plus_minus FROM player_stats"
    ):
        if row[0] in valid_mids:
            ps_by_match[row[0]].append(row[1:])
    print(f"   -> {len(ps_by_match):,} partidos con player_stats ({time.time()-t0:.1f}s)", flush=True)

    print("3. Agregando puntos por Jugador y Cuarto (Q1..Q4) usando idx_pbp_scoring_player_quarter...", flush=True)
    pbp_q_player = defaultdict(lambda: defaultdict(int))
    for mid, team, q, player, pts in cur.execute(
        """
        SELECT match_id, team, quarter, player, SUM(points)
        FROM play_by_play INDEXED BY idx_pbp_scoring_player_quarter
        WHERE points > 0 AND player IS NOT NULL AND quarter IN ('Q1','Q2','Q3','Q4')
        GROUP BY match_id, team, quarter, player
        """
    ):
        if mid in valid_mids:
            pbp_q_player[(mid, team, q)][player] = int(pts)
    print(f"   -> PBP por cuarto y jugador agregado ({time.time()-t0:.1f}s)", flush=True)

    team_elo_long = defaultdict(lambda: {"ft": 1500.0, "q1": 1500.0, "q2": 1500.0, "q3": 1500.0, "q4": 1500.0, "n": 0})
    team_q_profile = defaultdict(
        lambda: {
            "q1_net": deque(maxlen=12),
            "q2_net": deque(maxlen=12),
            "q3_net": deque(maxlen=12),
            "q4_net": deque(maxlen=12),
            "q1_pts": deque(maxlen=12),
            "q2_pts": deque(maxlen=12),
            "q3_pts": deque(maxlen=12),
            "q4_pts": deque(maxlen=12),
            "recent_stars": deque(maxlen=8),
        }
    )

    player_state = defaultdict(
        lambda: {
            "elo_ft": 1500.0,
            "elo_q4": 1500.0,
            "elo_q1": 1500.0,
            "elo_q2": 1500.0,
            "elo_q3": 1500.0,
            "n": 0,
            "ratings": deque(maxlen=10),
            "pm": deque(maxlen=10),
            "pts": deque(maxlen=10),
            "mins": deque(maxlen=10),
            "q1_pts": deque(maxlen=10),
            "q2_pts": deque(maxlen=10),
            "q3_pts": deque(maxlen=10),
            "q4_pts": deque(maxlen=10),
        }
    )

    records = []

    print("4. Simulando cronológicamente ELOs por Cuarto, Estrategias de Dosificación y Estrellas...", flush=True)
    for mid, dt, lg, ht, at, hkey, akey, qs in valid_matches:
        q1_h, q1_a = qs["Q1"]
        q2_h, q2_a = qs["Q2"]
        q3_h, q3_a = qs["Q3"]
        q4_h, q4_a = qs["Q4"]

        q1_diff = q1_h - q1_a
        q2_diff = q2_h - q2_a
        q3_diff = q3_h - q3_a
        q4_diff = q4_h - q4_a
        y_q4 = 1 if q4_diff > 0 else 0
        cur_diff = q1_diff + q2_diff + q3_diff
        ft_diff = cur_diff + q4_diff

        th = team_elo_long[hkey]
        ta = team_elo_long[akey]
        tprof_h = team_q_profile[hkey]
        tprof_a = team_q_profile[akey]

        def get_team_strategy(tprof):
            if len(tprof["q4_net"]) < 4:
                return {
                    "q1_net": 0.0, "q2_net": 0.0, "q3_net": 0.0, "q4_net": 0.0,
                    "q4_vs_q1q3_surge": 0.0, "q4_pace_surge": 0.0, "q3_vs_q2_surge": 0.0,
                }
            m_q1 = fast_mean(tprof["q1_net"])
            m_q2 = fast_mean(tprof["q2_net"])
            m_q3 = fast_mean(tprof["q3_net"])
            m_q4 = fast_mean(tprof["q4_net"])
            p_q1q3 = (fast_mean(tprof["q1_pts"]) + fast_mean(tprof["q2_pts"]) + fast_mean(tprof["q3_pts"])) / 3.0
            p_q4 = fast_mean(tprof["q4_pts"])
            return {
                "q1_net": m_q1,
                "q2_net": m_q2,
                "q3_net": m_q3,
                "q4_net": m_q4,
                "q4_vs_q1q3_surge": m_q4 - ((m_q1 + m_q2 + m_q3) / 3.0),
                "q4_pace_surge": p_q4 - p_q1q3,
                "q3_vs_q2_surge": m_q3 - m_q2,
            }

        strat_h = get_team_strategy(tprof_h)
        strat_a = get_team_strategy(tprof_a)

        ps_list = ps_by_match.get(mid, [])
        if ps_list:
            side_info = {}
            for side, tkey, tprof in [("home", hkey, tprof_h), ("away", akey, tprof_a)]:
                side_ps = [r for r in ps_list if r[0] == side]
                q1_map = pbp_q_player.get((mid, side, "Q1"), {})
                q2_map = pbp_q_player.get((mid, side, "Q2"), {})
                q3_map = pbp_q_player.get((mid, side, "Q3"), {})
                q4_map = pbp_q_player.get((mid, side, "Q4"), {})

                players_today = []
                for _, pid, pname, rat, mins_str, pts, fouls, pm in side_ps:
                    pkey = pid if pid else pname
                    pst = player_state[pkey]
                    mins_val = parse_mins(mins_str)
                    avg_rat = fast_mean(pst["ratings"], 6.5)
                    avg_pm = fast_mean(pst["pm"], 0.0)
                    avg_pts = fast_mean(pst["pts"], 6.0)
                    avg_mins = fast_mean(pst["mins"], 15.0)

                    h_q1 = fast_mean(pst["q1_pts"], 1.2)
                    h_q2 = fast_mean(pst["q2_pts"], 1.2)
                    h_q3 = fast_mean(pst["q3_pts"], 1.2)
                    h_q4 = fast_mean(pst["q4_pts"], 1.2)
                    h_tot_pbp = h_q1 + h_q2 + h_q3 + h_q4

                    q4_clutch_share = (h_q4 / h_tot_pbp) if h_tot_pbp >= 4.0 else 0.25
                    q4_star_surge_hist = h_q4 - ((h_q1 + h_q2 + h_q3) / 3.0)

                    td_q1 = q1_map.get(pname, 0)
                    td_q2 = q2_map.get(pname, 0)
                    td_q3 = q3_map.get(pname, 0)
                    td_q4 = q4_map.get(pname, 0)

                    players_today.append(
                        {
                            "pkey": pkey,
                            "pname": pname,
                            "elo_ft": pst["elo_ft"],
                            "elo_q4": pst["elo_q4"],
                            "elo_q1": pst["elo_q1"],
                            "elo_q2": pst["elo_q2"],
                            "elo_q3": pst["elo_q3"],
                            "n_hist": pst["n"],
                            "avg_rat": avg_rat,
                            "avg_pm": avg_pm,
                            "avg_pts": avg_pts,
                            "avg_mins": avg_mins,
                            "h_q1": h_q1,
                            "h_q2": h_q2,
                            "h_q3": h_q3,
                            "h_q4": h_q4,
                            "q4_clutch_share": q4_clutch_share,
                            "q4_star_surge_hist": q4_star_surge_hist,
                            "td_q1": td_q1,
                            "td_q2": td_q2,
                            "td_q3": td_q3,
                            "td_q4": td_q4,
                            "_act_rat": float(rat) if rat is not None else None,
                            "_act_pm": float(pm) if pm is not None else None,
                            "_act_pts": float(pts) if pts is not None else 0.0,
                            "_act_mins": mins_val,
                        }
                    )

                players_today.sort(
                    key=lambda x: (x["n_hist"] > 0, x["elo_q4"] + x["avg_rat"] * 25.0 + x["avg_mins"] * 2.5),
                    reverse=True,
                )
                top5 = players_today[:5]
                bench = players_today[5:10]
                top3_stars = players_today[:3]

                starters_elo_ft = (sum(p["elo_ft"] for p in top5) / len(top5)) if top5 else 1500.0
                starters_elo_q4 = (sum(p["elo_q4"] for p in top5) / len(top5)) if top5 else 1500.0
                starters_elo_q2 = (sum(p["elo_q2"] for p in top5) / len(top5)) if top5 else 1500.0
                starters_elo_q3 = (sum(p["elo_q3"] for p in top5) / len(top5)) if top5 else 1500.0
                bench_elo_q4 = (sum(p["elo_q4"] for p in bench) / len(bench)) if bench else 1500.0
                top3_elo_q4 = (sum(p["elo_q4"] for p in top3_stars) / len(top3_stars)) if top3_stars else 1500.0

                stars_q4_surge_hist = sum(p["q4_star_surge_hist"] for p in top3_stars)
                stars_expected_q4_pts = sum(p["h_q4"] for p in top3_stars)

                stars_exp_q1q3 = sum(p["h_q1"] + p["h_q2"] + p["h_q3"] for p in top3_stars)
                stars_act_q1q3 = sum(p["td_q1"] + p["td_q2"] + p["td_q3"] for p in top3_stars)
                stars_energy_saved_q1q3 = stars_exp_q1q3 - stars_act_q1q3

                stars_exp_q3 = sum(p["h_q3"] for p in top3_stars)
                stars_act_q3 = sum(p["td_q3"] for p in top3_stars)
                stars_saved_in_q3 = stars_exp_q3 - stars_act_q3

                missing_star_loss = 0.0
                if len(tprof["recent_stars"]) >= 3:
                    p_freq = defaultdict(list)
                    for g_stars in tprof["recent_stars"]:
                        for pk, pelo in g_stars:
                            p_freq[pk].append(pelo)
                    usual_top = sorted(
                        [(pk, sum(el) / len(el)) for pk, el in p_freq.items() if len(el) >= 2],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    active_today_keys = {p["pkey"] for p in players_today if p["_act_mins"] > 0}
                    for pk, pelo in usual_top:
                        if pk not in active_today_keys:
                            missing_star_loss += max(0.0, pelo - 1500.0)

                side_info[side] = {
                    "starters_elo_ft": starters_elo_ft,
                    "starters_elo_q4": starters_elo_q4,
                    "starters_elo_q2": starters_elo_q2,
                    "starters_elo_q3": starters_elo_q3,
                    "bench_elo_q4": bench_elo_q4,
                    "top3_elo_q4": top3_elo_q4,
                    "stars_q4_surge_hist": stars_q4_surge_hist,
                    "stars_expected_q4_pts": stars_expected_q4_pts,
                    "stars_energy_saved_q1q3": stars_energy_saved_q1q3,
                    "stars_saved_in_q3": stars_saved_in_q3,
                    "missing_star_loss": missing_star_loss,
                    "players_today": players_today,
                }

            sh = side_info["home"]
            sa = side_info["away"]

            if th["n"] >= 5 and ta["n"] >= 5:
                records.append(
                    {
                        "mid": mid,
                        "cur_diff": cur_diff,
                        "q1_diff": q1_diff,
                        "q2_diff": q2_diff,
                        "q3_diff": q3_diff,
                        "q4_diff": q4_diff,
                        "y_q4": y_q4,
                        "d_team_elo_ft": th["ft"] - ta["ft"],
                        "d_team_elo_q4": th["q4"] - ta["q4"],
                        "d_team_elo_q2": th["q2"] - ta["q2"],
                        "d_team_elo_q3": th["q3"] - ta["q3"],
                        "d_lineup_starters_elo_ft": sh["starters_elo_ft"] - sa["starters_elo_ft"],
                        "d_lineup_starters_elo_q4": sh["starters_elo_q4"] - sa["starters_elo_q4"],
                        "d_lineup_starters_elo_q2": sh["starters_elo_q2"] - sa["starters_elo_q2"],
                        "d_lineup_starters_elo_q3": sh["starters_elo_q3"] - sa["starters_elo_q3"],
                        "d_lineup_bench_elo_q4": sh["bench_elo_q4"] - sa["bench_elo_q4"],
                        "d_lineup_top3_elo_q4": sh["top3_elo_q4"] - sa["top3_elo_q4"],
                        "d_missing_star_adv": sa["missing_star_loss"] - sh["missing_star_loss"],
                        "d_strat_q4_net": strat_h["q4_net"] - strat_a["q4_net"],
                        "d_strat_q4_vs_q1q3_surge": strat_h["q4_vs_q1q3_surge"] - strat_a["q4_vs_q1q3_surge"],
                        "d_strat_q4_pace_surge": strat_h["q4_pace_surge"] - strat_a["q4_pace_surge"],
                        "d_lineup_q4_minus_q3_elo": (sh["starters_elo_q4"] - sh["starters_elo_q3"]) - (sa["starters_elo_q4"] - sa["starters_elo_q3"]),
                        "d_stars_expected_q4_pts": sh["stars_expected_q4_pts"] - sa["stars_expected_q4_pts"],
                        "d_stars_q4_surge_hist": sh["stars_q4_surge_hist"] - sa["stars_q4_surge_hist"],
                        "d_stars_energy_saved_q1q3": sh["stars_energy_saved_q1q3"] - sa["stars_energy_saved_q1q3"],
                        "d_stars_saved_in_q3": sh["stars_saved_in_q3"] - sa["stars_saved_in_q3"],
                        "d_star_slingshot_q4": (
                            (sh["stars_saved_in_q3"] * max(0.0, (sh["top3_elo_q4"] - 1480.0) / 100.0))
                            - (sa["stars_saved_in_q3"] * max(0.0, (sa["top3_elo_q4"] - 1480.0) / 100.0))
                        ),
                        "q3_tactical_surprise": q3_diff - (strat_h["q3_net"] - strat_a["q3_net"]),
                    }
                )

            for side, tprof, won_ft, won_q1, won_q2, won_q3, won_q4 in [
                ("home", tprof_h, 1.0 if ft_diff > 0 else 0.0, 1.0 if q1_diff > 0 else 0.5 if q1_diff == 0 else 0.0, 1.0 if q2_diff > 0 else 0.5 if q2_diff == 0 else 0.0, 1.0 if q3_diff > 0 else 0.5 if q3_diff == 0 else 0.0, float(y_q4)),
                ("away", tprof_a, 0.0 if ft_diff > 0 else 1.0, 0.0 if q1_diff > 0 else 0.5 if q1_diff == 0 else 1.0, 0.0 if q2_diff > 0 else 0.5 if q2_diff == 0 else 1.0, 0.0 if q3_diff > 0 else 0.5 if q3_diff == 0 else 1.0, 1.0 - float(y_q4)),
            ]:
                opp_side = "away" if side == "home" else "home"
                opp_ft_elo = side_info[opp_side]["starters_elo_ft"]
                opp_q4_elo = side_info[opp_side]["starters_elo_q4"]
                opp_q3_elo = side_info[opp_side]["starters_elo_q3"]
                core_game = []
                for p in side_info[side]["players_today"]:
                    if p["_act_mins"] >= 4.0 or p["_act_pts"] > 0 or p["_act_rat"] is not None:
                        pst = player_state[p["pkey"]]
                        min_w = min(1.25, max(0.25, p["_act_mins"] / 24.0))
                        k_p = 15.0 * min_w
                        exp_ft = 1.0 / (1.0 + 10.0 ** ((opp_ft_elo - pst["elo_ft"]) / 400.0))
                        exp_q4 = 1.0 / (1.0 + 10.0 ** ((opp_q4_elo - pst["elo_q4"]) / 400.0))
                        exp_q3 = 1.0 / (1.0 + 10.0 ** ((opp_q3_elo - pst["elo_q3"]) / 400.0))
                        pm_val = p["_act_pm"] or 0.0
                        pm_bonus = 0.35 if pm_val > 8.75 else (-0.35 if pm_val < -8.75 else pm_val / 25.0)
                        pst["elo_ft"] += k_p * ((won_ft - exp_ft) + 0.25 * pm_bonus)
                        pst["elo_q4"] += k_p * ((won_q4 - exp_q4) + 0.25 * pm_bonus)
                        pst["elo_q3"] += k_p * (won_q3 - exp_q3)
                        pst["elo_q1"] += k_p * (won_q1 - 0.5)
                        pst["elo_q2"] += k_p * (won_q2 - 0.5)
                        pst["n"] += 1
                        if p["_act_rat"] is not None:
                            pst["ratings"].append(p["_act_rat"])
                        if p["_act_pm"] is not None:
                            pst["pm"].append(p["_act_pm"])
                        pst["pts"].append(p["_act_pts"])
                        pst["mins"].append(p["_act_mins"])
                        pst["q1_pts"].append(p["td_q1"])
                        pst["q2_pts"].append(p["td_q2"])
                        pst["q3_pts"].append(p["td_q3"])
                        pst["q4_pts"].append(p["td_q4"])
                        if p["_act_mins"] >= 16.0:
                            core_game.append((p["pkey"], pst["elo_q4"]))
                if core_game:
                    tprof["recent_stars"].append(core_game)

        for q_tag, diff_val in [("ft", ft_diff), ("q1", q1_diff), ("q2", q2_diff), ("q3", q3_diff), ("q4", q4_diff)]:
            res_h = 1.0 if diff_val > 0 else (0.5 if diff_val == 0 else 0.0)
            exp_h = 1.0 / (1.0 + 10.0 ** ((ta[q_tag] - th[q_tag]) / 400.0))
            th[q_tag] += 20.0 * (res_h - exp_h)
            ta[q_tag] += 20.0 * ((1.0 - res_h) - (1.0 - exp_h))
        th["n"] += 1
        ta["n"] += 1

        tprof_h["q1_net"].append(q1_diff)
        tprof_h["q2_net"].append(q2_diff)
        tprof_h["q3_net"].append(q3_diff)
        tprof_h["q4_net"].append(q4_diff)
        tprof_h["q1_pts"].append(q1_h)
        tprof_h["q2_pts"].append(q2_h)
        tprof_h["q3_pts"].append(q3_h)
        tprof_h["q4_pts"].append(q4_h)

        tprof_a["q1_net"].append(-q1_diff)
        tprof_a["q2_net"].append(-q2_diff)
        tprof_a["q3_net"].append(-q3_diff)
        tprof_a["q4_net"].append(-q4_diff)
        tprof_a["q1_pts"].append(q1_a)
        tprof_a["q2_pts"].append(q2_a)
        tprof_a["q3_pts"].append(q3_a)
        tprof_a["q4_pts"].append(q4_a)

    print(f"   -> Simulación completada en {time.time()-t0:.1f}s | Partidos con alineación completa: {len(records):,}", flush=True)

    print("\n" + "=" * 95)
    print("RESULTADO 1: CORRELACIÓN DE NUEVAS VARIABLES DE PLANTILLA Y ESTRATEGIA POR CUARTO vs Q4")
    print("=" * 95)
    q4_all = np.array([r["q4_diff"] for r in records])
    close_mask = np.array([abs(r["cur_diff"]) <= 12 for r in records])

    feat_list = [
        ("d_team_elo_ft", "ELO Equipo FT (Logo Histórico)"),
        ("d_team_elo_q4", "ELO Equipo Q4 (Logo Histórico)"),
        ("d_team_elo_q2", "ELO Equipo Q2 (Logo Histórico)"),
        ("d_team_elo_q3", "ELO Equipo Q3 (Logo Histórico)"),
        ("d_lineup_starters_elo_ft", "ELO Titulares de HOY (FT)"),
        ("d_lineup_starters_elo_q4", "ELO Titulares de HOY (Específico Q4)"),
        ("d_lineup_starters_elo_q2", "ELO Titulares de HOY (Específico Q2)"),
        ("d_lineup_starters_elo_q3", "ELO Titulares de HOY (Específico Q3)"),
        ("d_lineup_top3_elo_q4", "ELO Top-3 Estrellas de HOY (Específico Q4)"),
        ("d_lineup_bench_elo_q4", "ELO Banca (#6-#10) de HOY (Específico Q4)"),
        ("d_missing_star_adv", "Ventaja por Estrella Ausente del Rival HOY"),
        ("d_strat_q4_net", "Agresividad Neta Q4 Reciente (12 juegos)"),
        ("d_strat_q4_vs_q1q3_surge", "Estrategia Diesel Equipo (Salto Q4 vs Q1-Q3)"),
        ("d_strat_q4_pace_surge", "Aceleración Ofensiva Q4 vs Q1-Q3 del Equipo"),
        ("d_lineup_q4_minus_q3_elo", "Salto ELO Q4 vs Q3 de Titulares de HOY"),
        ("d_stars_expected_q4_pts", "Puntos Esperados en Q4 de las Top-3 Estrellas"),
        ("d_stars_q4_surge_hist", "Dosificación Histórica de Estrellas (Q4 - Q1..3)"),
        ("d_stars_energy_saved_q1q3", "Energía Guardada por Estrellas Hoy en Q1-Q3"),
        ("d_stars_saved_in_q3", "Dosificación de Estrellas Hoy en el Q3"),
        ("d_star_slingshot_q4", "Efecto Resorte Estrella (Dosificó Q3 * ELO Q4)"),
        ("q3_tactical_surprise", "Sorpresa Táctica en Q3 vs Guion del Rival"),
    ]

    print(f"{'FeatureCode':<27} | {'Descripción':<44} | {'r(q4 All)':>9} | {'r(|diff|<=12)':>13}")
    print("-" * 102)
    for fk, desc in feat_list:
        vals = np.array([r[fk] for r in records], dtype=float)
        r_all, _ = pearsonr(vals, q4_all)
        r_cl, _ = pearsonr(vals[close_mask], q4_all[close_mask])
        print(f"{fk:<27} | {desc:<44} | {r_all:>+9.4f} | {r_cl:>+13.4f}")

    print("\n" + "=" * 95)
    print("RESULTADO 2: COMPARATIVA DIRECTA: ¿ROSTER/LINEUP ELO SUPERA AL LOGO ELO CUANDO CAMBIAN JUGADORES?")
    print("=" * 95)
    c1 = [r for r in records if abs(r["cur_diff"]) <= 10 and r["d_team_elo_q4"] > 30 and r["d_lineup_starters_elo_q4"] > 40]
    c2 = [r for r in records if abs(r["cur_diff"]) <= 10 and r["d_team_elo_q4"] > 30 and r["d_lineup_starters_elo_q4"] < -20]
    c3 = [r for r in records if abs(r["cur_diff"]) <= 10 and r["d_team_elo_q4"] < -30 and r["d_lineup_starters_elo_q4"] > 30]
    print(f"  - Logo Q4 ELO Local (+30) + Alineación de Hoy Local (+40) [N={len(c1):,}]: WR Q4 Local = {np.mean([r['y_q4'] for r in c1])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in c1]):+.2f}")
    print(f"  - Logo Q4 ELO dice Local (+30), PERO Alineación de Hoy favorece Visita (<-20) [N={len(c2):,}]: WR Q4 Local cae a = {np.mean([r['y_q4'] for r in c2])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in c2]):+.2f}")
    print(f"  - Logo Q4 ELO dice Visita (<-30), PERO Alineación de Hoy favorece Local (>+30) [N={len(c3):,}]: WR Q4 Local sube a = {np.mean([r['y_q4'] for r in c3])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in c3]):+.2f}")

    print("\n" + "=" * 95)
    print("RESULTADO 3: ESTRATEGIA DE DOSIFICACIÓN DE ESTRELLAS ('GUARDAR GALLETA PARA EL Q4')")
    print("=" * 95)
    s_base = [r for r in records if -12 <= r["cur_diff"] <= 2 and r["d_lineup_top3_elo_q4"] > 35]
    s_dosified = [r for r in s_base if r["d_stars_energy_saved_q1q3"] > 3.0 and r["d_stars_expected_q4_pts"] > 1.5]
    s_diesel_team = [r for r in s_dosified if r["d_strat_q4_vs_q1q3_surge"] > 1.0]
    s_plus_bench = [r for r in s_dosified if r["d_lineup_bench_elo_q4"] > 15]

    print(f"  - Base (Marcador -12 a +2, Top-3 Estrellas con ELO Q4 > +35) [N={len(s_base):,}]: WR Q4 = {np.mean([r['y_q4'] for r in s_base])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in s_base]):+.2f}")
    print(f"  - + Estrellas dosificadas en Q1-Q3 (ahorraron >3 pts de energía y son anotadoras de Q4) [N={len(s_dosified):,}]: WR Q4 = {np.mean([r['y_q4'] for r in s_dosified])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in s_dosified]):+.2f}")
    print(f"  - + Equipo con Estrategia Diesel (acelera en Q4 vs Q1-Q3) [N={len(s_diesel_team):,}]: WR Q4 = {np.mean([r['y_q4'] for r in s_diesel_team])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in s_diesel_team]):+.2f}")
    print(f"  - + Banca de Hoy también superior (Bench ELO Q4 > +15) [N={len(s_plus_bench):,}]: WR Q4 = {np.mean([r['y_q4'] for r in s_plus_bench])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in s_plus_bench]):+.2f}")

    print("\n" + "=" * 95)
    print("RESULTADO 4: CHOQUE TÁCTICO POR CUARTO (EQUIPO AGRESIVO EN Q4 vs EQUIPO QUE YA GASTÓ SU CUARTO FUERTE)")
    print("=" * 95)
    tactical_clash = [
        r for r in records
        if -12 <= r["cur_diff"] <= -1
        and r["d_lineup_starters_elo_q4"] > 35
        and r["d_lineup_bench_elo_q4"] > 15
        and r["d_strat_q4_net"] > 1.5
    ]
    print(f"  - Remontada Táctica en Q4 (Perdiendo 1-12 pts al m30, pero con Titulares ELO Q4 >+35, Banca ELO Q4 >+15 y Estrategia Q4 >+1.5) [N={len(tactical_clash):,}]:")
    if tactical_clash:
        print(f"    -> WR Q4 = {np.mean([r['y_q4'] for r in tactical_clash])*100:.2f}% | Margen Q4 Promedio = {np.mean([r['q4_diff'] for r in tactical_clash]):+.2f} pts")


if __name__ == "__main__":
    main()
