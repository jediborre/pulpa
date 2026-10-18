"""
Investigación profunda de patrones de Jugadores, Alineaciones (Lineups), Rotación por Cuarto
(Q1, Q2, Q3, Q4) y ELO Dinámico de Plantilla (Roster ELO vs Team Logo ELO) en matches.db.

Objetivo:
Responder a la hipótesis central del usuario:
1. "Un mismo equipo histórico no tiene los mismos jugadores en las temporadas":
   Comparar el poder predictivo del ELO tradicional del equipo (Team Logo ELO) contra un
   **Roster/Lineup ELO** construido directamente a partir del ELO individual de los jugadores
   presentes en la alineación de HOY (`lineups` / `player_stats`), ponderado por su rol
   (titular vs banca y minutos esperados).
2. "Darle contexto de cómo juegan los Q's cada ELO y buscar patrones reales de juego (en Q2 están
   las estrellas, en Q3 descansan, etc.)":
   Cruzar `player_stats` + `lineups` + `match_events` / `play_by_play` + `quarter_scores` para medir:
   - Patrón 1 (Roster Drift / Cambio de Plantilla): ¿Qué pasa cuando un equipo cambia de jugadores
     o juega sin su Estrella #1/#2 (Missing Star Shock)?
   - Patrón 2 (Huella de Cuartos Q1, Q2, Q3, Q4 del Equipo y de la Plantilla): ELO por cuarto
     (`elo_q1`, `elo_q2`, `elo_q3`, `elo_q4`) de ventana corta (temporada actual) vs largo plazo.
   - Patrón 3 (Presencia / Descanso de Estrellas en Q2 vs Q3 vía PBP):
     ¿Qué pasa en el Q4 cuando los Top-2 jugadores de mayor ELO del equipo descansan o bajan su
     participación en la segunda mitad del Q3 (entran frescos al Q4) vs cuando cargan con todo el Q3
     (llegan fundidos al Q4)?
   - Patrón 4 (Estrella Fría / Regresión a la Media Individual en Q1-Q3):
     Cuando la estrella titular (Rating > 7.2 / alto PPG histórico) lleva pocos puntos en Q1-Q3 pero
     el partido sigue vivo, ¿explota en el Q4?
   - Patrón 5 (Profundidad de Banca vs Dependencia de 1-2 Estrellas / Gini de Plantilla):
     ¿Cómo impacta tener una banca profunda (jugadores #6-#10 con alto ELO/ +/-) en los primeros
     minutos del Q4 (m30-m34) y en el cierre del Q4?
   - Patrón 6 (Peligro de Faltas / Foul Trouble de Titulares):
     Impacto de las faltas acumuladas por los titulares en el rendimiento del Q4.
"""
import sys
import sqlite3
from pathlib import Path
from collections import defaultdict, deque
import numpy as np
from scipy.stats import pearsonr

sys.stdout.reconfigure(encoding="utf-8")

DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"


def main():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    print("=" * 95)
    print("1. INVENTARIO DE TABLAS DE JUGADORES, ALINEACIONES Y EVENTOS EN matches.db")
    print("=" * 95)
    for tbl in ["player_stats", "lineups", "match_events", "play_by_play", "team_statistics"]:
        cnt = cur.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        m_cnt = cur.execute(f"SELECT COUNT(DISTINCT match_id) FROM {tbl}").fetchone()[0]
        cols = [r[1] for r in cur.execute(f"PRAGMA table_info({tbl})").fetchall()]
        print(f"  - {tbl:<16}: {cnt:>10,} filas | {m_cnt:>7,} partidos únicos | cols={cols[:8]}...")

    # Check incident types in match_events
    inc_types = cur.execute(
        "SELECT incident_type, subtype, COUNT(*) FROM match_events GROUP BY incident_type, subtype ORDER BY COUNT(*) DESC LIMIT 12"
    ).fetchall()
    print("\n  Tipos de incidentes en `match_events`:")
    for it, st, c in inc_types:
        print(f"    * {str(it):<15} | subtype={str(st):<15} -> {c:>10,}")

    # Load matches + quarter_scores chronologically
    print("\nCargando partidos cronológicos con Q1..Q4...")
    q_rows = cur.execute(
        "SELECT match_id, quarter, home, away FROM quarter_scores WHERE quarter IN ('Q1','Q2','Q3','Q4')"
    ).fetchall()
    q_map = defaultdict(dict)
    for mid, q, h, a in q_rows:
        if h is not None and a is not None:
            q_map[mid][q] = (int(h), int(a))

    m_rows = cur.execute(
        "SELECT match_id, date, league, home_team, away_team, home_team_id, away_team_id FROM matches WHERE date IS NOT NULL ORDER BY date ASC, match_id ASC"
    ).fetchall()

    valid_matches = []
    for mid, dt, lg, ht, at, htid, atid in m_rows:
        qs = q_map.get(mid)
        if not qs or len(qs) < 4:
            continue
        if qs["Q4"][0] == qs["Q4"][1]:
            continue
        valid_matches.append((mid, dt, lg, ht, at, htid or ht, atid or at, qs))

    print(f"  -> Partidos válidos con Q1..Q4 sin empate en Q4: {len(valid_matches):,}")

    # Load player_stats grouped by match_id
    print("Cargando `player_stats` y `lineups`...")
    ps_by_match = defaultdict(list)
    for row in cur.execute(
        "SELECT match_id, team, player_id, player_name, sofascore_rating, minutes_played, points, fouls, plus_minus, rebounds, assists, turnovers FROM player_stats"
    ):
        ps_by_match[row[0]].append(row[1:])

    starter_map = {}
    for mid, team, pid, pname, is_st in cur.execute(
        "SELECT match_id, team, player_id, player_name, is_starter FROM lineups"
    ):
        key = pid if pid else pname
        starter_map[(mid, team, key)] = int(is_st or 0)

    # Load PBP player activity by quarter (Q1, Q2, Q3)
    # Using play_by_play (and match_events where player_id exists)
    print("Cargando actividad de jugadores por cuarto (Q1, Q2, Q3) desde `play_by_play`...")
    pbp_q_pts = defaultdict(lambda: defaultdict(int))  # (mid, team, q) -> {player_name: pts}
    pbp_q3_late_pts = defaultdict(lambda: defaultdict(int))  # (mid, team) -> {player_name: pts in last 4m of Q3}

    # Parse time helper
    for mid, q, tm, player, pts, team in cur.execute(
        "SELECT match_id, quarter, time, player, points, team FROM play_by_play WHERE quarter IN ('Q1','Q2','Q3') AND player IS NOT NULL AND points > 0"
    ):
        if mid not in q_map:
            continue
        pbp_q_pts[(mid, team, q)][player] += int(pts)
        if q == "Q3" and tm and ":" in str(tm):
            try:
                mm = int(str(tm).split(":")[0])
                if mm <= 4:  # Last 4:59 of Q3
                    pbp_q3_late_pts[(mid, team)][player] += int(pts)
            except Exception:
                pass

    print("Calculando ELO de Equipos (Largo vs Corto Plazo por Q1..Q4) y ELO Individual de Jugadores...")

    # State trackers:
    # 1. Team ELOs: Long-term (K=20, decay) vs Short-term Season/Form ELO (K=36, fast adaptation to new roster)
    team_elo_long = defaultdict(lambda: {"ft": 1500.0, "q1": 1500.0, "q2": 1500.0, "q3": 1500.0, "q4": 1500.0, "n": 0})
    team_elo_short = defaultdict(lambda: {"ft": 1500.0, "q1": 1500.0, "q2": 1500.0, "q3": 1500.0, "q4": 1500.0})
    # Team recent roster history: track top players of each team in last 8 games
    team_recent_rosters = defaultdict(lambda: deque(maxlen=8))  # team_key -> deque of sets of (player_key, impact_score)

    # 2. Individual Player ELO & Rolling Stats:
    # Each player has their own ELO for FT, Q1, Q2, Q3, Q4 (updated when they play >=10 mins),
    # plus rolling PPG, rating, plus_minus, minutes, and per-quarter scoring share (Q1, Q2, Q3)!
    player_state = defaultdict(
        lambda: {
            "elo_ft": 1500.0,
            "elo_q4": 1500.0,
            "elo_q1": 1500.0,
            "elo_q2": 1500.0,
            "elo_q3": 1500.0,
            "n": 0,
            "ratings": deque(maxlen=12),
            "pm": deque(maxlen=12),
            "pts": deque(maxlen=12),
            "mins": deque(maxlen=12),
            "q1_pts": deque(maxlen=12),
            "q2_pts": deque(maxlen=12),
            "q3_pts": deque(maxlen=12),
        }
    )

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

    records = []

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
        cur_diff = (q1_h + q2_h + q3_h) - (q1_a + q2_a + q3_a)
        ft_diff = cur_diff + q4_diff

        th_L = team_elo_long[hkey]
        ta_L = team_elo_long[akey]
        th_S = team_elo_short[hkey]
        ta_S = team_elo_short[akey]

        # Extract player features if match has player_stats
        ps_list = ps_by_match.get(mid, [])
        if ps_list:
            side_data = {}
            for side, tkey in [("home", hkey), ("away", akey)]:
                side_ps = [r for r in ps_list if r[0] == side]
                # Pre-match priors for players dressed today
                dressed_players = []
                for _, pid, pname, rat, mins_str, pts, fouls, pm, reb, ast, tov in side_ps:
                    pkey = pid if pid else pname
                    mins_actual = parse_mins(mins_str)
                    is_st = starter_map.get((mid, side, pkey), 1 if mins_actual >= 20 else 0)
                    pst = player_state[pkey]
                    avg_rat = float(np.mean(pst["ratings"])) if pst["ratings"] else 6.5
                    avg_pm = float(np.mean(pst["pm"])) if pst["pm"] else 0.0
                    avg_pts = float(np.mean(pst["pts"])) if pst["pts"] else 6.0
                    avg_mins = float(np.mean(pst["mins"])) if pst["mins"] else (22.0 if is_st else 10.0)
                    avg_q1 = float(np.mean(pst["q1_pts"])) if pst["q1_pts"] else 1.5
                    avg_q2 = float(np.mean(pst["q2_pts"])) if pst["q2_pts"] else 1.5
                    avg_q3 = float(np.mean(pst["q3_pts"])) if pst["q3_pts"] else 1.5

                    # In-game Q1, Q2, Q3 points from PBP for this player today
                    q1_today = pbp_q_pts[(mid, side, "Q1")].get(pname, 0)
                    q2_today = pbp_q_pts[(mid, side, "Q2")].get(pname, 0)
                    q3_today = pbp_q_pts[(mid, side, "Q3")].get(pname, 0)
                    q3_late_today = pbp_q3_late_pts[(mid, side)].get(pname, 0)

                    dressed_players.append(
                        {
                            "pkey": pkey,
                            "pname": pname,
                            "is_st": is_st,
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
                            "avg_q1": avg_q1,
                            "avg_q2": avg_q2,
                            "avg_q3": avg_q3,
                            "q1_today": q1_today,
                            "q2_today": q2_today,
                            "q3_today": q3_today,
                            "q3_late_today": q3_late_today,
                            # Post-match actuals for updating state AFTER recording features
                            "_actual_rat": float(rat) if rat is not None else None,
                            "_actual_pm": float(pm) if pm is not None else None,
                            "_actual_pts": float(pts) if pts is not None else 0.0,
                            "_actual_mins": mins_actual,
                            "_actual_fouls": int(fouls) if fouls is not None else 0,
                        }
                    )

                # Sort dressed players by pre-match importance (expected minutes * rating / ELO)
                dressed_players.sort(key=lambda x: (x["n_hist"] > 0, x["elo_q4"] + x["avg_rat"] * 25.0 + x["avg_mins"] * 2.0), reverse=True)
                top5 = dressed_players[:5]
                bench = dressed_players[5:10]
                top2_stars = dressed_players[:2]

                # 1. Lineup-based ELO (weighted by expected minutes of dressed players)
                if top5:
                    starters_elo_ft = float(np.mean([p["elo_ft"] for p in top5]))
                    starters_elo_q4 = float(np.mean([p["elo_q4"] for p in top5]))
                    starters_elo_q2 = float(np.mean([p["elo_q2"] for p in top5]))
                    starters_elo_q3 = float(np.mean([p["elo_q3"] for p in top5]))
                else:
                    starters_elo_ft = starters_elo_q4 = starters_elo_q2 = starters_elo_q3 = 1500.0

                bench_elo_q4 = float(np.mean([p["elo_q4"] for p in bench])) if bench else 1500.0
                bench_pm = float(np.mean([p["avg_pm"] for p in bench])) if bench else 0.0
                top2_elo_q4 = float(np.mean([p["elo_q4"] for p in top2_stars])) if top2_stars else 1500.0

                # 2. Roster Continuity & Missing Star Shock:
                # Compare dressed players today against team's recent 8-game core stars
                recent_core = team_recent_rosters[tkey]
                missing_star_elo_loss = 0.0
                roster_continuity = 1.0
                if len(recent_core) >= 3:
                    # Find players who appeared in >=40% of recent games
                    player_counts = defaultdict(list)
                    for game_roster in recent_core:
                        for pk, p_elo in game_roster:
                            player_counts[pk].append(p_elo)
                    core_stars = sorted(
                        [(pk, float(np.mean(elos)), len(elos)) for pk, elos in player_counts.items() if len(elos) >= 2],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    dressed_keys = {p["pkey"] for p in dressed_players if p["_actual_mins"] > 0 or p["is_st"] == 1}
                    if core_stars:
                        present = sum(1 for pk, _, _ in core_stars if pk in dressed_keys)
                        roster_continuity = present / len(core_stars)
                        for pk, star_elo, _ in core_stars:
                            if pk not in dressed_keys:
                                missing_star_elo_loss += max(0.0, star_elo - 1500.0)

                # 3. Star Q1-Q3 Activity vs Rest Pattern in PBP:
                # Did the top 2 stars play/score heavily in Q1+Q2, and rest in late Q3 (fresh for Q4)?
                # Or did they carry 100% of Q3 (fatigued) or underperform their usual scoring (cold star regression)?
                star_q1q2_pts = sum(p["q1_today"] + p["q2_today"] for p in top2_stars)
                star_q3_pts = sum(p["q3_today"] for p in top2_stars)
                star_q3_late_pts = sum(p["q3_late_today"] for p in top2_stars)
                star_expected_q1q3_pts = sum(p["avg_q1"] + p["avg_q2"] + p["avg_q3"] for p in top2_stars)
                star_actual_q1q3_pts = star_q1q2_pts + star_q3_pts
                # Scoring deficit of stars vs their own average (positive = stars scored LESS than usual in Q1-Q3 -> due for Q4 regression/fresh)
                star_scoring_reserve = star_expected_q1q3_pts - star_actual_q1q3_pts
                # Rest in late Q3: stars who were active in Q1+Q2 (>=6 pts) but scored 0 in last 4 mins of Q3 (sat on bench before Q4!)
                star_rested_late_q3 = 1.0 if (star_q1q2_pts >= 6 and star_q3_late_pts == 0) else 0.0
                # Overloaded star in Q3: top 2 stars scored >= 10 pts in Q3 alone (played entire Q3 at high usage -> Q4 fatigue)
                star_overloaded_q3 = 1.0 if (star_q3_pts >= 10 and star_q3_late_pts >= 5) else 0.0

                # 4. Bench Q2+Q3 scoring contribution (is the bench hot today?)
                bench_q1q3_pts = sum(p["q1_today"] + p["q2_today"] + p["q3_today"] for p in bench)

                side_data[side] = {
                    "starters_elo_ft": starters_elo_ft,
                    "starters_elo_q4": starters_elo_q4,
                    "starters_elo_q2": starters_elo_q2,
                    "starters_elo_q3": starters_elo_q3,
                    "bench_elo_q4": bench_elo_q4,
                    "bench_pm": bench_pm,
                    "top2_elo_q4": top2_elo_q4,
                    "missing_star_elo_loss": missing_star_elo_loss,
                    "roster_continuity": roster_continuity,
                    "star_scoring_reserve": star_scoring_reserve,
                    "star_rested_late_q3": star_rested_late_q3,
                    "star_overloaded_q3": star_overloaded_q3,
                    "bench_q1q3_pts": bench_q1q3_pts,
                    "dressed_players": dressed_players,
                }

            sh = side_data["home"]
            sa = side_data["away"]

            if th_L["n"] >= 5 and ta_L["n"] >= 5:
                records.append(
                    {
                        "mid": mid,
                        "cur_diff": cur_diff,
                        "q1_diff": q1_diff,
                        "q2_diff": q2_diff,
                        "q3_diff": q3_diff,
                        "q4_diff": q4_diff,
                        "y_q4": y_q4,
                        # Team Logo ELOs (Long vs Short)
                        "d_team_elo_ft_long": th_L["ft"] - ta_L["ft"],
                        "d_team_elo_q4_long": th_L["q4"] - ta_L["q4"],
                        "d_team_elo_q2_long": th_L["q2"] - ta_L["q2"],
                        "d_team_elo_q3_long": th_L["q3"] - ta_L["q3"],
                        "d_team_elo_ft_short": th_S["ft"] - ta_S["ft"],
                        "d_team_elo_q4_short": th_S["q4"] - ta_S["q4"],
                        # Team Quarter Profile (Q4 closer vs Q3/Q1)
                        "d_team_q4_vs_q3_elo": (th_L["q4"] - th_L["q3"]) - (ta_L["q4"] - ta_L["q3"]),
                        "d_team_q4_vs_q1_elo": (th_L["q4"] - th_L["q1"]) - (ta_L["q4"] - ta_L["q1"]),
                        # Lineup / Player-Driven ELOs
                        "d_lineup_starters_elo_ft": sh["starters_elo_ft"] - sa["starters_elo_ft"],
                        "d_lineup_starters_elo_q4": sh["starters_elo_q4"] - sa["starters_elo_q4"],
                        "d_lineup_bench_elo_q4": sh["bench_elo_q4"] - sa["bench_elo_q4"],
                        "d_lineup_top2_elo_q4": sh["top2_elo_q4"] - sa["top2_elo_q4"],
                        "d_lineup_bench_pm": sh["bench_pm"] - sa["bench_pm"],
                        # Roster Drift: difference between Today's Lineup ELO and Historical Team Logo ELO!
                        "d_roster_vs_logo_elo": (sh["starters_elo_ft"] - sa["starters_elo_ft"]) - (th_L["ft"] - ta_L["ft"]),
                        # Missing Star Shock
                        "d_missing_star_advantage": sa["missing_star_elo_loss"] - sh["missing_star_elo_loss"],
                        "d_roster_continuity": sh["roster_continuity"] - sa["roster_continuity"],
                        # In-game Star Q1-Q3 Usage / Rest / Reserve Patterns
                        "d_star_scoring_reserve": sh["star_scoring_reserve"] - sa["star_scoring_reserve"],
                        "d_star_rested_late_q3": sh["star_rested_late_q3"] - sa["star_rested_late_q3"],
                        "d_star_overloaded_q3": sh["star_overloaded_q3"] - sa["star_overloaded_q3"],
                        "d_bench_q1q3_pts": sh["bench_q1q3_pts"] - sa["bench_q1q3_pts"],
                    }
                )

            # Update player states AFTER match
            for side, tkey, won_ft, won_q1, won_q2, won_q3, won_q4 in [
                ("home", hkey, 1.0 if ft_diff > 0 else 0.0, 1.0 if q1_diff > 0 else 0.5 if q1_diff == 0 else 0.0, 1.0 if q2_diff > 0 else 0.5 if q2_diff == 0 else 0.0, 1.0 if q3_diff > 0 else 0.5 if q3_diff == 0 else 0.0, float(y_q4)),
                ("away", akey, 0.0 if ft_diff > 0 else 1.0, 0.0 if q1_diff > 0 else 0.5 if q1_diff == 0 else 1.0, 0.0 if q2_diff > 0 else 0.5 if q2_diff == 0 else 1.0, 0.0 if q3_diff > 0 else 0.5 if q3_diff == 0 else 1.0, 1.0 - float(y_q4)),
            ]:
                opp_side = "away" if side == "home" else "home"
                opp_ft_elo = side_data[opp_side]["starters_elo_ft"]
                opp_q4_elo = side_data[opp_side]["starters_elo_q4"]
                active_core = []
                for p in side_data[side]["dressed_players"]:
                    if p["_actual_mins"] >= 5.0 or p["_actual_pts"] > 0 or p["_actual_rat"] is not None:
                        pst = player_state[p["pkey"]]
                        # Weight K by minutes played (more minutes = more credit/blame for team quarter results + individual PM)
                        min_w = min(1.2, max(0.25, p["_actual_mins"] / 24.0))
                        k_p = 14.0 * min_w
                        exp_ft = 1.0 / (1.0 + 10.0 ** ((opp_ft_elo - pst["elo_ft"]) / 400.0))
                        exp_q4 = 1.0 / (1.0 + 10.0 ** ((opp_q4_elo - pst["elo_q4"]) / 400.0))
                        # Also blend individual plus_minus signal into player ELO!
                        pm_bonus = np.clip((p["_actual_pm"] or 0.0) / 25.0, -0.35, 0.35)
                        pst["elo_ft"] += k_p * ((won_ft - exp_ft) + 0.25 * pm_bonus)
                        pst["elo_q4"] += k_p * ((won_q4 - exp_q4) + 0.25 * pm_bonus)
                        pst["elo_q1"] += k_p * (won_q1 - 0.5)
                        pst["elo_q2"] += k_p * (won_q2 - 0.5)
                        pst["elo_q3"] += k_p * (won_q3 - 0.5)
                        pst["n"] += 1
                        if p["_actual_rat"] is not None:
                            pst["ratings"].append(p["_actual_rat"])
                        if p["_actual_pm"] is not None:
                            pst["pm"].append(p["_actual_pm"])
                        pst["pts"].append(p["_actual_pts"])
                        pst["mins"].append(p["_actual_mins"])
                        pst["q1_pts"].append(p["q1_today"])
                        pst["q2_pts"].append(p["q2_today"])
                        pst["q3_pts"].append(p["q3_today"])
                        if p["_actual_mins"] >= 15.0:
                            active_core.append((p["pkey"], pst["elo_q4"]))
                if active_core:
                    team_recent_rosters[tkey].append(active_core)

        # Update Team ELOs (Long K=20, Short K=38)
        for q_tag, diff_val in [("ft", ft_diff), ("q1", q1_diff), ("q2", q2_diff), ("q3", q3_diff), ("q4", q4_diff)]:
            res_h = 1.0 if diff_val > 0 else (0.5 if diff_val == 0 else 0.0)
            exp_L = 1.0 / (1.0 + 10.0 ** ((ta_L[q_tag] - th_L[q_tag]) / 400.0))
            th_L[q_tag] += 20.0 * (res_h - exp_L)
            ta_L[q_tag] += 20.0 * ((1.0 - res_h) - (1.0 - exp_L))

            exp_S = 1.0 / (1.0 + 10.0 ** ((ta_S[q_tag] - th_S[q_tag]) / 400.0))
            th_S[q_tag] += 38.0 * (res_h - exp_S)
            ta_S[q_tag] += 38.0 * ((1.0 - res_h) - (1.0 - exp_S))
        th_L["n"] += 1
        ta_L["n"] += 1

    print(f"\nTotal de partidos analizados con Alineaciones + PBP + ELOs: {len(records):,}")

    print("\n" + "=" * 95)
    print("2. CORRELACIONES CONTRA GANAR EL Q4 (y_q4) Y MARGEN Q4 (q4_diff)")
    print("   (Controlando también por partidos parejos |cur_diff| <= 12 donde más se apuesta)")
    print("=" * 95)
    y_all = np.array([r["y_q4"] for r in records])
    q4_all = np.array([r["q4_diff"] for r in records])
    close_mask = np.array([abs(r["cur_diff"]) <= 12 for r in records])

    feat_keys = [
        "d_team_elo_ft_long",
        "d_team_elo_q4_long",
        "d_team_elo_q2_long",
        "d_team_elo_q3_long",
        "d_team_elo_ft_short",
        "d_team_elo_q4_short",
        "d_team_q4_vs_q3_elo",
        "d_team_q4_vs_q1_elo",
        "d_lineup_starters_elo_ft",
        "d_lineup_starters_elo_q4",
        "d_lineup_bench_elo_q4",
        "d_lineup_top2_elo_q4",
        "d_lineup_bench_pm",
        "d_roster_vs_logo_elo",
        "d_missing_star_advantage",
        "d_roster_continuity",
        "d_star_scoring_reserve",
        "d_star_rested_late_q3",
        "d_star_overloaded_q3",
        "d_bench_q1q3_pts",
    ]

    print(f"{'Feature / Patrón Propuesto':<28} | {'r (y_q4 All)':>12} | {'r (q4_diff All)':>15} | {'r (q4_diff |diff|<=12)':>22}")
    print("-" * 88)
    for fk in feat_keys:
        vals = np.array([r[fk] for r in records], dtype=float)
        r_y, _ = pearsonr(vals, y_all)
        r_q4, _ = pearsonr(vals, q4_all)
        r_close, _ = pearsonr(vals[close_mask], q4_all[close_mask])
        print(f"{fk:<28} | {r_y:>+12.4f} | {r_q4:>+15.4f} | {r_close:>+22.4f}")

    # Test specific situational buckets!
    print("\n" + "=" * 95)
    print("3. HALLAZGOS DE PATRONES DE JUEGO (ESCENARIOS CONDICIONALES EN Q4)")
    print("=" * 95)

    # Scenario A: Team Logo says Home is better (d_team_elo_ft_long > 50),
    # BUT Today's Lineup ELO says Away has better players dressed today (d_lineup_starters_elo_q4 < -20) vs aligned!
    aligned_elo = [r for r in records if r["d_team_elo_ft_long"] > 50 and r["d_lineup_starters_elo_q4"] > 30 and abs(r["cur_diff"]) <= 12]
    divergent_elo = [r for r in records if r["d_team_elo_ft_long"] > 50 and r["d_lineup_starters_elo_q4"] < -15 and abs(r["cur_diff"]) <= 12]
    print(f"  A) ¿Logo ELO vs Lineup ELO de los Jugadores de Hoy (partidos |diff|<=12)?")
    if aligned_elo:
        print(f"     - Cuando Logo ELO y Lineup Q4 ELO coinciden a favor del Local (N={len(aligned_elo):,}): WR Q4 Local = {np.mean([r['y_q4'] for r in aligned_elo])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in aligned_elo]):+.2f}")
    if divergent_elo:
        print(f"     - Cuando Logo ELO dice Local (>+50), PERO la Alineación de Hoy favorece a la Visita (N={len(divergent_elo):,}): WR Q4 Local cae a = {np.mean([r['y_q4'] for r in divergent_elo])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in divergent_elo]):+.2f}")

    # Scenario B: Missing Star Shock
    missing_away_star = [r for r in records if r["d_missing_star_advantage"] > 40 and abs(r["cur_diff"]) <= 12]
    print(f"\n  B) Shock por Ausencia de Estrella (cuando al rival le falta 1-2 estrellas de su rotación reciente, |diff|<=12):")
    if missing_away_star:
        print(f"     - Ventaja por Estrella Ausente del Rival (N={len(missing_away_star):,}): WR Q4 = {np.mean([r['y_q4'] for r in missing_away_star])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in missing_away_star]):+.2f}")

    # Scenario C: Star Rested in Late Q3 vs Overloaded in Q3 (in close/moderate games |cur_diff| <= 12)
    rested_h = [r for r in records if r["d_star_rested_late_q3"] > 0 and r["d_lineup_starters_elo_q4"] > 20 and abs(r["cur_diff"]) <= 12]
    overloaded_h = [r for r in records if r["d_star_overloaded_q3"] > 0 and r["d_lineup_starters_elo_q4"] > 20 and abs(r["cur_diff"]) <= 12]
    print(f"\n  C) Patrón de Descanso vs Sobrecarga de Estrellas en el Q3 (partidos competitivos |diff|<=12):")
    if rested_h:
        print(f"     - Estrellas activas en Q1/Q2 que DESCANSARON al cierre del Q3 y entran frescas al Q4 (N={len(rested_h):,}): WR Q4 = {np.mean([r['y_q4'] for r in rested_h])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in rested_h]):+.2f}")
    if overloaded_h:
        print(f"     - Estrellas SOBRECARGADAS todo el Q3 (jugaron al 100% el cierre de Q3) (N={len(overloaded_h):,}): WR Q4 = {np.mean([r['y_q4'] for r in overloaded_h])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in overloaded_h]):+.2f}")

    # Scenario D: Cold Star Regression (Star with high ELO scored below their Q1-Q3 average while team trails by 1..12 pts)
    cold_star_comeback = [r for r in records if -12 <= r["cur_diff"] <= -1 and r["d_lineup_top2_elo_q4"] > 30 and r["d_star_scoring_reserve"] > 4.0]
    hot_star_exhausted = [r for r in records if 1 <= r["cur_diff"] <= 12 and r["d_lineup_top2_elo_q4"] < -20 and r["d_star_scoring_reserve"] < -5.0]
    print(f"\n  D) Regresión a la Media de la Estrella ('Estrella Fría' en Q1-Q3 vs 'Estrella Inflada'):")
    if cold_star_comeback:
        print(f"     - Equipo con mejores estrellas (Top2 ELO Q4 > +30) va perdiendo por 1-12 pts y sus estrellas tienen reserva de puntos (>4 pts debajo de su media en Q1-Q3) (N={len(cold_star_comeback):,}): WR Q4 = {np.mean([r['y_q4'] for r in cold_star_comeback])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in cold_star_comeback]):+.2f}")
    if hot_star_exhausted:
        print(f"     - Equipo inferior va ganando por 1-12 pts porque su estrella anotó >5 pts por encima de su media en Q1-Q3 (N={len(hot_star_exhausted):,}): WR Q4 del rival (favorito) = {(1.0 - np.mean([r['y_q4'] for r in hot_star_exhausted]))*100:.2f}%")

    # Scenario E: Bench Depth in Q4
    deep_bench = [r for r in records if r["d_lineup_bench_elo_q4"] > 25 and r["d_lineup_starters_elo_q4"] > 20 and -12 <= r["cur_diff"] <= 8]
    print(f"\n  E) Sinergia Titulares + Banca Profunda (Titulares ELO Q4 > +20 Y Banca ELO Q4 > +25, marcador -12 a +8):")
    if deep_bench:
        print(f"     - Plantilla completa superior (Titulares + Banca) (N={len(deep_bench):,}): WR Q4 = {np.mean([r['y_q4'] for r in deep_bench])*100:.2f}% | Margen Q4 = {np.mean([r['q4_diff'] for r in deep_bench]):+.2f}")


if __name__ == "__main__":
    main()
