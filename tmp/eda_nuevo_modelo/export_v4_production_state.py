"""
Exporta el estado final de ELO estacional, estadísticos rolling por equipo, estadísticos rolling
por jugador (`player_priors`) y clasificación de ligas (`lc_map`) a
`models/m27_v4/model_outputs/team_priors_state.pkl`, y copia los bundles entrenados de la
Tríada Canónica (`fiba_men_10m`, `fiba_women_10m`, `nba_12m`) para inferencia en producción O(1) (< 1 ms).
"""
import sys
import math
import shutil
import pickle
import sqlite3
from datetime import datetime
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
DB_PATH = ROOT_DIR / "matches.db"
SRC_DIR = Path(__file__).resolve().parent / "v4_artifacts"
DST_DIR = ROOT_DIR / "models" / "m27_v4" / "model_outputs"
DST_DIR.mkdir(parents=True, exist_ok=True)


def main():
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=60)
    cur = conn.cursor()

    cur.execute("""
        SELECT league, quarter_duration_minutes, is_women, avg_total_points, avg_q4_total_points
        FROM leagues_classification
    """)
    lc_map = {}
    for lg, q_dur, is_w, avg_ft, avg_q4 in cur.fetchall():
        qd = int(q_dur or 10)
        iw = int(is_w or 0)
        arch = "nba_12m" if qd == 12 else ("fiba_women_10m" if iw == 1 else "fiba_men_10m")
        lc_map[lg] = {
            "arch": arch,
            "q_dur": qd,
            "lg_avg_ft": float(avg_ft or (215.0 if qd == 12 else 158.0)),
            "lg_avg_q4": float(avg_q4 or (53.0 if qd == 12 else 39.5)),
        }

    cur.execute("""
        SELECT m.date, m.home_team, m.away_team,
               q1.home, q1.away, q2.home, q2.away, q3.home, q3.away, q4.home, q4.away
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)

    elo_ft = defaultdict(lambda: 1500.0)
    elo_q4 = defaultdict(lambda: 1500.0)
    elo_q1 = defaultdict(lambda: 1500.0)
    last_team_date = {}
    team_q4_hist = defaultdict(list)
    team_ft_hist = defaultdict(list)

    for dt, ht, at, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a in cur.fetchall():
        if None in (q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a):
            continue
        clean_dt = str(dt).split(" ")[0]
        try:
            cur_date = datetime.strptime(clean_dt, "%Y-%m-%d")
        except ValueError:
            cur_date = datetime(2025, 1, 1)

        for t in (ht, at):
            if t in last_team_date:
                days_gap = (cur_date - last_team_date[t]).days
                if days_gap > 300:
                    elo_ft[t] = 1500.0 + 0.25 * (elo_ft[t] - 1500.0)
                    elo_q4[t] = 1500.0 + 0.20 * (elo_q4[t] - 1500.0)
                    elo_q1[t] = 1500.0 + 0.20 * (elo_q1[t] - 1500.0)
                    team_q4_hist[t] = []
                    team_ft_hist[t] = []
                elif days_gap > 60:
                    elo_ft[t] = 1500.0 + 0.65 * (elo_ft[t] - 1500.0)
                    elo_q4[t] = 1500.0 + 0.50 * (elo_q4[t] - 1500.0)
                    elo_q1[t] = 1500.0 + 0.55 * (elo_q1[t] - 1500.0)
                    team_q4_hist[t] = team_q4_hist[t][-5:]
                    team_ft_hist[t] = team_ft_hist[t][-5:]
            last_team_date[t] = cur_date

        ft_m = (q1h + q2h + q3h + q4h) - (q1a + q2a + q3a + q4a)
        q4_m = q4h - q4a
        q1_m = q1h - q1a

        rh_ft, ra_ft = elo_ft[ht], elo_ft[at]
        rh_q4, ra_q4 = elo_q4[ht], elo_q4[at]
        rh_q1, ra_q1 = elo_q1[ht], elo_q1[at]

        exp_h_ft = 1.0 / (1.0 + 10.0 ** ((ra_ft - (rh_ft + 35.0)) / 400.0))
        act_h_ft = 1.0 if ft_m > 0 else (0.0 if ft_m < 0 else 0.5)
        mov_mult = math.log(abs(ft_m) + 1.0) * (2.2 / ((abs(rh_ft - ra_ft) * 0.001) + 2.2))
        elo_ft[ht] += 20.0 * mov_mult * (act_h_ft - exp_h_ft)
        elo_ft[at] += 20.0 * mov_mult * ((1.0 - act_h_ft) - (1.0 - exp_h_ft))

        exp_h_q4 = 1.0 / (1.0 + 10.0 ** ((ra_q4 - rh_q4) / 400.0))
        act_h_q4 = 1.0 if q4_m > 0 else (0.0 if q4_m < 0 else 0.5)
        elo_q4[ht] += 15.0 * (act_h_q4 - exp_h_q4)
        elo_q4[at] += 15.0 * ((1.0 - act_h_q4) - (1.0 - exp_h_q4))

        exp_h_q1 = 1.0 / (1.0 + 10.0 ** ((ra_q1 - (rh_q1 + 12.0)) / 400.0))
        act_h_q1 = 1.0 if q1_m > 0 else (0.0 if q1_m < 0 else 0.5)
        elo_q1[ht] += 14.0 * (act_h_q1 - exp_h_q1)
        elo_q1[at] += 14.0 * ((1.0 - act_h_q1) - (1.0 - exp_h_q1))

        team_q4_hist[ht].append(q4_m)
        team_q4_hist[at].append(-q4_m)
        team_ft_hist[ht].append(ft_m)
        team_ft_hist[at].append(-ft_m)

    # Calcular estadísticos rolling por jugador (últimos 10 partidos: rating y plus_minus)
    cur.execute("""
        SELECT ps.player_id, ps.sofascore_rating, ps.plus_minus
        FROM player_stats ps
        JOIN matches m ON m.match_id = ps.match_id
        WHERE ps.player_id IS NOT NULL AND ps.sofascore_rating IS NOT NULL
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)
    p_hist = defaultdict(list)
    for pid, rat, pm in cur:
        s_pid = str(pid)
        p_hist[s_pid].append((float(rat or 6.5), float(pm or 0.0)))
        if len(p_hist[s_pid]) > 12:
            p_hist[s_pid] = p_hist[s_pid][-10:]

    # Último estado de clasificación / forma (team_strength) por equipo
    cur.execute("""
        SELECT ts.team_name, ts.wins, ts.losses, ts.form
        FROM team_strength ts
        JOIN matches m ON m.match_id = ts.match_id
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)
    ts_latest = {}
    for tname, w, l, fm in cur:
        if not tname:
            continue
        w_i, l_i = int(w or 0), int(l or 0)
        if (w_i + l_i) > 0:
            wp_b = (w_i + 3.0) / (w_i + l_i + 6.0)
            fv = 0.5
            if fm:
                wc, lc_ = str(fm).count("W"), str(fm).count("L")
                if (wc + lc_) > 0:
                    fv = (wc + 1.5) / (wc + lc_ + 3.0)
            ts_latest[str(tname)] = (round(float(wp_b), 4), round(float(fv), 4))

    conn.close()

    player_priors = {}
    for pid, hist in p_hist.items():
        if len(hist) >= 2:
            rec = hist[-10:]
            m_rat = sum(x[0] for x in rec) / len(rec)
            m_pm = sum(x[1] for x in rec) / len(rec)
            player_priors[pid] = (round(float(m_rat), 3), round(float(m_pm), 3), len(rec))

    team_priors = {}
    ts_priors = dict(ts_latest)
    for t in elo_ft.keys():
        q4_l = team_q4_hist[t][-15:]
        ft_l = team_ft_hist[t][-15:]
        n_g = len(team_ft_hist[t])
        q4_b = (sum(q4_l) / (len(q4_l) + 6.0)) if q4_l else 0.0
        wr_b = (sum(1.0 if m > 0 else 0.0 for m in ft_l) + 3.0) / (len(ft_l) + 6.0)
        team_priors[t] = (
            round(float(elo_ft[t]), 2),
            round(float(elo_q4[t]), 2),
            round(float(elo_q1[t]), 2),
            round(float(q4_b), 3),
            round(float(wr_b), 4),
            int(n_g),
        )
        if t not in ts_priors and ft_l:
            ft_l5 = ft_l[-5:]
            fv_5 = (sum(1.0 if m > 0 else 0.0 for m in ft_l5) + 1.5) / (len(ft_l5) + 3.0)
            ts_priors[t] = (round(float(wr_b), 4), round(float(fv_5), 4))

    state_path = DST_DIR / "team_priors_state.pkl"
    with open(state_path, "wb") as f:
        pickle.dump(
            {
                "team_priors": team_priors,
                "lc_map": lc_map,
                "player_priors": player_priors,
                "ts_priors": ts_priors,
            },
            f,
        )

    for fname in [
        "model_v4_fiba_men_10m.pkl",
        "model_v4_fiba_women_10m.pkl",
        "model_v4_nba_12m.pkl",
        "v4_training_report.json",
    ]:
        src = SRC_DIR / fname
        if src.exists():
            shutil.copy2(src, DST_DIR / fname)

    print(
        f"Exportados {len(team_priors):,} equipos, {len(player_priors):,} jugadores, {len(ts_priors):,} ts_priors y {len(lc_map):,} ligas en {DST_DIR}"
    )


if __name__ == "__main__":
    main()
