"""
Diagnóstico específico de `nba_12m` (N=28,250 partidos de 2015 a 2026):
1. ¿Por qué `nba_12m` sobreajustó en V4.3 (Train AUC 0.789 vs Val AUC 0.594)?
2. ¿Qué features tienen correlación real con Q4 Winner en `nba_12m` en Train vs Val vs Test?
3. Efecto del Time-Decay en ELO entre temporadas (>90 días de pausa) y regularización fuerte para NBA.
"""
import sys
import math
import sqlite3
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.stdout.reconfigure(encoding="utf-8")
DB_PATH = Path(__file__).resolve().parents[2] / "matches.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        SELECT m.match_id, m.date, m.time, m.league, m.home_team, m.away_team,
               q1.home, q1.away, q2.home, q2.away, q3.home, q3.away, q4.home, q4.away
        FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        JOIN leagues_classification lc ON m.league = lc.league
        WHERE lc.quarter_duration_minutes = 12 AND q4.home != q4.away
        ORDER BY m.date ASC, m.time ASC, m.match_id ASC
    """)
    nba_rows = cur.fetchall()
    print(f"Total partidos nba_12m: {len(nba_rows):,}")

    cur.execute("""
        SELECT match_id, home_value, away_value
        FROM match_odds
        WHERE odds_type = 'Home/Away' AND market_period = 'Match'
          AND home_value > 1.0 AND away_value > 1.0
    """)
    odds_map = {r[0]: math.log(r[2] / r[1]) for r in cur.fetchall()}

    # Veamos cobertura de match_odds en nba_12m por año:
    by_yr_odds = defaultdict(list)
    for r in nba_rows:
        yr = r[1][:4]
        by_yr_odds[yr].append(1 if r[0] in odds_map else 0)
    print("\nCobertura de match_odds en nba_12m por año:")
    for yr in sorted(by_yr_odds.keys()):
        print(f"  {yr}: N={len(by_yr_odds[yr]):>5,} | %Odds = {np.mean(by_yr_odds[yr])*100:>5.1f}%")

    # Construir features con decay estacional (si pasan > 60 días sin jugar, ELO regresa 35% hacia 1500)
    from datetime import datetime
    elo_ft = defaultdict(lambda: 1500.0)
    elo_q4 = defaultdict(lambda: 1500.0)
    last_dt_map = {}
    q4_hist = defaultdict(list)

    X_list, y_list, ym_list, yr_list = [], [], [], []
    for row in nba_rows:
        mid, dt_s, tm, lg, ht, at, q1h, q1a, q2h, q2a, q3h, q3a, q4h, q4a = row
        dt_clean = dt_s.split(" ")[0]
        cur_date = datetime.strptime(dt_clean, "%Y-%m-%d")

        for t in (ht, at):
            if t in last_dt_map and (cur_date - last_dt_map[t]).days > 60:
                elo_ft[t] = 1500.0 + 0.65 * (elo_ft[t] - 1500.0)
                elo_q4[t] = 1500.0 + 0.50 * (elo_q4[t] - 1500.0)
                q4_hist[t] = q4_hist[t][-5:]
            last_dt_map[t] = cur_date

        q1_d = float(q1h - q1a)
        q2_d = float(q2h - q2a)
        ht_d = q1_d + q2_d
        q3_d = float(q3h - q3a)
        q3_end_d = ht_d + q3_d
        q3_tot = float(q1h + q1a + q2h + q2a + q3h + q3a)
        q4_m = float(q4h - q4a)
        ft_m = q3_end_d + q4_m

        d_elo_ft = elo_ft[ht] - elo_ft[at]
        d_elo_q4 = elo_q4[ht] - elo_q4[at]
        h_q4_b = sum(q4_hist[ht][-15:]) / (len(q4_hist[ht][-15:]) + 8.0)
        a_q4_b = sum(q4_hist[at][-15:]) / (len(q4_hist[at][-15:]) + 8.0)
        d_q4_b = h_q4_b - a_q4_b

        has_o = 1.0 if mid in odds_map else 0.0
        o_lr = odds_map.get(mid, 0.0)
        exp_ft = (0.4 * (d_elo_ft * 0.035) + 0.6 * (o_lr * 5.5)) if has_o else (d_elo_ft * 0.035)
        rubber_band = exp_ft - q3_end_d

        # En NBA el efecto rubber-band del marcador Q3 es fortísimo incluso entre equipos parejos
        X_list.append([
            q3_end_d,
            abs(q3_end_d),
            ht_d,
            q3_d,
            q3_d - 0.5 * ht_d,
            q3_tot / 36.0,
            d_elo_ft,
            d_elo_q4,
            d_q4_b,
            has_o,
            o_lr,
            exp_ft,
            rubber_band,
            1.0 if q3_end_d <= -15.0 else ( -1.0 if q3_end_d >= 15.0 else 0.0 ),  # NBA blowout bench comeback
        ])
        y_list.append(1 if q4_m > 0 else 0)
        ym_list.append(q4_m)
        yr_list.append(int(dt_clean[:4]))

        # Update ELO
        exp_h = 1.0 / (1.0 + 10.0 ** ((elo_ft[at] - (elo_ft[ht] + 30.0)) / 400.0))
        act_h = 1.0 if ft_m > 0 else 0.0
        elo_ft[ht] += 18.0 * (act_h - exp_h)
        elo_ft[at] += 18.0 * ((1.0 - act_h) - (1.0 - exp_h))

        exp_q4 = 1.0 / (1.0 + 10.0 ** ((elo_q4[at] - elo_q4[ht]) / 400.0))
        act_q4 = 1.0 if q4_m > 0 else 0.0
        elo_q4[ht] += 12.0 * (act_q4 - exp_q4)
        elo_q4[at] += 12.0 * ((1.0 - act_q4) - (1.0 - exp_q4))
        q4_hist[ht].append(q4_m)
        q4_hist[at].append(-q4_m)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    ym = np.array(ym_list, dtype=np.float32)
    N = len(X)
    ntr = int(N * 0.70)
    nva = int(N * 0.85)

    names = [
        "q3_end_d", "abs_q3_d", "ht_d", "q3_d", "q3_vs_ht", "pace",
        "d_elo_ft", "d_elo_q4", "d_q4_b", "has_o", "o_lr", "exp_ft", "rubber_band", "blowout_rev"
    ]
    print("\nCorrelaciones con Q4 Margin en NBA 12m (Train vs Val vs Test):")
    for idx, nm in enumerate(names):
        r_tr = np.corrcoef(X[:ntr, idx], ym[:ntr])[0, 1]
        r_va = np.corrcoef(X[ntr:nva, idx], ym[ntr:nva])[0, 1]
        r_te = np.corrcoef(X[nva:, idx], ym[nva:])[0, 1]
        print(f"  {nm:<15} | r_train={r_tr:+.4f} | r_val={r_va:+.4f} | r_test={r_te:+.4f}")

    # Probemos modelos con regularización fuerte (sin repetir snapshots x4)
    for m_name, clf in [
        ("LogisticRegression L2", make_pipeline(StandardScaler(), LogisticRegression(C=0.1))),
        ("HistGB Shallow (max_leaf=7, min_leaf=400, l2=50)", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.03, max_leaf_nodes=7, min_samples_leaf=400, l2_regularization=50.0, random_state=42)),
        ("HistGB Medium (max_leaf=12, min_leaf=300, l2=30)", HistGradientBoostingClassifier(max_iter=150, learning_rate=0.03, max_leaf_nodes=12, min_samples_leaf=300, l2_regularization=30.0, random_state=42)),
    ]:
        clf.fit(X[:ntr], y[:ntr])
        ptr = clf.predict_proba(X[:ntr])[:, 1]
        pva = clf.predict_proba(X[ntr:nva])[:, 1]
        pte = clf.predict_proba(X[nva:])[:, 1]
        print(f"\n  [{m_name}]")
        print(f"    Train AUC={roc_auc_score(y[:ntr], ptr):.4f} | Val AUC={roc_auc_score(y[ntr:nva], pva):.4f} | Test AUC={roc_auc_score(y[nva:], pte):.4f} | Gap(Tr-Va)={roc_auc_score(y[:ntr], ptr)-roc_auc_score(y[ntr:nva], pva):+.4f}")

    conn.close()


if __name__ == "__main__":
    main()
