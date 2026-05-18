"""Analyze whether team history features carry signal for Q4 prediction."""

import math
import sys
from collections import defaultdict
from datetime import datetime as dt_mod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\borre\OneDrive\OLD\Escritorio\pulpa")
sys.path.insert(0, str(ROOT / "match" / "training"))
import train_q3_q4_models_v6 as v6


def _point_biserial(values, targets):
    v = np.asarray(values, dtype=float)
    t = np.asarray(targets, dtype=float)
    if len(v) < 2 or v.std(ddof=0) == 0 or t.std(ddof=0) == 0:
        return None
    corr = np.corrcoef(v, t)[0, 1]
    return float(corr) if not math.isnan(corr) else None


def _parse_dt(s):
    return dt_mod.strptime(s.split(" ")[0], "%Y-%m-%d")


def _win_rate(wins, total):
    return wins / total if total > 0 else 0.5


def _last_n_form(team_games, team, before_date, n):
    """Compute win% in last n games before before_date."""
    games = [g for g in team_games.get(team, []) if g["dt"] < before_date]
    games = sorted(games, key=lambda g: g["dt"], reverse=True)[:n]
    if not games:
        return None, 0
    wins = sum(1 for g in games if g["won"])
    return _win_rate(wins, len(games)), len(games)


def _h2h(team_games, team_a, team_b, before_date):
    """H2H record of team_a vs team_b before before_date (from team_a's perspective)."""
    games = [
        g for g in team_games.get(team_a, [])
        if g["opp"] == team_b and g["dt"] < before_date
    ]
    if not games:
        return None, 0
    wins = sum(1 for g in games if g["won"])
    return _win_rate(wins, len(games)), len(games)


def _rest_days(team_games, team, before_date):
    """Days since team's last game before before_date."""
    games = [g for g in team_games.get(team, []) if g["dt"] < before_date]
    if not games:
        return None
    last = max(g["dt"] for g in games)
    delta = (before_date - last).days
    return min(delta, 14)  # cap at 14


def _avg_pts_for_against(team_games, team, before_date, n):
    """Avg points for/against in last n games."""
    games = [g for g in team_games.get(team, []) if g["dt"] < before_date]
    games = sorted(games, key=lambda g: g["dt"], reverse=True)[:n]
    if not games:
        return None, None, 0
    gf = sum(g["gf"] for g in games) / len(games)
    ga = sum(g["ga"] for g in games) / len(games)
    return gf, ga, len(games)


def main():
    conn = v6.db_mod.get_conn(str(v6.DB_PATH))
    v6.db_mod.init_db(conn)

    # 1. Build team game history
    print("Building team game history...")
    team_games = defaultdict(list)
    all_matches = conn.execute(
        "SELECT match_id, home_team, away_team, home_score, away_score, date FROM matches ORDER BY date"
    ).fetchall()

    for mid, ht, at, hs, aw, dt in all_matches:
        if hs is None or aw is None:
            continue
        hs, aw = int(hs), int(aw)
        dt_obj = _parse_dt(dt)
        team_games[ht].append({
            "dt": dt_obj, "opp": at, "gf": hs, "ga": aw, "won": hs > aw, "home": True,
        })
        team_games[at].append({
            "dt": dt_obj, "opp": ht, "gf": aw, "ga": hs, "won": aw > hs, "home": False,
        })

    # Also build match_id -> teams lookup
    match_teams = {}
    for mid, ht, at, _, _, _ in all_matches:
        match_teams[mid] = (ht, at)

    conn.close()
    print(f"  {len(team_games)} teams, {sum(len(v) for v in team_games.values())} games")

    # 2. Load training samples
    cache = joblib.load(
        str(ROOT / "match" / "training" / "model_outputs_m27_v2" / "dynamic_rows_cache.joblib")
    )
    print(f"  {len(cache)} cache rows")

    # 3. Compute features for each sample
    rows = []
    skipped_no_teams = 0
    skipped_no_history = 0
    for cache_row in cache:
        tgt = cache_row.get("target")
        if tgt is None:
            continue
        match_id = cache_row["match_id"]
        sample_dt = _parse_dt(cache_row["dt"])

        teams = match_teams.get(match_id)
        if not teams:
            skipped_no_teams += 1
            continue
        ht, at = teams

        # --- Team recent form ---
        f5h, n5h = _last_n_form(team_games, ht, sample_dt, 5)
        f5a, n5a = _last_n_form(team_games, at, sample_dt, 5)
        f10h, n10h = _last_n_form(team_games, ht, sample_dt, 10)
        f10a, n10a = _last_n_form(team_games, at, sample_dt, 10)
        f20h, n20h = _last_n_form(team_games, ht, sample_dt, 20)
        f20a, n20a = _last_n_form(team_games, at, sample_dt, 20)

        # --- Season win% ---
        seas_h, nsh = _last_n_form(team_games, ht, sample_dt, 9999)
        seas_a, nsa = _last_n_form(team_games, at, sample_dt, 9999)

        # --- Rest days ---
        rest_h = _rest_days(team_games, ht, sample_dt)
        rest_a = _rest_days(team_games, at, sample_dt)

        # --- Points for/against ---
        pts5_h = _avg_pts_for_against(team_games, ht, sample_dt, 5)
        pts5_a = _avg_pts_for_against(team_games, at, sample_dt, 5)

        # --- H2H ---
        h2h_ht, n_h2h = _h2h(team_games, ht, at, sample_dt)

        # --- Home/away splits ---
        home_games = [g for g in team_games.get(ht, []) if g["home"] and g["dt"] < sample_dt]
        home_wr = _win_rate(sum(1 for g in home_games if g["won"]), len(home_games)) if home_games else None
        away_games = [g for g in team_games.get(at, []) if not g["home"] and g["dt"] < sample_dt]
        away_wr = _win_rate(sum(1 for g in away_games if g["won"]), len(away_games)) if away_games else None

        if f5h is None or f5a is None:
            skipped_no_history += 1
            continue

        rows.append({
            "target": int(tgt),
            "team_home_win_rate_last5": f5h,
            "team_away_win_rate_last5": f5a,
            "team_win_rate_last5_diff": (f5h or 0) - (f5a or 0),
            "team_home_win_rate_last10": f10h,
            "team_away_win_rate_last10": f10a,
            "team_home_win_rate_last20": f20h,
            "team_away_win_rate_last20": f20a,
            "team_home_season_wr": seas_h,
            "team_away_season_wr": seas_a,
            "season_wr_diff": (seas_h or 0) - (seas_a or 0),
            "rest_home": rest_h,
            "rest_away": rest_a,
            "rest_diff": (rest_h or 0) - (rest_a or 0),
            "h2h_home_wr": h2h_ht,
            "h2h_n": n_h2h,
            "home_home_wr": home_wr,  # how home team plays at home
            "away_away_wr": away_wr,  # how away team plays away
            "pts_for_home_last5": pts5_h[0] if pts5_h[0] is not None else None,
            "pts_against_home_last5": pts5_h[1] if pts5_h[1] is not None else None,
            "pts_for_away_last5": pts5_a[0] if pts5_a[0] is not None else None,
            "pts_against_away_last5": pts5_a[1] if pts5_a[1] is not None else None,
            "pt_diff_home_last5": (pts5_h[0] - pts5_h[1]) if pts5_h[0] is not None else None,
            "pt_diff_away_last5": (pts5_a[0] - pts5_a[1]) if pts5_a[0] is not None else None,
        })

    df = pd.DataFrame(rows)
    print(f"\n  {len(df)} samples with history, skipped (no teams={skipped_no_teams}, no hist={skipped_no_history})")
    print(f"  Home wins: {int(df['target'].sum())}/{len(df)}")

    # 4. Compute correlations
    results = []
    for col in df.columns:
        if col == "target":
            continue
        valid = df[col].notna()
        n_valid = valid.sum()
        if n_valid < 100:
            continue
        corr = _point_biserial(df.loc[valid, col], df.loc[valid, "target"])
        if corr is not None:
            results.append({
                "feature": col,
                "corr": round(corr, 4),
                "abs_corr": round(abs(corr), 4),
                "n": n_valid,
            })

    results.sort(key=lambda r: r["abs_corr"], reverse=True)

    HR = "=" * 60
    print(f"\n{HR}")
    print("  Team History Feature Correlations with Q4 Target")
    print(HR)
    print(f"  {'Feature':<35s} {'Corr':>8s} {'|Corr|':>8s} {'N':>6s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*6}")
    for r in results:
        print(f"  {r['feature']:<35s} {r['corr']:+8.4f} {r['abs_corr']:8.4f} {r['n']:>6d}")

    # Compare vs baseline recent_2m_points_diff
    print(f"\n  Reference: recent_2m_points_diff corr = +0.2396 (from earlier analysis)")
    best_abs = max(r['abs_corr'] for r in results) if results else 0
    print(f"  Best team-history feature |corr| = {best_abs:.4f}")

    # Save
    out = ROOT / "findings" / "team_history_correlation.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved to {out}")


if __name__ == "__main__":
    main()
