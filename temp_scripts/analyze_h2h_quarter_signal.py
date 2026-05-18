"""Analyze whether H2H quarter-by-quarter scores carry signal for Q4 prediction."""

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


def main():
    conn = v6.db_mod.get_conn(str(v6.DB_PATH))
    v6.db_mod.init_db(conn)

    # 1. Load all matches with teams + dates
    print("Loading match data...")
    all_matches = conn.execute(
        "SELECT match_id, home_team, away_team, date FROM matches ORDER BY date"
    ).fetchall()
    match_info = {}
    for mid, ht, at, dt in all_matches:
        match_info[mid] = {"home_team": ht, "away_team": at, "date": _parse_dt(dt)}

    # 2. Load quarter scores
    print("Loading quarter scores...")
    qs_rows = conn.execute(
        "SELECT match_id, quarter, home, away FROM quarter_scores"
    ).fetchall()
    match_quarters = defaultdict(dict)
    for mid, qtr, hs, aw in qs_rows:
        if hs is not None and aw is not None:
            match_quarters[mid][qtr] = (int(hs), int(aw))

    conn.close()

    # 3. Build H2H history with quarter-level detail
    # For each pair (team_a, team_b), store list of past matches with quarter data
    h2h_games = defaultdict(list)
    for mid, info in match_info.items():
        ht, at = info["home_team"], info["away_team"]
        qs = match_quarters.get(mid, {})
        if not qs:
            continue
        pair = tuple(sorted([ht, at]))
        h2h_games[pair].append({
            "mid": mid,
            "dt": info["date"],
            "home_team": ht,
            "away_team": at,
            "home_won": None,  # will compute from Q total
            "quarters": qs,
        })

    # Compute who won the game from quarter scores
    for pair, games in h2h_games.items():
        for g in games:
            total_h = sum(v[0] for v in g["quarters"].values())
            total_a = sum(v[1] for v in g["quarters"].values())
            g["home_won"] = total_h > total_a

    # Compute Q4 winner for each game
    for pair, games in h2h_games.items():
        for g in games:
            q4 = g["quarters"].get("Q4")
            if q4:
                g["q4_home_won"] = q4[0] > q4[1]
            else:
                g["q4_home_won"] = None

    print(f"  {len(h2h_games)} H2H pairs, "
          f"{sum(len(v) for v in h2h_games.values())} total H2H games")

    # 4. Load training samples
    cache = joblib.load(
        str(ROOT / "match" / "training" / "model_outputs_m27_v2"
            / "dynamic_rows_cache.joblib")
    )
    print(f"  {len(cache)} training samples")

    # 5. For each sample, compute H2H quarter features
    rows = []
    for cache_row in cache:
        tgt = cache_row.get("target")
        if tgt is None:
            continue
        mid = cache_row["match_id"]
        sample_dt = _parse_dt(cache_row["dt"])

        info = match_info.get(mid)
        if not info:
            continue
        ht, at = info["home_team"], info["away_team"]

        # Look up past H2H games before this sample date
        pair = tuple(sorted([ht, at]))
        past = [g for g in h2h_games.get(pair, [])
                if g["dt"] < sample_dt and g["mid"] != mid]
        # Also exclude the current match itself
        past = [g for g in past if g["mid"] != mid]

        if not past:
            continue

        # Features from past meetings (from home team perspective)
        n_h2h = len(past)

        # Q1 diffs in past meetings
        q1_diffs = []
        q2_diffs = []
        q3_diffs = []
        q4_diffs = []
        home_won_count = 0
        home_won_q4_count = 0
        home_won_q1_count = 0
        home_won_q2_count = 0
        home_won_q3_count = 0

        for g in past:
            qs = g["quarters"]
            # Determine perspective: is our home team the home team in this past game?
            is_home_in_past = (g["home_team"] == ht)
            sign = 1 if is_home_in_past else -1

            q1 = qs.get("Q1")
            q2 = qs.get("Q2")
            q3 = qs.get("Q3")
            q4 = qs.get("Q4")

            if q1:
                q1_diffs.append((q1[0] - q1[1]) * sign)
                if q1[0] > q1[1]:
                    home_won_q1_count += 1 if is_home_in_past else 0
                elif q1[0] < q1[1]:
                    home_won_q1_count += 0 if is_home_in_past else 1
            if q2:
                q2_diffs.append((q2[0] - q2[1]) * sign)
                if q2[0] > q2[1]:
                    home_won_q2_count += 1 if is_home_in_past else 0
                elif q2[0] < q2[1]:
                    home_won_q2_count += 0 if is_home_in_past else 1
            if q3:
                q3_diffs.append((q3[0] - q3[1]) * sign)
                if q3[0] > q3[1]:
                    home_won_q3_count += 1 if is_home_in_past else 0
                elif q3[0] < q3[1]:
                    home_won_q3_count += 0 if is_home_in_past else 1
            if q4:
                q4_diffs.append((q4[0] - q4[1]) * sign)
                if q4[0] > q4[1]:
                    home_won_q4_count += 1 if is_home_in_past else 0
                elif q4[0] < q4[1]:
                    home_won_q4_count += 0 if is_home_in_past else 1

            home_won_count += 1 if g["home_won"] == is_home_in_past else 0

        avg_q1_diff = sum(q1_diffs) / len(q1_diffs) if q1_diffs else None
        avg_q2_diff = sum(q2_diffs) / len(q2_diffs) if q2_diffs else None
        avg_q3_diff = sum(q3_diffs) / len(q3_diffs) if q3_diffs else None
        avg_q4_diff = sum(q4_diffs) / len(q4_diffs) if q4_diffs else None
        avg_game_diff = (sum(q1_diffs) + sum(q2_diffs) + sum(q3_diffs) + sum(q4_diffs)) / (len(q1_diffs) + len(q2_diffs) + len(q3_diffs) + len(q4_diffs)) if (q1_diffs or q2_diffs or q3_diffs or q4_diffs) else None
        home_wr = home_won_count / n_h2h if n_h2h > 0 else None
        q4_wr = home_won_q4_count / len(q4_diffs) if q4_diffs else None
        q1_wr = home_won_q1_count / len(q1_diffs) if q1_diffs else None
        q2_wr = home_won_q2_count / len(q2_diffs) if q2_diffs else None
        q3_wr = home_won_q3_count / len(q3_diffs) if q3_diffs else None

        # Feature: is home team on a H2H losing streak?
        recent_3 = past[-3:] if len(past) >= 3 else past
        home_won_recent_3 = sum(
            1 for g in recent_3 if g["home_won"] == (g["home_team"] == ht)
        )

        # Feature: last H2H result
        if past:
            last = past[-1]
            last_h2h_home_won = 1 if (last["home_won"] == (last["home_team"] == ht)) else 0
        else:
            last_h2h_home_won = None

        rows.append({
            "target": int(tgt),
            "n_h2h": n_h2h,
            "h2h_home_wr": home_wr,
            "h2h_q4_home_wr": q4_wr,
            "h2h_q1_home_wr": q1_wr,
            "h2h_q2_home_wr": q2_wr,
            "h2h_q3_home_wr": q3_wr,
            "h2h_avg_q1_diff": avg_q1_diff,
            "h2h_avg_q2_diff": avg_q2_diff,
            "h2h_avg_q3_diff": avg_q3_diff,
            "h2h_avg_q4_diff": avg_q4_diff,
            "h2h_avg_game_diff": avg_game_diff,
            "h2h_last_home_won": last_h2h_home_won,
            "h2h_recent3_home_won": home_won_recent_3 / len(recent_3) if recent_3 else None,
        })

    df = pd.DataFrame(rows)
    print(f"\n  {len(df)} samples with H2H history")

    # 6. Correlations
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

    HR = "=" * 80
    print(f"\n{HR}")
    print("  H2H Quarter-by-Quarter Feature Correlations with Q4 Target")
    print(HR)
    print(f"  {'Feature':<30s} {'Corr':>8s} {'|Corr|':>8s} {'N':>8s}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        print(f"  {r['feature']:<30s} {r['corr']:+8.4f} {r['abs_corr']:8.4f} {r['n']:>8d}")

    # Compare quarters
    print(f"\n{HR}")
    print("  Comparison by quarter (where available)")
    print(HR)
    for q in ["q1", "q2", "q3", "q4"]:
        wr_feat = f"h2h_{q}_home_wr"
        diff_feat = f"h2h_avg_{q}_diff"
        wr = next((r for r in results if r["feature"] == wr_feat), None)
        diff = next((r for r in results if r["feature"] == diff_feat), None)
        if wr:
            print(f"  {q.upper()} win rate: {wr['corr']:+7.4f} (n={wr['n']})"
                  f"  |  avg diff: {diff['corr']:+7.4f} (n={diff['n']})" if diff else "")

    # Save
    out = ROOT / "findings" / "h2h_quarter_correlation.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved to {out}")


def _parse_dt(s):
    return dt_mod.strptime(s.split(" ")[0], "%Y-%m-%d")


if __name__ == "__main__":
    main()
