from collections import defaultdict
from statistics import quantiles
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "match" / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import train_q3_q4_models_v6 as v6
import infer_match as infer_live


def winner(home, away):
    if home is None or away is None:
        return "M"
    if home > away:
        return "H"
    if away > home:
        return "A"
    return "T"


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((value - mean_x) ** 2 for value in xs)
    var_y = sum((value - mean_y) ** 2 for value in ys)
    if var_x <= 1e-12 or var_y <= 1e-12:
        return float("nan")
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def rate(values):
    return sum(values) / len(values) if values else float("nan")


def deficit_bin(diff_abs):
    if diff_abs <= 3:
        return "01_03"
    if diff_abs <= 7:
        return "04_07"
    return "08_plus"


def main():
    conn = v6.db_mod.get_conn(str(v6.DB_PATH))
    v6.db_mod.init_db(conn)
    try:
        samples = v6._build_samples(v6.DB_PATH)
        rows = [sample for sample in samples if sample.target_q4 is not None]
        print(f"TOTAL_ROWS {len(rows)}")

        combo_stats = defaultdict(lambda: [0, 0])
        pattern_stats = defaultdict(lambda: [0, 0])
        pressure_stats = defaultdict(lambda: [0, 0])
        snap_features = {
            27: defaultdict(list),
            30: defaultdict(list),
        }

        for sample in rows:
            data = v6.db_mod.get_match(conn, str(sample.match_id))
            if not data:
                continue

            q1h, q1a = v6._quarter_points(data, "Q1")
            q2h, q2a = v6._quarter_points(data, "Q2")
            q3h, q3a = v6._quarter_points(data, "Q3")
            q4h, q4a = v6._quarter_points(data, "Q4")
            w1 = winner(q1h, q1a)
            w2 = winner(q2h, q2a)
            w3 = winner(q3h, q3a)
            w4 = winner(q4h, q4a)
            if "M" in (w1, w2, w3, w4):
                continue

            target_home_q4 = 1 if w4 == "H" else 0
            combo = f"{w1}{w2}{w3}"
            combo_stats[combo][0] += 1
            combo_stats[combo][1] += target_home_q4

            score_3q_home = int(q1h or 0) + int(q2h or 0) + int(q3h or 0)
            score_3q_away = int(q1a or 0) + int(q2a or 0) + int(q3a or 0)
            leader = "H" if score_3q_home > score_3q_away else ("A" if score_3q_away > score_3q_home else "T")

            home_wins_first3 = sum(1 for value in (w1, w2, w3) if value == "H")
            away_wins_first3 = sum(1 for value in (w1, w2, w3) if value == "A")
            ties_first3 = sum(1 for value in (w1, w2, w3) if value == "T")

            keys = [
                f"q3_{w3}",
                f"leader_after_q3_{leader}",
                f"home_wins_first3_{home_wins_first3}",
                f"away_wins_first3_{away_wins_first3}",
                f"ties_first3_{ties_first3}",
                f"q3_same_as_leader_{int(w3 == leader)}",
                f"sweep_first3_home_{int((w1, w2, w3) == ('H', 'H', 'H'))}",
                f"sweep_first3_away_{int((w1, w2, w3) == ('A', 'A', 'A'))}",
            ]
            for key in keys:
                pattern_stats[key][0] += 1
                pattern_stats[key][1] += target_home_q4

            ht_home = int(q1h or 0) + int(q2h or 0)
            ht_away = int(q1a or 0) + int(q2a or 0)
            halftime_diff = ht_home - ht_away
            for snap in (27, 30):
                est_home, est_away = infer_live._score_upto(data, snap)
                graph = infer_live._graph_stats_upto(data.get("graph_points", []), snap)
                store = snap_features[snap]
                store["target"].append(target_home_q4)
                store["score_diff"].append(est_home - est_away)
                store["abs_score_diff"].append(abs(est_home - est_away))
                store["gp_last"].append(graph.get("gp_last", 0))
                store["abs_gp_last"].append(abs(graph.get("gp_last", 0)))
                store["gp_mean_abs"].append(graph.get("gp_mean_abs", 0.0))
                store["gp_swings"].append(graph.get("gp_swings", 0))
                if snap == 27:
                    q3_partial_diff = (est_home - ht_home) - (est_away - ht_away)
                    store["q3_partial_diff"].append(q3_partial_diff)

                    current_diff = est_home - est_away
                    run_last_3m = q3_partial_diff
                    if current_diff != 0 and w4 in ("H", "A"):
                        trailing_side = "H" if current_diff < 0 else "A"
                        trailing_won_q4 = int(w4 == trailing_side)
                        pressure_stats["trail_now_any"][0] += 1
                        pressure_stats["trail_now_any"][1] += trailing_won_q4
                        pressure_stats[
                            f"trail_now_{trailing_side}_def_{deficit_bin(abs(current_diff))}"
                        ][0] += 1
                        pressure_stats[
                            f"trail_now_{trailing_side}_def_{deficit_bin(abs(current_diff))}"
                        ][1] += trailing_won_q4
                        pressure_stats[
                            f"trail_now_any_def_{deficit_bin(abs(current_diff))}"
                        ][0] += 1
                        pressure_stats[
                            f"trail_now_any_def_{deficit_bin(abs(current_diff))}"
                        ][1] += trailing_won_q4

                        trailing_has_recent_run = (
                            (trailing_side == "H" and run_last_3m > 0)
                            or (trailing_side == "A" and run_last_3m < 0)
                        )
                        pressure_stats[
                            f"trail_now_recent_run_{int(trailing_has_recent_run)}"
                        ][0] += 1
                        pressure_stats[
                            f"trail_now_recent_run_{int(trailing_has_recent_run)}"
                        ][1] += trailing_won_q4

                        trailing_also_trailed_ht = (
                            (trailing_side == "H" and halftime_diff < 0)
                            or (trailing_side == "A" and halftime_diff > 0)
                        )
                        pressure_stats[
                            f"trail_now_also_ht_{int(trailing_also_trailed_ht)}"
                        ][0] += 1
                        pressure_stats[
                            f"trail_now_also_ht_{int(trailing_also_trailed_ht)}"
                        ][1] += trailing_won_q4

                    if halftime_diff != 0 and w4 in ("H", "A"):
                        halftime_trailer = "H" if halftime_diff < 0 else "A"
                        halftime_trailer_won_q4 = int(w4 == halftime_trailer)
                        pressure_stats["trail_ht_any"][0] += 1
                        pressure_stats["trail_ht_any"][1] += halftime_trailer_won_q4
                        pressure_stats[
                            f"trail_ht_any_def_{deficit_bin(abs(halftime_diff))}"
                        ][0] += 1
                        pressure_stats[
                            f"trail_ht_any_def_{deficit_bin(abs(halftime_diff))}"
                        ][1] += halftime_trailer_won_q4

                        halftime_trailer_is_cutting = (
                            (halftime_trailer == "H" and q3_partial_diff > 0)
                            or (halftime_trailer == "A" and q3_partial_diff < 0)
                        )
                        pressure_stats[
                            f"trail_ht_cutting_in_q3_{int(halftime_trailer_is_cutting)}"
                        ][0] += 1
                        pressure_stats[
                            f"trail_ht_cutting_in_q3_{int(halftime_trailer_is_cutting)}"
                        ][1] += halftime_trailer_won_q4

        print("TOP_COMBOS")
        combos_sorted = sorted(combo_stats.items(), key=lambda item: (-item[1][0], item[0]))
        for combo, (count, home_q4_wins) in combos_sorted[:15]:
            print(combo, count, round(home_q4_wins / count, 4))

        print("COMBO_EXTREMES_MIN100")
        combo_extremes = [
            (combo, values)
            for combo, values in combo_stats.items()
            if values[0] >= 100
        ]
        combo_extremes.sort(
            key=lambda item: (-abs(item[1][1] / item[1][0] - 0.5), -item[1][0], item[0])
        )
        for combo, (count, home_q4_wins) in combo_extremes[:12]:
            print(combo, count, round(home_q4_wins / count, 4))

        print("PATTERNS")
        for key in sorted(pattern_stats):
            count, home_q4_wins = pattern_stats[key]
            print(key, count, round(home_q4_wins / count, 4))

        print("PRESSURE_M27")
        for key in sorted(pressure_stats):
            count, wins = pressure_stats[key]
            if count < 100:
                continue
            print(key, count, round(wins / count, 4))

        for snap in (27, 30):
            print(f"CORR_SNAP {snap}")
            target = snap_features[snap]["target"]
            feature_names = [name for name in snap_features[snap] if name != "target"]
            for feature_name in feature_names:
                value = corr(snap_features[snap][feature_name], target)
                print(feature_name, round(value, 4))

            for feature_name in ("abs_score_diff", "abs_gp_last", "gp_mean_abs", "gp_swings"):
                values = snap_features[snap][feature_name]
                q1, _, q3 = quantiles(values, n=4)
                low_bucket = [y for value, y in zip(values, target) if value <= q1]
                high_bucket = [y for value, y in zip(values, target) if value >= q3]
                print(
                    "BUCKET",
                    feature_name,
                    "low_n",
                    len(low_bucket),
                    "low_home_q4",
                    round(rate(low_bucket), 4),
                    "high_n",
                    len(high_bucket),
                    "high_home_q4",
                    round(rate(high_bucket), 4),
                )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
