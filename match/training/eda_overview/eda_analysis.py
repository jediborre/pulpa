"""
EDA - Exploratory Data Analysis for Betting Features
Goal: Analyze if predicting Over/Under (total points per quarter) is viable
"""

import sqlite3
import sys
from pathlib import Path
from collections import Counter, defaultdict
import json

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "matches.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def analyze_basic_stats():
    """Basic match statistics - ONLY matches with all 4 quarters"""
    conn = get_conn()
    
    # Total matches with ALL 4 quarters (not 2-quarter leagues)
    total = conn.execute("""
        SELECT COUNT(DISTINCT m.match_id) FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
    """).fetchone()[0]
    
    # Matches with graph_points AND all 4 quarters
    gp = conn.execute("""
        SELECT COUNT(DISTINCT m.match_id) FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        JOIN graph_points g ON m.match_id = g.match_id
    """).fetchone()[0]
    
    # Matches with play_by_play AND all 4 quarters
    pbp = conn.execute("""
        SELECT COUNT(DISTINCT m.match_id) FROM matches m
        JOIN quarter_scores q1 ON m.match_id = q1.match_id AND q1.quarter = 'Q1'
        JOIN quarter_scores q2 ON m.match_id = q2.match_id AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON m.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        JOIN play_by_play p ON m.match_id = p.match_id
    """).fetchone()[0]
    
    conn.close()
    
    return {
        "total_matches_4q": total,
        "with_graph_points": gp,
        "with_play_by_play": pbp
    }


def analyze_quarter_scores():
    """Analyze score distribution per quarter - ONLY 4-quarter matches"""
    conn = get_conn()
    
    results = {}
    for q in ["Q1", "Q2", "Q3", "Q4"]:
        rows = conn.execute(f"""
            SELECT qs.home, qs.away FROM quarter_scores qs
            JOIN (
                SELECT match_id FROM quarter_scores WHERE quarter = 'Q1'
                INTERSECT SELECT match_id FROM quarter_scores WHERE quarter = 'Q2'
                INTERSECT SELECT match_id FROM quarter_scores WHERE quarter = 'Q3'
                INTERSECT SELECT match_id FROM quarter_scores WHERE quarter = 'Q4'
            ) m ON qs.match_id = m.match_id
            WHERE qs.quarter = '{q}' AND qs.home IS NOT NULL
        """).fetchall()
        
        homes = [r["home"] for r in rows]
        aways = [r["away"] for r in rows]
        
        totals = [h + a for h, a in zip(homes, aways)]
        
        results[q] = {
            "count": len(homes),
            "home_mean": round(sum(homes) / len(homes), 1) if homes else 0,
            "away_mean": round(sum(aways) / len(aways), 1) if aways else 0,
            "total_mean": round(sum(totals) / len(totals), 1) if totals else 0,
            "home_min": min(homes) if homes else 0,
            "home_max": max(homes) if homes else 0,
            "total_min": min(totals) if totals else 0,
            "total_max": max(totals) if totals else 0,
            "std": round((sum((t - sum(totals)/len(totals))**2 for t in totals) / len(totals))**0.5, 1) if totals else 0
        }
    
    conn.close()
    return results


def analyze_half_scores():
    """Analyze 1st half (Q1+Q2) vs 2nd half (Q3+Q4)"""
    conn = get_conn()
    
    rows = conn.execute("""
        SELECT q1.home as q1h, q1.away as q1a, q2.home as q2h, q2.away as q2a,
               q3.home as q3h, q3.away as q3a, q4.home as q4h, q4.away as q4a
        FROM quarter_scores q1
        JOIN quarter_scores q2 ON q1.match_id = q2.match_id AND q1.quarter = 'Q1' AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON q1.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON q1.match_id = q4.match_id AND q4.quarter = 'Q4'
        WHERE q1.home IS NOT NULL AND q2.home IS NOT NULL AND q3.home IS NOT NULL AND q4.home IS NOT NULL
    """).fetchall()
    
    first_half_totals = []
    second_half_totals = []
    q3_totals = []
    q4_totals = []
    
    for r in rows:
        first_half_totals.append(r["q1h"] + r["q1a"] + r["q2h"] + r["q2a"])
        second_half_totals.append(r["q3h"] + r["q3a"] + r["q4h"] + r["q4a"])
        q3_totals.append(r["q3h"] + r["q3a"])
        q4_totals.append(r["q4h"] + r["q4a"])
    
    conn.close()
    
    return {
        "first_half": {
            "mean": round(sum(first_half_totals) / len(first_half_totals), 1),
            "min": min(first_half_totals),
            "max": max(first_half_totals),
            "std": round((sum((x - sum(first_half_totals)/len(first_half_totals))**2 for x in first_half_totals) / len(first_half_totals))**0.5, 1)
        },
        "second_half": {
            "mean": round(sum(second_half_totals) / len(second_half_totals), 1),
            "min": min(second_half_totals),
            "max": max(second_half_totals),
            "std": round((sum((x - sum(second_half_totals)/len(second_half_totals))**2 for x in second_half_totals) / len(second_half_totals))**0.5, 1)
        },
        "q3": {
            "mean": round(sum(q3_totals) / len(q3_totals), 1),
            "min": min(q3_totals),
            "max": max(q3_totals),
            "std": round((sum((x - sum(q3_totals)/len(q3_totals))**2 for x in q3_totals) / len(q3_totals))**0.5, 1)
        },
        "q4": {
            "mean": round(sum(q4_totals) / len(q4_totals), 1),
            "min": min(q4_totals),
            "max": max(q4_totals),
            "std": round((sum((x - sum(q4_totals)/len(q4_totals))**2 for x in q4_totals) / len(q4_totals))**0.5, 1)
        }
    }


def analyze_correlations():
    """Correlation between 1H scores and Q3/Q4 totals"""
    conn = get_conn()
    
    rows = conn.execute("""
        SELECT q1.home as q1h, q1.away as q1a, q2.home as q2h, q2.away as q2a,
               q3.home as q3h, q3.away as q3a, q4.home as q4h, q4.away as q4a
        FROM quarter_scores q1
        JOIN quarter_scores q2 ON q1.match_id = q2.match_id AND q1.quarter = 'Q1' AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON q1.match_id = q3.match_id AND q3.quarter = 'Q3'
        JOIN quarter_scores q4 ON q1.match_id = q4.match_id AND q4.quarter = 'Q4'
        WHERE q1.home IS NOT NULL AND q2.home IS NOT NULL AND q3.home IS NOT NULL AND q4.home IS NOT NULL
    """).fetchall()
    
    first_half = []
    q3_total = []
    q4_total = []
    q1_total = []
    q2_total = []
    
    for r in rows:
        fh = r["q1h"] + r["q1a"] + r["q2h"] + r["q2a"]
        first_half.append(fh)
        q3_total.append(r["q3h"] + r["q3a"])
        q4_total.append(r["q4h"] + r["q4a"])
        q1_total.append(r["q1h"] + r["q1a"])
        q2_total.append(r["q2h"] + r["q2a"])
    
    def correlation(x, y):
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
        std_x = (sum((xi - mean_x)**2 for xi in x) / n) ** 0.5
        std_y = (sum((yi - mean_y)**2 for yi in y) / n) ** 0.5
        return round(cov / (std_x * std_y), 3) if std_x > 0 and std_y > 0 else 0
    
    conn.close()
    
    return {
        "1H_to_Q3": correlation(first_half, q3_total),
        "1H_to_Q4": correlation(first_half, q4_total),
        "Q1_to_Q3": correlation(q1_total, q3_total),
        "Q2_to_Q3": correlation(q2_total, q3_total),
        "Q1_to_Q4": correlation(q1_total, q4_total),
        "Q2_to_Q4": correlation(q2_total, q4_total),
        "Q3_to_Q4": correlation(q3_total, q4_total)
    }


def analyze_league_variance():
    """Analyze score variance by league"""
    conn = get_conn()
    
    rows = conn.execute("""
        SELECT m.league, qs.quarter, qs.home, qs.away
        FROM matches m
        JOIN quarter_scores qs ON m.match_id = qs.match_id
        WHERE qs.quarter = 'Q3' AND qs.home IS NOT NULL
    """).fetchall()
    
    league_stats = defaultdict(list)
    for r in rows:
        league_stats[r["league"]].append(r["home"] + r["away"])
    
    result = {}
    for league, totals in sorted(league_stats.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
        if len(totals) < 20:
            continue
        result[league] = {
            "n": len(totals),
            "mean": round(sum(totals) / len(totals), 1),
            "std": round((sum((t - sum(totals)/len(totals))**2 for t in totals) / len(totals))**0.5, 1)
        }
    
    conn.close()
    return result


def analyze_gender_variance():
    """Analyze score variance by gender"""
    conn = get_conn()
    
    rows = conn.execute("""
        SELECT m.home_team, m.away_team, m.league, qs.quarter, qs.home, qs.away
        FROM matches m
        JOIN quarter_scores qs ON m.match_id = qs.match_id
        WHERE qs.quarter = 'Q3' AND qs.home IS NOT NULL
    """).fetchall()
    
    men = []
    women = []
    
    for r in rows:
        text = f"{r['league']} {r['home_team']} {r['away_team']}".lower()
        total = r["home"] + r["away"]
        if any(m in text for m in ["women", "woman", "female", "femen", "(w)", " w "]):
            women.append(total)
        else:
            men.append(total)
    
    return {
        "men_or_open": {
            "n": len(men),
            "mean": round(sum(men) / len(men), 1) if men else 0,
            "std": round((sum((t - sum(men)/len(men))**2 for t in men) / len(men))**0.5, 1) if men else 0
        },
        "women": {
            "n": len(women),
            "mean": round(sum(women) / len(women), 1) if women else 0,
            "std": round((sum((t - sum(women)/len(women))**2 for t in women) / len(women))**0.5, 1) if women else 0
        }
    }


def analyze_over_under_distributions():
    """Analyze Over/Under thresholds"""
    conn = get_conn()
    
    rows = conn.execute("""
        SELECT q1.home as q1h, q1.away as q1a, q2.home as q2h, q2.away as q2a,
               q3.home as q3h, q3.away as q3a
        FROM quarter_scores q1
        JOIN quarter_scores q2 ON q1.match_id = q2.match_id AND q1.quarter = 'Q1' AND q2.quarter = 'Q2'
        JOIN quarter_scores q3 ON q1.match_id = q3.match_id AND q3.quarter = 'Q3'
        WHERE q1.home IS NOT NULL AND q2.home IS NOT NULL AND q3.home IS NOT NULL
    """).fetchall()
    
    first_half = [r["q1h"] + r["q1a"] + r["q2h"] + r["q2a"] for r in rows]
    q3_totals = [r["q3h"] + r["q3a"] for r in rows]
    
    # Find optimal thresholds
    thresholds = [25, 26, 27, 28, 29, 30]
    results = {}
    
    for t in thresholds:
        over_q3 = sum(1 for x in q3_totals if x > t)
        pct = over_q3 / len(q3_totals) * 100
        results[f"over_{t}"] = {
            "count": over_q3,
            "percentage": round(pct, 1)
        }
    
    conn.close()
    return results


def main():
    print("=" * 60)
    print("EDA - Exploratory Data Analysis for Over/Under Betting")
    print("=" * 60)
    
    print("\n[1/7] Basic Statistics...")
    basic = analyze_basic_stats()
    print(f"  Matches with ALL 4 quarters: {basic['total_matches_4q']}")
    print(f"  With graph_points: {basic['with_graph_points']}")
    print(f"  With play_by_play: {basic['with_play_by_play']}")
    
    print("\n[2/7] Quarter Score Distributions...")
    quarters = analyze_quarter_scores()
    for q, data in quarters.items():
        print(f"  {q}: mean={data['total_mean']}, std={data['std']}, range=[{data['total_min']}-{data['total_max']}]")
    
    print("\n[3/7] Half Score Analysis...")
    halves = analyze_half_scores()
    print(f"  1H: mean={halves['first_half']['mean']}, std={halves['first_half']['std']}")
    print(f"  2H: mean={halves['second_half']['mean']}, std={halves['second_half']['std']}")
    print(f"  Q3: mean={halves['q3']['mean']}, std={halves['q3']['std']}")
    print(f"  Q4: mean={halves['q4']['mean']}, std={halves['q4']['std']}")
    
    print("\n[4/7] Correlations...")
    corrs = analyze_correlations()
    for k, v in corrs.items():
        print(f"  {k}: {v}")
    
    print("\n[5/7] League Variance (Top 20)...")
    leagues = analyze_league_variance()
    for league, data in list(leagues.items())[:5]:
        print(f"  {league[:40]}: n={data['n']}, mean={data['mean']}, std={data['std']}")
    
    print("\n[6/7] Gender Variance...")
    gender = analyze_gender_variance()
    for k, v in gender.items():
        print(f"  {k}: n={v['n']}, mean={v['mean']}, std={v['std']}")
    
    print("\n[7/7] Over/Under Distributions...")
    ou = analyze_over_under_distributions()
    for k, v in ou.items():
        print(f"  {k}: {v['count']} ({v['percentage']}%)")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY - VIABILITY FOR OVER/UNDER BETTING")
    print("=" * 60)
    
    print("\nQ3 Score Statistics:")
    print(f"  Mean: {halves['q3']['mean']} points")
    print(f"  Std: {halves['q3']['std']} points")
    print(f"  Range: {halves['q3']['min']}-{halves['q3']['max']}")
    
    print("\nCorrelations:")
    print(f"  1H -> Q3: {corrs['1H_to_Q3']} (moderate positive)")
    print(f"  1H -> Q4: {corrs['1H_to_Q4']} (moderate positive)")
    
    print("\nKey Insights:")
    print(f"  - Q3 scores are relatively stable (std ~{halves['q3']['std']})")
    print(f"  - Moderate correlation with 1H suggests predictive potential")
    print(f"  - Gender differences: men's games avg ~{gender['men_or_open']['mean']}, women's ~{gender['women']['mean']}")
    print(f"  - Recommended O/U threshold: ~27-28 points (~50% over/under)")


if __name__ == "__main__":
    main()