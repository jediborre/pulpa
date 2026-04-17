"""
Daily Evaluation Report by Model and Quarter

Genera un reporte desglosado por:
- Fecha
- Modelo (v4, v5, v6, v7, etc.)
- Quarter (Q3, Q4)
"""

import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict

DB_PATH = Path(__file__).resolve().parents[1] / "matches.db"
ODDS = 1.91


def get_models():
    """Get list of available model columns"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cols = conn.execute("PRAGMA table_info(eval_match_results)").fetchall()
    conn.close()
    
    models = set()
    for c in cols:
        name = c[1]
        if name.startswith("q3_pick__"):
            model = name.replace("q3_pick__", "")
            models.add(model)
    return sorted(models)


def list_dates():
    """List available dates in eval_match_results"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    dates = conn.execute(
        "SELECT DISTINCT event_date FROM eval_match_results ORDER BY event_date DESC"
    ).fetchall()
    conn.close()
    
    print("Available dates:")
    for d in dates:
        print(f"  {d[0]}")


def generate_day_stats(date: str, models: list) -> dict:
    """Get stats for a single day"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute(
        "SELECT * FROM eval_match_results WHERE event_date = ?",
        (date,)
    ).fetchall()
    conn.close()
    
    if not rows:
        return {}
    
    all_stats = {}
    
    for model in models:
        q3_stats = {"samples": 0, "bets": 0, "hits": 0, "misses": 0}
        q4_stats = {"samples": 0, "bets": 0, "hits": 0, "misses": 0}
        
        for row in rows:
            row_dict = dict(row)
            
            # Q3
            available = row_dict.get(f"q3_available__{model}")
            pick = row_dict.get(f"q3_pick__{model}")
            outcome = row_dict.get(f"q3_outcome__{model}")
            
            if available == 1:
                q3_stats["samples"] += 1
                if pick in ("home", "away"):
                    q3_stats["bets"] += 1
                    if outcome == "hit":
                        q3_stats["hits"] += 1
                    elif outcome == "miss":
                        q3_stats["misses"] += 1
            
            # Q4
            available = row_dict.get(f"q4_available__{model}")
            pick = row_dict.get(f"q4_pick__{model}")
            outcome = row_dict.get(f"q4_outcome__{model}")
            
            if available == 1:
                q4_stats["samples"] += 1
                if pick in ("home", "away"):
                    q4_stats["bets"] += 1
                    if outcome == "hit":
                        q4_stats["hits"] += 1
                    elif outcome == "miss":
                        q4_stats["misses"] += 1
        
        all_stats[model] = {"q3": q3_stats, "q4": q4_stats}
    
    return all_stats


def generate_month_report(year: int, month: int, model_filter: str = None):
    """Generate report for a full month"""
    import datetime
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    # Get dates in the month
    date_pattern = f"{year}-{month:02d}%"
    rows = conn.execute(
        "SELECT DISTINCT event_date FROM eval_match_results WHERE event_date LIKE ? ORDER BY event_date",
        (date_pattern,)
    ).fetchall()
    conn.close()
    
    if not rows:
        print(f"No data for {year}-{month:02d}")
        return
    
    dates = [r[0] for r in rows]
    
    print("=" * 100)
    print(f"MONTHLY REPORT - {year}-{month:02d}")
    print("=" * 100)
    print(f"Days with data: {len(dates)}")
    print()
    
    # Get models
    all_models = get_models()
    if model_filter:
        models = [m for m in all_models if model_filter in m]
    else:
        models = ["v4", "v5", "v6", "v7"]
    
    # Header
    print(f"{'Date':<12} {'Model':<8} {'Q':<3} {'Samples':>8} {'Bets':>8} {'Hits':>7} {'Rate':>7} {'Profit':>10} {'ROI':>7}")
    print("-" * 100)
    
    daily_summary = {}
    monthly_totals = defaultdict(lambda: {"bets": 0, "hits": 0, "profit": 0})
    
    for date in dates:
        stats = generate_day_stats(date, models)
        daily_summary[date] = {"bets": 0, "hits": 0, "profit": 0}
        
        for model in models:
            for q in ["q3", "q4"]:
                s = stats.get(model, {}).get(q, {})
                samples = s.get("samples", 0)
                bets = s.get("bets", 0)
                hits = s.get("hits", 0)
                misses = s.get("misses", 0)
                
                if samples == 0:
                    continue
                
                if bets == 0:
                    rate = 0
                    profit = 0
                    roi = 0
                else:
                    rate = hits / bets * 100
                    profit = hits * (ODDS - 1) - misses
                    roi = profit / bets * 100
                
                quarter = "Q3" if q == "q3" else "Q4"
                print(f"{date:<12} {model:<8} {quarter:<3} {samples:>8} {bets:>8} {hits:>7} {rate:>6.1f}% {profit:>10.2f} {roi:>6.1f}%")
                
                daily_summary[date]["bets"] += bets
                daily_summary[date]["hits"] += hits
                daily_summary[date]["profit"] += profit
                
                monthly_totals[model + quarter]["bets"] += bets
                monthly_totals[model + quarter]["hits"] += hits
                monthly_totals[model + quarter]["profit"] += profit
    
    print("-" * 100)
    
    # Daily totals
    print(f"\n{'DAILY TOTALS':<20}")
    print("-" * 60)
    total_profit = 0
    for date in dates:
        d = daily_summary.get(date, {})
        if d.get("bets", 0) == 0:
            continue
        rate = d["hits"] / d["bets"] * 100
        print(f"{date}: {d['bets']:>4} bets, {d['hits']:>4} hits ({rate:.1f}%), Profit: {d['profit']:>8.2f}")
        total_profit += d["profit"]
    
    print("-" * 60)
    print(f"TOTAL PROFIT: {total_profit:.2f}")
    """Get list of available model columns"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cols = conn.execute("PRAGMA table_info(eval_match_results)").fetchall()
    conn.close()
    
    models = set()
    for c in cols:
        name = c[1]
        if name.startswith("q3_pick__"):
            model = name.replace("q3_pick__", "")
            models.add(model)
    return sorted(models)


def generate_report(date: str, model_filter: str = None):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    rows = conn.execute(
        "SELECT * FROM eval_match_results WHERE event_date = ?",
        (date,)
    ).fetchall()
    conn.close()
    
    if not rows:
        print(f"No data for date: {date}")
        list_dates()
        return
    
    models = get_models()
    if model_filter:
        models = [m for m in models if model_filter in m]
    
    print("=" * 100)
    print(f"EVALUATION REPORT - {date}")
    print("=" * 100)
    print(f"Total matches: {len(rows)}")
    print()
    
    all_stats = {}
    
    for model in models:
        q3_stats = {"samples": 0, "bets": 0, "hits": 0, "misses": 0}
        q4_stats = {"samples": 0, "bets": 0, "hits": 0, "misses": 0}
        
        for row in rows:
            row_dict = dict(row)
            
            # Q3
            available = row_dict.get(f"q3_available__{model}")
            pick = row_dict.get(f"q3_pick__{model}")
            outcome = row_dict.get(f"q3_outcome__{model}")
            
            if available == 1:
                q3_stats["samples"] += 1
                if pick in ("home", "away"):
                    q3_stats["bets"] += 1
                    if outcome == "hit":
                        q3_stats["hits"] += 1
                    elif outcome == "miss":
                        q3_stats["misses"] += 1
            
            # Q4
            available = row_dict.get(f"q4_available__{model}")
            pick = row_dict.get(f"q4_pick__{model}")
            outcome = row_dict.get(f"q4_outcome__{model}")
            
            if available == 1:
                q4_stats["samples"] += 1
                if pick in ("home", "away"):
                    q4_stats["bets"] += 1
                    if outcome == "hit":
                        q4_stats["hits"] += 1
                    elif outcome == "miss":
                        q4_stats["misses"] += 1
        
        all_stats[model] = {"q3": q3_stats, "q4": q4_stats}
    
    # Print summary
    print(f"{'Model':<20} {'Quarter':<8} {'Samples':>8} {'Bets':>8} {'Hits':>8} {'Rate':>8} {'Profit':>10} {'ROI':>8}")
    print("-" * 100)
    
    total_bets = 0
    total_profit = 0
    
    for model in models:
        stats = all_stats[model]
        
        for quarter, s in [("Q3", stats["q3"]), ("Q4", stats["q4"])]:
            samples = s["samples"]
            bets = s["bets"]
            hits = s["hits"]
            misses = s["misses"]
            
            if bets == 0:
                rate = 0
                profit = 0
                roi = 0
            else:
                rate = hits / bets * 100
                profit = hits * (ODDS - 1) - misses
                roi = profit / bets * 100
            
            if samples > 0:
                print(f"{model:<20} {quarter:<8} {samples:>8} {bets:>8} {hits:>8} {rate:>7.1f}% {profit:>10.2f} {roi:>7.1f}%")
                total_bets += bets
                total_profit += profit
    
    print("-" * 100)
    print(f"{'TOTAL':<28} {total_bets:>8} {'':>8} {'':>8} {'':>8} {total_profit:>10.2f}")
    print()
    
    return all_stats


def main():
    import datetime
    
    parser = argparse.ArgumentParser(description="Evaluation report by model and quarter")
    parser.add_argument("--date", type=str, help="Specific date (YYYY-MM-DD)")
    parser.add_argument("--month", type=str, help="Month (YYYY-MM)")
    parser.add_argument("--model", type=str, help="Filter by model name (e.g., 'v4', 'v6')")
    parser.add_argument("--list", action="store_true", help="List available dates")
    args = parser.parse_args()
    
    if args.list:
        list_dates()
        return
    
    if args.month:
        parts = args.month.split("-")
        year = int(parts[0])
        month = int(parts[1])
        generate_month_report(year, month, args.model)
        return
    
    if args.date:
        generate_report(args.date, args.model)
        return
    
    # Default: current month
    now = datetime.datetime.now()
    generate_month_report(now.year, now.month, args.model)


if __name__ == "__main__":
    main()
