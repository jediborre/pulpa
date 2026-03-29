import sqlite3
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "matches.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    out_lines = []
    
    # Check if table exists
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='eval_match_results'")
    if not cursor.fetchone():
        out_lines.append("Table 'eval_match_results' not found or empty.")
    else:
        df = pd.read_sql_query("SELECT * FROM eval_match_results", conn)
        versions = ["v4", "v5", "v6", "v7"]
        
        if len(df) == 0:
            out_lines.append("No evaluations found in database.")
        else:
            out_lines.append(f"Total Rows Evaluated in Database: {len(df)}")
            
            for quarter in ["q3", "q4"]:
                out_lines.append(f"\n=====================================")
                out_lines.append(f"    MODEL COMPARISON FOR {quarter.upper()} (Threshold > 65%)")
                out_lines.append(f"=====================================")
                out_lines.append(f"{'Version':<7} | {'Total Bets':<10} | {'Hits':<5} | {'Misses':<6} | {'Hit Rate':<8} | {'ROI (1.91)':<10}")
                out_lines.append("-" * 65)
                
                for v in versions:
                    signal_col = f"{quarter}_signal__{v}"
                    outcome_col = f"{quarter}_outcome__{v}"
                    
                    if signal_col not in df.columns or outcome_col not in df.columns:
                        out_lines.append(f"{v:<7} | Not calculated yet")
                        continue
                        
                    filtered = df[df[signal_col] == "BET"].copy()
                    total_bets = len(filtered)
                    
                    if total_bets == 0:
                        out_lines.append(f"{v:<7} | 0          | 0     | 0      | 0.00%    | 0.00%")
                        continue
                    
                    hits = len(filtered[filtered[outcome_col] == "hit"])
                    misses = len(filtered[filtered[outcome_col] == "miss"])
                    
                    if (hits + misses) == 0:
                        out_lines.append(f"{v:<7} | {total_bets:<10} | {hits:<5} | {misses:<6} | 0.00%    | 0.00%")
                        continue
                        
                    hit_rate = hits / (hits + misses)
                    profit = (hits * 0.91) - misses
                    roi = profit / (hits + misses)
                    
                    out_lines.append(f"{v:<7} | {hits+misses:<10} | {hits:<5} | {misses:<6} | {hit_rate*100:>6.2f}%  | {roi*100:>7.2f}%")
                    
    with open("results_roi.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))

if __name__ == "__main__":
    main()
