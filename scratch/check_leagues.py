import sqlite3
import json

def main():
    conn = sqlite3.connect('match/matches.db')
    cur = conn.cursor()
    cur.execute('SELECT DISTINCT league FROM matches WHERE date LIKE "2026-04-17%"')
    db_leagues = {r[0] for r in cur.fetchall()}
    conn.close()

    print(f"Leagues in DB (17-Apr): {len(db_leagues)}")
    
    with open('match/training/v17/model_outputs/training_summary_v17.json') as f:
        summary = json.load(f)
    
    model_leagues = {m['league'] for m in summary['models']}
    print(f"Leagues in Model Summary: {len(model_leagues)}")
    
    matches = db_leagues.intersection(model_leagues)
    print(f"Intersection: {len(matches)}")
    
    missing = db_leagues - model_leagues
    if missing:
        print("\nMissing in Summary (found in DB):")
        for m in sorted(missing):
            print(f" - {m}")

if __name__ == "__main__":
    main()
