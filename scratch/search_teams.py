import sqlite3

dbs = {
    "matches": r"c:\Users\App\Desktop\pulpa\match\matches.db",
    "schedule": r"c:\Users\App\Desktop\pulpa\match\bet_monitor_schedule.db"
}

teams = [
    "Amambay", "Félix Pérez", "Cavaliers", "Knicks", "Osos de Manatí", "Cangrejeros De Santurce", "Manatí", "Santurce"
]

for db_name, db_path in dbs.items():
    print(f"\n=================== DB: {db_name} ({db_path}) ===================")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cursor.fetchall()]
    print("Tables:", tables)
    
    for table in tables:
        # Check if table has home_team/away_team or similar columns
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [c[1] for c in cursor.fetchall()]
        
        # Build search query if team-like columns exist
        search_cols = []
        if "home_team" in cols:
            search_cols.append("home_team")
        if "away_team" in cols:
            search_cols.append("away_team")
        if "home" in cols:
            search_cols.append("home")
        if "away" in cols:
            search_cols.append("away")
            
        if search_cols:
            where_clauses = []
            for t in teams:
                for col in search_cols:
                    where_clauses.append(f"{col} LIKE '%{t}%'")
            
            query = f"SELECT * FROM {table} WHERE " + " OR ".join(where_clauses)
            try:
                cursor.execute(query)
                rows = cursor.fetchall()
                if rows:
                    print(f"  Table '{table}' matches ({len(rows)} rows):")
                    for row in rows[:5]:
                        row_dict = dict(row)
                        # Remove potentially huge columns for clean printing
                        row_dict.pop("raw_json", None)
                        row_dict.pop("inference_json", None)
                        print("    ", row_dict)
            except Exception as e:
                print(f"  Error querying {table}: {e}")
                
    conn.close()
