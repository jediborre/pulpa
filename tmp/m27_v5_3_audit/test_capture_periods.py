# =====================================================================
# PRUEBA DE LA CAPTURA AMPLIADA (stats por cuarto) CON UN PAYLOAD FALSO
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.models.evaluator import _record_live_snapshot  # noqa: E402

payload = {
    "match": {"status_description": "3rd quarter"},
    "score": {"home": 60, "away": 55},
    "odds_live": {"ml_home": 1.8, "ml_away": 1.95},
    "lineups": [{"team": "home", "player_id": "1", "is_starter": True}],
    "player_stats": [{"team": "home", "player_id": "1", "is_starter": True,
                       "minutes_played": 20, "points": 12, "fouls": 3,
                       "plus_minus": 5, "sofascore_rating": 7.4}],
    # aggregated endpoint: period "1Q".."3Q" + "ALL"
    "team_statistics": [
        {"period": "3Q", "stat_name": "Fouls", "home_value": 10, "away_value": 6},
        {"period": "3Q", "stat_name": "Time spent in lead", "home_value": 0, "away_value": 600},
        {"period": "ALL", "stat_name": "Fouls", "home_value": 21, "away_value": 20},
        {"period": "ALL", "stat_name": "Biggest lead", "home_value": 8, "away_value": 12},
    ],
    # reconstruction from incidents: period "Q1".."Q4"
    "period_stats": [
        {"period": "Q1", "stat_name": "Fouls", "home_value": 2, "away_value": 4},
        {"period": "Q2", "stat_name": "Fouls", "home_value": 6, "away_value": 7},
        {"period": "Q3", "stat_name": "Fouls", "home_value": 10, "away_value": 6},
        {"period": "Q3", "stat_name": "Timeouts", "home_value": 1, "away_value": 2},
        {"period": "Q3", "stat_name": "Points", "home_value": 24, "away_value": 18},
        {"period": "Q4", "stat_name": "Fouls", "home_value": 0, "away_value": 0},
    ],
}

MID = "99999999"
_record_live_snapshot(MID, 27.0, payload)

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
r = conn.execute("SELECT * FROM live_snapshots_v3 WHERE match_id=? ORDER BY id DESC LIMIT 1",
                 (MID,)).fetchone()
print("SNAPSHOT GUARDADO:")
print(f"   match {r['match_id']} min {r['game_minute']} score {r['home_score']}-{r['away_score']}")
print("   team_stats_json:")
for x in json.loads(r["team_stats_json"]):
    print(f"      {x[0]:<5} {x[1]:<22} {x[2]} - {x[3]}")
print("   players_json:", r["players_json"])
print("   odds_live_json:", r["odds_live_json"])
conn.execute("DELETE FROM live_snapshots_v3 WHERE match_id=?", (MID,))
conn.commit()
print("(snapshot de prueba borrado)")
conn.close()
