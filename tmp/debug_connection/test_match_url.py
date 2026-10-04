"""
tmp/debug_connection/test_match_url.py
Proposito: Probar _sofascore_match_url (v3) con los datos reales de la BD: con
match_data (event_slug/custom_id), sin match_data (fallback) y sin custom_id.
"""
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.notifications.telegram_bot import _sofascore_match_url

match_data = {
    "match": {
        "event_slug": "ldlc-asvel-lyon-villeurbanne-bcm-gravelines-dunkerque",
        "custom_id": "HvbsLvb",
        "home_slug": "bcm-gravelines-dunkerque",
        "away_slug": "ldlc-asvel-lyon-villeurbanne",
        "home_team": "BCM Gravelines-Dunkerque",
        "away_team": "LDLC ASVEL Lyon-Villeurbanne",
    }
}

print("con match_data :", _sofascore_match_url("16988998", match_data=match_data,
      home_team="BCM Gravelines-Dunkerque", away_team="LDLC ASVEL Lyon-Villeurbanne"))
print("sin match_data :", _sofascore_match_url("16988998",
      home_team="BCM Gravelines-Dunkerque", away_team="LDLC ASVEL Lyon-Villeurbanne"))
no_cid = {"match": dict(match_data["match"], custom_id="")}
print("sin custom_id  :", _sofascore_match_url("16988998", match_data=no_cid,
      home_team="BCM Gravelines-Dunkerque", away_team="LDLC ASVEL Lyon-Villeurbanne"))
print("al-ula         :", _sofascore_match_url("17249511", match_data={"match": {
    "event_slug": "al-ula-al-kuwait", "custom_id": "fEicsGRpi",
    "home_slug": "al-ula", "away_slug": "al-kuwait"}},
    home_team="Al-Ula", away_team="Al-Kuwait"))
