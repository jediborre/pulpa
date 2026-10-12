"""
tmp/missing_data_audit/status.py
Proposito: Ver, para cada tabla, cuantos partidos sin datos estan MARCADOS como
revisados (details_checked_at, ya intentados y con 404) vs PENDIENTES (sin marcar).
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
con = sqlite3.connect(ROOT / "matches.db")
total = con.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
checked_total = con.execute(
    "SELECT COUNT(*) FROM matches WHERE details_checked_at IS NOT NULL"
).fetchone()[0]
print(f"matches totales: {total} | marcados: {checked_total} | pendientes: {total - checked_total}\n")
print(f"{'tabla':<18} {'marcado(404)':>13} {'pendiente':>10}")
print("-" * 45)
for t in ["lineups", "player_stats", "team_statistics", "match_odds",
          "play_by_play", "graph_points", "team_strength"]:
    checked = con.execute(
        "SELECT COUNT(*) FROM matches m WHERE m.details_checked_at IS NOT NULL "
        f"AND NOT EXISTS (SELECT 1 FROM {t} x WHERE x.match_id = m.match_id)"
    ).fetchone()[0]
    unchecked = con.execute(
        "SELECT COUNT(*) FROM matches m WHERE m.details_checked_at IS NULL "
        f"AND NOT EXISTS (SELECT 1 FROM {t} x WHERE x.match_id = m.match_id)"
    ).fetchone()[0]
    print(f"{t:<18} {checked:>13} {unchecked:>10}")
con.close()
