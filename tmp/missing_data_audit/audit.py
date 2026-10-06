"""
tmp/missing_data_audit/audit.py
Proposito: Auditar matches.db para detectar partidos a los que les faltan datos de
detalle. Usa diferencias de conjuntos (mas rapido que NOT EXISTS sin indices).
"""
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "matches.db"

TABLES = [
    "quarter_scores", "play_by_play", "graph_points", "match_events",
    "match_h2h", "player_stats", "lineups", "team_statistics",
    "match_odds", "team_strength",
]


def main() -> None:
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    all_ids = {r[0] for r in con.execute("SELECT match_id FROM matches")}
    total = len(all_ids)
    print(f"matches totales: {total}\n")
    print(f"{'tabla':<18} {'con datos':>10} {'sin datos':>10} {'% falta':>8}   tiempo")
    print("-" * 60)
    for t in TABLES:
        t0 = time.perf_counter()
        try:
            present = {r[0] for r in con.execute(f"SELECT DISTINCT match_id FROM {t}")}
        except Exception as e:
            print(f"{t:<18} ERROR: {e}")
            continue
        missing = len(all_ids - present)
        dt = time.perf_counter() - t0
        pct = f"{int(round(missing * 100 / total))}%" if total else "-"
        print(f"{t:<18} {len(present):>10} {missing:>10} {pct:>8}   {dt:.1f}s")

    con.close()


if __name__ == "__main__":
    main()
