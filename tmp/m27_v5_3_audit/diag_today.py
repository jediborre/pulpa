# =====================================================================
# DIAGNOSTICO DE HOY: resultados de m27_v5_3 en el log de apuestas
# =====================================================================

import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row

HOY = "2026-10-20"

print("1) TODAS LAS APUESTAS DE HOY POR MODELO:")
for r in conn.execute(
        "SELECT model_version, signal_type, COUNT(*) n, "
        "SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) ganadas, "
        "SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) perdidas, "
        "SUM(CASE WHEN result IS NULL OR result='' THEN 1 ELSE 0 END) sin_resultado "
        "FROM bet_monitor_log_v3 WHERE substr(created_at,1,10)=? "
        "GROUP BY model_version, signal_type ORDER BY model_version", (HOY,)):
    d = dict(r)
    res = d["ganadas"] + d["perdidas"]
    wr = f"{d['ganadas'] * 100 / res:.1f}%" if res else "-"
    print(f"   {d['model_version']:<12} {d['signal_type']:<12} n={d['n']:>3} | "
          f"ganadas {d['ganadas']:>3} perdidas {d['perdidas']:>3} sin res {d['sin_resultado']:>3} | WR {wr}")

print()
print("2) APUESTAS DE HOY DE m27_v5_3 (detalle):")
print()
print("   columnas disponibles:", [d[1] for d in conn.execute("PRAGMA table_info(bet_monitor_log_v3)")])
for r in conn.execute(
        "SELECT b.*, m.home_team, m.away_team, m.league, m.date, m.status_type "
        "FROM bet_monitor_log_v3 b LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id "
        "WHERE substr(b.created_at,1,10)=? AND b.model_version='m27_v5_3' "
        "ORDER BY b.id", (HOY,)):
    print(f"   id {r['id']:>5} | {r['match_id']} | {str(r['home_team'])[:20]:<20} vs "
          f"{str(r['away_team'])[:20]:<20} | {str(r['league'])[:24]:<24} | "
          f"{r['picked_side']:<5} conf {r['confidence']:.4f} | {r['result']} | "
          f"creado {r['created_at'][11:19]}")

print()
print("3) APUESTAS DE HOY DE m27_v5_2 (para comparar en los mismos partidos):")
for r in conn.execute(
        "SELECT b.*, m.home_team, m.away_team, m.league "
        "FROM bet_monitor_log_v3 b LEFT JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id "
        "WHERE substr(b.created_at,1,10)=? AND b.model_version='m27_v5_2' "
        "ORDER BY b.id", (HOY,)):
    print(f"   id {r['id']:>5} | {r['match_id']} | {str(r['home_team'])[:22]:<22} vs "
          f"{str(r['away_team'])[:22]:<22} | {r['picked_side']:<5} conf {r['confidence']:.4f} | "
          f"{r['result']} | {r['created_at'][11:19]}")

print()
print("4) PARTIDOS DE HOY CON APUESTA: marcador final y estado (para validar la liquidacion):")
for r in conn.execute(
        "SELECT b.match_id, m.home_team, m.away_team, m.home_score, m.away_score, "
        "m.status_type, m.status_description, m.league, "
        "GROUP_CONCAT(b.model_version || ':' || b.picked_side || ':' || b.result) bets "
        "FROM bet_monitor_log_v3 b JOIN matches m ON CAST(m.match_id AS INTEGER)=b.match_id "
        "WHERE substr(b.created_at,1,10)=? GROUP BY b.match_id ORDER BY b.match_id", (HOY,)):
    print(f"   {r['match_id']} | {str(r['home_team'])[:20]:<20} {r['home_score']}-{r['away_score']} "
          f"{str(r['away_team'])[:20]:<20} | {r['status_type']:<12} | {str(r['status_description'])[:18]:<18} | "
          f"{r['bets']}")
conn.close()
