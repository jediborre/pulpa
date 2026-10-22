# Inspección temporal (parte 2): logs del partido 16665310 (Anadolu Efes vs
# Besiktas JK) para verificar que modelos loguearon y con que senal.
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB = Path(__file__).resolve().parents[2] / "matches.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
mid = "16665310"

print("--- bet_monitor_log_v3 ---")
for r in conn.execute(
    "SELECT id, model_version, signal_type, picked_side, confidence, result, target_quarter "
    "FROM bet_monitor_log_v3 WHERE match_id = ? ORDER BY id",
    (mid,),
):
    print(dict(r))

cols = [r[1] for r in conn.execute("PRAGMA table_info(model_inference_v3)")]
print("\ncolumnas model_inference_v3:", cols)
print("\n--- model_inference_v3 ---")
for r in conn.execute(
    "SELECT model_version, COUNT(*) n FROM model_inference_v3 WHERE match_id = ? GROUP BY model_version",
    (mid,),
):
    print(dict(r))

print("\n--- ultimos envios (si existe tabla de auditoria) ---")
tablas = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
print([t for t in tablas if "notif" in t or "telegram" in t or "audit" in t])
conn.close()
