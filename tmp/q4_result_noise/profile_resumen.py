# Perfilado temporal del "Resumen" del bot (/resumen).
# Mide: tamaño de las tablas implicadas y tiempo de compute_live_summary,
# separando el coste de cada consulta para localizar el cuello de botella.
import cProfile
import io
import pstats
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3.models.live_summary import compute_live_summary  # noqa: E402

conn = sqlite3.connect(ROOT / "matches.db")
conn.row_factory = sqlite3.Row

for q, label in (
    ("SELECT COUNT(*) FROM matches", "matches"),
    ("SELECT COUNT(*) FROM model_inference_v3", "model_inference_v3"),
    ("SELECT COUNT(*) FROM bet_monitor_log_v3", "bet_monitor_log_v3"),
    ("SELECT COUNT(*) FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3'", "logs v5_3"),
    ("SELECT COUNT(*) FROM quarter_scores_v2", "quarter_scores_v2"),
):
    t0 = time.time()
    n = conn.execute(q).fetchone()[0]
    print(f"{label}: {n:,} filas ({time.time()-t0:.2f}s)")

# Tamaño medio del inference_json de los NO_BET (los que parsea el resumen)
row = conn.execute(
    "SELECT AVG(LENGTH(inference_json)), MAX(LENGTH(inference_json)), COUNT(*) "
    "FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3' AND signal_type LIKE '%NO_BET%'"
).fetchone()
print(f"inference_json NO_BET: media={row[0] or 0:.0f} B, max={row[1] or 0:,} B, filas={row[2]}")

print("\n--- consulta pesada (join minuto/quarter) ---")
t0 = time.time()
n = conn.execute("""
    SELECT COUNT(*) FROM model_inference_v3 i JOIN quarter_scores_v2 q
      ON CAST(q.match_id AS INTEGER)=i.match_id
    WHERE i.model_version='m27_v5_3' AND q.q4_home IS NOT NULL
""").fetchone()[0]
print(f"filas del join: {n:,} ({time.time()-t0:.2f}s)")

conn.close()

print("\n--- compute_live_summary completo ---")
conn = sqlite3.connect(ROOT / "matches.db")
conn.row_factory = sqlite3.Row
t0 = time.time()
pr = cProfile.Profile()
pr.enable()
data = compute_live_summary(conn, "m27_v5_3")
pr.disable()
conn.close()
total = time.time() - t0
print(f"total: {total:.2f}s")

s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(18)
print(s.getvalue())
