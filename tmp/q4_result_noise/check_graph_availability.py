# Diagnóstico temporal: ¿existe curva de momentum (graph_points) para el partido
# 17330673 en matches.db, aunque la API FT no la trajera? Sirve para decidir si
# se puede reenviar una tarjeta de resultado CON gráfico en cierres "no_graph".
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
conn = sqlite3.connect(Path(__file__).resolve().parents[2] / "matches.db")
conn.row_factory = sqlite3.Row

for mid in ("17330673", "17324789", "17220608", "17330191", "16875280", "17330192"):
    n = conn.execute("SELECT COUNT(*) FROM graph_points WHERE match_id = ?", (mid,)).fetchone()[0]
    print(f"{mid}: graph_points = {n}")

print("\ncolumnas graph_points:", [r[1] for r in conn.execute("PRAGMA table_info(graph_points)")])
row = conn.execute("SELECT * FROM graph_points WHERE match_id = '17330673' LIMIT 3").fetchall()
for r in row:
    print(dict(r))
conn.close()
