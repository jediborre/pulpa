# Medición temporal tras optimizar el join del /resumen.
# Dos pasadas: la primera incluye imports pesados (numpy/features), la segunda
# es el coste real recurrente de calcular el resumen.
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
for i in (1, 2):
    t0 = time.time()
    d = compute_live_summary(conn, "m27_v5_3")
    print(f"pasada {i}: {time.time()-t0:.2f}s | partidos_evaluados={d['partidos_evaluados']} "
          f"| filas_curva={len(d['curva'])} | bets={d['acumulado']['n']}")
conn.close()
