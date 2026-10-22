# Fechas de los BETs de m27_v5_3 por debajo de 0.75, para confirmar que son
# del modelo/umbral anterior (0.72) y no de la configuracion actual (0.75).
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
conn = sqlite3.connect(Path(__file__).resolve().parents[2] / "matches.db")
q = ("SELECT MIN(created_at), MAX(created_at), COUNT(*), MIN(confidence) "
     "FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3' "
     "AND signal_type LIKE 'BET%' AND signal_type NOT LIKE '%NO_BET%' "
     "AND confidence < 0.75")
print("BETs < 0.75 :", conn.execute(q).fetchone())
q2 = q.replace("confidence < 0.75", "confidence >= 0.75")
print("BETs >= 0.75:", conn.execute(q2).fetchone())
conn.close()
