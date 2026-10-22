# =====================================================================
# ¿DONDE SE DESCARTAN LAS LIGAS DE 12 MIN Y LOS FORMATOS BLOQUEADOS?
#   (para poner la nota correcta en el resumen)
# =====================================================================

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.competition_stage import classify_stage  # noqa: E402
from models.m27_v5_3.features import BLOCKED_STAGE_CATEGORIES  # noqa: E402

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
liga_de = {str(r["match_id"]): r["league"] for r in
           conn.execute("SELECT match_id, league FROM matches WHERE league IS NOT NULL")}

vistos = set()
bloq_rows = 0
for r in conn.execute("""
    SELECT match_id, signal_type, inference_json FROM bet_monitor_log_v3
    WHERE model_version='m27_v5_3'"""):
    vistos.add(str(r["match_id"]))
    cat, _ = classify_stage(liga_de.get(str(r["match_id"])) or "")
    if cat in BLOCKED_STAGE_CATEGORIES:
        bloq_rows += 1
print(f"partidos vistos por el modelo (log): {len(vistos):,}")
print(f"filas del log en formatos que AHORA se bloquean: {bloq_rows} "
      f"(eran válidos antes del arreglo)")

# ¿el monitor llega a ver partidos de ligas de 12 min?
docena = [l for l in liga_de.values() if l in ("NBA", "NBA G League", "WNBA", "China CBA")]
vistos_12 = [m for m in vistos if liga_de.get(m) in docena]
print(f"partidos de ligas de 12 min vistos por el modelo: {len(vistos_12)} "
      f"(el filtro va antes de la inferencia)")

# ¿y los formatos bloqueados que ahora se filtran?
bloq_actual = Counter()
for m in vistos:
    cat, _ = classify_stage(liga_de.get(m) or "")
    if cat in BLOCKED_STAGE_CATEGORIES:
        bloq_actual[cat] += 1
print(f"de los vistos, los que ahora caen en formato bloqueado: {sum(bloq_actual.values())} "
      f"{dict(bloq_actual)}")
conn.close()
