import sys
from pathlib import Path

# Load workspace path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bet_monitor_v2.utils.logger import log_info, log_warning, log_error

print("Testing console coloring for DESCARGA component:")
log_info("DESCARGA", "[FT] Final y liquidación de apuestas persistidos | Mets de Guaynabo vs Indios de Mayagüez (gp=40)")
log_warning("DESCARGA", "[FT] Cierre con gráfica corta (gp=0) | Reales De La Vega vs Indios de San Francisco")
log_error("DESCARGA", "[FT] Error en fetch final | San Salvador BC vs Santiagueño BC")
