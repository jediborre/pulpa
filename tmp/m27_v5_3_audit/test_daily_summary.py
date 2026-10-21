# =====================================================================
# PRUEBA DEL RESUMEN DIARIO CON LA CURVA DE UMBRAL INTEGRADA
# =====================================================================

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.notifications.telegram_bot import build_daily_summary_text  # noqa: E402

print(build_daily_summary_text())
