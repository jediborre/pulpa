# =====================================================================
# VERIFICA QUE m27_v5_2 NO APARECE EN NADA DE TELEGRAM
#   1) Tarjeta de resultado (logs del partido, filtrado)
#   2) Resumen diario (por modelo y totales)
#   3) La alerta de apuesta (filtro de|display)
# =====================================================================

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.config.constants import ACTIVE_MODELS, SHADOW_MODELS  # noqa: E402
from monitor_v3.notifications.telegram_bot import (  # noqa: E402
    _daily_summary_data, _render_daily_summary_text, format_combined_final_message,
)

print("1) CONFIGURACION")
print(f"   ACTIVE_MODELS : {ACTIVE_MODELS}")
print(f"   SHADOW_MODELS : {SHADOW_MODELS}")
print(f"   -> visible en Telegram: "
      f"{[m for m in ACTIVE_MODELS if m not in SHADOW_MODELS]}")

print()
print("2) TARJETA DE RESULTADO (simulando los logs de un partido):")
logs = [
    {"model_version": "m27_v5_3", "signal_type": "NO_BET", "confidence": 0.59,
     "picked_side": "HOME", "result": "push", "target_quarter": 4, "inference_minute": 27},
    {"model_version": "m27_v5_2", "signal_type": "NO_BET", "confidence": 0.53,
     "picked_side": "HOME", "result": "push", "target_quarter": 4, "inference_minute": 27},
]
visibles = [l for l in logs if l["model_version"] not in SHADOW_MODELS]
print(format_combined_final_message(visibles, "Salvadorenos BC", "San Salvador BC"))

print()
print("3) RESUMEN DIARIO (hoy):")
data = _daily_summary_data()
if data:
    text = _render_daily_summary_text(data)
    print(text)
    print()
    print("   ¿aparece m27_v5_2 en el resumen? ->",
          "SI (mal)" if "m27_v5_2" in text else "NO (bien)")
else:
    print("   (sin datos de hoy)")
