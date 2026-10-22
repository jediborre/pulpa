# =====================================================================
# PRUEBA DEL RESUMEN (texto que vera el bot con /resumen)
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.models.live_summary import (  # noqa: E402
    compute_live_summary, render_live_summary_text,
)

conn = sqlite3.connect(str(ROOT / "matches.db"))
conn.row_factory = sqlite3.Row
data = compute_live_summary(conn, "m27_v5_3")
text = render_live_summary_text(data)
print(text)
print()
print(f"longitud: {len(text)} caracteres (limite de Telegram ~4096)")
conn.close()
