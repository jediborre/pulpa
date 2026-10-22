# Test temporal del filtro de modelos en /status (monitor_v3).
#
# Problema: el bloque MES mostraba m27_v4, m27_v3 y v6_2 (modelos retirados)
# con 700+ apuestas historicas, dominando el total y el % de acierto.
# Ahora `fetch_v3_stats` solo cuenta STATUS_MODELS = ["m27_v5_3"] en filas Y
# totales, tanto en el texto como en la imagen del /status.
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3.config.constants import STATUS_MODELS  # noqa: E402
from monitor_v3.notifications.bot_runner import build_status_text, fetch_v3_stats  # noqa: E402

month = datetime.now().strftime("%Y-%m")
stats = fetch_v3_stats(created_month=month)

print(f"STATUS_MODELS = {STATUS_MODELS}")
print(f"mes {month}: per_model = {stats['per_model']}")
print(f"mes {month}: total     = {stats['total']}")

assert set(stats["per_model"]) <= set(STATUS_MODELS), "hay modelos no permitidos en per_model"
for banned in ("m27_v4", "m27_v3", "v6_2", "m27_v5_2"):
    assert banned not in stats["per_model"], f"{banned} no deberia aparecer"

# Los totales deben ser la suma de las filas mostradas
bets = sum(m["bets"] for m in stats["per_model"].values())
wins = sum(m["win"] for m in stats["per_model"].values())
losses = sum(m["loss"] for m in stats["per_model"].values())
assert stats["total"]["bets"] == bets, (stats["total"]["bets"], bets)
assert stats["total"]["win"] == wins, (stats["total"]["win"], wins)
assert stats["total"]["loss"] == losses, (stats["total"]["loss"], losses)

# El texto del /status no debe nombrar a los modelos retirados
text = build_status_text(day_stats={}, month_stats=stats)
for banned in ("m27_v4", "m27_v3", "v6_2", "m27_v5_2"):
    assert banned not in text, f"{banned} aparece en el texto del status"
print("\n" + text)

# Cuantificar lo que se deja fuera del mes
conn = sqlite3.connect(ROOT / "matches.db")
row = conn.execute(
    "SELECT COUNT(*) FROM bet_monitor_log_v3 "
    "WHERE strftime('%Y-%m', created_at) = ? "
    "AND model_version IN ('m27_v4','m27_v3','v6_2')",
    (month,),
).fetchone()
print(f"\nlogs del mes excluidos (m27_v4/m27_v3/v6_2): {row[0]}")
conn.close()

print("OK: /status (texto e imagen) solo muestra m27_v5_3 en filas y totales")
