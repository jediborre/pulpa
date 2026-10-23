# Test temporal del fix "resultado sin gráfica" (monitor_v3).
#
# Problema real (2026-10-22): el partido 17330673 Vasto Verde Blumenau vs AJAB
# Jaraguá terminó 81-36 (Q4 14-7) y m27_v5_3 había apostado BET_HOME al 84.1 %.
# El FT llegó sin `graph_points`, así que `_final_fetch_and_save` tomaba el
# branch `gp_total == 0`, guardaba marcadores y hacía `return` ANTES de enviar
# la tarjeta de resultado. El usuario nunca recibió el ✅.
#
# Este test llama a `_notify_final_result` (la ruta que ahora también usa el
# branch sin gráfica) con datos reales de la BD y el transporte mockeado.
import asyncio
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3 import main as mon  # noqa: E402

sent: list[dict] = []


async def fake_final(**kwargs):
    sent.append(kwargs)
    return {"ok": True}


async def fake_combined(**kwargs):
    sent.append({"combined": kwargs})
    return {"ok": True}


mon.send_final_confirmation = fake_final
mon.send_combined_final_confirmation = fake_combined

MID = "17330673"
conn = sqlite3.connect(ROOT / "matches.db")
conn.row_factory = sqlite3.Row
qs = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id = ?", (MID,)).fetchone()
conn.close()
assert qs, "falta quarter_scores_v2 del partido de prueba"
quarters = {
    "Q1": {"home": qs["q1_home"], "away": qs["q1_away"]},
    "Q2": {"home": qs["q2_home"], "away": qs["q2_away"]},
    "Q3": {"home": qs["q3_home"], "away": qs["q3_away"]},
    "Q4": {"home": qs["q4_home"], "away": qs["q4_away"]},
}
data = {"match": {"status_type": "finished"}, "score": {"quarters": quarters}, "graph_points": []}

asyncio.run(mon._notify_final_result(MID, data, "Vasto Verde Blumenau", "AJAB Jaraguá", quarters))

assert len(sent) == 1, f"deberia enviarse 1 tarjeta, se enviaron {len(sent)}"
call = sent[0]
assert call.get("model") == "m27_v5_3", call
assert call.get("signal") == "BET_HOME", call
assert call.get("outcome") == "win", call
assert call.get("q_home") == qs["q4_home"] and call.get("q_away") == qs["q4_away"], call
print("OK: el resultado (win, Q4 %s-%s) se notifica aunque el FT no traiga gráfica"
      % (qs["q4_home"], qs["q4_away"]))
