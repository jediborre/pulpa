# Reenvío manual de una tarjeta de RESULTADO Q4 que no llegó a Telegram.
#
# Uso: .venv\Scripts\python.exe tmp\q4_result_noise\resend_missing_result.py <match_id>
#
# Contexto: los cierres con `skip_reason='no_graph'` retornaban antes de enviar
# la tarjeta de resultado (bug corregido el 2026-10-22). Este script reutiliza
# `_notify_final_result` con los marcadores de `quarter_scores_v2` y el
# transporte real del bot, respetando la política de umbral válido: solo envía
# si hay un BET real de un modelo visible. No modifica la BD.
#
# Ejemplo del caso reportado:
#   .venv\Scripts\python.exe tmp\q4_result_noise\resend_missing_result.py 17330673
import asyncio
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3 import main as mon  # noqa: E402


def _load(match_id: str) -> tuple[dict, dict, str, str]:
    conn = sqlite3.connect(ROOT / "matches.db")
    conn.row_factory = sqlite3.Row
    sched = conn.execute(
        "SELECT home_team, away_team FROM bet_monitor_schedule_v3 WHERE match_id = ?",
        (match_id,),
    ).fetchone()
    qs = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id = ?", (match_id,)).fetchone()
    conn.close()
    if not sched or not qs:
        raise SystemExit(f"Sin datos de schedule/quarter_scores para {match_id}")
    quarters = {
        "Q1": {"home": qs["q1_home"], "away": qs["q1_away"]},
        "Q2": {"home": qs["q2_home"], "away": qs["q2_away"]},
        "Q3": {"home": qs["q3_home"], "away": qs["q3_away"]},
        "Q4": {"home": qs["q4_home"], "away": qs["q4_away"]},
    }
    data = {"match": {"status_type": "finished"}, "score": {"quarters": quarters}, "graph_points": []}
    return data, quarters, sched["home_team"], sched["away_team"]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Uso: resend_missing_result.py <match_id>")
    match_id = str(sys.argv[1])
    data, quarters, home, away = _load(match_id)
    print(f"Reenviando resultado de {match_id} {home} vs {away} (Q4 {quarters['Q4']})...")
    asyncio.run(mon._notify_final_result(match_id, data, home, away, quarters))
    print("Listo (si no había apuesta con umbral válido, no se envía nada).")


if __name__ == "__main__":
    main()
