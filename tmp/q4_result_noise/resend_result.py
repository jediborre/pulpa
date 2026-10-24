# Reenvío manual de una tarjeta de RESULTADO Q4 (con su gráfica) cuando el
# envío original se perdió por una caída de Telegram.
#
# Uso: .venv\Scripts\python.exe tmp\q4_result_noise\resend_result.py <match_id>
#
# Descarga el payload FT real (event + graph), por lo que la tarjeta sale
# completa (panel + cuartos + curva de momentum). No toca la BD: solo envía a
# los suscriptores mediante `_notify_final_result`, respetando la política de
# umbral válido y el filtro de modelos en sombra.
import asyncio
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from monitor_v3 import main as mon  # noqa: E402
from monitor_v3.scrapers.match_detail_scraper import fetch_match_by_id  # noqa: E402


def _quarters_from_db(match_id: str) -> dict:
    conn = sqlite3.connect(ROOT / "matches.db")
    conn.row_factory = sqlite3.Row
    qs = conn.execute("SELECT * FROM quarter_scores_v2 WHERE match_id = ?", (match_id,)).fetchone()
    conn.close()
    if not qs:
        return {}
    return {
        "Q1": {"home": qs["q1_home"], "away": qs["q1_away"]},
        "Q2": {"home": qs["q2_home"], "away": qs["q2_away"]},
        "Q3": {"home": qs["q3_home"], "away": qs["q3_away"]},
        "Q4": {"home": qs["q4_home"], "away": qs["q4_away"]},
    }


async def _run(match_id: str) -> None:
    data = await fetch_match_by_id(match_id, is_ft=True)
    match = data.get("match") or {}
    home = match.get("home_team") or "?"
    away = match.get("away_team") or "?"
    quarters = (data.get("score") or {}).get("quarters") or _quarters_from_db(match_id)
    print(f"Reenviando resultado {match_id} {home} vs {away} "
          f"(gp={len(data.get('graph_points') or [])}, Q4={quarters.get('Q4')})...")
    await mon._notify_final_result(match_id, data, home, away, quarters)

    # Si Telegram está caído (502/504), el aviso queda en la cola interna: hay
    # que mantener vivo el proceso para que la vacíe (cada 60 s, hasta 30 min).
    from monitor_v3.notifications.telegram_bot import pending_notifications_count

    waited = 0
    while pending_notifications_count() > 0 and waited < 1860:
        await asyncio.sleep(30)
        waited += 30
        print(f"  esperando a Telegram... {waited}s (pendientes={pending_notifications_count()})")
    if pending_notifications_count():
        print("No se pudo entregar: sigue en cola al agotarse el tiempo de espera.")
    else:
        print("Listo: entregado (o no había apuesta válida que notificar).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Uso: resend_result.py <match_id>")
    asyncio.run(_run(str(sys.argv[1])))
