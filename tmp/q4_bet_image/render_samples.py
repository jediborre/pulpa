# =====================================================================
# PRUEBA DE LA IMAGEN FUSIONADA DE APUESTA Q4 (tarjeta + momentum)
# ---------------------------------------------------------------------
# QUÉ HACE:
#   1. Descarga un partido real vía la API móvil (fetch_full_match) y busca
#      uno que tenga `graph_points` (momentum) para poder fusionar la gráfica.
#   2. Construye la imagen fusionada con monitor_v3.notifications.bet_image
#      en tres variantes: apuesta simple, combinada (BET + NO_BET) y tardía.
#   3. Guarda los JPEG en tmp/q4_bet_image/ para revisión visual.
#   4. Con --send la envía al chat de Telegram configurado (usa el mismo
#      camino real: broadcast_photo), útil para ver cómo se ve en el móvil.
#
# HIPÓTESIS QUE ABORDA:
#   Validar el layout, la paleta (espejo de match/ml_tools.plot_graph) y el
#   degradado seguro a texto plano antes de activarlo en el daemon.
#
# USO:
#   .venv\Scripts\python.exe tmp\q4_bet_image\render_samples.py
#   .venv\Scripts\python.exe tmp\q4_bet_image\render_samples.py --send
# =====================================================================

import argparse
import asyncio
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.notifications import bet_image  # noqa: E402
from monitor_v3.scrapers.match_detail_scraper import fetch_match_by_id  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
# Candidatos: partidos recientes con gráfica de momentum (gp) confirmada en logs.
CANDIDATOS = ["17221472", "17221476", "17221477", "17015142", "17227862"]


async def _get_match_with_graph() -> tuple[str, dict] | tuple[None, None]:
    for mid in CANDIDATOS:
        try:
            data = await fetch_match_by_id(mid, is_ft=False)
        except Exception as e:  # noqa: BLE001
            print(f"  {mid}: fallo de descarga ({type(e).__name__}: {e})")
            continue
        gp = len(data.get("graph_points") or [])
        match = data.get("match") or {}
        print(f"  {mid}: gp={gp} | {match.get('home_team')} vs {match.get('away_team')}")
        if gp >= 2:
            return mid, data
    return None, None


async def _get_live_match_with_graph() -> tuple[str, dict, str] | tuple[None, None, None]:
    """Toma un partido EN VIVO real (para ver Q4/minuto/marcador de producción)."""
    from monitor_v3.scrapers.live_scraper import fetch_live_events

    try:
        events = await fetch_live_events()
    except Exception as e:  # noqa: BLE001
        print(f"  no se pudieron listar los partidos en vivo: {e}")
        return None, None, None

    for ev in events:
        mid = str(ev.get("id"))
        desc = ((ev.get("status") or {}).get("description") or "")
        try:
            data = await fetch_match_by_id(mid, is_ft=False)
        except Exception:
            continue
        gp = len(data.get("graph_points") or [])
        match = data.get("match") or {}
        if gp >= 2:
            print(f"  EN VIVO {mid}: gp={gp} | {match.get('home_team')} vs {match.get('away_team')} | {desc}")
            q_key = "Q4"
            for number, word in ((1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th")):
                if word in desc:
                    q_key = f"Q{number}"
                    break
            return mid, data, q_key
    return None, None, None


def _bets(home: str, away: str, variant: str) -> list[dict]:
    """Datos de prueba con valores reales de producción (umbrales de raw_json)."""
    thr_v5_3 = {"raw_json": {"min_conf_threshold": 0.72}}
    thr_v5_2 = {"raw_json": {"min_conf_threshold": 0.75}}
    thr_v6_2 = {"raw_json": {"threshold_bet": 0.14}}

    bet_v5_3 = {
        "model": "m27_v5_3", "signal": "BET_HOME", "picked_team": home, "is_home": True,
        "confidence": 0.78, "decimal_odds": 1.85, "inference_json": thr_v5_3,
    }
    nobet_v5_2 = {  # cerca del umbral (0.507/0.75) → ámbar
        "model": "m27_v5_2", "signal": "NO_BET", "picked_team": away, "is_home": False,
        "confidence": 0.51, "decimal_odds": 2.10, "inference_json": thr_v5_2,
    }
    nobet_v6_2 = {  # lejos del umbral (0.05/0.14) → rojo
        "model": "v6_2", "signal": "NO_BET", "picked_team": away, "is_home": False,
        "confidence": 0.05, "decimal_odds": 2.10, "inference_json": thr_v6_2,
    }
    nobet_mid_v6_2 = {  # a medio camino (0.10/0.14) → ámbar anaranjado
        "model": "v6_2", "signal": "NO_BET", "picked_team": home, "is_home": True,
        "confidence": 0.10, "decimal_odds": 1.85, "inference_json": thr_v6_2,
    }

    if variant == "apuesta":
        return [bet_v5_3]
    if variant == "apuesta_combinada":
        return [bet_v5_3, dict(nobet_v5_2), dict(nobet_mid_v6_2)]
    if variant == "resultado":
        return [
            {**bet_v5_3, "result": "win"},
            {**nobet_v5_2, "result": "would_win"},
            {**nobet_v6_2, "result": "would_loss"},
        ]
    return [
        {
            "model": "m27_v5_3", "signal": "BET_AWAY_LATE", "picked_team": away,
            "is_home": False, "confidence": 0.62, "decimal_odds": 2.10,
            "inference_json": thr_v5_3,
        }
    ]


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--send", action="store_true", help="enviar las imágenes por Telegram")
    ap.add_argument("--live", action="store_true", help="usar un partido EN VIVO real (caso de producción)")
    ap.add_argument("--match-id", help="renderizar un partido concreto (p. ej. uno ya finalizado)")
    args = ap.parse_args()

    q_key = "Q4"
    if args.match_id:
        print(f"Descargando partido {args.match_id}...")
        data = await fetch_match_by_id(args.match_id, is_ft=False)
        mid = args.match_id
    elif args.live:
        print("Buscando un partido EN VIVO con gráfica de momentum...")
        mid, data, q_key = await _get_live_match_with_graph()
    else:
        print("Buscando un partido con gráfica de momentum...")
        mid, data = await _get_match_with_graph()
    if not data:
        print("❌ Ningún candidato tiene graph_points; no se puede probar la fusión.")
        return

    match = data.get("match") or {}
    home, away = str(match.get("home_team")), str(match.get("away_team"))
    score = data.get("score") or {}
    # Minuto real: reloj del API; si no lo expone, el último punto de la gráfica.
    minute = None
    if match.get("game_seconds_played") is not None:
        minute = int(int(match["game_seconds_played"]) // 60)
    elif data.get("graph_points"):
        minute = int(max(float(p.get("minute") or 0) for p in data["graph_points"]))
    # Hora de inicio plausible (la tarjeta la usa como etiqueta del eje).
    ahora = int(__import__("time").time())
    scheduled_ts = ahora - max(1, minute or 30) * 60
    print(f"\nPartido de prueba: {mid} | {home} vs {away} | marcador {score.get('home')}-{score.get('away')} "
          f"| {q_key} | min={minute}")

    for variant in ("apuesta", "apuesta_combinada", "tardia", "resultado"):
        is_result = variant == "resultado"
        photo = bet_image.build_q4_alert_image(
            data,
            _bets(home, away, variant),
            minute=minute,
            q_key=q_key,
            scheduled_ts=scheduled_ts,
            is_result=is_result,
            # Los avisos de apuesta en vivo van SIN gráfica (solo la tarjeta);
            # las confirmaciones FT llevan la gráfica completa.
            include_chart=is_result,
        )
        if not photo:
            print(f"❌ variante {variant}: no se generó imagen")
            continue
        out = OUT_DIR / f"q4_{variant}_{mid}.jpg"
        out.write_bytes(photo)
        print(f"✅ variante {variant}: {len(photo)} bytes -> {out.name}")

        if args.send:
            from monitor_v3.notifications.telegram_bot import broadcast_photo

            res = await broadcast_photo(photo, f"PRUEBA imagen Q4 ({variant}) - {home} vs {away}")
            print(f"   Telegram: {[r.get('ok') for r in res]}")


if __name__ == "__main__":
    asyncio.run(main())
