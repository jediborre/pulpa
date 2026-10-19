# =====================================================================
# PRUEBA DE CASOS LÍMITE DE LA IMAGEN Q4 (sin red)
# ---------------------------------------------------------------------
# QUÉ VALIDA:
#   1. `build_q4_alert_image` devuelve None si no hay match_data o bets
#      (el aviso de Telegram debe caer a texto plano, nunca romperse).
#   2. Con match_data pero SIN graph_points igualmente genera la TARJETA
#      (JPEG válido) aunque no haya momentum que fusionar.
#   3. Equipos/nombres larguísimos no rompen el layout (recorte por tamaño).
#   4. Conversión de cuota decimal -> americana y proyección de ganancia.
#   5. `_decimal_odds_for_side` lee ml_home/ml_away de odds_live y descarta
#      valores inválidos.
#   6. `_photo_caption` recorta a 1024 caracteres (límite de Telegram).
#
# USO: .venv\Scripts\python.exe tmp\q4_bet_image\test_edge_cases.py
# =====================================================================

import io
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from monitor_v3.notifications import bet_image as bi  # noqa: E402
from monitor_v3.notifications.telegram_bot import (  # noqa: E402
    _decimal_odds_for_side,
    _hypothetical_result,
    _photo_caption,
)

OUT_DIR = Path(__file__).resolve().parent
ok = True


def check(label: str, cond: bool) -> None:
    global ok
    ok = ok and cond
    print(f"  {'✅' if cond else '❌'} {label}")


def _match_data(gp: list | None = None) -> dict:
    return {
        "match": {
            "home_team": "Equipo Local Con Nombre Absurdamente Largo Que No Cabe",
            "away_team": "Visitante Igualmente Largo S.A. de C.V.",
            "league": "Liga de Prueba Profesional de Baloncesto",
            "country": "México",
            "status_type": "inprogress",
            "game_seconds_played": 1680,
        },
        "score": {"home": 71, "away": 68, "quarters": {"Q1": {"home": 20, "away": 18}}},
        "graph_points": gp if gp is not None else [],
    }


BETS = [
    {
        "model": "m27_v5_3",
        "signal": "HOME_BET_Q4",
        "picked_team": "Equipo Local",
        "is_home": True,
        "confidence": 0.72,
        "decimal_odds": 1.85,
    }
]


def main() -> None:
    print("\n[1] Sin datos -> None (degradado a texto plano)")
    check("match_data=None", bi.build_q4_alert_image(None, BETS) is None)
    check("bets vacío", bi.build_q4_alert_image(_match_data(), []) is None)
    check("sin equipos", bi.build_q4_alert_image({"match": {}, "score": {}}, BETS) is None)

    print("\n[2] Sin momentum -> tarjeta sola (JPEG válido)")
    photo = bi.build_q4_alert_image(_match_data(), BETS, minute=27, q_key="Q4")
    check("devuelve bytes", isinstance(photo, (bytes, bytearray)) and len(photo) > 5000)
    if photo:
        with Image.open(io.BytesIO(photo)) as im:
            check(f"JPEG {im.format} {im.size[0]}px de ancho", im.format == "JPEG" and im.size[0] == bi.WIDTH)
        (OUT_DIR / "edge_sin_momentum.jpg").write_bytes(photo)

    print("\n[3] Nombres largos con momentum de 2 puntos")
    photo2 = bi.build_q4_alert_image(
        _match_data([{"minute": 1, "value": 3}, {"minute": 2, "value": -4}]),
        BETS,
        minute=27,
        q_key="Q4",
    )
    check("genera imagen sin excepción", isinstance(photo2, (bytes, bytearray)) and len(photo2) > 5000)

    print("\n[4] Cuota decimal -> americana y proyección")
    check("1.85 -> -118", bi.decimal_to_american(1.85) == -118)
    check("2.10 -> +110", bi.decimal_to_american(2.10) == 110)
    check("1.00 -> None (inválida)", bi.decimal_to_american(1.0) is None)
    gain, total = bi.payout(1.85, 500.0)
    check(f"payout 1.85/500 = +{gain:.2f} / {total:.2f}", abs(gain - 425.0) < 0.01 and abs(total - 925.0) < 0.01)

    print("\n[5] Cuota in-play por lado")
    md = {"odds_live": {"ml_home": 1.85, "ml_away": 2.10}}
    check("home -> 1.85", _decimal_odds_for_side(md, True) == 1.85)
    check("away -> 2.10", _decimal_odds_for_side(md, False) == 2.10)
    check("sin odds -> None", _decimal_odds_for_side({}, True) is None)
    check("cuota inválida -> None", _decimal_odds_for_side({"odds_live": {"ml_home": 1.0}}, True) is None)

    print("\n[6] Caption recortado a 1024")
    check("texto corto intacto", _photo_caption("hola") == "hola")
    largo = "x" * 2000
    cap = _photo_caption(largo)
    check(f"largo recortado ({len(cap)} chars)", len(cap) <= 1024 and cap.endswith("..."))

    print("\n[7] Umbral real del modelo desde raw_json")
    check(
        "m27_v5_3 -> 0.72",
        abs(bi._model_threshold("m27_v5_3", {"raw_json": {"min_conf_threshold": 0.72}}) - 0.72) < 1e-9,
    )
    check("v6_2 -> threshold_bet", abs(bi._model_threshold("v6_2", {"threshold_bet": 0.14}) - 0.14) < 1e-9)
    check("m27_v3 sin raw_json -> 0.68", abs(bi._model_threshold("m27_v3", None) - 0.68) < 1e-9)
    check("desconocido -> 0.65", abs(bi._model_threshold("m99", {}) - 0.65) < 1e-9)

    print("\n[8] Semáforo del porcentaje")
    check("apostó -> verde", bi._conf_color(0.55, 0.72, True) == bi.SS_CONF_BET)
    cerca = bi._conf_color(0.70, 0.72, False)
    lejos = bi._conf_color(0.05, 0.72, False)
    check(f"cerca del umbral -> amarillento {cerca}", cerca[0] > 200 and cerca[1] > 150 and cerca[2] < 120)
    check(f"lejos del umbral -> rojizo {lejos}", lejos[0] > 200 and lejos[1] < 110)

    print("\n[9] Resultado hipotético de un NO_BET")
    md_local = {"score": {"home": 86, "away": 76}}
    md_visita = {"score": {"home": 70, "away": 80}}
    check("pick local y ganó local -> would_win", _hypothetical_result(True, md_local) == "would_win")
    check("pick visita y ganó local -> would_loss", _hypothetical_result(False, md_local) == "would_loss")
    check("pick visita y ganó visita -> would_win", _hypothetical_result(False, md_visita) == "would_win")
    check("empate -> None", _hypothetical_result(True, {"score": {"home": 80, "away": 80}}) is None)
    check("sin marcador -> None", _hypothetical_result(True, {}) is None)
    check("etiquetas", bi._result_tag("would_win")[0] == "HUBIERA GANADO" and bi._result_tag("would_loss")[1] == bi.SS_CONF_FAR)

    print("\nRESULTADO:", "✅ TODO OK" if ok else "❌ HAY FALLOS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
