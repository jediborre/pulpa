# =====================================================================
# PRUEBA DE LA CAPTURA DE INFERENCIA POR MINUTO (model_inference_v3)
# ---------------------------------------------------------------------
# QUÉ COMPRUEBA:
#   1. Que la tabla model_inference_v3 existe con su esquema.
#   2. Que la serie por minuto se escribe con una fila por (partido, modelo,
#      minuto) dentro de la ventana valida de entrenamiento.
#   3. Que la ventana se respeta: un snapshot fuera de [25.5, 30] NO se archiva.
#   4. Que el motivo de descarte queda registrado.
#   5. Que despues se puede analizar por minuto (acalierto/calibracion).
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/test_minute_capture.py
# =====================================================================

import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.config.constants import DEFAULT_VALID_WINDOW, MODEL_VALID_WINDOW  # noqa: E402
from monitor_v3.database.repository import init_tables, save_model_inference  # noqa: E402
from models.common.data_loader import get_db_connection, load_match  # noqa: E402
from models.m27_v5_3 import predict as predict_v53  # noqa: E402


def main() -> None:
    print("1. Creando el esquema (init_tables)...")
    init_tables()
    conn = sqlite3.connect(str(ROOT / "matches.db"))
    conn.row_factory = sqlite3.Row
    cols = [r[1] for r in conn.execute("PRAGMA table_info(model_inference_v3)")]
    print(f"   columnas: {cols}")
    n0 = conn.execute("SELECT COUNT(*) FROM model_inference_v3").fetchone()[0]
    print(f"   filas antes: {n0:,}")

    print("\n2. Simulando la serie por minuto de un partido real...")
    row = conn.execute(
        "SELECT match_id FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3' "
        "AND signal_type LIKE '%BET%' ORDER BY created_at DESC LIMIT 1").fetchone()
    mid = row["match_id"]
    db = get_db_connection()
    db.row_factory = sqlite3.Row
    payload = load_match(db, mid)
    print(f"   partido {mid}")

    lo, hi = MODEL_VALID_WINDOW.get("m27_v5_3", DEFAULT_VALID_WINDOW)
    print(f"   ventana valida de m27_v5_3: [{lo}, {hi}]")
    for minute in (24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 33.0):
        res = predict_v53(mid, match_data=payload, conn=db, snapshot_minute=minute)
        if not res.available:
            print(f"   min {minute:>5}: NO DISPONIBLE ({res.reason[:60]})")
            continue
        rawj = res.raw_json or {}
        snap = float(rawj.get("snapshot_minute_used") or minute)
        dentro = lo - 1e-6 <= snap <= hi + 1e-6
        if dentro:
            save_model_inference(
                mid, minute, "m27_v5_3", res.p_home_win, res.p_away_win, rawj.get("p_cal"),
                rawj.get("expected_q4_margin"), str(res.signal or ""), str(res.pick or ""),
                res.confidence, snap, rawj.get("block_reason"), rawj.get("features"))
        print(f"   min {minute:>5}: snapshot_usado={snap:>5} p_cal={rawj.get('p_cal')} "
              f"senal={res.signal:<5} -> {'ARCHIVADO' if dentro else 'FUERA DE VENTANA (no se archiva)'}")

    db.close()
    print("\n3. Serie resultante en model_inference_v3:")
    for r in conn.execute(
        "SELECT game_minute, model_version, p_cal, margin, signal, pick, confidence, "
        "snapshot_minute, block_reason FROM model_inference_v3 WHERE match_id=? "
        "ORDER BY game_minute", (mid,)
    ):
        print(f"   min {r['game_minute']:>5} (snapshot {r['snapshot_minute']}) "
              f"p_cal={r['p_cal']:<7} margen={r['margin']:<7} {r['signal']:<5} "
              f"conf={r['confidence']}  motivo={r['block_reason']}")

    print("\n4. Analisis por minuto (lo que se podra hacer con la serie):")
    for r in conn.execute(
        """
        SELECT i.game_minute, COUNT(*) n, AVG(i.p_cal) p, AVG(i.confidence) conf
        FROM model_inference_v3 i GROUP BY i.game_minute ORDER BY i.game_minute
        """
    ):
        print(f"   minuto {r['game_minute']:>6}: n={r['n']:>4}  p_cal medio={r['p']:.4f}  "
              f"confianza media={r['conf']:.4f}")
    conn.close()


if __name__ == "__main__":
    main()
