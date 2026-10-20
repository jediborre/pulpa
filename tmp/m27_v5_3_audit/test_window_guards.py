# =====================================================================
# PRUEBA DE LOS GUARDS DE VENTANA DE ENTRENAMIENTO (v5_3 y v5_2)
# ---------------------------------------------------------------------
# QUÉ COMPRUEBA:
#   1. m27_v5_3: con snapshot_minute=27 (dentro de [25.5, 30]) la inferencia
#      debe ser available; con 31, 33 y 35 debe ser unavailable por ventana.
#   2. m27_v5_2: mismo comportamiento (su ventana es Q3 cerrado).
#   3. Que el umbral desplegado de v5_3 es 0.75 y que la inferencia NO cambia
#      con un snapshot valido (es decir, el guard no altera el modelo).
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/test_window_guards.py
# =====================================================================

import json
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.data_loader import get_db_connection, load_match  # noqa: E402
from models.m27_v5_2 import predict as predict_v52  # noqa: E402
from models.m27_v5_3 import predict as predict_v53  # noqa: E402


def main() -> None:
    conn = get_db_connection()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """
        SELECT match_id FROM bet_monitor_log_v3
        WHERE model_version = 'm27_v5_3' AND signal_type LIKE '%BET%'
        ORDER BY created_at DESC LIMIT 1
        """
    ).fetchone()
    mid = row["match_id"] if row else None
    if not mid:
        print("No hay partidos en el log; no se puede probar")
        return
    print(f"Partido de prueba: {mid}")
    payload = load_match(conn, mid)
    if not payload:
        print("El partido no esta en matches.db")
        return
    m = payload.get("match") or {}
    print(f"  league={m.get('league')} | status={m.get('status_description')} | "
          f"gsp={m.get('game_seconds_played')}")

    print("\n" + "=" * 74)
    print("m27_v5_3 (ventana [25.5, 30.0])")
    print("=" * 74)
    for snap in (24.0, 25.0, 25.5, 27.0, 29.0, 30.0, 30.5, 31.0, 33.0, 35.0):
        res = predict_v53(mid, match_data=payload, conn=conn, snapshot_minute=snap)
        raw = res.raw_json or {}
        used = raw.get("snapshot_minute_used")
        print(f"  entrada={snap:>5}  available={str(res.available):<5}  signal={str(res.signal):<8} "
              f"conf={res.confidence}  snapshot_usado={used}  {res.reason or ''}")

    print("\n" + "=" * 74)
    print("m27_v5_2 (ventana: marcador de Q3 cerrado)")
    print("=" * 74)
    for snap in (25.0, 27.0, 30.0, 30.5, 31.0, 33.0, 35.0):
        res = predict_v52(mid, match_data=payload, conn=conn, snapshot_minute=snap)
        print(f"  entrada={snap:>5}  available={str(res.available):<5}  signal={str(res.signal):<8} "
              f"conf={res.confidence}  {res.reason or ''}")

    print("\n" + "=" * 74)
    print("CONFIG desplegada de m27_v5_3")
    print("=" * 74)
    cfg = json.loads((ROOT / "models" / "m27_v5_3" / "model_outputs" / "model_config.json")
                     .read_text(encoding="utf-8"))
    print(f"  min_conf={cfg['min_conf']}  min_margin={cfg['min_margin']}  chosen_on={cfg.get('chosen_on')}")
    res = predict_v53(mid, match_data=payload, conn=conn, snapshot_minute=27.0)
    print(f"  inferencia a min 27: signal={res.signal} conf={res.confidence} "
          f"min_conf_threshold={(res.raw_json or {}).get('min_conf_threshold')}")
    conn.close()


if __name__ == "__main__":
    main()
