# Verificación temporal del umbral efectivo de m27_v5_3:
#   1) configuración cargada en producción (model_config.json),
#   2) confianza mínima observada en las apuestas reales registradas en vivo.
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from models.m27_v5_3.predict import _load_production_models  # noqa: E402

cfg = _load_production_models()["cfg"]
print(f"min_conf efectivo : {cfg['min_conf']}")
print(f"min_margin        : {cfg['min_margin']}")
print(f"block_toxic_leagues: {cfg['block_toxic_leagues']}")
print(f"snapshot_range    : {cfg.get('snapshot_range')}")

conn = sqlite3.connect(ROOT / "matches.db")
for sig_like in ("BET%",):
    row = conn.execute(
        "SELECT COUNT(*), MIN(confidence), MAX(confidence) FROM bet_monitor_log_v3 "
        "WHERE model_version = 'm27_v5_3' AND signal_type LIKE ? AND signal_type NOT LIKE '%NO_BET%'",
        (sig_like,),
    ).fetchone()
    print(f"BETs v5_3 registrados: n={row[0]} min_conf={row[1]} max_conf={row[2]}")

row = conn.execute(
    "SELECT COUNT(*) FROM bet_monitor_log_v3 WHERE model_version = 'm27_v5_3' "
    "AND signal_type NOT LIKE '%NO_BET%' AND signal_type LIKE 'BET%' AND confidence < 0.75"
).fetchone()
print(f"BETs por debajo de 0.75 (deberia ser 0): {row[0]}")
conn.close()
