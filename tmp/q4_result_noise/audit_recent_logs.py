# Auditoría temporal: cuántos partidos de las últimas 48 h tienen logs y en qué
# situación habrían generado tarjeta de resultado antes/después del fix de
# política de notificaciones (solo BET con umbral válido de m27_v5_3).
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DB = Path(__file__).resolve().parents[2] / "matches.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

rows = conn.execute(
    "SELECT match_id, model_version, signal_type, confidence, result "
    "FROM bet_monitor_log_v3 WHERE created_at >= datetime('now', '-3 days')"
).fetchall()

by_match: dict[str, list] = {}
for r in rows:
    by_match.setdefault(r["match_id"], []).append(dict(r))

print(f"partidos con log (3 días): {len(by_match)}")
for mid, logs in by_match.items():
    resumen = ", ".join(f"{l['model_version']}:{l['signal_type']}@{l['confidence']:.2f}" for l in logs)
    notificable = [l for l in logs
                   if l["model_version"] not in ("m27_v5_2",)
                   and "BET" in (l["signal_type"] or "")
                   and "NO_BET" not in (l["signal_type"] or "")]
    estado = "ENVIA" if notificable else "SILENCIADO"
    print(f"  {mid} [{estado}] {resumen}")

conn.close()
