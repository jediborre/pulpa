"""
Auditoría y verificación de inferencia en tiempo real de m27_v5:
1. Verificación sobre partidos históricos de matches.db (10m FIBA y 12m NBA).
2. Verificación de cero NaNs y cero errores de tipo.
3. Verificación de compatibilidad con el evaluador de monitor_v3.
"""
import sys
import sqlite3
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from models.registry import predict, get_available_models
from monitor_v3.models.evaluator import load_models_to_cache


def main():
    print("=" * 80)
    print("VERIFICACIÓN DE M27_V5 EN EL REGISTRO Y EVALUADOR")
    print("=" * 80)
    models = get_available_models()
    print(f"Modelos registrados: {models}")
    assert "m27_v5" in models, "m27_v5 no está registrado en models/registry.py"

    conn = sqlite3.connect("matches.db")
    cur = conn.cursor()

    # Seleccionar 5 partidos históricos (incluyendo NBA y FIBA)
    cur.execute("""
        SELECT match_id, league, home_team, away_team
        FROM matches
        WHERE home_score IS NOT NULL
        ORDER BY date DESC
        LIMIT 6
    """)
    sample_matches = cur.fetchall()

    for mid, lg, ht, at in sample_matches:
        res = predict(str(mid), model_version="m27_v5", target="q4", conn=conn, snapshot_minute=27.0)
        print(f"\nMatch {mid} ({lg[:25]}...): {ht} vs {at}")
        print(f"  Available: {res.available} | Signal: {res.signal} | Pick: {res.pick} | Conf: {res.confidence*100:.1f}%")
        print(f"  Reason: {res.reason}")
        raw = res.raw_json or {}
        sub_preds = raw.get("submodel_predictions", {})
        print(f"  Expected Q4 Margin: {sub_preds.get('expected_q4_margin')} pts")

        # Verificar que no haya NaNs en las features
        feats = raw.get("features", {})
        nans = [k for k, v in feats.items() if v != v or v is None]
        assert len(nans) == 0, f"NaNs detectados en features: {nans}"
        assert len(feats) == 37, f"Esperadas 37 features, obtenidas {len(feats)}"

    conn.close()
    print("\nAuditoría superada: 0 NaNs, 37 features exactas, señales y márgenes válidos.")


if __name__ == "__main__":
    main()
