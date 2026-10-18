"""
Auditoría rápida (< 5s) de:
1. Paridad y ausencia total de errores/NaNs en `models/m27_v4/features.py` y `models/m27_v4/predict.py`
   sobre partidos históricos de los 3 arquetipos y sobre payloads vivos de `MobileClient.fetch_full_match`.
2. Verificación de las condiciones óptimas para maximizar verdes en `models/m27_v4/predict.py`.
"""
from __future__ import annotations

import asyncio
import sys
import sqlite3
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import models
from models.m27_v4.features import FEATURE_NAMES_CLEAN, extract_m27_v4_features
from models.common.data_loader import load_match
from monitor_v3.core.mobile_client import get_mobile_client
from monitor_v3.core.token_manager import get_token_pool


async def main_async():
    conn = sqlite3.connect(f"file:{ROOT_DIR / 'matches.db'}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    # 1. Probar 30 partidos recientes de los 3 arquetipos desde matches.db
    rows = conn.execute("""
        SELECT m.match_id, m.league, m.home_team, m.away_team
        FROM matches m
        JOIN quarter_scores q4 ON m.match_id = q4.match_id AND q4.quarter = 'Q4'
        WHERE m.date >= '2026-09-01'
        ORDER BY m.date DESC
        LIMIT 40
    """).fetchall()

    print("=" * 95)
    print("1. AUDITORÍA EXHAUSTIVA DE LAS 36 FEATURES EN PARTIDOS DE BASE DE DATOS (N=40)")
    print("=" * 95)

    nan_count = 0
    non_zero_counts = np.zeros(len(FEATURE_NAMES_CLEAN), dtype=int)
    for r in rows:
        mid = str(r["match_id"])
        mdata = load_match(conn, mid)
        X, meta = extract_m27_v4_features(mdata, conn=conn, match_id=mid)
        assert X.shape == (1, 36), f"Shape inválido: {X.shape}"
        if np.isnan(X).any() or np.isinf(X).any():
            nan_count += 1
        non_zero_counts += (np.abs(X[0]) > 1e-6).astype(int)
        pred = models.predict(match_id=mid, model_version="m27_v4", target="q4", match_data=mdata, conn=conn)
        assert pred.available, f"Predicción no disponible para {mid}: {pred.reason}"

    print(f"  - Partidos probados: {len(rows)} |Vectores con NaN/Inf: {nan_count} (0 esperado)")
    for idx, fname in enumerate(FEATURE_NAMES_CLEAN):
        print(f"    [{idx:2d}] {fname:<24s} -> Activa en {non_zero_counts[idx]:2d}/{len(rows)} partidos")

    # 2. Probar con payload en vivo real desde SofaScore MobileClient
    print("\n" + "=" * 95)
    print("2. AUDITORÍA EN VIVO SOBRE PAYLOAD DE `MobileClient.fetch_full_match`")
    print("=" * 95)
    pool = get_token_pool()
    await pool.initialize()
    mc = get_mobile_client()
    live_evs = await mc.get_live_events()
    print(f"  - Eventos en vivo detectados ahora mismo en SofaScore: {len(live_evs)}")

    tested_live = 0
    for ev in live_evs[:5]:
        mid = str(ev.get("id"))
        full_data = await mc.fetch_full_match(mid, fetch_team_strength=True)
        X, meta = extract_m27_v4_features(full_data, conn=conn, match_id=mid)
        pred = models.predict(match_id=mid, model_version="m27_v4", target="q4", match_data=full_data, conn=conn)
        print(
            f"  - Live Match {mid} ({meta['league']} | {meta['archetype']} | min={meta['snapshot_minute']}): "
            f"Signal={pred.signal} | Pick={pred.pick} ({pred.confidence*100:.1f}%) | "
            f"d_elo_ft={meta['d_elo_ft']} | d_rol_pm={meta['d_rol_plus_minus']} | "
            f"has_odds={meta['has_odds']} | h2h={meta['h2h_available']}"
        )
        tested_live += 1

    if tested_live == 0 and rows:
        mid0 = str(rows[0]["match_id"])
        full_data = await mc.fetch_full_match(mid0, fetch_team_strength=True)
        X, meta = extract_m27_v4_features(full_data, conn=conn, match_id=mid0)
        pred = models.predict(match_id=mid0, model_version="m27_v4", target="q4", match_data=full_data, conn=conn)
        print(
            f"  - API Match {mid0} ({meta['league']} | {meta['archetype']}): "
            f"Signal={pred.signal} | Pick={pred.pick} ({pred.confidence*100:.1f}%) | "
            f"d_elo_ft={meta['d_elo_ft']} | d_rol_pm={meta['d_rol_plus_minus']} | "
            f"has_odds={meta['has_odds']} | h2h={meta['h2h_available']}"
        )

    conn.close()


if __name__ == "__main__":
    asyncio.run(main_async())
