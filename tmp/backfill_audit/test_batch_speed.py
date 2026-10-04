"""
Script de Prueba y Benchmark: Descarga Asíncrona Concurrente por Lotes

Propósito:
- Evaluar la velocidad y estabilidad de descargar partidos por lotes concurrentes (concurrency=8-10)
  utilizando MobileClient y TokenPool en lugar de uno por uno secuencial.
- Medir tiempo total para 20 partidos y verificar inserción atómica en matches.db.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "match"))

from monitor_v3.core.mobile_client import get_mobile_client
import db as db_mod
import scraper

async def benchmark_batch():
    date_str = "2026-06-12"
    mc = get_mobile_client()
    conn = db_mod.get_conn(str(ROOT / "matches.db"))
    db_mod.init_db(conn)

    print(f"Obteniendo partidos de {date_str}...")
    events = await mc.get_all_scheduled_events_for_date(date_str)
    finished = [e for e in events if (e.get("status") or {}).get("type") == "finished"]
    sample = finished[:20]
    print(f"Total finished: {len(finished)}. Probando lote de {len(sample)} partidos concurrentes...")

    sem = asyncio.Semaphore(10)
    saved = 0

    async def _download_one(ev):
        nonlocal saved
        mid = str(ev.get("id"))
        async with sem:
            try:
                data = await mc.fetch_full_match(mid)
                # Guardar en SQLite
                db_mod.save_match(conn, mid, data)
                saved += 1
                return mid, True
            except Exception as e:
                return mid, False

    t0 = time.perf_counter()
    tasks = [_download_one(ev) for ev in sample]
    results = await asyncio.gather(*tasks)
    conn.commit()
    conn.close()
    dur = time.perf_counter() - t0

    oks = sum(1 for _, ok in results if ok)
    print(f"\nResultado Benchmark:")
    print(f"  - Partidos procesados: {oks}/{len(sample)}")
    print(f"  - Tiempo total: {dur:.2f}s")
    print(f"  - Promedio por partido: {dur / len(sample):.3f}s (Equivalente a {(len(sample) / dur):.1f} partidos/segundo!)")

    await mc.close()

if __name__ == "__main__":
    asyncio.run(benchmark_batch())
