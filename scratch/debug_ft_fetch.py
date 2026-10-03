"""
Test de fetch individual para diagnosticar timeouts en partidos con final_fetch_failed.
Prueba un match_id con 'traditional' (backend FT actual) y con 'obscura'.
"""
import asyncio
import sys
import io
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from match.scraper import fetch_match_by_id

# Match a probar: Rain Or Shine vs Barangay Ginebra (PBA, partido de hoy)
TEST_MATCH_ID = "16193682"
TEST_NAME = "Rain Or Shine vs Barangay Ginebra San Miguel"

async def test_backend(backend: str, match_id: str, timeout_secs: float = 60.0):
    print(f"\n{'='*60}")
    print(f"  Backend: {backend.upper()}")
    print(f"  Partido: {TEST_NAME}")
    print(f"  Match ID: {match_id}")
    print(f"  Timeout: {timeout_secs}s")
    print(f"{'='*60}")
    
    start = time.time()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(fetch_match_by_id, match_id, backend=backend),
            timeout=timeout_secs
        )
        elapsed = time.time() - start
        
        score = result.get("score", {})
        quarters = score.get("quarters", {})
        gp = len(result.get("graph_points") or [])
        
        q4 = quarters.get("Q4", {})
        
        print(f"  ✅ ÉXITO en {elapsed:.1f}s")
        print(f"  Graph points: {gp}")
        print(f"  Score: {score.get('home')} - {score.get('away')}")
        print(f"  Q4: {q4.get('home')} - {q4.get('away')}")
        print(f"  Quarters: {list(quarters.keys())}")
        return True
        
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  ❌ TIMEOUT después de {elapsed:.1f}s")
        return False
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ❌ ERROR después de {elapsed:.1f}s: {type(e).__name__}: {e}")
        return False

async def main():
    print(f"\nDiagnóstico de fetch FT - {TEST_NAME}")
    
    # Test 1: traditional (backend actual para FT)
    ok_trad = await test_backend("traditional", TEST_MATCH_ID, timeout_secs=70.0)
    
    # Test 2: obscura (si traditional falló)
    print(f"\n--- Ahora probando con Obscura ---")
    ok_obs = await test_backend("obscura", TEST_MATCH_ID, timeout_secs=70.0)
    
    print(f"\n{'='*60}")
    print(f"  RESUMEN:")
    print(f"    traditional: {'✅ OK' if ok_trad else '❌ FAIL'}")
    print(f"    obscura:     {'✅ OK' if ok_obs else '❌ FAIL'}")
    print(f"{'='*60}")

asyncio.run(main())
