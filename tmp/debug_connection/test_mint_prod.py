"""
tmp/debug_connection/test_mint_prod.py
Proposito: Verificar que TokenPool._mint_token() de produccion emite un JWT nuevo
usando la huella OkHttp + UA firmado, sin tocar el telefono ni modificar tokens.json.
"""
import asyncio
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.core.token_manager import TokenPool


async def main() -> None:
    pool = TokenPool()
    print(f"tokens en disco: {len(pool.tokens)}")
    item = await pool._mint_token()
    if item:
        print(f"JWT emitido OK: ...{item.token[-14:]}")
        print(f"device_uuid: {item.device_uuid}")
    else:
        print("FALLO la emision")


if __name__ == "__main__":
    asyncio.run(main())
