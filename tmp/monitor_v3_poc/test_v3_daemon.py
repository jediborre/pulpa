"""
Script auxiliar: test_v3_daemon.py
Propósito: Validar el arranque y ejecución inicial del daemon monitor_v3 en vivo
durante 15 segundos y verificar que se ejecuta limpiamente.
"""

import asyncio
import sys
import time

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path
root_dir = str(Path(__file__).resolve().parents[2])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from monitor_v3.main import main_async

async def run_timed():
    print("Iniciando prueba de 12 segundos para monitor_v3...")
    task = asyncio.create_task(main_async())
    await asyncio.sleep(12)
    print("\nTiempo cumplido. Cancelando tarea para verificar apagado seguro...")
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("Tarea cancelada correctamente.")

if __name__ == '__main__':
    asyncio.run(run_timed())
