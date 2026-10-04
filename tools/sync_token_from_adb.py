"""
Sincronizador y Generador por Lotes de Tokens JWT directo desde Android vía ADB

Propósito:
- Conectarse al teléfono vía ADB con 'run-as com.sofascore.results'.
- Automatizar el ciclo de generación: resetear datos de la app, abrirla, esperar
  interacción del usuario (12s), extraer el nuevo AUTH_TOKEN y acumularlo en el pool.
- Permitir generar N tokens consecutivos de forma guiada y desatendida.
- Mostrar una tabla resumen con todos los tokens disponibles, expiración y UUIDs.
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
TOKENS_JSON = ROOT / "monitor_v3" / "config" / "tokens.json"
PACKAGE_NAME = "com.sofascore.results"
PREFS_FILE = "shared_prefs/com.sofascore.results_preferences.xml"


def find_adb() -> Path | None:
    which_adb = shutil.which("adb")
    if which_adb:
        return Path(which_adb)
    candidates = [
        Path(r"C:\Users\App\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_adb(adb: Path, args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    cmd = [str(adb)] + args
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)


def decode_jwt_payload(token: str) -> dict:
    try:
        parts = token.split(".")
        if len(parts) >= 2:
            payload_b64 = parts[1]
            payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
            data = base64.urlsafe_b64decode(payload_b64)
            return json.loads(data.decode("utf-8"))
    except Exception:
        pass
    return {}


def load_pool() -> list[dict]:
    if TOKENS_JSON.exists():
        try:
            with open(TOKENS_JSON, "r", encoding="utf-8") as f:
                d = json.load(f)
                return d.get("tokens", [])
        except Exception:
            pass
    return []


def save_pool(tokens: list[dict]) -> None:
    TOKENS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "count": len(tokens),
        "tokens": tokens,
    }
    with open(TOKENS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def display_pool_table(tokens: list[dict]) -> None:
    print("\n" + "=" * 82)
    print(f"RESUMEN DEL POOL DE TOKENS (TOTAL: {len(tokens)} TOKENS ACTIVOS)")
    print("=" * 82)
    if not tokens:
        print("  [!] El pool está actualmente vacío.")
        print("=" * 82)
        return

    print(f"{'#':<3} | {'Device UUID':<37} | {'Expiración':<19} | {'Token Sufijo':<15}")
    print("-" * 3 + "-+-" + "-" * 37 + "-+-" + "-" * 19 + "-+-" + "-" * 15)

    for idx, it in enumerate(tokens, 1):
        tok = it.get("token", "")
        uuid_str = it.get("device_uuid", "desconocido")[:36]
        meta = decode_jwt_payload(tok)
        exp = meta.get("exp")
        exp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp)) if exp else "N/A"
        suffix = f"...{tok[-12:]}" if len(tok) > 12 else tok
        print(f"{idx:<3} | {uuid_str:<37} | {exp_str:<19} | {suffix:<15}")

    print("=" * 82 + "\n")


def extract_token_from_device(adb: Path) -> str | None:
    res = run_adb(adb, ["shell", "run-as", PACKAGE_NAME, "cat", PREFS_FILE])
    if res.returncode != 0:
        return None
    match = re.search(r'name=["\']AUTH_TOKEN["\']>([^<]+)<', res.stdout)
    if match:
        return match.group(1).strip()
    return None


def generate_single_token(adb: Path, current_step: int, total_steps: int, wait_seconds: int = 12) -> str | None:
    print(f"\n--- [Token {current_step}/{total_steps}] Generando nueva sesión limpia ---")
    
    # 1. Resetear datos
    print(f"  [1/3] Reseteando datos de SofaScore vía ADB (pm clear)...")
    run_adb(adb, ["shell", "pm", "clear", PACKAGE_NAME])
    time.sleep(1)

    # 2. Iniciar app
    print(f"  [2/3] Abriendo SofaScore en el teléfono...")
    run_adb(adb, ["shell", "monkey", "-p", PACKAGE_NAME, "-c", "android.intent.category.LAUNCHER", "1"])

    # 3. Cuenta regresiva con mensaje
    print(f"  [3/3] 📱 Toca la pantalla o abre cualquier partido en el teléfono:")
    for s in range(wait_seconds, 0, -1):
        print(f"\r        ⏳ Esperando interacción del usuario... [{s:02d}s restantes] ", end="", flush=True)
        time.sleep(1)
    print("\r        ✅ Tiempo cumplido. Inspeccionando memoria de la app...            ")

    # 4. Extraer token (con reintentos)
    for retry in range(1, 4):
        token = extract_token_from_device(adb)
        if token:
            payload = decode_jwt_payload(token)
            exp = payload.get("exp")
            exp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(exp)) if exp else "N/A"
            dev_id = payload.get("id", "android-direct")
            print(f"  [OK] ¡Token capturado exitosamente!")
            print(f"       UUID:   {dev_id}")
            print(f"       Expira: {exp_str}")
            print(f"       Sufijo: ...{token[-12:]}")
            return token
        time.sleep(2)

    print("  [ERROR] No se pudo encontrar el AUTH_TOKEN tras esperar. ¿Se abrió la app correctamente?")
    return None


def sync_batch(num_tokens: int = 1, wait_seconds: int = 12) -> bool:
    print("=" * 65)
    print("EXTRACTOR Y GENERADOR DE TOKENS SOFASCORE VIA ADB (USB)")
    print("=" * 65)

    adb = find_adb()
    if not adb:
        print("[ERROR] No se encontró el ejecutable 'adb.exe'.")
        print("        Verifica que Android SDK esté instalado o adb en el PATH.")
        return False

    # Verificar dispositivo conectado
    res = run_adb(adb, ["devices"])
    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip() and not line.startswith("*")]
    device_lines = [l for l in lines[1:] if "\tdevice" in l]

    if not device_lines:
        print("\n[!] No hay ningún dispositivo Android conectado con depuración USB.")
        print("    Asegúrate de conectar el teléfono por USB y autorizar la conexión.")
        return False

    dev_id = device_lines[0].split("\t")[0]
    print(f"[+] Dispositivo detectado: {dev_id}")

    existing_tokens = load_pool()
    print(f"[+] Tokens existentes en pool: {len(existing_tokens)}")

    if num_tokens <= 0:
        display_pool_table(existing_tokens)
        return True

    new_tokens_captured = 0
    all_tokens = list(existing_tokens)

    for i in range(1, num_tokens + 1):
        token_str = generate_single_token(adb, i, num_tokens, wait_seconds=wait_seconds)
        if token_str:
            # Comprobar si ya existe
            already = next((t for t in all_tokens if t.get("token") == token_str), None)
            if already:
                print("  [AVISO] Este token ya estaba en el pool. Omitiendo duplicado.")
            else:
                payload = decode_jwt_payload(token_str)
                new_item = {
                    "token": token_str,
                    "created_at": time.time(),
                    "device_uuid": payload.get("id", f"android-{len(all_tokens)+1}"),
                    "advertising_id": "direct-adb",
                    "failures": 0,
                    "last_used": time.time(),
                }
                all_tokens.insert(0, new_item)
                save_pool(all_tokens)
                new_tokens_captured += 1
                print(f"  [+] Guardado en tokens.json. Total activos ahora: {len(all_tokens)}")

        if i < num_tokens:
            print("  ⏳ Pausa de 2 segundos antes del siguiente ciclo...")
            time.sleep(2)

    # Mostrar tabla resumen final
    display_pool_table(all_tokens)
    print(f"Generación por lote concluida: {new_tokens_captured} nuevos token(s) agregados.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extractor y generador por lotes de tokens SofaScore vía ADB")
    parser.add_argument("-n", "--count", type=int, default=None, help="Cantidad de tokens a generar")
    parser.add_argument("-w", "--wait", type=int, default=12, help="Segundos de espera por interacción (default: 12)")
    parser.add_argument("--list", action="store_true", help="Solo listar los tokens actuales en el pool")
    args = parser.parse_args()

    if args.list:
        display_pool_table(load_pool())
        return

    num = args.count
    if num is None:
        existing = load_pool()
        print("=" * 65)
        print("SINCRONIZADOR DE TOKENS JWT VIA ADB")
        print("=" * 65)
        print(f"Actualmente tienes {len(existing)} token(s) en monitor_v3/config/tokens.json")
        try:
            raw = input("\n¿Cuántos tokens nuevos deseas generar en este lote? [1-10] (default: 1): ").strip()
            num = int(raw) if raw else 1
        except (ValueError, KeyboardInterrupt):
            num = 1

    sync_batch(num_tokens=num, wait_seconds=args.wait)


if __name__ == "__main__":
    main()
