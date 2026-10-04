"""
Herramienta de Desinstalación y Reinstalación Limpia de SofaScore (APK Parcheada)

Propósito:
- Localizar adb.exe automáticamente (en PATH o en Android SDK).
- Verificar la conexión con el dispositivo Android vía depuración USB.
- Desinstalar completamente el paquete com.sofascore.results para borrar el Installation ID,
  Keystore y credenciales locales marcadas por Cloudflare.
- Reinstalar de forma limpia el paquete parcheado con sus splits (arm64_v8a y xxxhdpi).
- Lanzar automáticamente la app en el dispositivo para que solicite un nuevo JWT limpio.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "com.sofascore.results"

APK_DIR = ROOT / "tmp" / "monitor_v3_poc"
BASE_APK = APK_DIR / "Sofascore-patched.apk"
SPLIT_ARM64 = APK_DIR / "split_config.arm64_v8a.apk"
SPLIT_DPI = APK_DIR / "split_config.xxxhdpi.apk"


def find_adb() -> Path | None:
    """Busca el binario de adb en el sistema."""
    which_adb = shutil.which("adb")
    if which_adb:
        return Path(which_adb)

    candidates = [
        Path(r"C:\Users\App\AppData\Local\Android\Sdk\platform-tools\adb.exe"),
        Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "HTTP Toolkit" / "resources" / "httptoolkit-server" / "bin" / "adb.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def run_adb(adb: Path, args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Ejecuta un comando de adb y captura su salida."""
    cmd = [str(adb)] + args
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")


def reinstall() -> bool:
    print("=" * 65)
    print("REINSTALADOR LIMPIO DE SOFASCORE (APK PARCHEADA)")
    print("=" * 65)

    adb = find_adb()
    if not adb or not adb.exists():
        print("[ERROR] No se encontró el ejecutable 'adb.exe'.")
        print("        Verifica que Android SDK esté instalado o adb en el PATH.")
        return False
    print(f"[+] ADB detectado en: {adb}")

    # Verificar archivos APK
    for f in [BASE_APK, SPLIT_ARM64, SPLIT_DPI]:
        if not f.exists():
            print(f"[ERROR] Archivo APK requerido no encontrado: {f}")
            return False
    print(f"[+] APKs listas en: {APK_DIR}")

    # Verificar dispositivos conectados
    res = run_adb(adb, ["devices"])
    lines = [line.strip() for line in res.stdout.strip().splitlines() if line.strip() and not line.startswith("*")]
    device_lines = [l for l in lines[1:] if "\tdevice" in l]

    if not device_lines:
        print("\n[!] No hay ningún dispositivo Android conectado con depuración USB.")
        print("    Asegúrate de:")
        print("    1. Conectar tu teléfono por USB.")
        print("    2. Habilitar 'Depuración USB' en Opciones de desarrollador.")
        print("    3. Aceptar la huella digital RSA en la pantalla de tu teléfono.")
        return False

    dev_id = device_lines[0].split("\t")[0]
    print(f"[+] Dispositivo Android detectado: {dev_id}")

    # Paso 1: Desinstalar app previa
    print("\n[1/3] Desinstalando versión previa de SofaScore...")
    uninst = run_adb(adb, ["uninstall", PACKAGE_NAME])
    out_uninst = (uninst.stdout + uninst.stderr).strip()
    if "Success" in out_uninst:
        print("  [OK] Desinstalación completada con éxito (datos y sesión purgados).")
    else:
        print(f"  [AVISO] {out_uninst} (Es normal si la app no estaba instalada).")

    time.sleep(1)

    # Paso 2: Instalar APKs divididas
    print("\n[2/3] Instalando APK parcheada limpia con splits...")
    print(f"      - Base: {BASE_APK.name}")
    print(f"      - Arch: {SPLIT_ARM64.name}")
    print(f"      - Res:  {SPLIT_DPI.name}")

    install_args = [
        "install-multiple",
        "-r",
        "-d",
        str(BASE_APK),
        str(SPLIT_ARM64),
        str(SPLIT_DPI),
    ]
    inst = run_adb(adb, install_args, timeout=120)
    out_inst = (inst.stdout + inst.stderr).strip()

    if "Success" in out_inst:
        print("  [OK] ¡Instalación exitosa!")
    else:
        print(f"  [ERROR] Falló la instalación: {out_inst}")
        return False

    # Paso 3: Iniciar la app
    print("\n[3/3] Abriendo SofaScore en el teléfono...")
    launch = run_adb(adb, ["shell", "monkey", "-p", PACKAGE_NAME, "-c", "android.intent.category.LAUNCHER", "1"])
    time.sleep(2)
    print("  [OK] Aplicación iniciada con nuevo Installation ID.")

    print("\n" + "=" * 65)
    print("LISTO. Ahora realiza estos pasos en tu teléfono:")
    print(" 1. Verifica que HTTP Toolkit esté activo e interceptando el teléfono.")
    print(" 2. Navega o toca cualquier partido en SofaScore durante 5 segundos.")
    print(" 3. Ejecuta la opción de sincronización en menu.bat (o 'menu.bat token').")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = reinstall()
    sys.exit(0 if success else 1)
