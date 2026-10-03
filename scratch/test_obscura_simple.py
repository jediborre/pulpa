"""Test simple: verificar que Obscura se inicia y acepta conexiones CDP"""
import subprocess
import time
import socket
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OBSCURA_EXE = ROOT / "tools" / "obscura" / "v0.1.5" / "obscura.exe"
OBSCURA_DIR = ROOT / "tools" / "obscura" / "v0.1.5"
CERT_PATH = ROOT / ".venv" / "Lib" / "site-packages" / "certifi" / "cacert.pem"

os.environ["SSL_CERT_FILE"] = str(CERT_PATH)

print("[1] Matando procesos obscura existentes...")
subprocess.run(["taskkill", "/IM", "obscura.exe", "/F"],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1)

print(f"[2] Iniciando Obscura desde {OBSCURA_DIR}...")
print(f"    Executable: {OBSCURA_EXE}")
print(f"    Exists: {OBSCURA_EXE.exists()}")

proc = subprocess.Popen(
    [str(OBSCURA_EXE), "serve", "--port", "9222", "--stealth"],
    cwd=str(OBSCURA_DIR),
    env=os.environ,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
)

print(f"[3] Proceso iniciado con PID: {proc.pid}")

print("[4] Esperando 5 segundos para que inicie...")
time.sleep(5)

print("[5] Verificando si el puerto 9222 está escuchando...")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    result = sock.connect_ex(('127.0.0.1', 9222))
    sock.close()
    if result == 0:
        print("[OK] Puerto 9222 está escuchando!")
    else:
        print(f"[FAIL] Puerto 9222 NO está escuchando (código: {result})")
except Exception as e:
    print(f"[ERROR] {e}")

print("[6] Leyendo stdout/stderr del proceso...")
try:
    stdout, stderr = proc.communicate(timeout=2)
    print(f"STDOUT: {stdout.decode('utf-8', errors='ignore')[:500]}")
    print(f"STDERR: {stderr.decode('utf-8', errors='ignore')[:500]}")
except subprocess.TimeoutExpired:
    print("[INFO] Proceso sigue corriendo (timeout esperado)")

print("[7] Limpiando...")
proc.terminate()
try:
    proc.wait(timeout=3)
except:
    proc.kill()

print("[DONE]")
