"""
Script auxiliar: test_token_generation.py
Propósito: Probar la generación bajo demanda de tokens JWT mediante POST a /api/v1/token/init
con UUIDs generados aleatoriamente, para validar la creación de un pool rotativo de tokens
auto-regenerable para monitor_v3.
"""

import sys
import time
import uuid
import requests
import json

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROXIES = {
    'http': 'http://127.0.0.1:8000',
    'https': 'http://127.0.0.1:8000'
}
CERT = r'C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem'

def generate_new_jwt():
    session = requests.Session()
    session.proxies.update(PROXIES)
    session.verify = CERT

    headers = {
        'User-Agent': 'com.sofascore.results/260921/022538',
        'x-timestamp': str(int(time.time() * 1000)),
        'Content-Type': 'application/json; charset=UTF8',
        'Accept-Encoding': 'gzip',
        'Cache-Control': 'max-age=0',
        'Connection': 'Keep-Alive',
        'Host': 'api.sofascore.com',
    }

    device_uuid = str(uuid.uuid4())
    ad_id = str(uuid.uuid4())

    payload = {
        "deviceType": "android",
        "version": 260921,
        "sdk": 29,
        "language": "en",
        "country": "MX",
        "timezone": -18000,
        "advertisingId": ad_id,
        "uuid": device_uuid
    }

    url = "https://api.sofascore.com/api/v1/token/init"
    t0 = time.time()
    res = session.post(url, headers=headers, json=payload, timeout=10)
    elapsed_ms = (time.time() - t0) * 1000

    print(f"Status POST /token/init: {res.status_code} ({elapsed_ms:.1f}ms)")
    if res.status_code == 200:
        token = res.json().get('token')
        print(f"-> Nuevo JWT obtenido exitosamente (Longitud: {len(token)} chars):")
        print(f"   {token[:40]}...{token[-40:]}")
        return token
    else:
        print(f"-> Error: {res.text}")
        return None

if __name__ == '__main__':
    print("Probando generación de 2 tokens nuevos con UUIDs distintos...")
    t1 = generate_new_jwt()
    t2 = generate_new_jwt()
    if t1 and t2 and t1 != t2:
        print("\n✅ ¡ÉXITO TOTAL! El endpoint /token/init puede generar tokens válidos bajo demanda.")
