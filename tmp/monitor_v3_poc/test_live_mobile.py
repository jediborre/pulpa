import sys
import time
import urllib.request
import json
import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

headers = {
    'User-Agent': 'com.sofascore.results/260921/2b47a6',
    'x-timestamp': str(int(time.time() * 1000)),
    'Accept-Encoding': 'gzip',
    'Cache-Control': 'max-age=0',
    'Connection': 'Keep-Alive',
    'Host': 'api.sofascore.com',
}

endpoints = [
    'https://api.sofascore.com/api/v1/event/newly-added-events',
    'https://api.sofascore.com/api/v1/sport/basketball/events/live',
]

print("=== 1. Probando con requests estándar (HTTP/1.1) ===")
s = requests.Session()
for url in endpoints:
    headers['x-timestamp'] = str(int(time.time() * 1000))
    t0 = time.time()
    try:
        res = s.get(url, headers=headers, timeout=10)
        dur = (time.time() - t0) * 1000
        print(f"[{res.status_code}] ({dur:.1f}ms) {url}")
        if res.status_code == 200:
            print("   -> Éxito:", res.text[:150])
        else:
            print("   -> Error:", res.text[:200])
    except Exception as e:
        print("   -> Exception:", e)

print("\n=== 2. Probando a través del proxy local de HTTP Toolkit (127.0.0.1:8000) ===")
proxies = {'http': 'http://127.0.0.1:8000', 'https': 'http://127.0.0.1:8000'}
for url in endpoints:
    headers['x-timestamp'] = str(int(time.time() * 1000))
    t0 = time.time()
    try:
        res = requests.get(url, headers=headers, proxies=proxies, verify=r'C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem', timeout=10)
        dur = (time.time() - t0) * 1000
        print(f"[Proxy {res.status_code}] ({dur:.1f}ms) {url}")
        if res.status_code == 200:
            print("   -> Éxito vía Proxy:", res.text[:150])
        else:
            print("   -> Error vía Proxy:", res.text[:200])
    except Exception as e:
        print("   -> Exception:", e)

