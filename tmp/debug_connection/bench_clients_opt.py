"""
tmp/debug_connection/bench_clients_opt.py
Proposito: Benchmark optimizado. Corrige el sesgo del benchmark anterior (que creaba
una sesion por thread y recreaba el ThreadPool en cada partido, perdiendo keep-alive).
Aqui: (a) pool de threads persistente, (b) sesion curl_cffi COMPARTIDA (curl handle
thread-local interno => reutiliza conexiones), (c) variante AsyncSession sin threads.
Compara con tls_client en el mismo patron (7 endpoints en paralelo por partido).
IDs de partido disjuntos por cliente.
"""
import asyncio
import concurrent.futures
import hashlib
import sqlite3
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import tls_client
from curl_cffi import requests as cffi

ROOT = Path(__file__).resolve().parents[2]
JA3 = "771,4865-4866-4867-49195-49196-52393-49199-49200-52392-49171-49172-156-157-47-53,0-23-65281-10-11-35-16-5-13-51-45-43-21,29-23-24,0"
PKG = "com.sofascore.results"
VER6 = "260921"
BASE = "https://api.sofascore.com/api/v1"
MATCHES = 15


def signed_ua() -> str:
    bucket = int(time.time() // 100)
    return f"{PKG}/{VER6}/{hashlib.md5(f'{bucket}sofa2012'.encode()).hexdigest()[:6]}"


def headers(token=None):
    h = {
        "User-Agent": signed_ua(),
        "X-Timestamp": str(int(time.time() * 1000)),
        "app-version": VER6,
        "Cache-Control": "max-age=0",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def payload():
    return {
        "deviceType": "android", "version": 260921, "sdk": 29, "language": "en",
        "country": "MX", "timezone": -18000,
        "advertisingId": str(uuid.uuid4()), "uuid": str(uuid.uuid4()),
    }


def eps(mid):
    return [f"event/{mid}", f"event/{mid}/incidents", f"event/{mid}/graph", f"event/{mid}/h2h",
            f"event/{mid}/statistics", f"event/{mid}/lineups", f"event/{mid}/odds/1/all"]


def get_ids(n):
    con = sqlite3.connect(ROOT / "matches.db")
    rows = con.execute(
        "select distinct match_id from matches "
        "where date(datetime(date || ' ' || time, '-6 hours')) between '2026-08-01' and '2026-10-03' "
        "order by match_id desc limit ?", (n,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def report(name, times):
    ms = sorted(t * 1000 for t in times)
    p90 = ms[int(len(ms) * 0.9) - 1]
    print(f"  {name:<20} media={statistics.mean(ms):7.1f}ms  mediana={statistics.median(ms):7.1f}ms  min={ms[0]:6.1f}  p90={p90:7.1f}  max={ms[-1]:7.1f}")


def bench_tls(ids):
    local = threading.local()

    def sess():
        s = getattr(local, "s", None)
        if s is None:
            s = tls_client.Session(client_identifier="okhttp4_android_13", random_tls_extension_order=False)
            local.s = s
        return s

    h = headers(); h["Content-Type"] = "application/json; charset=UTF8"
    tok = sess().post(f"{BASE}/token/init", headers=h, json=payload(), timeout_seconds=15).json()["token"]

    def fetch(ep):
        return sess().get(f"{BASE}/{ep}", headers=headers(tok), timeout_seconds=20)

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=7)
    list(ex.map(fetch, eps(ids[0])))
    times = []
    for mid in ids:
        t0 = time.perf_counter()
        list(ex.map(fetch, eps(mid)))
        times.append(time.perf_counter() - t0)
    ex.shutdown()
    return times


def bench_cffi_shared(ids):
    # UNA sesion compartida: curl_cffi usa un handle curl thread-local por thread
    # y reutiliza conexiones entre llamadas.
    s = cffi.Session(ja3=JA3, impersonate="chrome")
    h = headers(); h["Content-Type"] = "application/json; charset=UTF8"
    tok = s.post(f"{BASE}/token/init", headers=h, json=payload(), timeout=15).json()["token"]

    def fetch(ep):
        return s.get(f"{BASE}/{ep}", headers=headers(tok), timeout=20)

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=7)
    list(ex.map(fetch, eps(ids[0])))
    times = []
    for mid in ids:
        t0 = time.perf_counter()
        list(ex.map(fetch, eps(mid)))
        times.append(time.perf_counter() - t0)
    ex.shutdown()
    return times


def bench_cffi_ja3(ids):
    # Solo JA3 (sin impersonate) para medir si el perfil completo añade overhead.
    s = cffi.Session(ja3=JA3)
    h = headers(); h["Content-Type"] = "application/json; charset=UTF8"
    tok = s.post(f"{BASE}/token/init", headers=h, json=payload(), timeout=15).json()["token"]

    def fetch(ep):
        return s.get(f"{BASE}/{ep}", headers=headers(tok), timeout=20)

    ex = concurrent.futures.ThreadPoolExecutor(max_workers=7)
    list(ex.map(fetch, eps(ids[0])))
    times = []
    for mid in ids:
        t0 = time.perf_counter()
        list(ex.map(fetch, eps(mid)))
        times.append(time.perf_counter() - t0)
    ex.shutdown()
    return times


def bench_cffi_async(ids):
    async def run():
        async with cffi.AsyncSession(ja3=JA3, impersonate="chrome") as s:
            h = headers(); h["Content-Type"] = "application/json; charset=UTF8"
            r = await s.post(f"{BASE}/token/init", headers=h, json=payload(), timeout=15)
            tok = r.json()["token"]

            async def fetch(ep):
                return await s.get(f"{BASE}/{ep}", headers=headers(tok), timeout=20)

            await asyncio.gather(*[fetch(ep) for ep in eps(ids[0])])
            times = []
            for mid in ids:
                t0 = time.perf_counter()
                await asyncio.gather(*[fetch(ep) for ep in eps(mid)])
                times.append(time.perf_counter() - t0)
            return times

    return asyncio.run(run())


def main() -> None:
    ids = get_ids(MATCHES * 4)
    a, b, c, d = (ids[:MATCHES], ids[MATCHES:MATCHES * 2],
                  ids[MATCHES * 2:MATCHES * 3], ids[MATCHES * 3:MATCHES * 4])

    print(f"Partido completo ({MATCHES} partidos, IDs disjuntos):")
    report("tls_client", bench_tls(a))
    report("curl_cffi shared", bench_cffi_shared(b))
    report("curl_cffi ja3-only", bench_cffi_ja3(d))
    report("curl_cffi async", bench_cffi_async(c))


if __name__ == "__main__":
    main()
