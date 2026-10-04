"""
tmp/debug_connection/bench_clients.py
Proposito: Benchmark riguroso entre tls_client y curl_cffi. Para evitar sesgo de
cache del servidor: (1) usa IDs de partido DISJUNTOS por cliente, (2) alterna las
peticiones, (3) reporta media/mediana/min/p90 por request. Mide latencia de un
endpoint liviano (event/{id}) y de un partido completo (7 endpoints en paralelo).
"""
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


def get_ids(n):
    con = sqlite3.connect(ROOT / "matches.db")
    rows = con.execute(
        "select distinct match_id from matches "
        "where date(datetime(date || ' ' || time, '-6 hours')) between '2026-08-01' and '2026-10-03' "
        "order by match_id desc limit ?", (n,),
    ).fetchall()
    return [str(r[0]) for r in rows]


def mktls():
    local = threading.local()

    def sess():
        s = getattr(local, "s", None)
        if s is None:
            s = tls_client.Session(client_identifier="okhttp4_android_13", random_tls_extension_order=False)
            local.s = s
        return s

    h = headers()
    h["Content-Type"] = "application/json; charset=UTF8"
    tok = sess().post(f"{BASE}/token/init", headers=h, json=payload(), timeout_seconds=15).json()["token"]

    def one(ep):
        return sess().get(f"{BASE}/{ep}", headers=headers(tok), timeout_seconds=20)

    return one


def mkcffi():
    local = threading.local()

    def sess():
        s = getattr(local, "s", None)
        if s is None:
            s = cffi.Session(ja3=JA3, impersonate="chrome")
            local.s = s
        return s

    h = headers()
    h["Content-Type"] = "application/json; charset=UTF8"
    tok = sess().post(f"{BASE}/token/init", headers=h, json=payload(), timeout=15).json()["token"]

    def one(ep):
        return sess().get(f"{BASE}/{ep}", headers=headers(tok), timeout=20)

    return one


def report(name, times):
    times_ms = sorted(t * 1000 for t in times)
    p90 = times_ms[int(len(times_ms) * 0.9) - 1]
    print(f"  {name:<12} media={statistics.mean(times_ms):6.1f}ms  mediana={statistics.median(times_ms):6.1f}ms  min={times_ms[0]:5.1f}  p90={p90:6.1f}  max={times_ms[-1]:6.1f}")


def main() -> None:
    ids = get_ids(80)
    tls_ids, cffi_ids = ids[:40], ids[40:80]
    tls, cffi_f = mktls(), mkcffi()

    # warmup
    tls(f"event/{tls_ids[0]}"); cffi_f(f"event/{cffi_ids[0]}")

    tls_t, cffi_t = [], []
    for i in range(40):
        t0 = time.perf_counter(); tls(f"event/{tls_ids[i]}"); tls_t.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); cffi_f(f"event/{cffi_ids[i]}"); cffi_t.append(time.perf_counter() - t0)

    print("== 1 request (event/{id}), IDs disjuntos, alternado ==")
    report("tls_client", tls_t)
    report("curl_cffi", cffi_t)

    # Partido completo: 7 endpoints en paralelo, 15 partidos disjuntos cada uno
    tls_m, cffi_m = [], []
    for i in range(15):
        mid = tls_ids[i]
        eps = [f"event/{mid}", f"event/{mid}/incidents", f"event/{mid}/graph", f"event/{mid}/h2h",
               f"event/{mid}/statistics", f"event/{mid}/lineups", f"event/{mid}/odds/1/all"]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
            list(ex.map(tls, eps))
        tls_m.append(time.perf_counter() - t0)

        mid = cffi_ids[i]
        eps = [f"event/{mid}", f"event/{mid}/incidents", f"event/{mid}/graph", f"event/{mid}/h2h",
               f"event/{mid}/statistics", f"event/{mid}/lineups", f"event/{mid}/odds/1/all"]
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=7) as ex:
            list(ex.map(cffi_f, eps))
        cffi_m.append(time.perf_counter() - t0)

    print("== Partido completo (7 endpoints en paralelo), IDs disjuntos ==")
    report("tls_client", tls_m)
    report("curl_cffi", cffi_m)


if __name__ == "__main__":
    main()
