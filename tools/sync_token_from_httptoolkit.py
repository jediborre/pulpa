"""
Script de Sincronización y Captura de JWT desde HTTP Toolkit

Propósito:
- Conectarse al socket/servidor local de HTTP Toolkit (//./pipe/httptoolkit-ctl).
- Inspeccionar las peticiones interceptadas de la app móvil de SofaScore.
- Extraer el token Bearer JWT legítimo emitido por la app oficial.
- Validar el token contra la API de SofaScore y guardarlo en monitor_v3/config/tokens.json.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NODE_EXE = Path(r"C:\Users\App\AppData\Local\Programs\HTTP Toolkit\resources\httptoolkit-server\bin\node.exe")
TOKENS_JSON = ROOT / "monitor_v3" / "config" / "tokens.json"

JS_SCRIPT = """
const http = require('http');
function postJson(body) {
    return new Promise((resolve) => {
        const payload = JSON.stringify(body);
        const req = http.request({
            socketPath: '//./pipe/httptoolkit-ctl',
            path: '/api/execute',
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(payload) }
        }, res => {
            let data = '';
            res.on('data', c => data += c);
            res.on('end', () => {
                try { resolve(JSON.parse(data)); } catch(e) { resolve(data); }
            });
        });
        req.on('error', err => resolve({ error: err.message }));
        req.write(payload);
        req.end();
    });
}
async function run() {
    const list = await postJson({ name: 'events.list', source: 'mcp', args: { filter: 'hostname=api.sofascore.com', limit: 100 } });
    if (!list || !list.data || !list.data.events) {
        process.stdout.write(JSON.stringify({ error: 'No events found or HTTP Toolkit not reachable' }));
        return;
    }
    const tokens = [];
    const seen = new Set();
    // Revisar de mas reciente a mas antiguo
    for (const ev of list.data.events.reverse()) {
        const outline = await postJson({ name: 'events.get-outline', source: 'mcp', args: { id: ev.id } });
        if (outline && outline.data && outline.data.request && outline.data.request.headers) {
            const auth = outline.data.request.headers.authorization;
            if (auth && auth.startsWith('Bearer eyJ')) {
                const tok = auth.substring(7).trim();
                if (!seen.has(tok)) {
                    seen.add(tok);
                    tokens.push({
                        token: tok,
                        url: ev.url,
                        method: ev.method,
                        timestamp: ev.timestamp
                    });
                }
            }
        }
    }
    process.stdout.write(JSON.stringify({ tokens }));
}
run();
"""

def extract_tokens_from_httptoolkit() -> list[dict]:
    if not NODE_EXE.exists():
        print(f"[ERROR] Node de HTTP Toolkit no encontrado en: {NODE_EXE}")
        return []

    try:
        proc = subprocess.run(
            [str(NODE_EXE), "-e", JS_SCRIPT],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode != 0:
            print(f"[ERROR] Node falló: {proc.stderr}")
            return []
        
        data = json.loads(proc.stdout)
        if "error" in data:
            print(f"[ADVERTENCIA] HTTP Toolkit: {data['error']}")
            return []
        return data.get("tokens", [])
    except Exception as e:
        print(f"[ERROR] Excepción consultando HTTP Toolkit: {e}")
        return []

def sync():
    print("=" * 65)
    print("SINCRONIZADOR DE JWT SOFASCORE DESDE HTTP TOOLKIT")
    print("=" * 65)
    print("Buscando peticiones autenticadas de SofaScore en HTTP Toolkit...")
    
    ht_tokens = extract_tokens_from_httptoolkit()
    if ht_tokens:
        print(f"[+] Se encontraron {len(ht_tokens)} token(s) en el historial de HTTP Toolkit.")
    else:
        print("[-] No se encontraron tokens nuevos en el historial inmediato de HTTP Toolkit.")

    # Cargar tokens existentes en disco
    existing_tokens = {}
    if TOKENS_JSON.exists():
        try:
            with open(TOKENS_JSON, "r", encoding="utf-8") as f:
                d = json.load(f)
                for t in d.get("tokens", []):
                    existing_tokens[t["token"]] = t
        except Exception:
            pass

    # Unificar candidatos sin duplicados
    candidates = []
    seen = set()

    for item in ht_tokens:
        tok = item["token"]
        if tok not in seen:
            seen.add(tok)
            candidates.append({"token": tok, "source": "HTTP Toolkit (Nuevo/Capturado)", "meta": existing_tokens.get(tok)})

    for tok, meta in existing_tokens.items():
        if tok not in seen:
            seen.add(tok)
            candidates.append({"token": tok, "source": "tokens.json (Previo)", "meta": meta})

    if not candidates:
        print("\n[!] No hay ningún token para evaluar (ni en tokens.json ni en HTTP Toolkit).")
        print("    Asegúrate de:")
        print("    1. Tener HTTP Toolkit abierto.")
        print("    2. Borrar datos de la app SofaScore en el teléfono y abrirla.")
        print("    3. Tocar o navegar en cualquier partido durante 5 segundos.")
        return False

    print(f"\nEvaluando {len(candidates)} token(s) candidato(s) contra API de SofaScore...")

    import httpx
    proxy = "http://127.0.0.1:8000"
    cert = r"C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem"
    verify_cert = cert if os.path.exists(cert) else True

    valid_pool_items = []

    for idx, cand in enumerate(candidates, 1):
        tok = cand["token"]
        src = cand["source"]
        print(f"\nProbando token #{idx} [{src}] (...{tok[-12:]})...")
        headers = {
            "User-Agent": "com.sofascore.results/260921/2b47a6",
            "Authorization": f"Bearer {tok}",
            "Connection": "Keep-Alive",
        }
        try:
            r = httpx.get(
                "https://api.sofascore.com/api/v1/event/16689322",
                headers=headers,
                proxy=proxy,
                verify=verify_cert,
                timeout=10.0,
            )
            if r.status_code == 200:
                print(f"  [OK] Token VÁLIDO y aceptado por SofaScore (HTTP 200)!")
                meta = cand["meta"]
                if meta:
                    meta["failures"] = 0
                    meta["last_used"] = time.time()
                    valid_pool_items.append(meta)
                else:
                    valid_pool_items.append({
                        "token": tok,
                        "created_at": time.time(),
                        "device_uuid": f"android-{len(valid_pool_items)+1}",
                        "advertising_id": f"ad-{len(valid_pool_items)+1}",
                        "failures": 0,
                        "last_used": time.time(),
                    })
            else:
                print(f"  [RECHAZADO] HTTP {r.status_code}: {r.text[:120]}")
        except Exception as ex:
            print(f"  [ERROR] Fallo de conexión: {ex}")

    # Guardar en tokens.json
    TOKENS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "count": len(valid_pool_items),
        "tokens": valid_pool_items,
    }
    with open(TOKENS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if not valid_pool_items:
        print("\n[!] Ninguno de los tokens probados es válido (todos caducados o desafiados por Cloudflare).")
        print("    Por favor borra datos de SofaScore en el teléfono, ábrela e intenta de nuevo.")
        return False

    print(f"\n[ÉXITO] Pool actualizado con {len(valid_pool_items)} token(s) válido(s) en: {TOKENS_JSON}")
    for i, it in enumerate(valid_pool_items, 1):
        print(f"  Token #{i}: ...{it['token'][-12:]}")
    return True

if __name__ == "__main__":
    sync()
