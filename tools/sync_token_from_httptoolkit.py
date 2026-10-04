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
    
    tokens = extract_tokens_from_httptoolkit()
    if not tokens:
        print("\n[!] No se encontraron peticiones con token Bearer en HTTP Toolkit.")
        print("    Asegúrate de:")
        print("    1. Tener HTTP Toolkit abierto.")
        print("    2. Abrir la app SofaScore en el teléfono o borrar datos y abrirla.")
        print("    3. Navegar en cualquier partido para que emita tráfico.")
        return False

    print(f"[+] Se encontraron {len(tokens)} token(s) únicos en el historial reciente.")
    
    # Probar el más reciente
    import httpx
    proxy = "http://127.0.0.1:8000"
    cert = r"C:\Users\App\AppData\Local\httptoolkit\Config\ca.pem"
    verify_cert = cert if os.path.exists(cert) else True

    # Cargar tokens existentes
    existing_tokens = {}
    if TOKENS_JSON.exists():
        try:
            with open(TOKENS_JSON, "r", encoding="utf-8") as f:
                d = json.load(f)
                for t in d.get("tokens", []):
                    existing_tokens[t["token"]] = t
        except Exception:
            pass

    valid_tokens = []
    # Evaluar tokens encontrados en HTTP Toolkit
    for idx, item in enumerate(tokens, 1):
        tok = item["token"]
        print(f"\nProbando token #{idx} (...{tok[-12:]})...")
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
                print(f"  [OK] Token válido y aceptado por SofaScore (HTTP 200)!")
                if tok not in valid_tokens:
                    valid_tokens.append(tok)
            else:
                print(f"  [RECHAZADO] HTTP {r.status_code}: {r.text[:120]}")
        except Exception as ex:
            print(f"  [ERROR] Fallo de red: {ex}")

    if not valid_tokens:
        print("\n[!] Ninguno de los tokens actuales en HTTP Toolkit es válido o están caducados/desafiados.")
        print("    Por favor, abre la app en el teléfono (o borra datos y ábrela) para que genere uno fresco.")
        return False

    # Combinar tokens válidos existentes y nuevos
    pool_items = []
    for tok in valid_tokens:
        if tok in existing_tokens:
            pool_items.append(existing_tokens[tok])
        else:
            pool_items.append({
                "token": tok,
                "created_at": time.time(),
                "device_uuid": f"android-{len(pool_items)+1}",
                "advertising_id": f"ad-{len(pool_items)+1}",
                "failures": 0,
                "last_used": time.time(),
            })

    TOKENS_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": time.time(),
        "count": len(pool_items),
        "tokens": pool_items,
    }
    with open(TOKENS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[ÉXITO] Pool actualizado con {len(pool_items)} token(s) válido(s) en: {TOKENS_JSON}")
    for i, it in enumerate(pool_items, 1):
        print(f"  Token #{i}: ...{it['token'][-12:]}")
    return True

if __name__ == "__main__":
    sync()
