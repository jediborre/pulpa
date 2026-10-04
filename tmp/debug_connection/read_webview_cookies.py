"""
tmp/debug_connection/read_webview_cookies.py
Proposito: Conectarse por CDP al WebView de la app (webview_devtools_remote) via
adb forward tcp:9222 y consultar Network.getAllCookies para descubrir cookies de
sesion (p. ej. cf_clearance / cookies de sofascore) que la app pudiera inyectar en
OkHttp mediante SCSWebviewCookieJar.
"""
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import urllib.request
import websocket


def targets():
    with urllib.request.urlopen("http://127.0.0.1:9222/json", timeout=10) as r:
        return json.load(r)


def main() -> None:
    tgs = targets()
    if not tgs:
        print("sin targets")
        return
    ws_url = tgs[0]["webSocketDebuggerUrl"]
    ws = websocket.create_connection(ws_url, timeout=15, suppress_origin=True)
    ws.send(json.dumps({"id": 1, "method": "Network.enable"}))
    ws.recv()
    ws.send(json.dumps({"id": 2, "method": "Network.getAllCookies"}))
    resp = json.loads(ws.recv())
    ws.close()
    cookies = resp.get("result", {}).get("cookies", [])
    print(f"total cookies: {len(cookies)}")
    for c in cookies:
        dom = c.get("domain", "")
        name = c.get("name", "")
        val = c.get("value", "")
        flag = "  <<< RELEVANTE" if ("sofascore" in dom or "clearance" in name.lower()) else ""
        print(f"  {dom:<30} {name:<24} len={len(val):<4} http={c.get('httpOnly')} sec={c.get('secure')}{flag}")


if __name__ == "__main__":
    main()
