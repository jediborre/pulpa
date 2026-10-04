"""
tmp/debug_connection/capture_ja3.py
Proposito: Capturar el ClientHello TLS del perfil okhttp4_android_13 de tls_client
mediante un listener TCP local, calcular su JA3 y guardarlo. Sirve para alimentar
curl_cffi (libcurl) con `ja3=` y demostrar que se puede prescindir del wrapper Go.
"""
import socket
import struct
import sys
import threading

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import tls_client

PORT = 44443
GREASE = {0x0a0a, 0x1a1a, 0x2a2a, 0x3a3a, 0x4a4a, 0x5a5a, 0x6a6a, 0x7a7a,
          0x8a8a, 0x9a9a, 0xaaaa, 0xbaba, 0xcaca, 0xdada, 0xeaea, 0xfafa}


def parse(hello: bytes):
    hs = hello[5:5 + struct.unpack(">H", hello[3:5])[0]]
    ver = struct.unpack(">H", hs[4:6])[0]
    p = 6 + 32
    sid = hs[p]
    p += 1 + sid
    cs_len = struct.unpack(">H", hs[p:p + 2])[0]
    p += 2
    ciphers = [struct.unpack(">H", hs[p + i:p + i + 2])[0] for i in range(0, cs_len, 2)]
    p += cs_len
    comp_len = hs[p]
    p += 1 + comp_len
    ext_len = struct.unpack(">H", hs[p:p + 2])[0]
    p += 2
    exts, curves, alpn = [], [], ""
    end = p + ext_len
    while p < end:
        et = struct.unpack(">H", hs[p:p + 2])[0]
        el = struct.unpack(">H", hs[p + 2:p + 4])[0]
        body = hs[p + 4:p + 4 + el]
        if et not in GREASE:
            exts.append(et)
        if et == 0x000a and body:
            cl = struct.unpack(">H", body[0:2])[0]
            curves = [struct.unpack(">H", body[2 + i:4 + i])[0] for i in range(0, cl, 2)]
        if et == 0x0010 and body:
            alpn = body[3:].decode("latin1", "ignore")
        p += 4 + el
    cs = [c for c in ciphers if c not in GREASE]
    ja3 = f"{ver},{'-'.join(map(str, cs))},{'-'.join(map(str, exts))},{'-'.join(map(str, curves))},0"
    return ja3, alpn, ver


def main() -> None:
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(1)
    result = {}

    def client():
        try:
            s = tls_client.Session(client_identifier="okhttp4_android_13", random_tls_extension_order=False)
            s.proxies = {"http": f"http://127.0.0.1:{PORT}", "https": f"http://127.0.0.1:{PORT}"}
            s.get("https://api.sofascore.com/api/v1/event/15935071", timeout_seconds=6)
        except Exception:
            pass

    threading.Thread(target=client, daemon=True).start()
    conn, _ = srv.accept()
    # Leer el CONNECT del proxy y responder 200 antes del ClientHello.
    req = b""
    while b"\r\n\r\n" not in req:
        req += conn.recv(4096)
    conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    data = b""
    while len(data) < 5 or len(data) < 5 + struct.unpack(">H", data[3:5])[0]:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data += chunk
    conn.close()
    srv.close()
    ja3, alpn, ver = parse(data)
    print(f"JA3 : {ja3}")
    print(f"ALPN: {alpn}")
    print(f"VER : 0x{ver:04x}")
    open(__file__.replace("capture_ja3.py", "okhttp_ja3.txt"), "w").write(ja3)


if __name__ == "__main__":
    main()
