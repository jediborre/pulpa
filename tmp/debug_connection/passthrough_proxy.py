"""
tmp/debug_connection/passthrough_proxy.py
Proposito: Proxy TCP passthrough (sin MITM) que se configura como proxy del telefono
para capturar el ClientHello TLS real de la app SofaScore (SNI, ALPN, cipher suites,
extensiones y JA3). Como no descifra, la app sigue funcionando y Cloudflare ve su
huella nativa. Sirve para saber que fingerprint hay que replicar desde la PC.
Uso: python passthrough_proxy.py [port] [seconds]
"""
import socket
import ssl
import struct
import sys
import threading
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
SECONDS = int(sys.argv[2]) if len(sys.argv) > 2 else 45
OUT = Path(__file__).resolve().parent / "clienthello.bin"

GREASE = {0x0a0a, 0x1a1a, 0x2a2a, 0x3a3a, 0x4a4a, 0x5a5a, 0x6a6a, 0x7a7a,
          0x8a8a, 0x9a9a, 0xaaaa, 0xbaba, 0xcaca, 0xdada, 0xeaea, 0xfafa}


def parse_client_hello(data: bytes) -> str:
    try:
        # TLS record header: type(1) version(2) length(2)
        if data[0] != 0x16:
            return "no-handshake"
        rec_len = struct.unpack(">H", data[3:5])[0]
        hs = data[5:5 + rec_len]
        # handshake: type(1) len(3) version(2) random(32)
        if hs[0] != 0x01:
            return "no-clienthello"
        ver = struct.unpack(">H", hs[4:6])[0]
        p = 6 + 32
        sid_len = hs[p]
        p += 1 + sid_len
        cs_len = struct.unpack(">H", hs[p:p + 2])[0]
        p += 2
        ciphers = []
        for i in range(0, cs_len, 2):
            c = struct.unpack(">H", hs[p + i:p + i + 2])[0]
            if c not in GREASE:
                ciphers.append(c)
        p += cs_len
        comp_len = hs[p]
        p += 1 + comp_len
        ext_len = struct.unpack(">H", hs[p:p + 2])[0]
        p += 2
        exts = []
        sni = ""
        alpn = ""
        end = p + ext_len
        while p < end:
            et = struct.unpack(">H", hs[p:p + 2])[0]
            el = struct.unpack(">H", hs[p + 2:p + 4])[0]
            body = hs[p + 4:p + 4 + el]
            if et not in GREASE:
                exts.append(et)
            if et == 0x0000 and body:
                name_len = struct.unpack(">H", body[3:5])[0]
                sni = body[5:5 + name_len].decode("latin1", "ignore")
            if et == 0x0010 and body:
                alpn = body[3:].decode("latin1", "ignore")
            p += 4 + el
        ja3 = f"{ver},|{'-'.join(map(str, ciphers))}|{'-'.join(map(str, exts))}|0-11-10|0"
        return f"ver=0x{ver:04x} sni={sni} alpn={alpn}\n    JA3={ja3}\n    ciphers={ciphers}\n    exts={exts}"
    except Exception as exc:
        return f"parse-error {exc}"


def handle(client: socket.socket, addr) -> None:
    try:
        client.settimeout(10)
        req = b""
        while b"\r\n\r\n" not in req:
            chunk = client.recv(4096)
            if not chunk:
                return
            req += chunk
        line = req.split(b"\r\n", 1)[0].decode("latin1", "ignore")
        parts = line.split()
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            return
        host, port = parts[1].split(":")
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        # Read ClientHello from client
        hello = b""
        while len(hello) < 5 or len(hello) < 5 + struct.unpack(">H", hello[3:5])[0]:
            chunk = client.recv(4096)
            if not chunk:
                return
            hello += chunk
            if len(hello) > 8000:
                break
        print(f"\n=== ClientHello -> {host}:{port} desde {addr[0]} ({len(hello)} bytes) ===")
        print(parse_client_hello(hello))
        OUT.write_bytes(hello)
        # Tunnel to real host
        upstream = socket.create_connection((host, int(port)), timeout=10)
        upstream.sendall(hello)

        def pump(a, b):
            try:
                while True:
                    d = a.recv(65536)
                    if not d:
                        break
                    b.sendall(d)
            except Exception:
                pass
            finally:
                try:
                    b.shutdown(socket.SHUT_WR)
                except Exception:
                    pass

        t = threading.Thread(target=pump, args=(upstream, client), daemon=True)
        t.start()
        pump(client, upstream)
        upstream.close()
    except Exception as exc:
        print(f"[proxy] error {addr}: {exc}")
    finally:
        try:
            client.close()
        except Exception:
            pass


def main() -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", PORT))
    srv.listen(50)
    srv.settimeout(1.0)
    print(f"[proxy] escuchando en 0.0.0.0:{PORT} durante {SECONDS}s")
    end = time.time() + SECONDS
    while time.time() < end:
        try:
            c, a = srv.accept()
            threading.Thread(target=handle, args=(c, a), daemon=True).start()
        except socket.timeout:
            continue
        except KeyboardInterrupt:
            break
    srv.close()
    print("[proxy] fin")


if __name__ == "__main__":
    main()
