"""
tmp/debug_connection/read_cookies.py
Proposito: Leer la base SQLite de cookies del WebView extraida del telefono e imprimir
las cookies de sofascore/cloudflare (name, length(value), length(encrypted_value)).
Se usa para verificar si existe 'cf_clearance' y si su valor esta cifrado por Android.
"""
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DB = Path(__file__).resolve().parent / "mobile_cookies" / "Cookies"


def main() -> None:
    con = sqlite3.connect(DB)
    tables = [r[0] for r in con.execute("select name from sqlite_master where type='table'")]
    print("tablas:", tables)
    rows = con.execute(
        "select host_key, name, value, length(encrypted_value), is_secure, is_httponly "
        "from cookies order by host_key, name"
    ).fetchall()
    print(f"cookies: {len(rows)}")
    for host, name, value, enc_len, secure, httponly in rows:
        val = (value[:40] + "...") if value else ""
        flag = ""
        if "clearance" in name.lower() or "cf_" in name.lower():
            flag = "  <<< CLOUDFLARE"
        print(f"  {host:<28} {name:<22} vlen={len(value or ''):<4} enclen={enc_len:<4} sec={secure} http={httponly} val={val}{flag}")


if __name__ == "__main__":
    main()
