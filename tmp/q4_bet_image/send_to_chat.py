# =====================================================================
# ENVÍO DE PRUEBA DE LA IMAGEN Q4 A UN CHAT CONCRETO
# ---------------------------------------------------------------------
# QUÉ HACE:
#   1. Lee los suscriptores de `monitor_subscribers` (settings en matches.db).
#   2. Resuelve el NOMBRE de cada chat_id con la propia Bot API (`getChat`),
#      para poder identificar al destinatario sin adivinar.
#   3. Envía una imagen de prueba usando el MISMO camino de producción
#      (`telegram_bot._send_photo_request`, multipart/form-data), lo que
#      valida de verdad el upload de fotos de la alerta de apuesta Q4.
#
# USO:
#   # Solo listar quién es quién (no envía nada):
#   .venv\Scripts\python.exe tmp\q4_bet_image\send_to_chat.py --dry-run
#   # Enviar al chat cuyo nombre contenga "Borrego":
#   .venv\Scripts\python.exe tmp\q4_bet_image\send_to_chat.py --name Borrego
#   # Enviar un archivo concreto a un chat_id explícito:
#   .venv\Scripts\python.exe tmp\q4_bet_image\send_to_chat.py --chat-id 123456789 --file ruta.jpg
# =====================================================================

import argparse
import asyncio
import json
import sys
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from monitor_v3.config.constants import TELEGRAM_BOT_TOKEN  # noqa: E402
from monitor_v3.notifications.telegram_bot import (  # noqa: E402
    _get_subscribers_from_db,
    _send_photo_request,
)

OUT_DIR = Path(__file__).resolve().parent


def _get_chat_name(chat_id: int) -> str:
    """Resuelve el nombre visible del chat con la Bot API (getChat)."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getChat?chat_id={chat_id}"
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        return f"<error: {e}>"
    if not data.get("ok"):
        return f"<error: {data.get('description')}>"
    result = data.get("result") or {}
    parts = [
        result.get("first_name") or "",
        result.get("last_name") or "",
        result.get("title") or "",
    ]
    name = " ".join(p for p in parts if p).strip()
    if result.get("username"):
        name = f"{name} (@{result['username']})" if name else f"@{result['username']}"
    return name or "<sin nombre>"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="Borrego", help="fragmento de nombre del destinatario (case-insensitive)")
    ap.add_argument("--chat-id", type=int, help="enviar a este chat_id en vez de buscar por nombre")
    ap.add_argument("--file", help="ruta del JPEG a enviar (default: la muestra más reciente en tmp/q4_bet_image)")
    ap.add_argument("--dry-run", action="store_true", help="solo listar suscriptores y nombres")
    args = ap.parse_args()

    if not TELEGRAM_BOT_TOKEN:
        print("❌ Falta TELEGRAM_BOT_TOKEN (.env)")
        return

    subs = _get_subscribers_from_db()
    if not subs:
        print("❌ Sin suscriptores en monitor_subscribers")
        return

    print("Suscriptores y nombres resueltos vía getChat:")
    resueltos: dict[int, str] = {}
    for cid in subs:
        name = _get_chat_name(int(cid))
        resueltos[int(cid)] = name
        print(f"  {cid}: {name}")

    if args.dry_run:
        return

    if args.chat_id:
        target = int(args.chat_id)
    else:
        matches = [cid for cid, name in resueltos.items() if args.name.lower() in name.lower()]
        if not matches:
            print(f"❌ Ningún suscriptor coincide con '{args.name}'. Usa --dry-run para revisar o --chat-id.")
            return
        if len(matches) > 1:
            print(f"⚠️ Varios coinciden ({matches}); usa --chat-id para desambiguar.")
            return
        target = matches[0]

    if args.file:
        img_path = Path(args.file)
    else:
        candidatos = sorted(OUT_DIR.glob("q4_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidatos:
            print("❌ No hay muestras en tmp/q4_bet_image; corre antes render_samples.py")
            return
        img_path = candidatos[0]

    photo = img_path.read_bytes()
    caption = (
        "🧪 <b>PRUEBA</b> — imagen fusionada de apuesta Q4 (tarjeta + momentum)\n"
        "Generada por monitor_v3. En producción llega automáticamente con cada apuesta Q4."
    )
    print(f"\nEnviando {img_path.name} ({len(photo)} bytes) a {target} ({resueltos.get(target, '?')})...")
    res = await asyncio.to_thread(_send_photo_request, target, photo, caption, None)
    if res.get("ok"):
        msg = res.get("result") or {}
        print(f"✅ Enviado. message_id={msg.get('message_id')}")
    else:
        print(f"❌ Falló: {res.get('description')}")


if __name__ == "__main__":
    asyncio.run(main())
