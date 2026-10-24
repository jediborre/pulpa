# Diagnóstico temporal: ¿la subida a Telegram falla por tamaño o por la red?
# Se prueban imágenes de ~1 KB y ~90 KB contra chat_id=1 (error esperado, sin
# entregar nada) con timeout amplio, midiendo tiempos.
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import requests  # noqa: E402
from PIL import Image  # noqa: E402

from monitor_v3.config.constants import TELEGRAM_BOT_TOKEN  # noqa: E402


def make_jpeg(size_px: int, color=(200, 80, 40)) -> bytes:
    img = Image.new("RGB", (size_px, size_px), color)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    return out.getvalue()


for label, blob in (("tiny ~1KB", make_jpeg(40)), ("grande ~90KB", make_jpeg(400))):
    t0 = time.time()
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
            data={"chat_id": "1", "caption": label},
            files={"photo": ("t.jpg", blob, "image/jpeg")},
            timeout=150,
        )
        print(f"{label} ({len(blob)} B): HTTP {r.status_code} en {time.time()-t0:.1f}s -> {r.text[:80]}")
    except Exception as e:
        print(f"{label} ({len(blob)} B): ERROR en {time.time()-t0:.1f}s -> {type(e).__name__}: {e}")
