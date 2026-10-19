# =====================================================================
# IMÁGENES DE ESTADÍSTICAS (STATUS / RESUMEN DIARIO)
# =====================================================================
# Propósito:
#   Generar en memoria (sin escribir a disco) imágenes JPEG con las
#   estadísticas de la BD _v3, para acompañar los avisos del bot
#   interactivo (`/status`) y el resumen diario del daemon.
#
#   ┌──────────────────────────────────────────────────┐
#   │ STATUS BD (V3)                      ACIERTO 56.7%│
#   │ Señales 12 480 · Apuestas 1 234 · No bet 11 246  │
#   │ ┌────────┐ ┌────────┐ ┌──────┐ ┌───────────┐     │
#   │ │GANADAS │ │PERDIDAS│ │ PUSH │ │PENDIENTES │     │  KPI cards
#   │ │  700   │ │  480   │ │  54  │ │     0     │     │
#   │ └────────┘ └────────┘ └──────┘ └───────────┘     │
#   │ POR MODELO                                       │
#   │ m27_v5_3   BET 320   ganadas 180  perdidas 140   │
#   │ ████████████████████░░░░░░░░░░░░░  56.2%         │  barra apilada
#   └──────────────────────────────────────────────────┘
#
# Paleta y utilidades: se reutilizan las de `bet_image.py` (mismo estilo
# oscuro que la tarjeta de apuesta) para que todas las imágenes del
# monitor se vean consistentes.
#
# Reglas:
#   - Nunca rompe el flujo: ante cualquier fallo devuelve None y el aviso
#     cae a texto plano.
#   - Sin emojis en el texto dibujado (las fuentes de texto no traen
#     glifos emoji); las categorías se marcan con círculos de color.
# =====================================================================

from __future__ import annotations

import io

from PIL import Image, ImageDraw

from monitor_v3.utils.logger import log_warning

from monitor_v3.notifications.bet_image import (
    COLOR_ACCENT,
    COLOR_BG,
    COLOR_BORDER,
    COLOR_MUTED,
    COLOR_PANEL,
    COLOR_PANEL_ALT,
    COLOR_TEXT,
    JPEG_QUALITY,
    SS_CONF_BET,
    SS_CONF_FAR,
    SS_CONF_NEAR,
    WIDTH,
    _bold,
    _font,
    _font_size,
    _text_len,
)

PAD = 32
KPI_H = 118
BAR_H = 12
ROW_H = 60

# Estilo de las categorías de apuesta (mismo criterio que los avisos):
# clave del clasificador → (etiqueta, colores de los puntos)
CATEGORY_STYLE: dict[str, tuple[str, list[tuple]]] = {
    "🟢": ("ALTA", [SS_CONF_BET]),
    "🟡": ("BAJA", [SS_CONF_NEAR]),
    "⚪️": ("TARDÍA", [COLOR_TEXT]),
    "🟡⚪️": ("TARDÍA BAJA", [SS_CONF_NEAR, COLOR_TEXT]),
    "🔴": ("NO BET", [COLOR_BORDER]),
}


def _fmt_count(value) -> str:
    """Entero con separador de miles (12 480)."""
    try:
        return f"{int(value):,}".replace(",", " ")
    except (TypeError, ValueError):
        return "—"


def _acc(win: int, loss: int) -> str:
    total = int(win) + int(loss)
    if total <= 0:
        return "—"
    return f"{int(round(win * 100 / total))}%"


def _draw_kpis(draw, x: float, y: float, kpis: list[tuple[str, object, tuple]], width: float) -> float:
    """Fila de tarjetas KPI. Devuelve la Y inferior."""
    n = max(1, len(kpis))
    gap = 16
    card_w = (width - gap * (n - 1)) / n
    for index, (label, value, color) in enumerate(kpis):
        cx = x + index * (card_w + gap)
        draw.rounded_rectangle(
            [cx, y, cx + card_w, y + KPI_H], radius=14, fill=COLOR_PANEL_ALT, outline=COLOR_BORDER, width=2
        )
        draw.text((cx + 18, y + 16), label.upper(), font=_font(20, True), fill=COLOR_MUTED)
        draw.text((cx + 18, y + 44), _fmt_count(value), font=_font(44, True), fill=color)
    return y + KPI_H


def _draw_bar(draw, x: float, y: float, width: float, win: int, loss: int, pending: int = 0) -> None:
    """Barra apilada: verde = ganadas, rojo = perdidas, gris = pendientes."""
    draw.rounded_rectangle([x, y, x + width, y + BAR_H], radius=BAR_H // 2, fill=COLOR_PANEL)
    total = int(win) + int(loss) + int(pending)
    if total <= 0:
        return
    cursor = x
    for value, color in ((win, SS_CONF_BET), (loss, SS_CONF_FAR), (pending, COLOR_BORDER)):
        if not value:
            continue
        segment = width * (value / total)
        segment = max(segment, 6.0)
        if cursor + segment > x + width:
            segment = (x + width) - cursor
        draw.rounded_rectangle([cursor, y, cursor + segment, y + BAR_H], radius=BAR_H // 2, fill=color)
        cursor += segment


def _draw_row(
    draw,
    x: float,
    y: float,
    width: float,
    label: str,
    label_color: tuple,
    win: int,
    loss: int,
    pending: int,
    right_text: str,
    dots: list[tuple] | None = None,
) -> float:
    """Una fila de estadística (modelo o categoría) con su barra. Devuelve la Y final."""
    cx = x
    if dots:
        for color in dots:
            r = 7
            draw.ellipse([cx, y + 8, cx + 2 * r, y + 8 + 2 * r], fill=color)
            cx += 2 * r + 6
        cx += 6

    draw.text((cx, y), label, font=_font(26, True), fill=label_color)
    cx += _text_len(draw, label, _font(26, True)) + 22

    parts = [
        (f"BET {_fmt_count(win + loss + pending)}", COLOR_MUTED),
        (f"ganadas {_fmt_count(win)}", SS_CONF_BET),
        (f"perdidas {_fmt_count(loss)}", SS_CONF_FAR),
    ]
    if pending:
        parts.append((f"pendientes {_fmt_count(pending)}", COLOR_MUTED))

    for index, (text, color) in enumerate(parts):
        if index:
            draw.text((cx, y + 4), "·", font=_font(22), fill=COLOR_BORDER)
            cx += _text_len(draw, "·", _font(22)) + 10
        font = _font(22, True)
        draw.text((cx, y + 4), text, font=font, fill=color)
        cx += _text_len(draw, text, font) + 10

    right_font = _font(26, True)
    draw.text((x + width, y), right_text, font=right_font, fill=COLOR_TEXT, anchor="ra")

    _draw_bar(draw, x, y + 38, width, win, loss, pending)
    return y + ROW_H


def build_stats_image(
    *,
    title: str,
    subtitle: str = "",
    accuracy: str = "",
    accuracy_label: str = "ACIERTO",
    kpis: list[tuple[str, object, tuple]] | None = None,
    models: list[dict] | None = None,
    categories: list[dict] | None = None,
    footer: str = "",
) -> bytes | None:
    """
    Construye la imagen de estadísticas y la devuelve como bytes JPEG.

    - `kpis`: [(etiqueta, valor, color)]
    - `models`: [{"name", "win", "loss", "pending", "label_color"?}]
    - `categories`: [{"label", "win", "loss", "dots": [color, ...]}]
    """
    try:
        models = models or []
        categories = categories or []
        kpis = kpis or []

        # Altura dinámica
        y_cursor = PAD + 48 + 40
        if kpis:
            y_cursor += KPI_H + 28
        if models:
            y_cursor += 34 + ROW_H * len(models) + 10
        if categories:
            y_cursor += 34 + ROW_H * len(categories) + 10
        if footer:
            y_cursor += 30
        height = int(y_cursor + PAD - 20)

        img = Image.new("RGB", (WIDTH, height), COLOR_BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, WIDTH, 5], fill=COLOR_ACCENT)

        # Cabecera
        y = PAD
        draw.text((PAD, y), title.upper(), font=_font(34, True), fill=COLOR_ACCENT)
        if accuracy:
            draw.text((WIDTH - PAD, y + 6), f"{accuracy_label} {accuracy}", font=_font(30, True), fill=SS_CONF_BET, anchor="ra")
        y += 48
        if subtitle:
            draw.text((PAD, y), subtitle, font=_font(22), fill=COLOR_MUTED)
        y += 40

        inner_w = WIDTH - 2 * PAD

        if kpis:
            _draw_kpis(draw, PAD, y, kpis, inner_w)
            y += KPI_H + 28

        if models:
            draw.text((PAD, y), "POR MODELO", font=_font(22, True), fill=COLOR_ACCENT)
            y += 34
            for model in models:
                win, loss = int(model.get("win") or 0), int(model.get("loss") or 0)
                y = _draw_row(
                    draw,
                    PAD,
                    y,
                    inner_w,
                    str(model.get("name") or "?"),
                    model.get("label_color") or COLOR_TEXT,
                    win,
                    loss,
                    int(model.get("pending") or 0),
                    _acc(win, loss),
                )
            y += 10

        if categories:
            draw.text((PAD, y), "POR CATEGORÍA", font=_font(22, True), fill=COLOR_ACCENT)
            y += 34
            for cat in categories:
                win, loss = int(cat.get("win") or 0), int(cat.get("loss") or 0)
                y = _draw_row(
                    draw,
                    PAD,
                    y,
                    inner_w,
                    str(cat.get("label") or "?"),
                    COLOR_TEXT,
                    win,
                    loss,
                    0,
                    _acc(win, loss),
                    dots=cat.get("dots"),
                )
            y += 10

        if footer:
            draw.text((PAD, y), footer, font=_font(20), fill=COLOR_MUTED)

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001
        log_warning("TELEGRAM", f"[IMG] No se pudo generar la imagen de estadísticas ({type(e).__name__}: {e})")
        return None


# ─────────────────────────────────────────────────────────────────────
# Envoltorios por caso de uso
# ─────────────────────────────────────────────────────────────────────

def status_image(stats: dict, active_models: list[str] | None = None, *, footer: str = "") -> bytes | None:
    """
    Imagen para el `/status` del bot interactivo, a partir del dict que
    devuelve `fetch_v3_stats()` de `bot_runner`:
    {"total": {signals, bets, win, loss, pending}, "per_model": {model: {...}}}
    """
    total = (stats or {}).get("total") or {}
    per_model = (stats or {}).get("per_model") or {}
    if not total and not per_model:
        return None

    bets = int(total.get("bets") or 0)
    signals = int(total.get("signals") or 0)
    nobet = max(0, signals - bets)

    kpis = [
        ("Ganadas", total.get("win"), SS_CONF_BET),
        ("Perdidas", total.get("loss"), SS_CONF_FAR),
        ("Push", total.get("push"), COLOR_MUTED),
        ("Pendientes", total.get("pending"), COLOR_MUTED),
    ]

    order = list(per_model.keys()) or list(active_models or [])
    models = []
    for name in order:
        data = per_model.get(name) or {}
        models.append(
            {
                "name": name,
                "win": data.get("win", 0),
                "loss": data.get("loss", 0),
                "pending": data.get("pending", 0),
            }
        )

    subtitle = f"Señales {_fmt_count(signals)}  ·  Apuestas {_fmt_count(bets)}  ·  No bet {_fmt_count(nobet)}"
    return build_stats_image(
        title="Status BD (v3)",
        subtitle=subtitle,
        accuracy=_acc(total.get("win") or 0, total.get("loss") or 0),
        kpis=kpis,
        models=models,
        footer=footer,
    )


def summary_image(
    total: dict,
    per_model: dict,
    *,
    day: str = "",
    categories: list[dict] | None = None,
    footer: str = "",
) -> bytes | None:
    """Imagen del resumen diario (mismas métricas que el texto del resumen)."""
    total = total or {}
    per_model = per_model or {}
    if not total:
        return None

    kpis = [
        ("Ganadas", total.get("win"), SS_CONF_BET),
        ("Perdidas", total.get("loss"), SS_CONF_FAR),
        ("Push", total.get("push"), COLOR_MUTED),
        ("Pendientes", total.get("pending"), COLOR_MUTED),
    ]

    models = [
        {
            "name": name,
            "win": (data or {}).get("win", 0),
            "loss": (data or {}).get("loss", 0),
            "pending": (data or {}).get("pending", 0),
        }
        for name, data in per_model.items()
    ]

    # Resolver el estilo de cada categoría (etiqueta + puntos de color)
    resolved_categories = []
    for cat in categories or []:
        key = str(cat.get("key") or cat.get("label") or "?")
        label, dots = CATEGORY_STYLE.get(key, (key, [COLOR_MUTED]))
        resolved_categories.append(
            {"label": cat.get("label") or label, "win": cat.get("win", 0), "loss": cat.get("loss", 0), "dots": dots}
        )

    bets = int(total.get("bets") or 0)
    nobet = int(total.get("nobet") or 0)
    subtitle = f"Apuestas {_fmt_count(bets)}  ·  No bet {_fmt_count(nobet)}"
    if day:
        subtitle = f"{day}  ·  {subtitle}"

    return build_stats_image(
        title="Resumen del día",
        subtitle=subtitle,
        accuracy=_acc(total.get("win") or 0, total.get("loss") or 0),
        accuracy_label="ACIERTO",
        kpis=kpis,
        models=models,
        categories=resolved_categories,
        footer=footer,
    )

