# =====================================================================
# IMÁGENES DE ESTADÍSTICAS (STATUS / RESUMEN DIARIO)
# =====================================================================
# Propósito:
#   Generar en memoria (sin escribir a disco) imágenes JPEG con las
#   estadísticas de la BD _v3, para acompañar los avisos del bot
#   interactivo (`/status`) y el resumen diario del daemon.
#
#   ┌──────────────────────────────────────────────────┐
#   │ STATUS BD (V3)                                   │
#   │ Mié 19 oct 2026 · octubre 2026                   │
#   │ HOY                                  ACIERTO 60%│
#   │ ┌──────────┐ ┌──────────┐ ┌──────┐ ┌───────────┐ │  KPI cards
#   │ │ GANADAS  │ │ PERDIDAS │ │ PUSH │ │PENDIENTES │ │
#   │ │   341    │ │   231    │ │  31  │ │    46     │ │
#   │ └──────────┘ └──────────┘ └──────┘ └───────────┘ │
#   │ m27_v5_3  BET 28 ganadas 16 perdidas 9 ...   64% │
#   │ ████████████████░░░░░░░░░▨▨▨▨▨                   │  barra apilada
#   │ ─────────────────────────────────────────────────│
#   │ MES                                  ACIERTO 58%│
#   │ ...                                              │
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
    _text_len,
)

PAD = 32
KPI_H = 108
BAR_H = 12
ROW_H = 58

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
    total = int(win or 0) + int(loss or 0)
    if total <= 0:
        return "—"
    return f"{int(round(int(win) * 100 / total))}%"


def _draw_kpis(draw, x: float, y: float, kpis, width: float) -> float:
    """Fila de tarjetas KPI. Devuelve la Y inferior."""
    n = max(1, len(kpis))
    gap = 16
    card_w = (width - gap * (n - 1)) / n
    for index, (label, value, color) in enumerate(kpis):
        cx = x + index * (card_w + gap)
        draw.rounded_rectangle(
            [cx, y, cx + card_w, y + KPI_H], radius=14, fill=COLOR_PANEL_ALT, outline=COLOR_BORDER, width=2
        )
        draw.text((cx + 16, y + 14), label.upper(), font=_font(19, True), fill=COLOR_MUTED)
        draw.text((cx + 16, y + 40), _fmt_count(value), font=_font(40, True), fill=color)
    return y + KPI_H


def _draw_bar(draw, x: float, y: float, width: float, win: int, loss: int, pending: int = 0) -> None:
    """Barra apilada: verde = ganadas, rojo = perdidas, gris = pendientes."""
    draw.rounded_rectangle([x, y, x + width, y + BAR_H], radius=BAR_H // 2, fill=COLOR_PANEL)
    total = int(win or 0) + int(loss or 0) + int(pending or 0)
    if total <= 0:
        return
    cursor = x
    for value, color in ((win, SS_CONF_BET), (loss, SS_CONF_FAR), (pending, COLOR_BORDER)):
        if not value:
            continue
        segment = max(width * (value / total), 6.0)
        if cursor + segment > x + width:
            segment = (x + width) - cursor
        if segment <= 0:
            break
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

    name_font = _font(25, True)
    draw.text((cx, y), label, font=name_font, fill=label_color)
    cx += _text_len(draw, label, name_font) + 20

    parts = [
        (f"BET {_fmt_count((win or 0) + (loss or 0) + (pending or 0))}", COLOR_MUTED),
        (f"ganadas {_fmt_count(win)}", SS_CONF_BET),
        (f"perdidas {_fmt_count(loss)}", SS_CONF_FAR),
    ]
    if pending:
        parts.append((f"pendientes {_fmt_count(pending)}", COLOR_MUTED))

    for index, (text, color) in enumerate(parts):
        if index:
            draw.text((cx, y + 3), "·", font=_font(21), fill=COLOR_BORDER)
            cx += _text_len(draw, "·", _font(21)) + 8
        font = _font(21, True)
        draw.text((cx, y + 3), text, font=font, fill=color)
        cx += _text_len(draw, text, font) + 8

    draw.text((x + width, y), right_text, font=_font(25, True), fill=COLOR_TEXT, anchor="ra")
    _draw_bar(draw, x, y + 36, width, win, loss, pending)
    return y + ROW_H


def _section_height(section: dict) -> int:
    height = 44  # etiqueta de la sección + acierto
    if section.get("subtitle"):
        height += 30
    if section.get("kpis"):
        height += KPI_H + 20
    if section.get("models"):
        height += ROW_H * len(section["models"])
    if section.get("categories"):
        height += 28 + ROW_H * len(section["categories"])
    return height + 20


def build_stats_image(*, title: str, sections: list[dict], subtitle: str = "", footer: str = "") -> bytes | None:
    """
    Construye la imagen de estadísticas con una o varias secciones.

    Cada sección es un dict con:
        label (str), subtitle (str, opcional), accuracy (str, opcional),
        kpis: [(etiqueta, valor, color)],
        models: [{"name", "win", "loss", "pending"}],
        categories: [{"label", "win", "loss", "dots": [color, ...]}]
    """
    try:
        sections = [s for s in (sections or []) if s]
        if not sections:
            return None

        height = PAD + 52
        if subtitle:
            height += 32
        height += sum(_section_height(s) for s in sections)
        if footer:
            height += 30
        height = int(height + PAD - 16)

        img = Image.new("RGB", (WIDTH, height), COLOR_BG)
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, WIDTH, 5], fill=COLOR_ACCENT)

        y = PAD
        draw.text((PAD, y), title.upper(), font=_font(34, True), fill=COLOR_ACCENT)
        y += 52
        if subtitle:
            draw.text((PAD, y), subtitle, font=_font(22), fill=COLOR_MUTED)
            y += 32

        inner_w = WIDTH - 2 * PAD
        for index, section in enumerate(sections):
            if index:
                draw.line([(PAD, y), (WIDTH - PAD, y)], fill=COLOR_BORDER, width=2)
                y += 16

            draw.text((PAD, y), str(section.get("label") or "").upper(), font=_font(26, True), fill=COLOR_ACCENT)
            accuracy = section.get("accuracy")
            if accuracy:
                draw.text(
                    (WIDTH - PAD, y + 2),
                    f"ACIERTO {accuracy}",
                    font=_font(24, True),
                    fill=SS_CONF_BET,
                    anchor="ra",
                )
            y += 38

            if section.get("subtitle"):
                draw.text((PAD, y), section["subtitle"], font=_font(21), fill=COLOR_MUTED)
                y += 30

            kpis = section.get("kpis") or []
            if kpis:
                _draw_kpis(draw, PAD, y, kpis, inner_w)
                y += KPI_H + 20

            models = section.get("models") or []
            for model in models:
                win, loss = int(model.get("win") or 0), int(model.get("loss") or 0)
                y = _draw_row(
                    draw,
                    PAD,
                    y,
                    inner_w,
                    str(model.get("name") or "?"),
                    COLOR_TEXT,
                    win,
                    loss,
                    int(model.get("pending") or 0),
                    _acc(win, loss),
                )
            if models:
                y += 8

            categories = section.get("categories") or []
            if categories:
                draw.text((PAD, y), "POR CATEGORÍA", font=_font(21, True), fill=COLOR_ACCENT)
                y += 28
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

        if footer:
            draw.text((PAD, int(height - PAD + 2)), footer, font=_font(19), fill=COLOR_MUTED)

        out = io.BytesIO()
        img.save(out, format="JPEG", quality=JPEG_QUALITY)
        return out.getvalue()
    except Exception as e:  # noqa: BLE001
        log_warning("TELEGRAM", f"[IMG] No se pudo generar la imagen de estadísticas ({type(e).__name__}: {e})")
        return None


# ─────────────────────────────────────────────────────────────────────
# Envoltorios por caso de uso
# ─────────────────────────────────────────────────────────────────────

def _model_rate(model: dict) -> tuple[float, int]:
    """Clave de orden por efectividad: % de acierto (desc) y resueltas (desc)."""
    win = int(model.get("win") or 0)
    loss = int(model.get("loss") or 0)
    resolved = win + loss
    rate = (win / resolved) if resolved else -1.0
    return (rate, resolved)


def _section_from_stats(label: str, stats: dict | None, active_models: list[str] | None = None) -> dict:
    """Sección (KPIs + modelos) a partir del dict de `fetch_v3_stats()`."""
    total = (stats or {}).get("total") or {}
    per_model = (stats or {}).get("per_model") or {}
    bets = int(total.get("bets") or 0)
    signals = int(total.get("signals") or 0)
    nobet = max(0, signals - bets)

    order = list(per_model.keys()) or list(active_models or [])
    models = [
        {
            "name": name,
            "win": (per_model.get(name) or {}).get("win", 0),
            "loss": (per_model.get(name) or {}).get("loss", 0),
            "pending": (per_model.get(name) or {}).get("pending", 0),
        }
        for name in order
    ]
    # Orden por efectividad: primero los de mejor % de acierto, al final los peores
    # (los que aún no tienen apuestas resueltas quedan al final).
    models.sort(key=_model_rate, reverse=True)

    subtitle = f"Señales {_fmt_count(signals)}  ·  Apuestas {_fmt_count(bets)}  ·  No bet {_fmt_count(nobet)}"
    void = int(total.get("void") or 0)
    if void:
        subtitle += f"  ·  Anuladas {_fmt_count(void)}"

    return {
        "label": label,
        "subtitle": subtitle,
        "accuracy": _acc(total.get("win") or 0, total.get("loss") or 0),
        "kpis": [
            ("Ganadas", total.get("win"), SS_CONF_BET),
            ("Perdidas", total.get("loss"), SS_CONF_FAR),
            ("Push", total.get("push"), COLOR_MUTED),
            ("Pendientes", total.get("pending"), COLOR_MUTED),
        ],
        "models": models,
    }


def status_image(
    day_stats: dict | None,
    month_stats: dict | None = None,
    active_models: list[str] | None = None,
    *,
    subtitle: str = "",
    footer: str = "",
) -> bytes | None:
    """Imagen del `/status`: resumen de HOY y del MES (tablas _v3)."""
    sections = []
    if day_stats:
        sections.append(_section_from_stats("Hoy", day_stats, active_models))
    if month_stats:
        sections.append(_section_from_stats("Mes", month_stats, active_models))
    if not sections:
        return None
    return build_stats_image(
        title="Status BD (v3)",
        subtitle=subtitle,
        sections=sections,
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

    models = [
        {
            "name": name,
            "win": (data or {}).get("win", 0),
            "loss": (data or {}).get("loss", 0),
            "pending": (data or {}).get("pending", 0),
        }
        for name, data in per_model.items()
    ]

    resolved_categories = []
    for cat in categories or []:
        key = str(cat.get("key") or cat.get("label") or "?")
        label, dots = CATEGORY_STYLE.get(key, (key, [COLOR_MUTED]))
        resolved_categories.append(
            {"label": cat.get("label") or label, "win": cat.get("win", 0), "loss": cat.get("loss", 0), "dots": dots}
        )

    sub = f"Apuestas {_fmt_count(total.get('bets'))}  ·  No bet {_fmt_count(total.get('nobet'))}"
    void = int(total.get("void") or 0)
    if void:
        sub += f"  ·  Anuladas {_fmt_count(void)}"

    section = {
        "label": day or "Hoy",
        "subtitle": sub,
        "accuracy": _acc(total.get("win") or 0, total.get("loss") or 0),
        "kpis": [
            ("Ganadas", total.get("win"), SS_CONF_BET),
            ("Perdidas", total.get("loss"), SS_CONF_FAR),
            ("Push", total.get("push"), COLOR_MUTED),
            ("Pendientes", total.get("pending"), COLOR_MUTED),
        ],
        "models": models,
        "categories": resolved_categories,
    }
    return build_stats_image(title="Resumen del día", sections=[section], footer=footer)
