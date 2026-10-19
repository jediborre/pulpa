# =====================================================================
# IMAGEN FUSIONADA DE APUESTA Q4 (TARJETA + GRÁFICA DE MOMENTUM)
# =====================================================================
# Propósito:
#   Generar en memoria (sin escribir a disco) un único JPEG vertical que
#   FUSIONA la tarjeta de la apuesta Q4 con la gráfica de momentum/presión
#   de SofaScore del mismo partido, para adjuntarla a la alerta de Telegram.
#
#   ┌───────────────────────────────┐
#   │ PAÍS | LIGA                   │  tarjeta de apuesta
#   │ LOCAL vs VISITANTE            │  (paleta oscura SofaScore)
#   │ 18:30 · Q4 · MIN~30 · 78-71   │
#   │ Q1 20-18 | Q2 22-21 | ...     │
#   │ ┃ APUESTA Q4                  │
#   │ ┃ m27_v3 72% → LOCAL 1.85     │
#   │ ┃ $500 -> $925.00 (+$425.00)  │
#   ├───────────────────────────────┤
#   │ gráfica de momentum (Q1..Q4)  │  reutiliza match/ml_tools.plot_graph
#   └───────────────────────────────┘
#
# Paleta: idéntica a la que ya usa `match/ml_tools.py::plot_graph` (el bot
#   interactivo de monitor_v1), para que tarjeta y gráfica se vean fusionadas:
#   fondo #0C141B, panel #111A22, banda alterna #132031, borde #2B3C4F,
#   texto #E2E8F0, texto atenuado #8FA0B4, local #4CD05E, visitante #7882EB,
#   acento #4FD3FF, alerta/tardía #FF4D57.
#
# Reglas:
#   - Nunca rompe el flujo: ante cualquier fallo devuelve None y el aviso de
#     Telegram cae a texto plano.
#   - Sin emojis en el texto dibujado (las fuentes de sistema no traen glifos
#     emoji); el lado se rotula con texto LOCAL / VISITANTE.
#   - La gráfica se renderiza con matplotlib en un hilo aparte (ver
#     telegram_bot) y protegida por candado: Agg no es thread-safe.
# =====================================================================

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone

from PIL import Image, ImageDraw, ImageFont

from monitor_v3.config.constants import UTC_OFFSET_HOURS
from monitor_v3.utils.logger import log_debug, log_warning

# ── Paleta (espejo de match/ml_tools.py::plot_graph) ──────────────────
COLOR_BG = (12, 20, 27)        # #0C141B
COLOR_PANEL = (17, 26, 34)     # #111A22
COLOR_PANEL_ALT = (19, 32, 49)  # #132031
COLOR_BORDER = (43, 60, 79)    # #2B3C4F
COLOR_TEXT = (226, 232, 240)   # #E2E8F0
COLOR_MUTED = (143, 160, 180)  # #8FA0B4
COLOR_HOME = (76, 208, 94)     # #4CD05E  (local / presión positiva)
COLOR_AWAY = (120, 130, 235)   # #7882EB  (visitante / presión negativa)
COLOR_ACCENT = (79, 211, 255)  # #4FD3FF  (acento / línea de tiempo)
COLOR_ALERT = (255, 77, 87)    # #FF4D57  (tardía / alerta)

# Semáforo de confianza para modelos que NO apostaron
SS_CONF_BET = (63, 203, 92)    # #3FCB5C  el modelo apostó → verde
SS_CONF_NEAR = (255, 210, 74)  # #FFD24A  cerca del umbral → amarillo
SS_CONF_FAR = (255, 77, 87)    # #FF4D57  lejísimos del umbral → rojo
SS_BET = (63, 203, 92)         # palabra BET en verde
SS_NO_BET = (255, 77, 87)      # palabra NO BET en rojo

# Layout
WIDTH = 1080
PAD = 32
STAKE = 500.0                  # fija de ejemplo para la proyección de ganancia
JPEG_QUALITY = 92

# Gráfica de momentum (estilo oficial de SofaScore modo dark)
CHART_HEIGHT = 320             # alto final de la gráfica dentro de la imagen
CHART_TOP_STRIP = 60           # franja superior: etiquetas Q1..Q4 + cuarto en vivo en rojo
CHART_BOTTOM_LABELS = 32       # franja inferior para "18:00" / "FT"
CHART_SS = 2                   # supersampling (se dibuja al doble y se reduce: bordes suaves)

# Colores de la gráfica estilo SofaScore
SS_BG = (12, 20, 27)           # #0C141B
SS_ZONE_HOME = (22, 50, 38)    # tinte verde de la mitad local
SS_ZONE_AWAY = (31, 40, 64)    # tinte azul de la mitad visitante
SS_FILL_HOME = (63, 203, 92)   # #3FCB5C relleno de presión local
SS_FILL_AWAY = (140, 156, 247)  # #8C9CF7 relleno de presión visitante
SS_BASELINE = (208, 214, 222)
SS_SEPARATOR = (0, 0, 0)       # separadores de cuarto en negro (como la web oficial)
SS_Y_LABEL = (76, 208, 94)     # etiquetas 11/0/11 en verde
SS_Y_ZERO = (200, 208, 218)
SS_X_LABEL = (76, 208, 94)     # hora de inicio en verde
SS_X_FT = (63, 203, 92)        # "FT" del eje destacado en verde
SS_LIVE = (255, 77, 87)        # marcador de tiempo real y cuarto en curso
SS_Q_LABEL = (79, 211, 255)    # etiquetas Q1..Q4 sobre la gráfica (destacadas)
SS_SCORE_WIN = (63, 203, 92)   # marcador del ganador
SS_SCORE_LOSS = (255, 77, 87)  # marcador del perdedor
SS_FT = (63, 203, 92)          # etiqueta FT destacada en verde

# Fuentes: Segoe UI en Windows, con respaldo Arial/DejaVu y, en último caso, default.
_FONT_CANDIDATES_REGULAR = [
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]
_FONT_CANDIDATES_BOLD = [
    r"C:\Windows\Fonts\seguisb.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
# Fuente de emoji en color (casa = local, avión = visitante)
_FONT_CANDIDATES_EMOJI = [
    r"C:\Windows\Fonts\seguiemj.ttf",
    "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf",
]
EMOJI_HOME = "\U0001F3E0"      # 🏠
EMOJI_AWAY = "\u2708\ufe0f"    # ✈️
_font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}
_emoji_font_cache: dict[int, ImageFont.FreeTypeFont | None] = {}


# ─────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────

def _font_size(font) -> int:
    return int(getattr(font, "size", 16) or 16)


def _emoji_font(size: int):
    """Fuente de emoji en color, o None si el sistema no la tiene."""
    if size in _emoji_font_cache:
        return _emoji_font_cache[size]
    font = None
    for path in _FONT_CANDIDATES_EMOJI:
        try:
            font = ImageFont.truetype(path, size=size)
            break
        except Exception:
            continue
    _emoji_font_cache[size] = font
    return font


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    """Interpola entre dos colores (t=0 → c1, t=1 → c2)."""
    t = max(0.0, min(1.0, t))
    return tuple(int(round(a + (b - a) * t)) for a, b in zip(c1, c2))


def _model_threshold(model: str, inference_json: dict | None, raw_payload: dict | None = None) -> float:
    """
    Umbral de confianza real del modelo (para el semáforo del porcentaje).
    Se lee de `raw_json` (min_conf_threshold / threshold_bet) y, si no está,
    se cae a los umbrales de referencia por familia de modelo.
    """
    raw: dict = {}
    for source in (inference_json, raw_payload):
        if isinstance(source, dict):
            maybe = source.get("raw_json") if isinstance(source.get("raw_json"), dict) else source
            if isinstance(maybe, dict) and maybe:
                raw = maybe
                break
    for key in ("min_conf_threshold", "threshold_bet", "bet_thr"):
        try:
            value = float(raw.get(key))
            if 0.0 < value < 1.0:
                return value
        except (TypeError, ValueError):
            continue
    thresholds = raw.get("thresholds")
    if isinstance(thresholds, dict):
        for key in ("bet", "min_conf"):
            try:
                value = float(thresholds.get(key))
                if 0.0 < value < 1.0:
                    return value
            except (TypeError, ValueError):
                continue

    name = str(model or "").lower()
    if "v6_" in name:
        return 0.18
    if "m27_v5" in name:
        return 0.75
    if "m27_v3" in name:
        return 0.68
    return 0.65


def _conf_color(confidence: float | None, threshold: float, is_bet: bool) -> tuple:
    """
    Color del porcentaje del modelo:
      - apostó → verde
      - no apostó → semáforo amarillo→rojo según lo cerca que estuvo del umbral
        (desde el 90% del umbral ya es amarillo; por debajo baja hacia rojo).
    """
    if is_bet:
        return SS_CONF_BET
    try:
        conf = float(confidence or 0.0)
        thr = float(threshold or 0.0)
    except (TypeError, ValueError):
        return SS_CONF_FAR
    if thr <= 0:
        return SS_CONF_FAR
    ratio = conf / thr
    return _lerp_color(SS_CONF_FAR, SS_CONF_NEAR, (ratio - 0.45) / 0.45)


def _panel_height(n_bets: int) -> int:
    """Altura total de la banda de apuesta (misma fórmula que el dibujo)."""
    pad, line1_h, line2_h, gap = 18, 40, 30, 10
    return pad + 40 + n_bets * (line1_h + line2_h + gap) + pad - gap


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    cached = _font_cache.get(key)
    if cached is not None:
        return cached
    for path in (_FONT_CANDIDATES_BOLD if bold else _FONT_CANDIDATES_REGULAR):
        try:
            font = ImageFont.truetype(path, size=size)
            _font_cache[key] = font
            return font
        except Exception:
            continue
    font = ImageFont.load_default()
    _font_cache[key] = font
    return font


def decimal_to_american(decimal: float | None) -> int | None:
    """Convierte una cuota decimal (1.85) a su equivalente americano (+120 / -118)."""
    if not decimal or decimal <= 1.0:
        return None
    dec = float(decimal)
    if dec >= 2.0:
        return int(round((dec - 1.0) * 100))
    return int(round(-100.0 / (dec - 1.0)))


def payout(decimal: float, stake: float = STAKE) -> tuple[float, float]:
    """Devuelve (ganancia, total a recibir) para una cuota decimal y una fija."""
    gain = stake * (float(decimal) - 1.0)
    return gain, stake + gain


def _text_len(draw: ImageDraw.ImageDraw, text: str, font) -> float:
    try:
        return draw.textlength(text, font=font)
    except Exception:
        return len(text) * font.size * 0.55


def _fit_font(draw, text: str, max_w: float, size: int, bold: bool = False):
    """Reduce el tamaño de fuente hasta que el texto quepa en `max_w`."""
    while size > 12:
        font = _font(size, bold)
        if _text_len(draw, text, font) <= max_w:
            return font
        size -= 1
    return _font(12, bold)


def _ellipsize(draw, text: str, font, max_w: float) -> str:
    """Recorta el texto con '...' si excede `max_w` (último recurso de layout)."""
    if _text_len(draw, text, font) <= max_w:
        return text
    ell = "..."
    base = text
    while base and _text_len(draw, base + ell, font) > max_w:
        base = base[:-1]
    return (base + ell) if base else ell


def _fit_teams(draw, home: str, away: str, max_w: float, size: int = 42):
    """
    Ajusta el tamaño de la línea de equipos midiendo la composición real
    (local + 'vs' + visitante con sus separaciones), no solo la cadena teórica.
    """
    while size > 16:
        team_font = _font(size, True)
        vs_font = _font(max(14, size - 10), True)
        total = (
            _text_len(draw, home, team_font)
            + 16
            + _text_len(draw, "vs", vs_font)
            + 16
            + _text_len(draw, away, team_font)
        )
        if total <= max_w:
            return team_font, vs_font
        size -= 1
    return _font(16, True), _font(13, True)


def _fmt_hhmm(scheduled_ts: int | None, match: dict) -> str:
    if scheduled_ts:
        tz = timezone(timedelta(hours=UTC_OFFSET_HOURS))
        return datetime.fromtimestamp(int(scheduled_ts), tz=tz).strftime("%H:%M")
    return str(match.get("time") or "").strip()


def _score_colors(home, away) -> tuple[tuple, tuple]:
    """Color del marcador: ganador en verde, perdedor en rojo (empate sin color)."""
    try:
        h_val = int(home)
        a_val = int(away)
    except (TypeError, ValueError):
        return COLOR_TEXT, COLOR_TEXT
    if h_val > a_val:
        return SS_SCORE_WIN, SS_SCORE_LOSS
    if a_val > h_val:
        return SS_SCORE_LOSS, SS_SCORE_WIN
    return COLOR_TEXT, COLOR_TEXT


def _quarters_items(score: dict) -> list[tuple[str, object, object]]:
    """[(etiqueta, home, away)] de los cuartos con marcador disponible."""
    quarters = score.get("quarters") or {}
    items: list[tuple[str, object, object]] = []
    for q in ("Q1", "Q2", "Q3", "Q4", "OT", "OT1", "OT2"):
        data = quarters.get(q)
        if isinstance(data, dict) and data.get("home") is not None and data.get("away") is not None:
            items.append((q, data["home"], data["away"]))
    return items


def _quarters_width(draw, items, font) -> float:
    total = 0.0
    for label, home, away in items:
        total += _text_len(draw, label, font) + 8
        total += _text_len(draw, str(home), font)
        total += _text_len(draw, "-", font) + 1
        total += _text_len(draw, str(away), font) + 26
    return total


def _draw_quarters(draw, x: float, y: float, items, font) -> float:
    """Dibuja 'Q1 20-18  Q2 ...' con la etiqueta Q destacada y el ganador en verde."""
    for label, home, away in items:
        draw.text((x, y), label, font=font, fill=SS_Q_LABEL)
        x += _text_len(draw, label, font) + 8

        h_color, a_color = _score_colors(home, away)
        draw.text((x, y), str(home), font=font, fill=h_color)
        x += _text_len(draw, str(home), font)
        draw.text((x, y), "-", font=font, fill=COLOR_MUTED)
        x += _text_len(draw, "-", font) + 1
        draw.text((x, y), str(away), font=font, fill=a_color)
        x += _text_len(draw, str(away), font) + 26
    return x


def _draw_runs(draw, x: float, y: float, runs, font, default_sep: str = "  ·  ") -> float:
    """
    Dibuja una secuencia de (texto, color[, separador]) separada por `default_sep`.
    Un separador vacío ("") permite agrupar piezas sin separación (p. ej. el 86-76).
    """
    for index, item in enumerate(runs):
        text, color = item[0], item[1]
        if len(item) > 2:
            sep = item[2]
        else:
            sep = default_sep if index else ""
        if sep:
            draw.text((x, y), sep, font=font, fill=COLOR_MUTED)
            x += _text_len(draw, sep, font)
        draw.text((x, y), text, font=font, fill=color)
        x += _text_len(draw, text, font)
    return x


def _conf_pct(confidence: float | None) -> int | None:
    if confidence is None:
        return None
    conf = float(confidence)
    if conf > 1.0:
        conf = conf / 100.0
    return int(round(conf * 100))


# ─────────────────────────────────────────────────────────────────────
# Gráfica de momentum (estilo oficial SofaScore dark, dibujada con Pillow)
# ─────────────────────────────────────────────────────────────────────

def _blend(c1: tuple, c2: tuple, alpha: float) -> tuple:
    """Mezcla el color c2 sobre c1 con la opacidad alpha (0..1)."""
    return tuple(int(round(a + (b - a) * alpha)) for a, b in zip(c1, c2))


def _period_label(status_description: str, q_key: str | None) -> str:
    """Etiqueta roja del cuarto en curso, tal como la rotula SofaScore."""
    desc = (status_description or "").strip()
    if desc and any(w in desc.lower() for w in ("quarter", "half", "overtime")):
        return desc[0].upper() + desc[1:]
    return q_key or ""


def _current_quarter(status_description: str, q_key: str | None) -> int | None:
    """Número de cuarto en curso (1..4) para resaltar su etiqueta Qn."""
    desc = (status_description or "").lower()
    for number, word in ((1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th")):
        if word in desc:
            return number
    if q_key:
        digits = q_key.strip().upper().lstrip("Q")
        if digits.isdigit() and 1 <= int(digits) <= 4:
            return int(digits)
    return None


def _is_real_bet(bet: dict) -> bool:
    """True si la señal es una apuesta operable (BET y no NO_BET)."""
    signal = str(bet.get("signal") or "")
    return "BET" in signal and "NO_BET" not in signal


def _result_tag(result: str | None) -> tuple[str, tuple]:
    """Etiqueta y color del resultado de una apuesta en la confirmación FT."""
    value = str(result or "").lower()
    if value in ("win", "hit"):
        return "GANADA", SS_CONF_BET
    if value in ("loss", "miss"):
        return "PERDIDA", SS_CONF_FAR
    if value == "would_win":
        return "HUBIERA GANADO", SS_CONF_BET
    if value == "would_loss":
        return "HUBIERA PERDIDO", SS_CONF_FAR
    if value == "push":
        return "PUSH", COLOR_MUTED
    return "PENDIENTE", COLOR_MUTED


def _total_minutes(match: dict, score: dict, graph_points: list[dict]) -> float:
    """
    Duración total de la gráfica (eje 0..FT). Usa `period_length` del API
    (10 min FIBA / 12 min NBA) y contempla partidos de 2 mitades (NCAA).
    """
    quarters = score.get("quarters") or {}
    n_played = sum(
        1
        for q in ("Q1", "Q2", "Q3", "Q4")
        if isinstance(quarters.get(q), dict) and quarters[q].get("home") is not None
    )
    status = str(match.get("status_type") or "").lower()
    n_periods = 2 if (status == "finished" and 0 < n_played <= 2) else 4

    qd = 10
    period_length = match.get("period_length")
    try:
        if period_length:
            value = float(period_length) / 60.0
            if 5 <= value <= 25:
                qd = int(round(value))
    except (TypeError, ValueError):
        pass

    total = float(n_periods * qd)
    last = max((float(p.get("minute") or 0) for p in graph_points), default=0.0)
    return max(total, last)


def _series_points(graph_points: list[dict]) -> list[tuple[float, float]]:
    """Normaliza graph_points a [(minuto, valor)] ordenado y sin duplicados de minuto."""
    pts: dict[float, float] = {}
    for p in graph_points or []:
        try:
            minute = float(p.get("minute"))
            value = float(p.get("value"))
        except (TypeError, ValueError, AttributeError):
            continue
        pts[minute] = value
    return sorted(pts.items())


def _split_runs(pts: list[tuple[float, float]]) -> list[tuple[int, list[tuple[float, float]]]]:
    """
    Divide la serie en tramos de signo constante, insertando los cruces por cero
    interpolados (equivalente a `fill_between(interpolate=True)` de la gráfica oficial).
    Devuelve [(signo, [(x, valor), ...]), ...].
    """
    runs: list[tuple[int, list[tuple[float, float]]]] = []
    if not pts:
        return runs

    def _push(sign: int, point: tuple[float, float]) -> None:
        if runs and runs[-1][0] == sign:
            runs[-1][1].append(point)
        else:
            previous = runs[-1][1][-1] if runs else None
            new_run = [previous, point] if previous else [point]
            runs.append((sign, new_run))

    _push(1 if pts[0][1] >= 0 else -1, pts[0])
    for (x0, v0), (x1, v1) in zip(pts, pts[1:]):
        if (v0 >= 0) == (v1 >= 0):
            _push(1 if v1 >= 0 else -1, (x1, v1))
            continue
        # Cruce por cero: se interpola el punto exacto y se corta el tramo ahí
        span = v1 - v0
        x_zero = x0 if span == 0 else x0 + (0.0 - v0) * (x1 - x0) / span
        _push(1 if v0 >= 0 else -1, (x_zero, 0.0))
        _push(1 if v1 >= 0 else -1, (x_zero, 0.0))
        _push(1 if v1 >= 0 else -1, (x1, v1))
    return runs


def momentum_chart(
    graph_points: list[dict],
    *,
    width: int = WIDTH,
    height: int = CHART_HEIGHT,
    start_label: str = "",
    period_label: str = "",
    current_minute: int | None = None,
    current_quarter: int | None = None,
    finished: bool = False,
    total_minutes: float | None = None,
) -> Image.Image | None:
    """
    Dibuja el gráfico de momentum con el estilo EXACTO de la web oficial de
    SofaScore (modo dark): relleno verde para presión local sobre la línea
    base, relleno añil para la visitante, tintes de zona, separadores de
    cuarto en NEGRO, etiquetas Q1..Q4 SOBRE la gráfica, escala en verde a la
    derecha, hora de inicio abajo-izquierda y marcador rojo del cuarto en curso.

    Devuelve una imagen RGBA de (width × height) o None si no hay datos.
    """
    pts = _series_points(graph_points)
    if len(pts) < 2:
        return None

    ss = CHART_SS
    w, h = width * ss, height * ss
    img = Image.new("RGB", (w, h), SS_BG)
    draw = ImageDraw.Draw(img)

    top = CHART_TOP_STRIP * ss
    bottom = h - CHART_BOTTOM_LABELS * ss
    if bottom - top < 40 * ss:
        return None
    baseline = (top + bottom) // 2
    half = min(baseline - top, bottom - baseline)

    max_abs = max(abs(v) for _, v in pts)
    if max_abs <= 0:
        return None

    # Rango temporal: el partido completo (o el minuto jugado si se pasa).
    last_minute = pts[-1][0]
    total = float(total_minutes) if total_minutes else max(last_minute, 1.0)
    if not finished and total < last_minute:
        total = last_minute

    def x_of(minute: float) -> float:
        return max(0.0, min(1.0, minute / total)) * w

    # Tintes de zona (mitad local arriba, mitad visitante abajo)
    draw.rectangle([0, top, w, baseline], fill=SS_ZONE_HOME)
    draw.rectangle([0, baseline, w, bottom], fill=SS_ZONE_AWAY)

    # Separadores de cuarto en negro (atraviesan toda la gráfica)
    n_quarters = 4
    for i in range(1, n_quarters):
        x = w * i / n_quarters
        draw.line([(x, top), (x, bottom)], fill=SS_SEPARATOR, width=max(2, ss))

    # Etiquetas Q1..Q4 SOBRE la gráfica (centradas por cuarto); el cuarto en
    # curso se resalta en rojo para ligarlo con el marcador de tiempo real.
    font_q = _font(22 * ss, True)
    for i in range(n_quarters):
        cx = w * (i + 0.5) / n_quarters
        is_current = current_quarter is not None and (i + 1) == current_quarter
        draw.text(
            (cx, 4 * ss),
            f"Q{i + 1}",
            font=font_q,
            fill=SS_LIVE if is_current else SS_Q_LABEL,
            anchor="ma",
        )

    # Rellenos por tramos de signo
    for sign, run in _split_runs(pts):
        poly: list[tuple[float, float]] = []
        for minute, value in run:
            y = baseline - (value / max_abs) * half if sign > 0 else baseline + (abs(value) / max_abs) * half
            poly.append((x_of(minute), y))
        if len(poly) < 2:
            continue
        area = [(x_of(run[0][0]), baseline)] + poly + [(x_of(run[-1][0]), baseline)]
        draw.polygon(area, fill=SS_FILL_HOME if sign > 0 else SS_FILL_AWAY)

    # Línea base
    draw.line([(0, baseline), (w, baseline)], fill=SS_BASELINE, width=max(2, ss))

    # Escala a la derecha: máximo arriba y abajo (verde) y 0 (gris claro)
    font_y = _font(21 * ss, True)
    label = f"{int(round(max_abs))}"
    draw.text((w - 6 * ss, top + 2 * ss), label, font=font_y, fill=SS_Y_LABEL, anchor="ra")
    draw.text((w - 6 * ss, baseline), "0", font=font_y, fill=SS_Y_ZERO, anchor="rm")
    draw.text((w - 6 * ss, bottom - 2 * ss), label, font=font_y, fill=SS_Y_LABEL, anchor="rd")

    # Hora de inicio (verde, abajo-izquierda) con el punto de origen y "FT" a la derecha
    font_x = _font(22 * ss, True)
    if start_label:
        draw.ellipse(
            [6 * ss - 3 * ss, baseline - 3 * ss, 6 * ss + 3 * ss, baseline + 3 * ss],
            outline=SS_X_LABEL,
            width=max(2, ss),
        )
        draw.text((6 * ss, bottom - 2 * ss), start_label, font=font_x, fill=SS_X_LABEL, anchor="la")
    draw.text((w - 6 * ss, bottom - 2 * ss), "FT", font=font_x, fill=SS_X_FT, anchor="ra")

    # Marcador de tiempo real: línea roja + punto + etiqueta del cuarto en curso
    if not finished:
        live_minute = float(current_minute) if current_minute is not None else last_minute
        x_live = x_of(live_minute)
        draw.line([(x_live, top), (x_live, bottom)], fill=SS_LIVE, width=max(2, ss))
        r = 5 * ss
        draw.ellipse([x_live - r, top - r, x_live + r, top + r], fill=SS_BG, outline=SS_LIVE, width=max(2, ss))
        if period_label:
            font_live = _font(23 * ss, True)
            draw.text((w - 6 * ss, 30 * ss), period_label, font=font_live, fill=SS_LIVE, anchor="ra")

    return img.resize((width, height), getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1))


# ─────────────────────────────────────────────────────────────────────
# Tarjeta de la apuesta
# ─────────────────────────────────────────────────────────────────────

def _payout_detail(decimal, confidence, probability, result: str, is_result: bool) -> str:
    """Línea informativa de la apuesta: cuota/proyección o confianza."""
    value = str(result or "").lower()
    if decimal:
        dec = float(decimal)
        american = decimal_to_american(dec)
        am_str = f" ({american:+d})" if american is not None else ""
        head = f"Cuota {dec:.2f}{am_str}"
        if is_result:
            if value in ("win", "hit"):
                gain, total = payout(dec, STAKE)
                return f"{head}   ·   ${STAKE:,.0f} -> ${total:,.2f}  (+${gain:,.2f})"
            if value in ("loss", "miss"):
                return f"{head}   ·   ${STAKE:,.0f} -> $0.00  (-${STAKE:,.2f})"
            if value == "push":
                return f"{head}   ·   ${STAKE:,.0f} -> ${STAKE:,.2f}  ($0.00)"
        gain, total = payout(dec, STAKE)
        return f"{head}   ·   ${STAKE:,.0f} -> ${total:,.2f}  (+${gain:,.2f})"

    prob_str = f" · P={float(probability):.2f}" if probability is not None else ""
    tail = {
        "win": "  ·   ganada",
        "hit": "  ·   ganada",
        "loss": "  ·   perdida",
        "miss": "  ·   perdida",
        "push": "  ·   push",
    }.get(value, "")
    if is_result and tail:
        return f"Confianza {confidence if confidence is not None else '?'}%{prob_str}{tail}"
    return (
        f"Confianza {confidence if confidence is not None else '?'}%{prob_str}"
        f"   ·   sin cuota in-play disponible"
    )


def _draw_side_icon(draw, x: float, y: float, is_home: bool, size: int = 28) -> float:
    """
    Dibuja el emoji del lado apostado (🏠 local / ✈️ visitante) en color.
    Si el sistema no tiene fuente de emoji, cae a la palabra LOCAL/VISITANTE.
    Devuelve el ancho consumido.
    """
    emoji = EMOJI_HOME if is_home else EMOJI_AWAY
    font = _emoji_font(size)
    if font is None:
        label = "LOCAL" if is_home else "VISITANTE"
        label_font = _font(size, True)
        draw.text((x, y + 2), label, font=label_font, fill=COLOR_TEXT)
        return _text_len(draw, label, label_font)
    try:
        draw.text((x, y), emoji, font=font, embedded_color=True)
    except TypeError:
        # Pillow antiguo sin embedded_color: se dibuja monocromo
        draw.text((x, y), emoji, font=font, fill=COLOR_TEXT)
    return size * 1.35


def _draw_bet_panel(
    draw,
    x0: int,
    y0: int,
    w: int,
    bets: list[dict],
    is_late: bool,
    has_real_bet: bool = True,
    is_result: bool = False,
) -> int:
    """Dibuja la banda de la apuesta/resultado. Devuelve la Y inferior del panel."""
    pad = 18
    accent = COLOR_ALERT if is_late else COLOR_ACCENT
    if is_result:
        tags = [_result_tag(b.get("result"))[1] for b in bets if b.get("result")]
        if tags and all(t == SS_CONF_BET for t in tags):
            accent = SS_CONF_BET
        elif tags and all(t == SS_CONF_FAR for t in tags):
            accent = SS_CONF_FAR
        elif any(t == SS_CONF_FAR for t in tags) and any(t == SS_CONF_BET for t in tags):
            accent = COLOR_ACCENT
        elif is_late:
            accent = COLOR_ALERT

    if not has_real_bet:
        title = "SIN APUESTA Q4"
    elif is_result:
        title = "RESULTADO TARDIO Q4" if is_late else "RESULTADO Q4"
    elif is_late:
        title = "APUESTA TARDIA Q4"
    else:
        title = "APUESTA Q4"

    line1_h, line2_h, gap = 40, 30, 10
    panel_h = _panel_height(len(bets))

    draw.rounded_rectangle(
        [x0, y0, x0 + w, y0 + panel_h], radius=14, fill=COLOR_PANEL_ALT, outline=COLOR_BORDER, width=2
    )
    draw.rectangle([x0, y0 + 10, x0 + 7, y0 + panel_h - 10], fill=accent)

    inner_x = x0 + pad + 10
    max_w = w - 2 * pad - 20

    draw.text((inner_x, y0 + pad), title, font=_font(27, True), fill=accent)

    y = y0 + pad + 46
    for bet in bets:
        is_real_bet = _is_real_bet(bet)
        is_home = bool(bet.get("is_home"))
        model = str(bet.get("model") or "?")
        conf = _conf_pct(bet.get("confidence"))
        team = str(bet.get("picked_team") or "")
        threshold = bet.get("threshold")
        if threshold is None:
            threshold = _model_threshold(model, bet.get("inference_json"), bet.get("raw_payload"))
        conf_color = _conf_color(bet.get("confidence"), threshold, is_real_bet)

        # Etiqueta de resultado (solo en las confirmaciones FT)
        tag_text, tag_color = _result_tag(bet.get("result")) if is_result else ("", COLOR_MUTED)

        # 1) Palabra BET (verde) / NO BET (rojo)
        x = inner_x
        bet_font = _font(28, True)
        bet_label = "BET" if is_real_bet else "NO BET"
        draw.text((x, y + 1), bet_label, font=bet_font, fill=SS_BET if is_real_bet else SS_NO_BET)
        x += _text_len(draw, bet_label, bet_font) + 18

        # 2) Modelo + porcentaje coloreado por semáforo
        model_font = _font(30, True)
        draw.text((x, y), model, font=model_font, fill=COLOR_TEXT)
        x += _text_len(draw, model, model_font) + 10
        if conf is not None:
            conf_text = f"{conf}%"
            draw.text((x, y), conf_text, font=model_font, fill=conf_color)
            x += _text_len(draw, conf_text, model_font) + 16

        tag_w = 0.0
        if tag_text:
            font_tag = _font(26, True)
            tag_w = _text_len(draw, tag_text, font_tag) + 16
            draw.text((inner_x + max_w, y + 6), tag_text, font=font_tag, fill=tag_color, anchor="ra")

        # 3) Flecha + emoji del lado (🏠/✈️) + equipo
        arrow_font = _font(28, True)
        draw.text((x, y + 2), "->", font=arrow_font, fill=COLOR_MUTED)
        x += _text_len(draw, "->", arrow_font) + 14

        side_w = _draw_side_icon(draw, x, y + 5, is_home, size=26)
        x += side_w + 10

        remaining = max(60.0, (inner_x + max_w) - x - tag_w)
        team_font = _fit_font(draw, team, remaining, 28, bold=is_real_bet)
        draw.text((x, y + 1), team, font=team_font, fill=COLOR_TEXT if is_real_bet else COLOR_MUTED)

        # 4) Línea 2: cuota + proyección/resultado, o confianza/probabilidad
        y2 = y + line1_h
        if is_real_bet:
            detail = _payout_detail(
                bet.get("decimal_odds"),
                conf,
                bet.get("probability"),
                bet.get("result"),
                is_result,
            )
        else:
            detail = f"no operó (umbral {int(round(float(threshold) * 100))}%)"

        draw.text((inner_x, y2 + 2), detail, font=_font(22), fill=COLOR_MUTED)
        y = y2 + line2_h + gap

    return y0 + panel_h


# ─────────────────────────────────────────────────────────────────────
# Composición final
# ─────────────────────────────────────────────────────────────────────

def build_q4_alert_image(
    match_data: dict | None,
    bets: list[dict],
    *,
    minute: int | None = None,
    q_key: str | None = None,
    scheduled_ts: int | None = None,
    is_result: bool = False,
    include_chart: bool = True,
) -> bytes | None:
    """
    Construye el JPEG (tarjeta + gráfica opcional) y lo devuelve como bytes.

    `bets` es una lista de diccionarios con:
        model, signal, picked_team, is_home, confidence,
        decimal_odds (opcional), probability (opcional), result (FT),
        threshold (opcional; umbral para el semáforo del porcentaje)

    `is_result=True` dibuja el panel como confirmación de resultado FT
    (título RESULTADO Q4 y etiqueta GANADA/PERDIDA/HUBIERA ...).
    `include_chart=False` deja solo la tarjeta (avisos de apuesta en vivo).

    Devuelve None si no hay datos suficientes o si falla el dibujo, para que el
    caller degrade a texto plano.
    """
    if not match_data or not bets:
        return None

    try:
        match = match_data.get("match") or {}
        score = match_data.get("score") or {}
        home = str(match.get("home_team") or "")
        away = str(match.get("away_team") or "")
        if not home or not away:
            return None

        league = str(match.get("league") or "")
        country = str(match.get("country") or "")
        status_type = str(match.get("status_type") or "").lower()

        # Cabecera de situación: hora · FT (verde) · cuarto · minuto · marcador
        hhmm = _fmt_hhmm(scheduled_ts, match)
        if minute is None and match.get("game_seconds_played") is not None:
            minute = int(int(match["game_seconds_played"]) // 60)

        situation_runs: list[tuple[str, tuple]] = []
        if hhmm:
            situation_runs.append((hhmm, COLOR_MUTED))
        if status_type == "finished":
            situation_runs.append(("FT", SS_FT))
        else:
            situation_runs.append((q_key or "Q4", COLOR_TEXT))
        if minute is not None:
            situation_runs.append((f"MIN~{minute}", COLOR_MUTED))

        home_score, away_score = score.get("home"), score.get("away")
        if home_score is not None and away_score is not None:
            h_color, a_color = _score_colors(home_score, away_score)
            situation_runs.extend(
                [
                    ("MARCADOR ", COLOR_MUTED),
                    (str(home_score), h_color, ""),
                    ("-", COLOR_MUTED, ""),
                    (str(away_score), a_color, ""),
                ]
            )

        quarters_items = _quarters_items(score)
        real_bets = [
            b
            for b in bets
            if "BET" in str(b.get("signal") or "") and "NO_BET" not in str(b.get("signal") or "")
        ]
        has_real_bet = bool(real_bets)
        is_late = any("LATE" in str(b.get("signal") or "") for b in real_bets)

        # ── Lienzo: primero la tarjeta, luego se anexa la gráfica ──
        probe = Image.new("RGB", (WIDTH, 10), COLOR_BG)
        probe_draw = ImageDraw.Draw(probe)

        has_head = bool(country or league)
        font_league = _font(22, True)
        font_teams, font_vs = _fit_teams(probe_draw, home, away, WIDTH - 2 * PAD)
        font_meta = _font(24)
        # La línea de cuartos se ajusta midiendo su composición real (etiqueta + marcadores)
        font_quarters = None
        if quarters_items:
            q_size = 21
            while q_size > 12:
                candidate = _font(q_size)
                if _quarters_width(probe_draw, quarters_items, candidate) <= WIDTH - 2 * PAD:
                    font_quarters = candidate
                    break
                q_size -= 1
            font_quarters = font_quarters or _font(12)

        y = PAD
        if has_head:
            y += 32
        y += (_font_size(font_teams) + 14)
        y += 38
        if quarters_items:
            y += 34
        y += 20  # separación antes del panel

        panel_h = _panel_height(len(bets))
        card_h = y + panel_h + PAD

        card = Image.new("RGB", (WIDTH, card_h), COLOR_BG)
        draw = ImageDraw.Draw(card)

        # Cabecera: país | liga
        cy = PAD
        if has_head:
            x = PAD
            if country:
                draw.text((x, cy), country.upper(), font=font_league, fill=COLOR_MUTED)
                x += _text_len(draw, country.upper(), font_league) + 14
            if country and league:
                draw.text((x, cy), "|", font=font_league, fill=COLOR_BORDER)
                x += _text_len(draw, "|", font_league) + 14
            if league:
                lg_font = _fit_font(draw, league.upper(), (WIDTH - PAD) - x, 22, bold=True)
                draw.text((x, cy), league.upper(), font=lg_font, fill=COLOR_ACCENT)
            cy += 32

        # Equipos: local en verde, "vs" atenuado, visitante en violeta
        x = PAD
        draw.text((x, cy), home, font=font_teams, fill=COLOR_HOME)
        x += _text_len(draw, home, font_teams) + 16
        draw.text((x, cy + 8), "vs", font=font_vs, fill=COLOR_MUTED)
        x += _text_len(draw, "vs", font_vs) + 16
        away_txt = _ellipsize(draw, away, font_teams, max(40.0, (WIDTH - PAD) - x))
        draw.text((x, cy), away_txt, font=font_teams, fill=COLOR_AWAY)
        cy += _font_size(font_teams) + 14

        # Situación (con FT y marcador coloreados) y cuartos (Q destacado)
        _draw_runs(draw, PAD, cy, situation_runs, font_meta)
        cy += 38
        if quarters_items and font_quarters:
            _draw_quarters(draw, PAD, cy, quarters_items, font_quarters)
            cy += 34

        cy += 20
        _draw_bet_panel(draw, PAD, cy, WIDTH - 2 * PAD, bets, is_late, has_real_bet, is_result)

        # ── Gráfica de momentum (solo cuando se pide; los avisos de apuesta en
        #    vivo van sin gráfica, y las confirmaciones FT con el partido completo) ──
        graph_points = match_data.get("graph_points") or []
        chart = None
        if include_chart:
            chart = momentum_chart(
                graph_points,
                start_label=hhmm,
                period_label=_period_label(str(match.get("status_description") or ""), q_key),
                current_minute=minute,
                current_quarter=(
                    None
                    if status_type == "finished"
                    else _current_quarter(str(match.get("status_description") or ""), q_key)
                ),
                finished=(status_type == "finished"),
                total_minutes=_total_minutes(match, score, graph_points),
            )
        if chart is None:
            out = io.BytesIO()
            card.save(out, format="JPEG", quality=JPEG_QUALITY)
            log_debug("TELEGRAM", "[IMG] Tarjeta Q4 generada sin gráfica")
            return out.getvalue()

        graph = chart.convert("RGB")
        sep_h = 2
        total_h = card_h + sep_h + graph.height
        canvas = Image.new("RGB", (WIDTH, total_h), COLOR_BG)
        canvas.paste(card, (0, 0))
        ImageDraw.Draw(canvas).rectangle([0, card_h, WIDTH, card_h + sep_h], fill=COLOR_BORDER)
        canvas.paste(graph, (0, card_h + sep_h))

        out = io.BytesIO()
        canvas.save(out, format="JPEG", quality=JPEG_QUALITY)
        log_debug("TELEGRAM", f"[IMG] Tarjeta Q4 fusionada {WIDTH}x{total_h} ({len(bets)} apuesta/s)")
        return out.getvalue()

    except Exception as e:  # noqa: BLE001
        log_warning("TELEGRAM", f"[IMG] No se pudo generar la tarjeta Q4 ({type(e).__name__}: {e})")
        return None
