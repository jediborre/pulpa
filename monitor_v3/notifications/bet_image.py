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
import tempfile
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

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

# Layout
WIDTH = 1080
PAD = 32
STAKE = 500.0                  # fija de ejemplo para la proyección de ganancia
JPEG_QUALITY = 92

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
_font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}
_plot_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────────────────────────────

def _font_size(font) -> int:
    return int(getattr(font, "size", 16) or 16)


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


def _score_line(score: dict) -> str:
    home = score.get("home")
    away = score.get("away")
    if home is None or away is None:
        return ""
    return f"{home}-{away}"


def _quarters_line(score: dict) -> str:
    quarters = score.get("quarters") or {}
    parts = []
    for q in ("Q1", "Q2", "Q3", "Q4", "OT", "OT1", "OT2"):
        data = quarters.get(q)
        if isinstance(data, dict) and data.get("home") is not None:
            parts.append(f"{q} {data['home']}-{data['away']}")
    return "   ".join(parts)


def _conf_pct(confidence: float | None) -> int | None:
    if confidence is None:
        return None
    conf = float(confidence)
    if conf > 1.0:
        conf = conf / 100.0
    return int(round(conf * 100))


# ─────────────────────────────────────────────────────────────────────
# Gráfica de momentum (reutiliza el motor del bot anterior)
# ─────────────────────────────────────────────────────────────────────

def momentum_png(match_data: dict | None) -> bytes | None:
    """
    Renderiza la gráfica de momentum reutilizando `match.ml_tools.plot_graph`
    (misma paleta y estilo que el bot interactivo anterior) y devuelve el PNG
    en memoria. Devuelve None si el partido no tiene `graph_points`.
    """
    if not match_data or not match_data.get("graph_points"):
        return None

    graph_points = match_data.get("graph_points") or []
    try:
        if len(graph_points) < 2:
            return None
    except TypeError:
        return None

    tmp_dir = Path(tempfile.gettempdir()) / "pulpa_q4_graph"
    mid = str((match_data.get("match") or {}).get("match_id") or "match")
    out_path = tmp_dir / f"momentum_{mid}.png"

    try:
        from match.ml_tools import plot_graph

        # Agg/matplotlib no es thread-safe: se serializa el render.
        # show_title=False: la tarjeta ya rotula equipos, cuartos y marcador.
        with _plot_lock:
            plot_graph(match_data, str(out_path), show_title=False)
        data = out_path.read_bytes()
        return data or None
    except Exception as e:  # noqa: BLE001
        log_warning("TELEGRAM", f"[IMG] Sin gráfica de momentum ({type(e).__name__}: {e})")
        return None
    finally:
        try:
            out_path.unlink()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────
# Tarjeta de la apuesta
# ─────────────────────────────────────────────────────────────────────

def _draw_bet_panel(
    draw, x0: int, y0: int, w: int, bets: list[dict], is_late: bool, has_real_bet: bool = True
) -> int:
    """Dibuja la banda de la apuesta. Devuelve la Y inferior del panel."""
    pad = 18
    accent = COLOR_ALERT if is_late else COLOR_ACCENT
    if not has_real_bet:
        title = "SIN APUESTA Q4"
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
        signal = str(bet.get("signal") or "")
        is_real_bet = "BET" in signal and "NO_BET" not in signal
        is_home = bool(bet.get("is_home"))
        side_color = COLOR_HOME if is_home else COLOR_AWAY
        side_label = "LOCAL" if is_home else "VISITANTE"
        model = str(bet.get("model") or "?")
        conf = _conf_pct(bet.get("confidence"))
        team = str(bet.get("picked_team") or "")

        # Línea 1: modelo + confianza + lado + equipo (colores por tramos)
        x = inner_x
        head = f"{model}"
        if conf is not None:
            head += f" {conf}%"
        font1 = _font(30, True)
        draw.text((x, y), head, font=font1, fill=COLOR_TEXT if is_real_bet else COLOR_MUTED)
        x += _text_len(draw, head, font1) + 18

        if is_real_bet:
            arrow = "->"
            font_arrow = _font(28, True)
            draw.text((x, y + 2), arrow, font=font_arrow, fill=COLOR_MUTED)
            x += _text_len(draw, arrow, font_arrow) + 18

            label = f"{side_label} "
            font2 = _font(28, True)
            draw.text((x, y + 1), label, font=font2, fill=side_color)
            x += _text_len(draw, label, font2)

            remaining = max(60.0, (inner_x + max_w) - x)
            team_font = _fit_font(draw, team, remaining, 28, bold=False)
            draw.text((x, y + 1), team, font=team_font, fill=COLOR_TEXT)
        else:
            tag = "NO BET"
            font_tag = _font(26, True)
            draw.text((x, y + 2), tag, font=font_tag, fill=COLOR_MUTED)
            x += _text_len(draw, tag, font_tag) + 16
            remaining = max(60.0, (inner_x + max_w) - x)
            team_font = _fit_font(draw, team, remaining, 26, bold=False)
            draw.text((x, y + 2), team, font=team_font, fill=COLOR_MUTED)

        # Línea 2: cuota + proyección (si hay cuota) o confianza/probabilidad
        y2 = y + line1_h
        decimal = bet.get("decimal_odds") if is_real_bet else None
        if decimal:
            american = decimal_to_american(decimal)
            gain, total = payout(float(decimal), STAKE)
            am_str = f" ({american:+d})" if american is not None else ""
            detail = (
                f"Cuota {float(decimal):.2f}{am_str}   ·   "
                f"${STAKE:,.0f} -> ${total:,.2f}  (+${gain:,.2f})"
            )
        elif is_real_bet:
            prob = bet.get("probability")
            prob_str = f" · P={float(prob):.2f}" if prob is not None else ""
            detail = (
                f"Confianza {conf if conf is not None else '?'}%{prob_str}"
                f"   ·   sin cuota in-play disponible"
            )
        else:
            detail = "sin valor suficiente para operar"

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
) -> bytes | None:
    """
    Construye el JPEG fusionado (tarjeta + gráfica) y lo devuelve como bytes.

    `bets` es una lista de diccionarios con:
        model, signal, picked_team, is_home, confidence,
        decimal_odds (opcional), probability (opcional)

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

        # Cabecera de situación: hora · cuarto · minuto · marcador
        bits = []
        hhmm = _fmt_hhmm(scheduled_ts, match)
        if hhmm:
            bits.append(hhmm)
        if status_type == "finished":
            bits.append("FT")
        else:
            bits.append(q_key or "Q4")
        if minute is None and match.get("game_seconds_played") is not None:
            minute = int(int(match["game_seconds_played"]) // 60)
        if minute is not None:
            bits.append(f"MIN~{minute}")
        score_str = _score_line(score)
        if score_str:
            bits.append(f"MARCADOR {score_str}")
        situation = "  ·  ".join(bits)

        quarters_str = _quarters_line(score)
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
        font_quarters = _fit_font(probe_draw, quarters_str, WIDTH - 2 * PAD, 21) if quarters_str else None

        y = PAD
        if has_head:
            y += 32
        y += (_font_size(font_teams) + 14)
        y += 38
        if quarters_str:
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

        # Situación y cuartos
        draw.text((PAD, cy), situation, font=font_meta, fill=COLOR_TEXT)
        cy += 38
        if quarters_str and font_quarters:
            draw.text((PAD, cy), quarters_str, font=font_quarters, fill=COLOR_MUTED)
            cy += 34

        cy += 20
        _draw_bet_panel(draw, PAD, cy, WIDTH - 2 * PAD, bets, is_late, has_real_bet)

        # ── Gráfica de momentum debajo ──
        graph_bytes = momentum_png(match_data)
        if not graph_bytes:
            out = io.BytesIO()
            card.save(out, format="JPEG", quality=JPEG_QUALITY)
            log_debug("TELEGRAM", "[IMG] Tarjeta Q4 generada sin gráfica (sin momentum)")
            return out.getvalue()

        with Image.open(io.BytesIO(graph_bytes)) as graph:
            graph = graph.convert("RGB")
            scale = WIDTH / graph.width
            resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", 1)
            graph = graph.resize((WIDTH, max(1, int(graph.height * scale))), resample)

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
