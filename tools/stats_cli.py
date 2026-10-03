import sys
import io
import os
import sqlite3
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional

# Configurar salida UTF-8 para consola Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Colores ANSI
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
BLUE = "\033[1;34m"
CYAN = "\033[1;36m"
MAGENTA = "\033[1;35m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Buscar base de datos
ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "matches.db"

@dataclass
class Prediction:
    model: str
    pick: str          # "HOME", "AWAY", "NO BET"
    confidence: float  # 0 - 100


def get_actual_winner(qs, deduped_logs) -> Optional[str]:
    """
    Determina el ganador real (HOME, AWAY, PUSH o None) del 4to cuarto (Q4) para basketball.
    Usa el marcador de Q4 en quarter_scores_v2 si está disponible.
    Si no, busca en los logs deduplicados de bet_monitor_log_v2 si hay algún resultado definitivo.
    """
    if qs and qs["q4_home"] is not None and qs["q4_away"] is not None:
        q4h = qs["q4_home"]
        q4a = qs["q4_away"]
        if q4h > q4a:
            return "HOME"
        elif q4a > q4h:
            return "AWAY"
        else:
            return "PUSH"
            
    if deduped_logs:
        for m_ver in ["v6_2", "m27_v3"]:
            l = deduped_logs.get(m_ver)
            if l and l["result"] in ("win", "loss", "hit", "miss", "push") and l["picked_side"] in ("HOME", "AWAY"):
                res = l["result"]
                picked = l["picked_side"]
                if res in ("win", "hit"):
                    return picked
                elif res in ("loss", "miss"):
                    return "AWAY" if picked == "HOME" else "HOME"
                elif res == "push":
                    return "PUSH"
    return None


@dataclass
class FusionResult:
    final_pick: str
    tier: str
    fusion_score: float
    avg_confidence: float
    recommended_stake: float
    reason: str


class FusionConsensusEngine:

    HOME = "HOME"
    AWAY = "AWAY"
    NO_BET = "NO BET"

    DIRECTION = {
        HOME: 1,
        AWAY: -1,
        NO_BET: 0
    }

    def __init__(
        self,
        v6_weight: float = 0.45,
        m27_weight: float = 0.55,
        consensus_threshold: float = 35,
        premium_threshold: float = 40,
        extreme_diff_threshold: float = 55,
        fusion_threshold: float = 25,
        toxicity_threshold: float = 25,
        overconfidence_threshold: float = 85
    ):

        self.v6_weight = v6_weight
        self.m27_weight = m27_weight
        self.consensus_threshold = consensus_threshold
        self.premium_threshold = premium_threshold
        self.extreme_diff_threshold = extreme_diff_threshold
        self.fusion_threshold = fusion_threshold
        self.toxicity_threshold = toxicity_threshold
        self.overconfidence_threshold = overconfidence_threshold

    def evaluate(
        self,
        v6: Prediction,
        m27: Prediction
    ) -> FusionResult:

        v6_dir = self.DIRECTION[v6.pick]
        m27_dir = self.DIRECTION[m27.pick]

        avg_conf = (v6.confidence + m27.confidence) / 2

        # =========================================================
        # 1. TOXICITY FILTER
        # =========================================================

        if (
            v6.confidence < self.toxicity_threshold and
            m27.confidence < self.toxicity_threshold
        ):
            return FusionResult(
                final_pick=self.NO_BET,
                tier="FILTERED",
                fusion_score=0,
                avg_confidence=avg_conf,
                recommended_stake=0,
                reason="Both models below toxicity threshold"
            )

        # =========================================================
        # 2. PREMIUM CONSENSUS
        # =========================================================

        if (
            v6.pick == m27.pick and
            v6.pick != self.NO_BET and
            v6.confidence >= self.premium_threshold and
            m27.confidence >= self.premium_threshold
        ):
            fusion_score = (
                (v6.confidence * v6_dir * self.v6_weight) +
                (m27.confidence * m27_dir * self.m27_weight)
            )

            return FusionResult(
                final_pick=v6.pick,
                tier="S",
                fusion_score=round(fusion_score, 2),
                avg_confidence=avg_conf,
                recommended_stake=1.0,
                reason="Premium consensus"
            )

        # =========================================================
        # 3. NORMAL CONSENSUS
        # =========================================================

        if (
            v6.pick == m27.pick and
            v6.pick != self.NO_BET and
            avg_conf >= self.consensus_threshold
        ):
            fusion_score = (
                (v6.confidence * v6_dir * self.v6_weight) +
                (m27.confidence * m27_dir * self.m27_weight)
            )

            return FusionResult(
                final_pick=v6.pick,
                tier="A",
                fusion_score=round(fusion_score, 2),
                avg_confidence=avg_conf,
                recommended_stake=0.75,
                reason="Consensus detected"
            )

        # =========================================================
        # 4. EXTREME DIFFERENCE
        # =========================================================

        if (
            v6.pick != m27.pick and
            v6.pick != self.NO_BET and
            m27.pick != self.NO_BET
        ):

            diff = abs(v6.confidence - m27.confidence)

            if diff >= self.extreme_diff_threshold:

                strongest = v6 if v6.confidence > m27.confidence else m27

                stake = 0.5

                # Anti overconfidence adjustment
                if (
                    v6.confidence > self.overconfidence_threshold or
                    m27.confidence > self.overconfidence_threshold
                ):
                    stake = 0.35

                fusion_score = (
                    (v6.confidence * v6_dir * self.v6_weight) +
                    (m27.confidence * m27_dir * self.m27_weight)
                )

                return FusionResult(
                    final_pick=strongest.pick,
                    tier="B",
                    fusion_score=round(fusion_score, 2),
                    avg_confidence=avg_conf,
                    recommended_stake=stake,
                    reason="Extreme confidence difference"
                )

        # =========================================================
        # 5. FUSION SCORE
        # =========================================================

        fusion_score = (
            (v6.confidence * v6_dir * self.v6_weight) +
            (m27.confidence * m27_dir * self.m27_weight)
        )

        # HOME
        if fusion_score > self.fusion_threshold:

            return FusionResult(
                final_pick=self.HOME,
                tier="C",
                fusion_score=round(fusion_score, 2),
                avg_confidence=avg_conf,
                recommended_stake=0.25,
                reason="Fusion score HOME"
            )

        # AWAY
        if fusion_score < -self.fusion_threshold:

            return FusionResult(
                final_pick=self.AWAY,
                tier="C",
                fusion_score=round(fusion_score, 2),
                avg_confidence=avg_conf,
                recommended_stake=0.25,
                reason="Fusion score AWAY"
            )

        # =========================================================
        # 6. DEFAULT NO BET
        # =========================================================

        return FusionResult(
            final_pick=self.NO_BET,
            tier="FILTERED",
            fusion_score=round(fusion_score, 2),
            avg_confidence=avg_conf,
            recommended_stake=0,
            reason="Fusion score inside neutral zone"
        )

class FusionBorregoEngine:
    HOME = "HOME"
    AWAY = "AWAY"
    NO_BET = "NO BET"

    def evaluate(self, v6: Prediction, m27: Prediction) -> str:
        v6_conf = v6.confidence
        m27_conf = m27.confidence
        v6_pick = v6.pick
        m27_pick = m27.pick
        
        diff = m27_conf - v6_conf
        
        # Rule 1: Si v6_2 Conf >= 50, Hacerle caso mejor a m27_v3 Pick
        if v6_conf >= 50:
            return m27_pick
            
        # Rule 2: Si Diff > 60 y v6_2 Pick = HOME, m27_v3 Pick = HOME. NO BET
        if diff > 60 and v6_pick == self.HOME and m27_pick == self.HOME:
            return self.NO_BET
            
        # Rule 3: Si Diff > 40 y v6_2 Pick = HOME, m27_v3 Pick = AWAY. Pick -> HOME
        if diff > 40 and v6_pick == self.HOME and m27_pick == self.AWAY:
            return self.HOME
            
        # Rule 4: Si Diff > 40 y v6_2 Pick = AWAY, m27_v3 Pick = AWAY. Pick -> HOME
        if diff > 40 and v6_pick == self.AWAY and m27_pick == self.AWAY:
            return self.HOME
            
        # Rule 4 (B): Pero si la Diff < 40, Si v6_2 Conf <= 30, m27_v3 Conf <= 30 y v6_2 Pick = AWAY, m27_v3 Pick = AWAY. Pick -> HOME
        if diff < 40 and v6_conf <= 30 and m27_conf <= 30 and v6_pick == self.AWAY and m27_pick == self.AWAY:
            return self.HOME

        # Rule 4 (C): Pero si la Diff < 40, Si v6_2 Conf <= 30, m27_v3 Conf >= 30 y v6_2 Pick = AWAY, m27_v3 Pick = AWAY. Pick -> HOME
        if diff < 40 and v6_conf <= 30 and m27_conf >= 30 and v6_pick == self.AWAY and m27_pick == self.AWAY:
            return self.HOME
            
        # Rule 5: Si v6_2 Conf > 30, m27_v3 Conf > 30 y v6_2 Pick = HOME, m27_v3 Pick = AWAY. Pick -> HOME
        if v6_conf > 30 and m27_conf > 30 and v6_pick == self.HOME and m27_pick == self.AWAY:
            return self.HOME
            
        # Rule 6: Si Diff < 0 y v6_2 Pick = HOME, m27_v3 Pick = HOME. Pick -> AWAY
        if diff < 0 and v6_pick == self.HOME and m27_pick == self.HOME:
            return self.AWAY
            
        # Rule 7: Si Diff < 0 y v6_2 Pick = AWAY, m27_v3 Pick = HOME. Pick -> AWAY
        if diff < 0 and v6_pick == self.AWAY and m27_pick == self.HOME:
            return self.AWAY

        # Default fallback
        if v6_pick == m27_pick:
            return v6_pick
        return self.NO_BET

def visual_len(s: str) -> int:
    """Calcula la longitud visual considerando emojis como 2 espacios de ancho e ignorando códigos ANSI."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_s = ansi_escape.sub('', s)
    clean_s = clean_s.replace("\ufe0f", "")
    
    emojis = ["✅", "❌", "🟢", "🟡", "⚪", "🔴", "🏠", "✈️", "🤖", "🏀", "⏳"]
    length = 0
    for char in clean_s:
        if char in emojis:
            length += 2
        else:
            length += 1
    return length

def pad_cell(text: str, target_width: int) -> str:
    """Rellena la celda con espacios basándose en su longitud visual."""
    v_len = visual_len(text)
    padding_needed = max(0, target_width - v_len)
    return text + (" " * padding_needed)

def colorize_scores(h, a) -> tuple[str, str]:
    """Colorea el marcador del ganador en verde y el perdedor en rojo."""
    try:
        h_val = int(h)
        a_val = int(a)
    except (ValueError, TypeError):
        return str(h), str(a)
    
    if h_val > a_val:
        return f"{GREEN}{h}{RESET}", f"{RED}{a}{RESET}"
    elif a_val > h_val:
        return f"{RED}{h}{RESET}", f"{GREEN}{a}{RESET}"
    else:
        return f"{h}", f"{a}"


def calculate_metrics_for_logs(conn: sqlite3.Connection, date_str: str = None) -> tuple[dict, dict]:
    """Calcula estadísticas por liga y por nivel de confianza (global o de una fecha)."""
    if date_str:
        cursor = conn.execute("""
            SELECT l.match_id, l.model_version, s.league, l.confidence, l.result
            FROM bet_monitor_log_v2 l
            JOIN bet_monitor_schedule_v2 s ON l.match_id = s.match_id
            WHERE l.result IN ('win', 'hit', 'loss', 'miss') AND s.event_date = ?
            ORDER BY l.id ASC
        """, (date_str,))
    else:
        cursor = conn.execute("""
            SELECT l.match_id, l.model_version, s.league, l.confidence, l.result
            FROM bet_monitor_log_v2 l
            JOIN bet_monitor_schedule_v2 s ON l.match_id = s.match_id
            WHERE l.result IN ('win', 'hit', 'loss', 'miss')
            ORDER BY l.id ASC
        """)
    rows = cursor.fetchall()
    
    # Deduplicar por (match_id, model_version) conservando la decisión final
    deduped = {}
    for r in rows:
        key = (r["match_id"], r["model_version"])
        deduped[key] = r
        
    models = ["v6_2", "m27_v3"]
    league_stats = {m: {} for m in models}
    conf_stats = {m: {
        "low": {"win": 0, "loss": 0},
        "high": {"win": 0, "loss": 0}
    } for m in models}
    
    for r in deduped.values():
        m = r["model_version"]
        if m not in models:
            continue
        league = r["league"] or "Desconocida"
        conf = r["confidence"] or 0.0
        res = r["result"] or ""
        
        if conf > 1.0:
            conf = conf / 100.0
            
        is_win = res in ("win", "hit")
        is_loss = res in ("loss", "miss")
        if not (is_win or is_loss):
            continue
            
        # Agrupar por liga
        if league not in league_stats[m]:
            league_stats[m][league] = {"win": 0, "loss": 0}
        if is_win:
            league_stats[m][league]["win"] += 1
        else:
            league_stats[m][league]["loss"] += 1
            
        # Agrupar por confianza: < 30% vs >= 30% (> 29%)
        conf_cat = "low" if conf < 0.30 else "high"
        if is_win:
            conf_stats[m][conf_cat]["win"] += 1
        else:
            conf_stats[m][conf_cat]["loss"] += 1
            
    return league_stats, conf_stats

def format_confidence_table(conf_stats: dict) -> str:
    """Formatea la distribución por confianza en una tabla de caja fina elegante."""
    col_widths = [12, 28, 28]
    top_border = "   ┌" + "┬".join("─" * w for w in col_widths) + "┐"
    header_line = "   │ " + " │ ".join(pad_cell(h, w - 2) for h, w in zip(["Modelo", "Confianza < 30% (<30)", "Confianza >= 30% (>29)"], col_widths)) + " │"
    mid_border = "   ├" + "┼".join("─" * w for w in col_widths) + "┤"
    bot_border = "   └" + "┴".join("─" * w for w in col_widths) + "┘"
    
    table_lines = [top_border, header_line, mid_border]
    for m in ["v6_2", "m27_v3"]:
        low_win = conf_stats[m]["low"]["win"]
        low_loss = conf_stats[m]["low"]["loss"]
        low_tot = low_win + low_loss
        if low_tot > 0:
            low_pct = int(round(low_win * 100.0 / low_tot))
            low_str = f"{GREEN}✅ {low_win}{RESET} | {RED}❌ {low_loss}{RESET} ({low_pct}%)"
        else:
            low_str = "N/D"
            
        high_win = conf_stats[m]["high"]["win"]
        high_loss = conf_stats[m]["high"]["loss"]
        high_tot = high_win + high_loss
        if high_tot > 0:
            high_pct = int(round(high_win * 100.0 / high_tot))
            high_str = f"{GREEN}✅ {high_win}{RESET} | {RED}❌ {high_loss}{RESET} ({high_pct}%)"
        else:
            high_str = "N/D"
            
        row_line = "   │ " + " │ ".join(pad_cell(cell, w - 2) for cell, w in zip([m, low_str, high_str], col_widths)) + " │"
        table_lines.append(row_line)
        
    table_lines.append(bot_border)
    return "\n".join(table_lines)

def format_league_table(league_stats: dict) -> str:
    """Formatea las estadísticas por liga ordenadas por volumen total en una tabla consolidada."""
    all_leagues = set(list(league_stats["v6_2"].keys()) + list(league_stats["m27_v3"].keys()))
    if not all_leagues:
        return f"   {YELLOW}Sin estadísticas por liga registradas.{RESET}"
        
    def league_volume(league):
        vol = 0
        for m in ["v6_2", "m27_v3"]:
            st = league_stats[m].get(league, {"win": 0, "loss": 0})
            vol += st["win"] + st["loss"]
        return vol
        
    sorted_leagues = sorted(all_leagues, key=league_volume, reverse=True)
    
    col_widths = [45, 23, 23]
    top_border = "   ┌" + "┬".join("─" * w for w in col_widths) + "┐"
    header_line = "   │ " + " │ ".join(pad_cell(h, w - 2) for h, w in zip(["Liga", "v6_2 (W/L | %)", "m27_v3 (W/L | %)"], col_widths)) + " │"
    mid_border = "   ├" + "┼".join("─" * w for w in col_widths) + "┤"
    bot_border = "   └" + "┴".join("─" * w for w in col_widths) + "┘"
    
    table_lines = [top_border, header_line, mid_border]
    
    for league in sorted_leagues:
        league_display = league
        if visual_len(league_display) > 43:
            league_display = league_display[:40] + "..."
            
        model_strs = []
        for m in ["v6_2", "m27_v3"]:
            st = league_stats[m].get(league, {"win": 0, "loss": 0})
            w = st["win"]
            l = st["loss"]
            tot = w + l
            if tot > 0:
                pct = int(round(w * 100.0 / tot))
                model_strs.append(f"{GREEN}✅ {w}{RESET} | {RED}❌ {l}{RESET} ({pct}%)")
            else:
                model_strs.append("N/D")
                
        row_line = "   │ " + " │ ".join(pad_cell(cell, w - 2) for cell, w in zip([league_display, model_strs[0], model_strs[1]], col_widths)) + " │"
        table_lines.append(row_line)
        
    table_lines.append(bot_border)
    return "\n".join(table_lines)

def export_matches_to_excel(conn: sqlite3.Connection, date_str: str) -> str:
    """Genera un archivo Excel (.xlsx) premium con justificación y colores de los partidos."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        return f"{RED}[ERROR]{RESET} No se pudo importar openpyxl. Instálalo con 'pip install openpyxl'."

    # 1. Fetch matches
    cursor = conn.execute("""
        SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league, s.scheduled_utc_ts
        FROM bet_monitor_schedule_v2 s
        JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
        WHERE s.event_date = ?
        ORDER BY s.scheduled_utc_ts ASC
    """, (date_str,))
    matches = cursor.fetchall()

    if not matches:
        return f"{YELLOW}No se encontraron partidos para la fecha: {date_str}{RESET}"

    # Initialize workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"Matches {date_str}"
    ws.views.sheetView[0].showGridLines = True

    # Define headers
    headers = [
        "Hora",
        "Partido",
        "Liga",
        "Score / Q4",
        "v6_2 Pick",
        "v6_2 Conf",
        "v6_2 Min",
        "v6_2 Res",
        "m27_v3 Pick",
        "m27_v3 Conf",
        "m27_v3 Min",
        "m27_v3 Res",
        "Fusion Pick",
        "Fusion Tier",
        "Fusion Stake",
        "Fusion Res",
        "Borrego Pick",
        "Borrego Res"
    ]
    ws.append(headers)

    # Style definitions
    font_header = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    font_regular = Font(name="Segoe UI", size=10)
    
    fill_header = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid") # Dark Blue
    fill_win = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")     # Light Green
    fill_loss = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")    # Light Red
    fill_pending = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid") # Light Yellow
    fill_filtered = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")# Light Gray
    
    font_win = Font(name="Segoe UI", size=10, color="375623")
    font_loss = Font(name="Segoe UI", size=10, color="C65911")
    font_pending = Font(name="Segoe UI", size=10, color="7F6000")
    font_gray = Font(name="Segoe UI", size=10, color="7F7F7F")

    # Tier specific premium colors
    fill_s = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")       # Light Blue for Tier S
    font_s = Font(name="Segoe UI", size=10, color="1F4E79")
    fill_c = PatternFill(start_color="E5F5F8", end_color="E5F5F8", fill_type="solid")       # Light Cyan for Tier C
    font_c = Font(name="Segoe UI", size=10, color="006666")

    border_thin = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    # Style Header Row
    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_thin
    ws.row_dimensions[1].height = 28

    engine = FusionConsensusEngine()

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    def strip_ansi(s: str) -> str:
        return ansi_escape.sub('', s).replace("\ufe0f", "")

    # Populate rows
    for m in matches:
        mid = m["match_id"]
        home = m["home_team"]
        away = m["away_team"]
        league = m["league"]
        sched_ts = m["scheduled_utc_ts"]
        
        sched_time = datetime.fromtimestamp(sched_ts, tz=timezone(timedelta(hours=-6))).strftime("%H:%M")

        qs = conn.execute("""
            SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
            FROM quarter_scores_v2 WHERE match_id = ?
        """, (mid,)).fetchone()

        score_str = "N/D"
        if qs:
            q4h = qs["q4_home"]
            q4a = qs["q4_away"]
            if q4h is not None and q4a is not None:
                if all(qs[c] is not None for c in ["q1_home", "q1_away", "q2_home", "q2_away", "q3_home", "q3_away"]):
                    tot_h = qs["q1_home"] + qs["q2_home"] + qs["q3_home"] + q4h
                    tot_a = qs["q1_away"] + qs["q2_away"] + qs["q3_away"] + q4a
                    score_str = f"{tot_h}-{tot_a} ({q4h}-{q4a})"
                else:
                    score_str = f"Q4:{q4h}-{q4a}"
            else:
                score_str = "Pendiente"

        logs_cursor = conn.execute("""
            SELECT model_version, signal_type, picked_side, confidence, result, inference_minute
            FROM bet_monitor_log_v2
            WHERE match_id = ?
            ORDER BY id ASC
        """, (mid,))
        logs = logs_cursor.fetchall()

        deduped = {}
        for l in logs:
            deduped[l["model_version"]] = l

        # Extract v6_2
        v6_log = deduped.get("v6_2")
        if not v6_log:
            v6_pick = "🔴 N/A"
            v6_conf = None
            v6_res = ""
            v6_min = None
        else:
            sig = v6_log["signal_type"] or ""
            picked = v6_log["picked_side"] or ""
            conf = v6_log["confidence"] or 0.0
            res = v6_log["result"] or "pending"
            v6_min = v6_log["inference_minute"]

            if "NO_BET" in sig or "BET" not in sig:
                v6_pick = "🔴 NO BET"
                v6_conf = None
                v6_res = ""
            else:
                side_emoji = "🏠" if picked == "HOME" else "✈️"
                v6_pick = f"{side_emoji} {picked}"
                v6_conf = int(round(conf * 100 if conf <= 1.0 else conf))
                v6_res = "✅" if res == "win" else ("❌" if res == "loss" else "⏳")

        # Extract m27_v3
        m27_log = deduped.get("m27_v3")
        if not m27_log:
            m27_pick = "🔴 N/A"
            m27_conf = None
            m27_res = ""
            m27_min = None
        else:
            sig = m27_log["signal_type"] or ""
            picked = m27_log["picked_side"] or ""
            conf = m27_log["confidence"] or 0.0
            res = m27_log["result"] or "pending"
            m27_min = m27_log["inference_minute"]

            if "NO_BET" in sig or "BET" not in sig:
                m27_pick = "🔴 NO BET"
                m27_conf = None
                m27_res = ""
            else:
                side_emoji = "🏠" if picked == "HOME" else "✈️"
                m27_pick = f"{side_emoji} {picked}"
                m27_conf = int(round(conf * 100 if conf <= 1.0 else conf))
                m27_res = "✅" if res == "win" else ("❌" if res == "loss" else "⏳")

        # Fusion consensus
        v6_pred = Prediction(model="v6_2", pick="NO BET", confidence=0.0)
        m27_pred = Prediction(model="m27_v3", pick="NO BET", confidence=0.0)
        
        v6_log_active = deduped.get("v6_2")
        if v6_log_active:
            sig = v6_log_active["signal_type"] or ""
            picked = v6_log_active["picked_side"] or ""
            conf = v6_log_active["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                v6_pred = Prediction(model="v6_2", pick=picked, confidence=conf)
                
        m27_log_active = deduped.get("m27_v3")
        if m27_log_active:
            sig = m27_log_active["signal_type"] or ""
            picked = m27_log_active["picked_side"] or ""
            conf = m27_log_active["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                m27_pred = Prediction(model="m27_v3", pick=picked, confidence=conf)

        # Determine actual winner
        actual_winner = get_actual_winner(qs, deduped)
                
        if not v6_log_active and not m27_log_active:
            fusion_pick = "🔴 N/A"
            fusion_tier = "N/A"
            fusion_stake = None
            fusion_res = ""
        else:
            res_fusion = engine.evaluate(v6_pred, m27_pred)
            
            if res_fusion.final_pick == "NO BET" or res_fusion.recommended_stake == 0:
                fusion_pick = "🔴 FILTERED"
                fusion_tier = "FILTERED"
                fusion_stake = 0.0
                fusion_res = ""
            else:
                side_emoji = "🏠" if res_fusion.final_pick == "HOME" else "✈️"
                fusion_pick = f"{side_emoji} {res_fusion.final_pick}"
                fusion_tier = res_fusion.tier
                fusion_stake = float(res_fusion.recommended_stake)
                
                if not actual_winner:
                    fusion_res = "⏳"
                elif res_fusion.final_pick == actual_winner:
                    fusion_res = "✅"
                else:
                    fusion_res = "❌"

        # Fusion Borrego
        borrego_engine = FusionBorregoEngine()
        borrego_pick_val = borrego_engine.evaluate(v6_pred, m27_pred)
        if not v6_log_active and not m27_log_active:
            borrego_pick = "🔴 N/A"
            borrego_res = ""
        elif borrego_pick_val == "NO BET":
            borrego_pick = "🔴 NO BET"
            borrego_res = ""
        else:
            side_emoji = "🏠" if borrego_pick_val == "HOME" else "✈️"
            borrego_pick = f"{side_emoji} {borrego_pick_val}"
            
            if not actual_winner:
                borrego_res = "⏳"
            elif borrego_pick_val == actual_winner:
                borrego_res = "✅"
            else:
                borrego_res = "❌"

        # Calculate if there is a big difference in confidence
        big_diff = False
        if v6_conf is not None and m27_conf is not None:
            big_diff = abs(v6_conf - m27_conf) > 40

        # Clean strings
        row_values = [
            sched_time,
            f"{home} vs {away}",
            league,
            strip_ansi(score_str),
            v6_pick,
            v6_conf,
            v6_min,
            v6_res,
            m27_pick,
            m27_conf,
            m27_min,
            m27_res,
            fusion_pick,
            fusion_tier,
            fusion_stake,
            fusion_res,
            borrego_pick,
            borrego_res
        ]
        ws.append(row_values)

        # Style populated row
        current_row = ws.max_row
        ws.row_dimensions[current_row].height = 20

        for col_idx in range(1, len(row_values) + 1):
            cell = ws.cell(row=current_row, column=col_idx)
            cell.font = font_regular
            cell.border = border_thin
            cell.alignment = Alignment(vertical="center")
            
            # Alignments
            if col_idx in (1, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18): # Hora, Score, Confs, Min, Res, Tier, Stake
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                cell.alignment = Alignment(horizontal="left", vertical="center")

            # Fills and Fonts based on column and value
            val = cell.value
            val_str = str(val or "")

            # 1. Results columns: 8 (v6_2 Res), 12 (m27_v3 Res), 16 (Fusion Res), 18 (Borrego Res)
            if col_idx in (8, 12, 16, 18):
                if val == "✅":
                    cell.fill = fill_win
                    cell.font = font_win
                elif val == "❌":
                    cell.fill = fill_loss
                    cell.font = font_loss
                elif val == "⏳":
                    cell.fill = fill_pending
                    cell.font = font_pending
                elif val_str == "FILTERED":
                    cell.fill = fill_filtered
                    cell.font = font_gray

            # 2. Confidence columns: 6 (v6_2 Conf), 10 (m27_v3 Conf)
            elif col_idx in (6, 10):
                if val is not None:
                    try:
                        conf_val = int(val)
                        if big_diff:
                            cell.fill = fill_loss
                            cell.font = font_loss
                        elif conf_val < 30:
                            cell.fill = fill_pending
                            cell.font = font_pending
                        else:
                            cell.fill = fill_win
                            cell.font = font_win
                    except ValueError:
                        pass

            # 3. Fusion Tier column: 14 (Fusion Tier)
            elif col_idx == 14:
                if val == "S":
                    cell.fill = fill_s
                    cell.font = font_s
                elif val == "A":
                    cell.fill = fill_win
                    cell.font = font_win
                elif val == "B":
                    cell.fill = fill_pending
                    cell.font = font_pending
                elif val == "C":
                    cell.fill = fill_c
                    cell.font = font_c
                elif val in ("FILTERED", "🔴 FILTERED"):
                    cell.fill = fill_filtered
                    cell.font = font_gray

            # 4. Gray out NO BET / FILTERED / N/A picks & stakes
            elif col_idx in (5, 9, 13, 17): # Pick columns
                if "NO_BET" in val_str or "NO BET" in val_str or "FILTERED" in val_str or "N/A" in val_str:
                    cell.fill = fill_filtered
                    cell.font = font_gray
            elif col_idx == 15: # Fusion Stake
                if val == 0.0 or val_str == "0.0" or val_str == "0":
                    cell.fill = fill_filtered
                    cell.font = font_gray

    # Auto-fit columns
    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            # We treat emojis as 2 chars long, others as 1
            v_len = len(str(cell.value or ''))
            # Simple check for emojis in values
            for emoji in ["🏠", "✈️", "✅", "❌", "🔴", "⏳"]:
                if emoji in str(cell.value or ''):
                    v_len += 1
            if v_len > max_len:
                max_len = v_len
        ws.column_dimensions[col_letter].width = max(max_len + 3, 10)

    # Save Excel with timestamp hash to prevent collisions
    import time
    ts = int(time.time())
    filename = f"Reporte_Matches_{date_str}_{ts}.xlsx"
    file_path = ROOT / filename
    wb.save(file_path)
    
    # Auto-open file on Windows
    open_msg = ""
    try:
        os.startfile(file_path)
        open_msg = " y abierto automáticamente"
    except Exception as e:
        open_msg = f" (no se pudo abrir automáticamente: {e})"
        
    return f"{GREEN}[ÉXITO] Excel generado{open_msg} en: {file_path}{RESET}"

def export_matches_monthly_to_excel(conn: sqlite3.Connection, year_month: str) -> str:
    """Genera un Excel con una sola hoja con todos los partidos del mes y columna Fecha."""
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        return f"{RED}[ERROR]{RESET} No se pudo importar openpyxl. Instálalo con 'pip install openpyxl'."

    cursor = conn.execute("""
        SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league, s.scheduled_utc_ts, s.event_date
        FROM bet_monitor_schedule_v2 s
        JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
        WHERE s.event_date LIKE ?
        ORDER BY s.event_date ASC, s.scheduled_utc_ts ASC
    """, (f"{year_month}%",))
    matches = cursor.fetchall()

    if not matches:
        return f"{YELLOW}No se encontraron partidos para el mes: {year_month}{RESET}"

    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    def strip_ansi(s: str) -> str:
        return ansi_escape.sub('', s).replace("\ufe0f", "")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"Mes {year_month}"

    headers = [
        "Fecha", "Hora", "Partido", "Liga", "Score / Q4",
        "v6_2 Pick", "v6_2 Conf", "v6_2 Min", "v6_2 Res",
        "m27_v3 Pick", "m27_v3 Conf", "m27_v3 Min", "m27_v3 Res",
        "Fusion Pick", "Fusion Tier", "Fusion Stake", "Fusion Res",
        "Borrego Pick", "Borrego Res"
    ]
    ws.append(headers)

    font_header = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
    font_regular = Font(name="Segoe UI", size=10)
    fill_header = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    fill_win = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    fill_loss = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    fill_pending = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")
    fill_filtered = PatternFill(start_color="F2F2F2", end_color="F2F2F2", fill_type="solid")
    font_win = Font(name="Segoe UI", size=10, color="375623")
    font_loss = Font(name="Segoe UI", size=10, color="C65911")
    font_pending = Font(name="Segoe UI", size=10, color="7F6000")
    font_gray = Font(name="Segoe UI", size=10, color="7F7F7F")
    fill_s = PatternFill(start_color="DDEBF7", end_color="DDEBF7", fill_type="solid")
    font_s = Font(name="Segoe UI", size=10, color="1F4E79")
    fill_c = PatternFill(start_color="E5F5F8", end_color="E5F5F8", fill_type="solid")
    font_c = Font(name="Segoe UI", size=10, color="006666")

    border_thin = Border(
        left=Side(style='thin', color='D9D9D9'),
        right=Side(style='thin', color='D9D9D9'),
        top=Side(style='thin', color='D9D9D9'),
        bottom=Side(style='thin', color='D9D9D9')
    )

    for col_idx in range(1, len(headers) + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = font_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = border_thin
    ws.row_dimensions[1].height = 28

    engine = FusionConsensusEngine()

    for m in matches:
        mid = m["match_id"]
        home = m["home_team"]
        away = m["away_team"]
        league = m["league"]
        sched_ts = m["scheduled_utc_ts"]
        event_date = m["event_date"]
        sched_time = datetime.fromtimestamp(sched_ts, tz=timezone(timedelta(hours=-6))).strftime("%H:%M")

        qs = conn.execute("""
            SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
            FROM quarter_scores_v2 WHERE match_id = ?
        """, (mid,)).fetchone()

        score_str = "N/D"
        if qs:
            q4h = qs["q4_home"]
            q4a = qs["q4_away"]
            if q4h is not None and q4a is not None:
                if all(qs[c] is not None for c in ["q1_home", "q1_away", "q2_home", "q2_away", "q3_home", "q3_away"]):
                    tot_h = qs["q1_home"] + qs["q2_home"] + qs["q3_home"] + q4h
                    tot_a = qs["q1_away"] + qs["q2_away"] + qs["q3_away"] + q4a
                    score_str = f"{tot_h}-{tot_a} ({q4h}-{q4a})"
                else:
                    score_str = f"Q4:{q4h}-{q4a}"
            else:
                score_str = "Pendiente"

        logs_cursor = conn.execute("""
            SELECT model_version, signal_type, picked_side, confidence, result, inference_minute
            FROM bet_monitor_log_v2
            WHERE match_id = ?
            ORDER BY id ASC
        """, (mid,))
        logs = logs_cursor.fetchall()
        deduped = {}
        for l in logs:
            deduped[l["model_version"]] = l

        v6_log = deduped.get("v6_2")
        if not v6_log:
            v6_pick = "🔴 N/A"
            v6_conf = None
            v6_res = ""
            v6_min = None
        else:
            sig = v6_log["signal_type"] or ""
            picked = v6_log["picked_side"] or ""
            conf = v6_log["confidence"] or 0.0
            res = v6_log["result"] or "pending"
            v6_min = v6_log["inference_minute"]
            if "NO_BET" in sig or "BET" not in sig:
                v6_pick = "🔴 NO BET"
                v6_conf = None
                v6_res = ""
            else:
                side_emoji = "🏠" if picked == "HOME" else "✈️"
                v6_pick = f"{side_emoji} {picked}"
                v6_conf = int(round(conf * 100 if conf <= 1.0 else conf))
                v6_res = "✅" if res == "win" else ("❌" if res == "loss" else "⏳")

        m27_log = deduped.get("m27_v3")
        if not m27_log:
            m27_pick = "🔴 N/A"
            m27_conf = None
            m27_res = ""
            m27_min = None
        else:
            sig = m27_log["signal_type"] or ""
            picked = m27_log["picked_side"] or ""
            conf = m27_log["confidence"] or 0.0
            res = m27_log["result"] or "pending"
            m27_min = m27_log["inference_minute"]
            if "NO_BET" in sig or "BET" not in sig:
                m27_pick = "🔴 NO BET"
                m27_conf = None
                m27_res = ""
            else:
                side_emoji = "🏠" if picked == "HOME" else "✈️"
                m27_pick = f"{side_emoji} {picked}"
                m27_conf = int(round(conf * 100 if conf <= 1.0 else conf))
                m27_res = "✅" if res == "win" else ("❌" if res == "loss" else "⏳")

        v6_pred = Prediction(model="v6_2", pick="NO BET", confidence=0.0)
        m27_pred = Prediction(model="m27_v3", pick="NO BET", confidence=0.0)
        v6_log_active = deduped.get("v6_2")
        if v6_log_active:
            sig = v6_log_active["signal_type"] or ""
            picked = v6_log_active["picked_side"] or ""
            conf = v6_log_active["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                v6_pred = Prediction(model="v6_2", pick=picked, confidence=conf)
        m27_log_active = deduped.get("m27_v3")
        if m27_log_active:
            sig = m27_log_active["signal_type"] or ""
            picked = m27_log_active["picked_side"] or ""
            conf = m27_log_active["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                m27_pred = Prediction(model="m27_v3", pick=picked, confidence=conf)

        actual_winner = get_actual_winner(qs, deduped)

        if not v6_log_active and not m27_log_active:
            fusion_pick = "🔴 N/A"
            fusion_tier = "N/A"
            fusion_stake = None
            fusion_res = ""
        else:
            res_fusion = engine.evaluate(v6_pred, m27_pred)
            if res_fusion.final_pick == "NO BET" or res_fusion.recommended_stake == 0:
                fusion_pick = "🔴 FILTERED"
                fusion_tier = "FILTERED"
                fusion_stake = 0.0
                fusion_res = ""
            else:
                side_emoji = "🏠" if res_fusion.final_pick == "HOME" else "✈️"
                fusion_pick = f"{side_emoji} {res_fusion.final_pick}"
                fusion_tier = res_fusion.tier
                fusion_stake = float(res_fusion.recommended_stake)
                if not actual_winner:
                    fusion_res = "⏳"
                elif res_fusion.final_pick == actual_winner:
                    fusion_res = "✅"
                else:
                    fusion_res = "❌"

        borrego_engine = FusionBorregoEngine()
        borrego_pick_val = borrego_engine.evaluate(v6_pred, m27_pred)
        if not v6_log_active and not m27_log_active:
            borrego_pick = "🔴 N/A"
            borrego_res = ""
        elif borrego_pick_val == "NO BET":
            borrego_pick = "🔴 NO BET"
            borrego_res = ""
        else:
            side_emoji = "🏠" if borrego_pick_val == "HOME" else "✈️"
            borrego_pick = f"{side_emoji} {borrego_pick_val}"
            if not actual_winner:
                borrego_res = "⏳"
            elif borrego_pick_val == actual_winner:
                borrego_res = "✅"
            else:
                borrego_res = "❌"

        big_diff = False
        if v6_conf is not None and m27_conf is not None:
            big_diff = abs(v6_conf - m27_conf) > 40

        # Mapeo de índices de columnas con Fecha agregada al inicio
        row_values = [
            event_date,
            sched_time,
            f"{home} vs {away}",
            league,
            strip_ansi(score_str),
            v6_pick, v6_conf, v6_min, v6_res,
            m27_pick, m27_conf, m27_min, m27_res,
            fusion_pick, fusion_tier, fusion_stake, fusion_res,
            borrego_pick, borrego_res
        ]
        ws.append(row_values)

        current_row = ws.max_row
        ws.row_dimensions[current_row].height = 20
        for col_idx in range(1, len(row_values) + 1):
            cell = ws.cell(row=current_row, column=col_idx)
            cell.font = font_regular
            cell.border = border_thin
            cell.alignment = Alignment(vertical="center")
            if col_idx in (2, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19):
                cell.alignment = Alignment(horizontal="center", vertical="center")
            val = cell.value
            val_str = str(val or "")
            if col_idx in (9, 13, 17, 19):
                if val == "✅":
                    cell.fill = fill_win
                    cell.font = font_win
                elif val == "❌":
                    cell.fill = fill_loss
                    cell.font = font_loss
                elif val == "⏳":
                    cell.fill = fill_pending
                    cell.font = font_pending
                elif val_str == "FILTERED":
                    cell.fill = fill_filtered
                    cell.font = font_gray
            elif col_idx in (7, 11):
                if val is not None:
                    try:
                        conf_val = int(val)
                        if big_diff:
                            cell.fill = fill_loss
                            cell.font = font_loss
                        elif conf_val < 30:
                            cell.fill = fill_pending
                            cell.font = font_pending
                        else:
                            cell.fill = fill_win
                            cell.font = font_win
                    except ValueError:
                        pass
            elif col_idx == 15:
                if val == "S":
                    cell.fill = fill_s
                    cell.font = font_s
                elif val == "A":
                    cell.fill = fill_win
                    cell.font = font_win
                elif val == "B":
                    cell.fill = fill_pending
                    cell.font = font_pending
                elif val == "C":
                    cell.fill = fill_c
                    cell.font = font_c
                elif val in ("FILTERED", "🔴 FILTERED"):
                    cell.fill = fill_filtered
                    cell.font = font_gray
            elif col_idx in (6, 10, 14, 18):
                if "NO_BET" in val_str or "NO BET" in val_str or "FILTERED" in val_str or "N/A" in val_str:
                    cell.fill = fill_filtered
                    cell.font = font_gray
            elif col_idx == 16:
                if val == 0.0 or val_str == "0.0" or val_str == "0":
                    cell.fill = fill_filtered
                    cell.font = font_gray

    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            v_len = len(str(cell.value or ''))
            for emoji in ["🏠", "✈️", "✅", "❌", "🔴", "⏳"]:
                if emoji in str(cell.value or ''):
                    v_len += 1
            if v_len > max_len:
                max_len = v_len
        ws.column_dimensions[col_letter].width = max(max_len + 3, 10)

    import time
    ts = int(time.time())
    filename = f"Reporte_Mensual_{year_month}_{ts}.xlsx"
    file_path = ROOT / filename
    wb.save(file_path)

    open_msg = ""
    try:
        os.startfile(file_path)
        open_msg = " y abierto automáticamente"
    except Exception as e:
        open_msg = f" (no se pudo abrir automáticamente: {e})"

    return f"{GREEN}[ÉXITO]{RESET} Excel mensual generado{open_msg} en: {file_path}\n   {BOLD}Hoja única:{RESET} Mes {year_month} ({len(matches)} partidos)"

def export_matches_to_text_ai(conn: sqlite3.Connection, date_str: str = None, year_month: str = None) -> str:
    """Exporta partidos a .txt compacto (pipe-delimited) para IA."""
    if year_month:
        cursor = conn.execute("""
            SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league,
                            s.scheduled_utc_ts, s.event_date
            FROM bet_monitor_schedule_v2 s
            JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
            WHERE s.event_date LIKE ?
            ORDER BY s.event_date ASC, s.scheduled_utc_ts ASC
        """, (f"{year_month}%",))
        label = year_month
        prefix = "Mensual"
    else:
        cursor = conn.execute("""
            SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league,
                            s.scheduled_utc_ts, s.event_date
            FROM bet_monitor_schedule_v2 s
            JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
            WHERE s.event_date = ?
            ORDER BY s.scheduled_utc_ts ASC
        """, (date_str,))
        label = date_str
        prefix = "Diario"

    matches = cursor.fetchall()
    if not matches:
        return f"{YELLOW}No se encontraron partidos para {label}{RESET}"

    engine = FusionConsensusEngine()

    def side_code(s):
        return "H" if s == "HOME" else ("A" if s == "AWAY" else "?")

    def res_code(r):
        return {"win": "W", "hit": "W", "loss": "L", "miss": "L", "pending": "P", "push": "D"}.get(r, "?")

    def fmt_model_code(log, deduped):
        if not log:
            return "X"
        sig = log["signal_type"] or ""
        picked = log["picked_side"] or ""
        conf = log["confidence"] or 0.0
        res = log["result"] or ""
        if "NO_BET" in sig or "BET" not in sig:
            return "N"
        c = int(round(conf * 100 if conf <= 1.0 else conf))
        return f"{side_code(picked)}{c}{res_code(res)}"

    lines = []
    lines.append(f"# {prefix} {label} | {len(matches)} matches")
    lines.append("# date|time|league|home_away|score|v6|m27|fusion|borrego")
    lines.append("# v6/m27: sid=PICK+CONF+RES (H=HOME,A=AWAY,N=NO BET,X=N/A) (W=win,L=loss,P=pending)")
    lines.append("# fusion: sid+PICK+TIER+STAKE+RES (F=FILTERED,X=N/A)")
    lines.append("# borrego: sid+PICK+RES (N=NO BET,X=N/A)")

    for idx, m in enumerate(matches, 1):
        mid = m["match_id"]
        home = m["home_team"]
        away = m["away_team"]
        league = m["league"]
        sched_ts = m["scheduled_utc_ts"]
        event_date = m["event_date"]
        sched_time = datetime.fromtimestamp(sched_ts, tz=timezone(timedelta(hours=-6))).strftime("%H:%M")

        qs = conn.execute("""
            SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
            FROM quarter_scores_v2 WHERE match_id = ?
        """, (mid,)).fetchone()

        score_str = "?"
        if qs and qs["q4_home"] is not None and qs["q4_away"] is not None:
            q4h = qs["q4_home"]
            q4a = qs["q4_away"]
            if all(qs[c] is not None for c in ["q1_home", "q1_away", "q2_home", "q2_away", "q3_home", "q3_away"]):
                tot_h = qs["q1_home"] + qs["q2_home"] + qs["q3_home"] + q4h
                tot_a = qs["q1_away"] + qs["q2_away"] + qs["q3_away"] + q4a
                score_str = f"{tot_h}-{tot_a}({q4h}-{q4a})"
            else:
                score_str = f"Q4:{q4h}-{q4a}"

        logs = conn.execute("""
            SELECT model_version, signal_type, picked_side, confidence, result, inference_minute
            FROM bet_monitor_log_v2 WHERE match_id = ? ORDER BY id ASC
        """, (mid,)).fetchall()
        deduped = {l["model_version"]: l for l in logs}

        v6_code = fmt_model_code(deduped.get("v6_2"), deduped)
        m27_code = fmt_model_code(deduped.get("m27_v3"), deduped)

        # Fusion
        v6_p = Prediction("v6_2", "NO BET", 0)
        m27_p = Prediction("m27_v3", "NO BET", 0)
        for k, p in [("v6_2", v6_p), ("m27_v3", m27_p)]:
            log = deduped.get(k)
            if log:
                sig = log["signal_type"] or ""
                pick = log["picked_side"] or ""
                conf = log["confidence"] or 0.0
                if conf <= 1.0:
                    conf *= 100.0
                if "BET" in sig and "NO_BET" not in sig and pick in ("HOME", "AWAY"):
                    p.pick = pick
                    p.confidence = conf
        if not deduped.get("v6_2") and not deduped.get("m27_v3"):
            fusion_code = "X"
        else:
            r = engine.evaluate(v6_p, m27_p)
            if r.final_pick == "NO BET" or r.recommended_stake == 0:
                fusion_code = "F"
            else:
                actual = get_actual_winner(qs, deduped)
                res = "P"
                if actual:
                    res = res_code("win" if r.final_pick == actual else "loss")
                fusion_code = f"{side_code(r.final_pick)}{r.tier}{r.recommended_stake}{res}"

        # Borrego
        if not deduped.get("v6_2") and not deduped.get("m27_v3"):
            borrego_code = "X"
        else:
            be = FusionBorregoEngine()
            bp = be.evaluate(v6_p, m27_p)
            if bp == "NO BET":
                borrego_code = "N"
            else:
                actual = get_actual_winner(qs, deduped)
                res = "P"
                if actual:
                    res = res_code("win" if bp == actual else "loss")
                borrego_code = f"{side_code(bp)}{res}"

        lines.append(f"{event_date}|{sched_time}|{league}|{home} vs {away}|{score_str}|{v6_code}|{m27_code}|{fusion_code}|{borrego_code}")

    import time
    ts = int(time.time())
    safe_label = label.replace("-", "")
    filename = f"Reporte_{prefix}_{safe_label}_{ts}.txt"
    file_path = ROOT / filename

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    open_msg = ""
    try:
        os.startfile(file_path)
        open_msg = " y abierto automáticamente"
    except Exception as e:
        open_msg = f" (no se pudo abrir automáticamente: {e})"

    return f"{GREEN}[ÉXITO]{RESET} TXT generado{open_msg} en: {file_path}\n   {BOLD}{len(matches)} partidos{RESET} en formato compacto"


def run_meta_model(conn: sqlite3.Connection, date_str: str = None, year_month: str = None,
                   _out_rows: list = None):
    """Meta-model: combina v6_2 + m27_v3 con filtro por liga y playoffs."""
    print(f"\n{BOLD}{CYAN}========================================================================================================================{RESET}")
    print(f"{BOLD}{CYAN}                                 META-MODEL V2: v6_2 + m27_v3 INTELIGENTE{RESET}")
    print(f"{BOLD}{CYAN}========================================================================================================================{RESET}\n")

    # 1. Fetch ALL historical data per league
    cur = conn.execute("""
        SELECT l.model_version, s.league, l.result, l.confidence, l.picked_side, l.signal_type, s.event_date
        FROM bet_monitor_log_v2 l
        JOIN bet_monitor_schedule_v2 s ON l.match_id = s.match_id
        WHERE l.result IN ('win','hit','loss','miss')
        ORDER BY s.event_date ASC
    """)
    rows = cur.fetchall()

    # Classify playoffs
    playoff_kw = ['playoff', 'play-off', 'play off', 'knockout', 'final', 'cuarto', 'semifinal',
                  'postseason', 'post-season', 'ronda', 'cup', 'coppa', 'pokal',
                  'championship', 'promovare', 'relegation', 'playout', 'placement', 'classifica']
    womens_kw = ['women', 'femen', 'femin', 'mujeres', 'dames', 'damen', 'lbf', 'lfb', 'lnbf', 'wnba']
    youth_kw = ['u18', 'u19', 'u20', 'u21', 'u17', 'u16', 'u14', 'u13', 'u9',
                'junior', 'juniors', 'kadet', 'pretkad', 'youth', 'next generation',
                'mini ko', 'espoir', 'dječak', 'djevoj']
    format_kw_r = ['regular season', 'temporada regular', 'main round', 'fase regular']
    format_kw_e = ['relegation', 'playout', 'spareggio']
    format_kw_p = ['placement', 'klasman', '3rd place', '3e place', 'classification', '5-8', '5-7', '5e', 'places']
    format_kw_g = ['group', 'girone', 'grupa', 'grupo', 'fase de classifica', 'round robin']
    format_kw_k = ['knockout', 'quarterfinal', 'cuartos', 'semifinal', 'semi-final',
                   'playoff', 'play-off', 'ronda', 'doigravanje', 'završni turnir', 'zavrsni turnir',
                   'promovare', 'promotion']
    format_kw_f = ['final', 'finale', 'finali', 'championship', 'copa', 'coppa', 'pokal']
    def get_round_type(league):
        ll = league.lower()
        if any(k in ll for k in format_kw_r): return 'R'
        if any(k in ll for k in format_kw_e): return 'E'
        if any(k in ll for k in format_kw_p): return 'P'
        if any(k in ll for k in format_kw_g): return 'G'
        if any(k in ll for k in format_kw_k): return 'K'
        if any(k in ll for k in format_kw_f):
            if 'qualifier' not in ll: return 'F'
        return '-'
    def is_womens_league(league):
        ll = league.lower()
        if any(k in ll for k in womens_kw):
            return True
        if any(part == '\u017d' for part in league.split()):
            return True
        return False
    def is_youth_league(league):
        ll = league.lower()
        return any(k in ll for k in youth_kw)

    twelve_min_kw = [
        "nba", "nbl", "pba", "cba", "euroleague", "b1 league", "b2 league",
        "germany bbl", "liga acb", "bnxt", "france pro a", "betsafe-lkl",
        "lkl", "poland 1st", "bulgaria nbl",
    ]
    def is_twelve_min_league(league):
        ll = league.lower()
        if "wnba" in ll: return False
        return any(k in ll for k in twelve_min_kw)

    league_data = {}
    total_v6_w = total_v6_l = 0
    total_m27_w = total_m27_l = 0
    po_v6_w = po_v6_l = 0
    po_m27_w = po_m27_l = 0

    for r in rows:
        model = r["model_version"]
        if model not in ("v6_2", "m27_v3"):
            continue
        league = (r["league"] or "Desconocida").strip()
        res = r["result"] or ""
        is_win = res in ("win", "hit")
        is_loss = res in ("loss", "miss")
        if not (is_win or is_loss):
            continue
        is_po = any(kw in league.lower() for kw in playoff_kw)
        key = (model, league, "playoff" if is_po else "regular")
        if key not in league_data:
            league_data[key] = {"w": 0, "l": 0}
        if is_win:
            league_data[key]["w"] += 1
        else:
            league_data[key]["l"] += 1
        if model == "v6_2":
            total_v6_w += is_win; total_v6_l += is_loss
            if is_po: po_v6_w += is_win; po_v6_l += is_loss
        else:
            total_m27_w += is_win; total_m27_l += is_loss
            if is_po: po_m27_w += is_win; po_m27_l += is_loss

    # Aggregate per league (regular season only - cleaner for meta)
    model_short = {"v6_2": "v6", "m27_v3": "m27"}
    league_agg = {}
    for (model, league, phase), d in league_data.items():
        short = model_short.get(model, model)
        if league not in league_agg:
            league_agg[league] = {"v6_w": 0, "v6_l": 0, "m27_w": 0, "m27_l": 0,
                                  "v6_po_w": 0, "v6_po_l": 0, "m27_po_w": 0, "m27_po_l": 0}
        if phase == "playoff":
            league_agg[league][f"{short}_po_w"] = d["w"]
            league_agg[league][f"{short}_po_l"] = d["l"]
        else:
            league_agg[league][f"{short}_w"] = d["w"]
            league_agg[league][f"{short}_l"] = d["l"]

    # Classify leagues
    good_leagues = []
    weak_leagues = []
    no_data_leagues = []

    for league, d in sorted(league_agg.items(), key=lambda x: x[1]["v6_w"] + x[1]["v6_l"] + x[1]["m27_w"] + x[1]["m27_l"], reverse=True):
        v6_t = d["v6_w"] + d["v6_l"]
        m27_t = d["m27_w"] + d["m27_l"]
        v6_wr = d["v6_w"] / v6_t * 100 if v6_t else None
        m27_wr = d["m27_w"] / m27_t * 100 if m27_t else None
        total = v6_t + m27_t
        # Combined WR weighted by volume
        combined_w = d["v6_w"] + d["m27_w"]
        combined_t = v6_t + m27_t
        combined_wr = combined_w / combined_t * 100 if combined_t else None
        playoff_w = d["v6_po_w"] + d["m27_po_w"]
        playoff_l = d["v6_po_l"] + d["m27_po_l"]
        playoff_t = playoff_w + playoff_l
        playoff_wr = playoff_w / playoff_t * 100 if playoff_t else None

        league_info = {
            "league": league, "v6_t": v6_t, "v6_wr": v6_wr,
            "m27_t": m27_t, "m27_wr": m27_wr,
            "combined_wr": combined_wr, "combined_t": combined_t,
            "playoff_wr": playoff_wr, "playoff_t": playoff_t,
        }

        if total < 5:
            no_data_leagues.append(league_info)
        elif combined_wr is not None and combined_wr < 45:
            weak_leagues.append(league_info)
        elif combined_wr is not None and combined_wr >= 50:
            good_leagues.append(league_info)
        else:
            weak_leagues.append(league_info)

    # Print league analysis
    def fmt_wr(wr, t):
        if wr is None or t == 0:
            return f"{YELLOW}N/A{RESET}"
        color = GREEN if wr >= 55 else (YELLOW if wr >= 45 else RED)
        return f"{color}{wr:.0f}%{RESET} ({t})"

    print(f"  {BOLD}LIGAS CONFIABLES (WR >= 50%, n >= 5):{RESET}")
    for lg in good_leagues[:15]:
        v6_s = fmt_wr(lg["v6_wr"], lg["v6_t"])
        m27_s = fmt_wr(lg["m27_wr"], lg["m27_t"])
        meta_s = fmt_wr(lg["combined_wr"], lg["combined_t"])
        po_s = fmt_wr(lg["playoff_wr"], lg["playoff_t"])
        name = lg["league"][:45]
        print(f"    {name:<47s} v6={v6_s}  m27={m27_s}  META={meta_s}  PO={po_s}")

    print(f"\n  {BOLD}LIGAS DÉBILES / OVERFITTING (WR < 50% o n < 10):{RESET}")
    for lg in (weak_leagues + no_data_leagues)[:10]:
        v6_s = fmt_wr(lg["v6_wr"], lg["v6_t"])
        m27_s = fmt_wr(lg["m27_wr"], lg["m27_t"])
        meta_s = fmt_wr(lg["combined_wr"], lg["combined_t"])
        po_s = fmt_wr(lg["playoff_wr"], lg["playoff_t"])
        name = lg["league"][:45]
        print(f"    {name:<47s} v6={v6_s}  m27={m27_s}  META={meta_s}  PO={po_s}")

    if len(weak_leagues) + len(no_data_leagues) > 10:
        print(f"    ... y {len(weak_leagues) + len(no_data_leagues) - 10} mas")

    # Global playoff stats
    print(f"\n  {BOLD}RENDIMIENTO EN PLAYOFFS (global):{RESET}")
    po_v6_t = po_v6_w + po_v6_l
    po_m27_t = po_m27_w + po_m27_l
    if po_v6_t:
        print(f"    v6_2:   {GREEN}W={po_v6_w}{RESET} {RED}L={po_v6_l}{RESET} WR={po_v6_w/po_v6_t*100:.1f}% (n={po_v6_t})")
    if po_m27_t:
        print(f"    m27_v3: {GREEN}W={po_m27_w}{RESET} {RED}L={po_m27_l}{RESET} WR={po_m27_w/po_m27_t*100:.1f}% (n={po_m27_t})")

    # Global regular season stats
    re_v6_t = total_v6_w + total_v6_l - po_v6_t
    re_m27_t = total_m27_w + total_m27_l - po_m27_t
    re_v6_w = total_v6_w - po_v6_w
    re_m27_w = total_m27_w - po_m27_w
    re_v6_l = total_v6_l - po_v6_l
    re_m27_l = total_m27_l - po_m27_l
    print(f"\n  {BOLD}RENDIMIENTO TEMPORADA REGULAR:{RESET}")
    if re_v6_t:
        print(f"    v6_2:   {GREEN}W={re_v6_w}{RESET} {RED}L={re_v6_l}{RESET} WR={re_v6_w/re_v6_t*100:.1f}% (n={re_v6_t})")
    if re_m27_t:
        print(f"    m27_v3: {GREEN}W={re_m27_w}{RESET} {RED}L={re_m27_l}{RESET} WR={re_m27_w/re_m27_t*100:.1f}% (n={re_m27_t})")

    # Build set of rejected leagues
    rejected = {lg["league"] for lg in weak_leagues + no_data_leagues}

    # 2. Show meta-model picks for a date, month, or all history
    all_history = date_str and date_str.upper() == "ALL"
    if date_str or year_month:
        if all_history:
            cur2 = conn.execute("""
                SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league,
                                s.scheduled_utc_ts, s.event_date
                FROM bet_monitor_schedule_v2 s
                JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
                ORDER BY s.event_date ASC, s.scheduled_utc_ts ASC
            """)
            label = "HISTORIAL COMPLETO"
        elif year_month:
            cur2 = conn.execute("""
                SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league,
                                s.scheduled_utc_ts, s.event_date
                FROM bet_monitor_schedule_v2 s
                JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
                WHERE s.event_date LIKE ?
                ORDER BY s.event_date ASC, s.scheduled_utc_ts ASC
            """, (f"{year_month}%",))
        else:
            cur2 = conn.execute("""
                SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league,
                                s.scheduled_utc_ts, s.event_date
                FROM bet_monitor_schedule_v2 s
                JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
                WHERE s.event_date = ?
                ORDER BY s.scheduled_utc_ts ASC
            """, (date_str,))
        matches = cur2.fetchall()
        if not matches:
            print(f"\n  {YELLOW}Sin partidos para el periodo seleccionado.{RESET}")
            return

        label = year_month if year_month else date_str
        print(f"\n  {BOLD}━━━ META-MODEL PICKS: {label} ━━━{RESET}\n")

        col_w = [16, 28, 18, 22, 9]
        sep = " ┃ "
        hdr = "  " + sep.join(f"{h:^{w}}" for h, w in zip(["Hora", "Partido", "Liga", "v6/m27", "META"], col_w))
        line_w = 4 + sum(col_w) + len(sep) * (len(col_w) - 1)
        print(f"  {'━' * line_w}")
        print(hdr)
        print(f"  {'━' * line_w}")

        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        def strip_a(s):
            return ansi_escape.sub('', s).replace("\ufe0f", "")

        def pad(text, width):
            v = visual_len(text)
            return text + " " * max(0, width - v)

        accepted = 0
        rejected_count = 0
        meta_wins = 0
        meta_losses = 0
        qlen_meta = {"10m": {"a": 0, "r": 0, "w": 0, "l": 0},
                     "12m": {"a": 0, "r": 0, "w": 0, "l": 0}}

        for m in matches:
            mid = m["match_id"]
            home = m["home_team"]
            away = m["away_team"]
            league = m["league"]
            sched_ts = m["scheduled_utc_ts"]
            event_date = m["event_date"]
            sched_time = datetime.fromtimestamp(sched_ts, tz=timezone(timedelta(hours=-6))).strftime("%H:%M")
            is_po = any(kw in league.lower() for kw in playoff_kw)
            is_w = is_womens_league(league)
            is_y = is_youth_league(league)
            round_type = get_round_type(league)
            is_12m = is_twelve_min_league(league)

            qs = conn.execute("""SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
                FROM quarter_scores_v2 WHERE match_id = ?""", (mid,)).fetchone()

            logs = conn.execute("""SELECT model_version, signal_type, picked_side, confidence, result, inference_minute
                FROM bet_monitor_log_v2 WHERE match_id = ? ORDER BY id ASC""", (mid,)).fetchall()
            deduped = {l["model_version"]: l for l in logs}

            def extract(model_key):
                log = deduped.get(model_key)
                if not log:
                    return None, None, None, None
                sig = log["signal_type"] or ""
                picked = log["picked_side"] or ""
                conf = log["confidence"] or 0.0
                raw_res = log["result"] or ""
                minute = log["inference_minute"]
                if "NO_BET" in sig or "BET" not in sig:
                    return None, None, None, None
                if conf <= 1.0:
                    conf = conf * 100.0
                if raw_res in ("win", "hit"):
                    res = "W"
                elif raw_res in ("loss", "miss", "push"):
                    res = "L"
                else:
                    res = None
                return picked, int(round(conf)), res, minute

            v6_pick, v6_conf, v6_res, v6_min = extract("v6_2")
            m27_pick, m27_conf, m27_res, m27_min = extract("m27_v3")
            actual_winner = get_actual_winner(qs, deduped)

            # ── MetaV2 decision ──
            base_league = league.split(",")[0].strip()
            league_rejected = base_league in rejected or any(r in league for r in rejected)
            meta_side = None
            meta_tier = ""
            status = "skip"

            v6c = v6_conf if v6_conf is not None else 0
            m27c = m27_conf if m27_conf is not None else 0
            both_active = v6_pick is not None and m27_pick is not None
            same_side = both_active and v6_pick == m27_pick
            min_conf = min(v6c, m27c) if both_active else 0
            max_conf = max(v6c, m27c) if both_active else 0

            if not v6_pick and not m27_pick:
                status = "skip"

            elif round_type == 'F':
                status = "rejected"
                meta_tier = "BLK-F"

            elif round_type == 'E':
                status = "rejected"
                meta_tier = "BLK-E"

            elif league_rejected:
                status = "rejected"
                meta_tier = "BLK-LG"

            elif both_active and same_side and v6c >= 50 and m27c >= 50:
                meta_side = v6_pick
                status = "accepted"
                meta_tier = "S"

            elif both_active and same_side and min_conf >= 30:
                meta_side = v6_pick
                status = "accepted"
                meta_tier = "A"

            elif both_active and same_side:
                if is_w:
                    meta_side = m27_pick
                else:
                    meta_side = v6_pick if v6c >= m27c else m27_pick
                status = "accepted"
                meta_tier = "B"

            elif both_active and not same_side:
                toxic_away_low = (v6_pick == "AWAY" and m27_pick == "AWAY"
                                  and v6c < 30 and m27c < 50)
                toxic_both_low = (min_conf < 30 and max_conf < 30)
                if toxic_both_low:
                    status = "rejected"
                    meta_tier = "BLK-TOX"
                else:
                    if is_w:
                        w_v6, w_m27 = 0.30, 0.70
                    else:
                        w_v6, w_m27 = 0.45, 0.55
                    v6_score = w_v6 * v6c
                    m27_score = w_m27 * m27c
                    if v6_pick == "HOME" and m27_pick == "AWAY" and v6c >= 30 and m27c >= 30:
                        meta_side = "HOME"
                        meta_tier = "C"
                    elif v6c >= 50 and v6c - m27c >= 30:
                        meta_side = v6_pick
                        meta_tier = "C"
                    elif m27c >= 50 and m27c - v6c >= 30:
                        meta_side = m27_pick
                        meta_tier = "C"
                    elif v6_score >= m27_score:
                        meta_side = v6_pick
                        meta_tier = "C"
                    else:
                        meta_side = m27_pick
                        meta_tier = "C"
                    status = "accepted"

            elif v6_pick and not m27_pick:
                if v6c >= 30:
                    meta_side = v6_pick
                    status = "accepted"
                    meta_tier = "B-solo"
                else:
                    status = "rejected"
                    meta_tier = "BLK-LO"

            elif m27_pick and not v6_pick:
                if is_w or m27c >= 30:
                    meta_side = m27_pick
                    status = "accepted"
                    meta_tier = "B-solo"
                else:
                    status = "rejected"
                    meta_tier = "BLK-LO"

            if is_y and status == "accepted" and meta_tier not in ("S",):
                meta_tier = meta_tier + "-Y"

            qk = "12m" if is_12m else "10m"
            if status == "accepted":
                accepted += 1
                qlen_meta[qk]["a"] += 1
            elif status == "rejected":
                rejected_count += 1
                qlen_meta[qk]["r"] += 1

            # Time tags (fixed 9-char: PO + W + Y + format)
            rt_colors = {'R': '', 'G': f'{BLUE}', 'K': f'{YELLOW}', 'F': f'{GREEN}',
                          'E': f'{RED}', 'P': f'{CYAN}', '-': '', '': ''}
            rt_c = rt_colors.get(round_type, '')
            po_part = f"{YELLOW}[PO]{RESET}" if is_po else "    "
            w_part = f"{MAGENTA}[W]{RESET}" if is_w else "   "
            y_part = f"{CYAN}Y{RESET}" if is_y else " "
            f_part = f"{rt_c}{round_type}{RESET}" if round_type not in ('-', '') else "-"
            qm_part = f"{BLUE}[12]{RESET}" if is_12m else "    "
            tags = f"{po_part}{w_part}{y_part}{f_part}{qm_part}"
            if year_month:
                time_str = f"{tags} {event_date[-5:]} {sched_time}"
            else:
                time_str = f"{tags} {sched_time}"

            # Match column
            match_str = f"{home[:18]} vs {away[:18]}"
            if visual_len(match_str) > 26:
                match_str = match_str[:23] + ".."
            # League column
            lg_short = league.split(",")[0].strip()
            if visual_len(lg_short) > 18:
                lg_short = lg_short[:17] + "."

            # v6 / m27 column with colored side + confidence + result + minute
            def fmt_pick(side, conf, res, minute=None):
                if side is None:
                    return f"{RED}—{RESET}"
                side_color = f"{CYAN}" if side == "HOME" else f"{MAGENTA}"
                conf_color = f"{RED}" if conf < 30 else (f"{YELLOW}" if conf < 50 else f"{GREEN}")
                res_str = ""
                if res:
                    res_color = f"{GREEN}" if res == "W" else f"{RED}"
                    res_str = f"{res_color}{res}{RESET}"
                min_str = ""
                if minute is not None:
                    min_color = f"{RED}" if minute < 30 else (f"{YELLOW}" if minute < 34 else f"{GREEN}")
                    min_str = f"@{min_color}{minute}{RESET}"
                return f"{side_color}{side[0]}{RESET}{conf_color}{conf}{RESET}{res_str}{min_str}"
            v6_code = fmt_pick(v6_pick, v6_conf, v6_res, v6_min)
            m27_code = fmt_pick(m27_pick, m27_conf, m27_res, m27_min)
            v6m27_str = f"v6:{v6_code}  m27:{m27_code}"

            # META column (MetaV2 with tier)
            if status == "skip":
                meta_str = f"{RED}—{RESET}"
            elif status == "rejected":
                blk_label = meta_tier if meta_tier else "BLK"
                meta_str = f"{RED}🚫{RESET}"
            elif meta_side:
                emoji = "🏠" if meta_side == "HOME" else "✈️"
                tier_display = ""
                if meta_tier:
                    tier_colors = {
                        "S": f"{GREEN}", "A": f"{GREEN}", "B": f"{YELLOW}",
                        "C": f"{CYAN}", "B-solo": f"{YELLOW}",
                    }
                    base_tier = meta_tier.split("-")[0]
                    tc = tier_colors.get(meta_tier, tier_colors.get(base_tier, f"{YELLOW}"))
                    tier_display = f" {tc}{meta_tier}{RESET}"
                res_emoji = ""
                if actual_winner:
                    if meta_side == actual_winner:
                        res_emoji = f"{GREEN}✅{RESET}"
                        meta_wins += 1
                        qlen_meta[qk]["w"] += 1
                    else:
                        res_emoji = f"{RED}❌{RESET}"
                        meta_losses += 1
                        qlen_meta[qk]["l"] += 1
                else:
                    res_emoji = f"{YELLOW}⏳{RESET}"
                meta_str = f"{emoji}{tier_display} {res_emoji}"
            else:
                meta_str = "?"

            row = sep.join([pad(time_str, col_w[0]),
                            pad(match_str, col_w[1]),
                            pad(lg_short, col_w[2]),
                            pad(v6m27_str, col_w[3]),
                            pad(meta_str, col_w[4])])
            print(f"  {row}")
            if _out_rows is not None:
                def sc(s):
                    return "H" if s == "HOME" else ("A" if s == "AWAY" else "?")
                def s_or_d(v):
                    return str(v) if v is not None else "-"
                tag_str = f"{'PO' if is_po else '--'},{'W' if is_w else '-'},{'Y' if is_y else '-'},{round_type},{'12m' if is_12m else '10m'}"
                v6s = f"{sc(v6_pick) if v6_pick else '-'}{v6_conf if v6_pick else '-'}{v6_res or '-'}{f'@{v6_min}' if v6_min is not None else '-'}"
                m27s = f"{sc(m27_pick) if m27_pick else '-'}{m27_conf if m27_pick else '-'}{m27_res or '-'}{f'@{m27_min}' if m27_min is not None else '-'}"
                meta_s = sc(meta_side) if meta_side else "-"
                meta_w = sc(actual_winner) if actual_winner else "-"
                status_c = "A" if status == "accepted" else ("R" if status == "rejected" else "S")
                tier_s = meta_tier if meta_tier else "-"
                _out_rows.append(f"{event_date}|{sched_time}|{home}|{away}|{league}|{tag_str}|{v6s}|{m27s}|{status_c}|{meta_s}|{meta_w}|{tier_s}")

        print(f"  {'━' * line_w}")

        # ── Combination stats ──
        print(f"\n  {BOLD}━━━ COMBINACIONES MODELOS ── HISTORIAL ── PREDICTIVO ━━━{RESET}\n")

        def conf_tier(conf):
            c = conf * 100.0 if conf is not None and conf <= 1.0 else (conf or 0)
            if c >= 50: return ">50"
            if c >= 30: return "30-50"
            return "<30"

        combo_data = conn.execute("""
            SELECT l1.picked_side AS v6s, l1.confidence AS v6c, l1.result AS v6r,
                   l2.picked_side AS m27s, l2.confidence AS m27c, l2.result AS m27r
            FROM bet_monitor_log_v2 l1
            JOIN bet_monitor_log_v2 l2 ON l1.match_id = l2.match_id
            WHERE l1.model_version = 'v6_2'
              AND l2.model_version = 'm27_v3'
              AND l1.signal_type LIKE '%BET%' AND l1.signal_type NOT LIKE '%NO_BET%'
              AND l2.signal_type LIKE '%BET%' AND l2.signal_type NOT LIKE '%NO_BET%'
        """).fetchall()

        combos = {}
        for r in combo_data:
            v6s, v6c, v6r, m27s, m27c, m27r = r
            if v6s not in ("HOME", "AWAY") or m27s not in ("HOME", "AWAY"):
                continue
            vk = f"{v6s[0]}{conf_tier(v6c)}"
            mk = f"{m27s[0]}{conf_tier(m27c)}"
            key = f"v6:{vk}  m27:{mk}"
            if key not in combos:
                combos[key] = {"v6w": 0, "v6l": 0, "m27w": 0, "m27l": 0, "n": 0}
            combos[key]["n"] += 1
            if v6r in ("win", "hit"):
                combos[key]["v6w"] += 1
            else:
                combos[key]["v6l"] += 1
            if m27r in ("win", "hit"):
                combos[key]["m27w"] += 1
            else:
                combos[key]["m27l"] += 1

        sorted_c = sorted(combos.items(), key=lambda x: -x[1]["n"])
        n_shown = 0
        for key, st in sorted_c:
            total = st["v6w"] + st["v6l"]
            if total < 2:
                continue
            if n_shown >= 20:
                extra = sum(1 for _, s in sorted_c if s["v6w"] + s["v6l"] >= 2) - 20
                if extra > 0:
                    print(f"  ... y {extra} combos mas")
                break
            n_shown += 1
            v6_wr = st["v6w"] / total * 100
            m27_wr = st["m27w"] / total * 100
            if st["v6w"] == st["m27w"] and st["v6l"] == st["m27l"]:
                clr = GREEN if v6_wr >= 60 else (YELLOW if v6_wr >= 40 else RED)
                print(f"  {key}  {GREEN}{st['v6w']}W{RESET} {RED}{st['v6l']}L{RESET} {clr}({v6_wr:.0f}%){RESET}  n={total}")
            else:
                v6_c = GREEN if v6_wr >= 60 else (YELLOW if v6_wr >= 40 else RED)
                m27_c = GREEN if m27_wr >= 60 else (YELLOW if m27_wr >= 40 else RED)
                print(f"  {key}  v6:{v6_c}{st['v6w']}W{RESET}{RED}{st['v6l']}L{RESET} {v6_c}({v6_wr:.0f}%){RESET}  "
                      f"m27:{m27_c}{st['m27w']}W{RESET}{RED}{st['m27l']}L{RESET} {m27_c}({m27_wr:.0f}%){RESET}  n={total}")

        # ── Performance by context (youth / women / regular) ──
        ctx_rows = conn.execute("""
            SELECT l1.result AS v6r, l2.result AS m27r, s.league
            FROM bet_monitor_log_v2 l1
            JOIN bet_monitor_log_v2 l2 ON l1.match_id = l2.match_id
            JOIN bet_monitor_schedule_v2 s ON l1.match_id = s.match_id
            WHERE l1.model_version = 'v6_2' AND l2.model_version = 'm27_v3'
              AND l1.signal_type LIKE '%BET%' AND l1.signal_type NOT LIKE '%NO_BET%'
              AND l2.signal_type LIKE '%BET%' AND l2.signal_type NOT LIKE '%NO_BET%'
        """).fetchall()

        ctx = {"regular": [0,0,0,0], "women": [0,0,0,0],
               "youth": [0,0,0,0], "women_youth": [0,0,0,0]}
        ctx_lbl = {"regular": "Regular", "women": "Solo Mujeres",
                   "youth": "Solo Jóvenes", "women_youth": "Mujeres+Jóvenes"}
        for v6r, m27r, league in ctx_rows:
            iw = is_womens_league(league)
            iy = is_youth_league(league)
            if iw and iy:
                bucket = "women_youth"
            elif iw:
                bucket = "women"
            elif iy:
                bucket = "youth"
            else:
                bucket = "regular"
            s = ctx[bucket]
            if v6r in ("win", "hit"): s[0] += 1
            else: s[1] += 1
            if m27r in ("win", "hit"): s[2] += 1
            else: s[3] += 1

        print(f"\n  {BOLD}━━━ RENDIMIENTO POR CONTEXTO ━━━{RESET}\n")
        print(f"  {'Categoría':<20s} {'v6':>12s} {'m27':>12s}")
        print(f"  {'─'*44}")
        for bucket in ["regular", "youth", "women", "women_youth"]:
            s = ctx[bucket]
            total = s[0]+s[1]
            if total == 0: continue
            v6_wr = s[0]/total*100
            m27_wr = s[2]/total*100
            v6_c = GREEN if v6_wr >= 55 else (YELLOW if v6_wr >= 45 else RED)
            m27_c = GREEN if m27_wr >= 55 else (YELLOW if m27_wr >= 45 else RED)
            n = total
            print(f"  {ctx_lbl[bucket]:<20s} {v6_c}{s[0]:>2d}W{RESET}{RED}{s[1]:>2d}L{RESET} {v6_c}({v6_wr:.0f}%){RESET}  "
                  f"{m27_c}{s[2]:>2d}W{RESET}{RED}{s[3]:>2d}L{RESET} {m27_c}({m27_wr:.0f}%){RESET}  n={n}")

        # ── Performance by format ──
        fmt_stats = {"R": [0,0,0,0], "G": [0,0,0,0], "K": [0,0,0,0],
                     "F": [0,0,0,0], "E": [0,0,0,0], "P": [0,0,0,0], "-": [0,0,0,0]}
        fmt_lbl = {"R": "Regular", "G": "Group", "K": "Knockout", "F": "Final",
                   "E": "Relegación", "P": "Placement", "-": "Otro"}
        for v6r, m27r, league in ctx_rows:
            rt = get_round_type(league)
            s = fmt_stats.get(rt, fmt_stats["-"])
            if v6r in ("win", "hit"): s[0] += 1
            else: s[1] += 1
            if m27r in ("win", "hit"): s[2] += 1
            else: s[3] += 1

        # ── Performance by quarter length (10m vs 12m) ──
        qlen_stats = {"10m": [0,0,0,0], "12m": [0,0,0,0]}
        for v6r, m27r, league in ctx_rows:
            is_12m = is_twelve_min_league(league)
            bucket = "12m" if is_12m else "10m"
            s = qlen_stats[bucket]
            if v6r in ("win", "hit"): s[0] += 1
            else: s[1] += 1
            if m27r in ("win", "hit"): s[2] += 1
            else: s[3] += 1

        print(f"\n  {BOLD}━━━ RENDIMIENTO POR DURACIÓN DE CUARTO ━━━{RESET}\n")
        print(f"  {'Cuarto':<10s} {'v6':>12s} {'m27':>12s}")
        print(f"  {'─'*34}")
        for bucket in ["10m", "12m"]:
            s = qlen_stats[bucket]
            total = s[0]+s[1]
            if total == 0: continue
            v6_wr = s[0]/total*100
            m27_wr = s[2]/total*100
            v6_c = GREEN if v6_wr >= 55 else (YELLOW if v6_wr >= 45 else RED)
            m27_c = GREEN if m27_wr >= 55 else (YELLOW if m27_wr >= 45 else RED)
            print(f"  {bucket+' min':<10s} {v6_c}{s[0]:>3d}W{RESET}{RED}{s[1]:>3d}L{RESET} {v6_c}({v6_wr:.0f}%){RESET}  "
                  f"{m27_c}{s[2]:>3d}W{RESET}{RED}{s[3]:>3d}L{RESET} {m27_c}({m27_wr:.0f}%){RESET}  n={total}")

        # ── Performance by format ──
        print(f"\n  {BOLD}━━━ RENDIMIENTO POR FORMATO ━━━{RESET}\n")
        print(f"  {'Formato':<14s} {'v6':>12s} {'m27':>12s}")
        print(f"  {'─'*38}")
        for rt in ["R", "G", "K", "F", "E", "P", "-"]:
            s = fmt_stats[rt]
            total = s[0]+s[1]
            if total == 0: continue
            v6_wr = s[0]/total*100
            m27_wr = s[2]/total*100
            v6_c = GREEN if v6_wr >= 55 else (YELLOW if v6_wr >= 45 else RED)
            m27_c = GREEN if m27_wr >= 55 else (YELLOW if m27_wr >= 45 else RED)
            print(f"  {fmt_lbl[rt]:<14s} {v6_c}{s[0]:>3d}W{RESET}{RED}{s[1]:>3d}L{RESET} {v6_c}({v6_wr:.0f}%){RESET}  "
                  f"{m27_c}{s[2]:>3d}W{RESET}{RED}{s[3]:>3d}L{RESET} {m27_c}({m27_wr:.0f}%){RESET}  n={total}")

        # Summary
        total_resolved = meta_wins + meta_losses
        if total_resolved:
            wr = meta_wins / total_resolved * 100
            print(f"\n  {BOLD}Aceptadas:{RESET} {GREEN}{accepted}{RESET}  {BOLD}Rechazadas:{RESET} {RED}{rejected_count}{RESET}  "
                  f"{BOLD}Record meta:{RESET} {GREEN}{meta_wins}W{RESET} {RED}{meta_losses}L{RESET} ({GREEN}{wr:.0f}%{RESET})")
        else:
            print(f"\n  {BOLD}Aceptadas:{RESET} {GREEN}{accepted}{RESET}  {BOLD}Rechazadas:{RESET} {RED}{rejected_count}{RESET}")

        # ── Quarter-length breakdown ──
        for qlen_bucket in ["10m", "12m"]:
            qs = qlen_meta[qlen_bucket]
            q_r = qs["a"] + qs["r"]
            q_resolved = qs["w"] + qs["l"]
            if q_r == 0: continue
            line = f"  {BOLD}{qlen_bucket}min:{RESET}  Aceptadas:{GREEN}{qs['a']}{RESET}/{RED}{qs['r']}{RESET}"
            if q_resolved:
                q_wr = qs["w"] / q_resolved * 100
                wc = GREEN if q_wr >= 60 else (YELLOW if q_wr >= 40 else RED)
                line += f"  Record:{GREEN}{qs['w']}W{RESET}{RED}{qs['l']}L{RESET} {wc}({q_wr:.0f}%){RESET}"
            print(line)


def run_meta_model_txt(conn: sqlite3.Connection, txt_path: str, date_str: str = None, year_month: str = None):
    """Run meta-model and export to text file (ANSI-stripped + raw data)."""
    import io, contextlib
    from datetime import date as dt_date
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    buf = io.StringIO()
    raw_rows = []
    with contextlib.redirect_stdout(buf):
        run_meta_model(conn, date_str, year_month, _out_rows=raw_rows)
    raw = buf.getvalue()
    sys.stdout.write(raw)
    sys.stdout.flush()
    clean = ansi_escape.sub('', raw).replace("\ufe0f", "")
    today_tag = dt_date.today().isoformat()
    clean = f"--- Exportado: {today_tag} ---\n\n{clean}"
    if raw_rows:
        clean += "\n\n# DATOS CRUDOS (pipe-delimited)\n"
        clean += "# date|time|home|away|league|po|w|y|format|qlen|v6_side|v6_conf|v6_res|v6_min|m27_side|m27_conf|m27_res|m27_min|meta_status|meta_side|meta_winner|meta_tier\n"
        clean += "# meta_status: A=accepted,R=rejected,S=skip  meta_winner: H/A/-  meta_tier: S/A/B/C/B-solo/BLK-F/BLK-E/BLK-LG/BLK-TOX/BLK-LO  min: @N (N=inference minute)  qlen: 10m|12m\n"
        clean += "\n".join(raw_rows)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(clean)
    print(f"\n  {GREEN}Exportado a:{RESET} {txt_path}")


def get_fusion_stats_for_date(conn: sqlite3.Connection, date_str: str) -> str:
    # 1. Fetch matches
    cursor = conn.execute("""
        SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league
        FROM bet_monitor_schedule_v2 s
        JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
        WHERE s.event_date = ?
    """, (date_str,))
    matches = cursor.fetchall()
    
    if not matches:
        return ""
        
    engine = FusionConsensusEngine()
    
    total_evaluated = 0
    filtered_count = 0
    
    tier_stats = {
        "S": {"win": 0, "loss": 0, "pending": 0},
        "A": {"win": 0, "loss": 0, "pending": 0},
        "B": {"win": 0, "loss": 0, "pending": 0},
        "C": {"win": 0, "loss": 0, "pending": 0},
    }
    
    for m in matches:
        mid = m["match_id"]
        
        # Get scores
        qs = conn.execute("""
            SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
            FROM quarter_scores_v2 WHERE match_id = ?
        """, (mid,)).fetchone()
        
        # Get logs
        logs = conn.execute("""
            SELECT model_version, signal_type, picked_side, confidence, result
            FROM bet_monitor_log_v2
            WHERE match_id = ?
            ORDER BY id ASC
        """, (mid,)).fetchall()
        
        deduped = {}
        for l in logs:
            deduped[l["model_version"]] = l
            
        v6_pred = Prediction(model="v6_2", pick="NO BET", confidence=0.0)
        m27_pred = Prediction(model="m27_v3", pick="NO BET", confidence=0.0)
        
        v6_log = deduped.get("v6_2")
        if v6_log:
            sig = v6_log["signal_type"] or ""
            picked = v6_log["picked_side"] or ""
            conf = v6_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                v6_pred = Prediction(model="v6_2", pick=picked, confidence=conf)
                
        m27_log = deduped.get("m27_v3")
        if m27_log:
            sig = m27_log["signal_type"] or ""
            picked = m27_log["picked_side"] or ""
            conf = m27_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                m27_pred = Prediction(model="m27_v3", pick=picked, confidence=conf)
                
        if not v6_log and not m27_log:
            continue
            
        res_fusion = engine.evaluate(v6_pred, m27_pred)
        total_evaluated += 1
        
        if res_fusion.final_pick == "NO BET" or res_fusion.recommended_stake == 0:
            filtered_count += 1
        else:
            # Determine actual winner
            actual_winner = get_actual_winner(qs, deduped)
            
            tier = res_fusion.tier
            if tier not in tier_stats:
                tier_stats[tier] = {"win": 0, "loss": 0, "pending": 0}
                
            if not actual_winner:
                tier_stats[tier]["pending"] += 1
            elif res_fusion.final_pick == actual_winner:
                tier_stats[tier]["win"] += 1
            else:
                tier_stats[tier]["loss"] += 1
                
    # Calculate global sums
    wins = sum(tier_stats[t]["win"] for t in ["S", "A", "B", "C"])
    losses = sum(tier_stats[t]["loss"] for t in ["S", "A", "B", "C"])
    pending = sum(tier_stats[t]["pending"] for t in ["S", "A", "B", "C"])
    bets_count = wins + losses + pending
    
    resolved = wins + losses
    win_rate = int(round(wins * 100.0 / resolved)) if resolved > 0 else 0
    
    # Helper to format tier percentages
    def fmt_tier_pct(t):
        w = tier_stats[t]["win"]
        l = tier_stats[t]["loss"]
        tot = w + l
        if tot > 0:
            return f"{int(round(w * 100.0 / tot))}%"
        return "N/D"

    col_widths = [7, 42, 18, 23, 23, 27]
    line_w = sum(col_widths) + len(col_widths) * 3 + 1
    border_eq = "=" * line_w

    s_pct = fmt_tier_pct("S")
    a_pct = fmt_tier_pct("A")
    b_pct = fmt_tier_pct("B")
    c_pct = fmt_tier_pct("C")

    # Colors
    cyan_col = f"{BOLD}{CYAN}"
    green_col = f"{GREEN}"
    red_col = f"{RED}"
    reset_col = f"{RESET}"
    yellow_col = f"{YELLOW}"

    lines = []
    lines.append(f"{cyan_col}{border_eq}{reset_col}")
    lines.append(f"{cyan_col}{' ' * ((line_w - 60) // 2)}REPORTE ESTADÍSTICO DE FUSION CONSENSUS: {date_str}{reset_col}")
    lines.append(f"{cyan_col}{border_eq}{reset_col}\n")
    
    lines.append(f"   {BOLD}📊 RESUMEN GLOBAL FUSION:{reset_col}")
    lines.append(f"      - Total Partidos Evaluados: {BOLD}{total_evaluated}{reset_col}")
    lines.append(f"      - Filtrados ({red_col}Filtered{reset_col}): {BOLD}{filtered_count}{reset_col}")
    lines.append(f"      - Operados ({green_col}Bets{reset_col})      : {BOLD}{bets_count}{reset_col} | {green_col}Ganados: {wins}{reset_col} | {red_col}Perdidos: {losses}{reset_col} | {yellow_col}Pendientes: {pending}{reset_col}")
    lines.append(f"      - Efectividad ({green_col}Win Rate{reset_col})  : {BOLD}{green_col}{win_rate}%{reset_col} (sobre resueltos)")
    lines.append("")
    lines.append(f"   {BOLD}🏅 DETALLE POR TIER:{reset_col}")
    lines.append(f"      - {BLUE}Tier S (Premium Consensus){reset_col} : {green_col}✅ {tier_stats['S']['win']}{reset_col} | {red_col}❌ {tier_stats['S']['loss']}{reset_col} | {yellow_col}⏳ {tier_stats['S']['pending']}{reset_col} ({BOLD}{s_pct}{reset_col})")
    lines.append(f"      - {green_col}Tier A (Normal Consensus){reset_col}  : {green_col}✅ {tier_stats['A']['win']}{reset_col} | {red_col}❌ {tier_stats['A']['loss']}{reset_col} | {yellow_col}⏳ {tier_stats['A']['pending']}{reset_col} ({BOLD}{a_pct}{reset_col})")
    lines.append(f"      - {yellow_col}Tier B (Extreme Diff){reset_col}      : {green_col}✅ {tier_stats['B']['win']}{reset_col} | {red_col}❌ {tier_stats['B']['loss']}{reset_col} | {yellow_col}⏳ {tier_stats['B']['pending']}{reset_col} ({BOLD}{b_pct}{reset_col})")
    lines.append(f"      - {CYAN}Tier C (Fusion Score Only){reset_col}  : {green_col}✅ {tier_stats['C']['win']}{reset_col} | {red_col}❌ {tier_stats['C']['loss']}{reset_col} | {yellow_col}⏳ {tier_stats['C']['pending']}{reset_col} ({BOLD}{c_pct}{reset_col})")
    lines.append(f"\n{cyan_col}{border_eq}{reset_col}")
    
    return "\n".join(lines)

def get_borrego_stats_for_date(conn: sqlite3.Connection, date_str: str) -> str:
    # 1. Fetch matches
    cursor = conn.execute("""
        SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league
        FROM bet_monitor_schedule_v2 s
        JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
        WHERE s.event_date = ?
    """, (date_str,))
    matches = cursor.fetchall()
    
    if not matches:
        return ""
        
    engine = FusionBorregoEngine()
    
    total_evaluated = 0
    no_bet_count = 0
    wins = 0
    losses = 0
    pending = 0
    
    home_wins = 0
    home_losses = 0
    home_pending = 0
    
    away_wins = 0
    away_losses = 0
    away_pending = 0
    
    for m in matches:
        mid = m["match_id"]
        
        # Get scores
        qs = conn.execute("""
            SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
            FROM quarter_scores_v2 WHERE match_id = ?
        """, (mid,)).fetchone()
        
        # Get logs
        logs = conn.execute("""
            SELECT model_version, signal_type, picked_side, confidence, result
            FROM bet_monitor_log_v2
            WHERE match_id = ?
            ORDER BY id ASC
        """, (mid,)).fetchall()
        
        deduped = {}
        for l in logs:
            deduped[l["model_version"]] = l
            
        v6_pred = Prediction(model="v6_2", pick="NO BET", confidence=0.0)
        m27_pred = Prediction(model="m27_v3", pick="NO BET", confidence=0.0)
        
        v6_log = deduped.get("v6_2")
        if v6_log:
            sig = v6_log["signal_type"] or ""
            picked = v6_log["picked_side"] or ""
            conf = v6_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                v6_pred = Prediction(model="v6_2", pick=picked, confidence=conf)
                
        m27_log = deduped.get("m27_v3")
        if m27_log:
            sig = m27_log["signal_type"] or ""
            picked = m27_log["picked_side"] or ""
            conf = m27_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                m27_pred = Prediction(model="m27_v3", pick=picked, confidence=conf)
                
        if not v6_log and not m27_log:
            continue
            
        pick = engine.evaluate(v6_pred, m27_pred)
        total_evaluated += 1
        
        if pick == "NO BET":
            no_bet_count += 1
        else:
            # Determine actual winner
            actual_winner = get_actual_winner(qs, deduped)
            
            is_win = (actual_winner is not None) and (pick == actual_winner)
            is_loss = (actual_winner is not None) and (pick != actual_winner)
            
            if pick == "HOME":
                if not actual_winner:
                    home_pending += 1
                elif is_win:
                    home_wins += 1
                    wins += 1
                else:
                    home_losses += 1
                    losses += 1
            elif pick == "AWAY":
                if not actual_winner:
                    away_pending += 1
                elif is_win:
                    away_wins += 1
                    wins += 1
                else:
                    away_losses += 1
                    losses += 1

    pending = home_pending + away_pending
    bets_count = wins + losses + pending
    resolved = wins + losses
    win_rate = int(round(wins * 100.0 / resolved)) if resolved > 0 else 0
    
    home_tot = home_wins + home_losses
    home_pct = int(round(home_wins * 100.0 / home_tot)) if home_tot > 0 else 0
    
    away_tot = away_wins + away_losses
    away_pct = int(round(away_wins * 100.0 / away_tot)) if away_tot > 0 else 0

    col_widths = [7, 42, 18, 23, 23, 27, 25]
    line_w = sum(col_widths) + len(col_widths) * 3 + 1
    border_eq = "=" * line_w

    # Colors
    magenta_col = f"{BOLD}{MAGENTA}"
    green_col = f"{GREEN}"
    red_col = f"{RED}"
    reset_col = f"{RESET}"
    yellow_col = f"{YELLOW}"

    lines = []
    lines.append(f"{magenta_col}{border_eq}{reset_col}")
    lines.append(f"{magenta_col}{' ' * ((line_w - 56) // 2)}REPORTE ESTADÍSTICO DE FUSION BORREGO: {date_str}{reset_col}")
    lines.append(f"{magenta_col}{border_eq}{reset_col}\n")
    
    lines.append(f"   {BOLD}📊 RESUMEN GLOBAL FUSION BORREGO:{reset_col}")
    lines.append(f"      - Total Partidos Evaluados: {BOLD}{total_evaluated}{reset_col}")
    lines.append(f"      - Sin Operación ({red_col}NO BET{reset_col})  : {BOLD}{no_bet_count}{reset_col}")
    lines.append(f"      - Operados ({green_col}Bets{reset_col})      : {BOLD}{bets_count}{reset_col} | {green_col}Ganados: {wins}{reset_col} | {red_col}Perdidos: {losses}{reset_col} | {yellow_col}Pendientes: {pending}{reset_col}")
    lines.append(f"      - Efectividad ({green_col}Win Rate{reset_col})  : {BOLD}{green_col}{win_rate}%{reset_col} (sobre resueltos)")
    lines.append("")
    lines.append(f"   {BOLD}🎯 RENDIMIENTO POR SELECCIÓN:{reset_col}")
    lines.append(f"      - Apuestas a {BOLD}HOME{reset_col} : {green_col}✅ {home_wins}{reset_col} | {red_col}❌ {home_losses}{reset_col} | {yellow_col}⏳ {home_pending}{reset_col} ({BOLD}{home_pct}%{reset_col})")
    lines.append(f"      - Apuestas a {BOLD}AWAY{reset_col} : {green_col}✅ {away_wins}{reset_col} | {red_col}❌ {away_losses}{reset_col} | {yellow_col}⏳ {away_pending}{reset_col} ({BOLD}{away_pct}%{reset_col})")
    lines.append(f"\n{magenta_col}{border_eq}{reset_col}")
    
    return "\n".join(lines)

def get_stats_for_date(conn: sqlite3.Connection, date_str: str) -> str:
    models = ["v6_2", "m27_v3"]
    BET_CATS = ["🟢", "🟡", "⚪️", "🟡⚪️"]
    FUSION_TIERS = ["S", "A", "B", "C"]
    COL_W = 24

    empty_stats = lambda: {
        "🟢":    {"win": 0, "loss": 0},
        "🟡":    {"win": 0, "loss": 0},
        "⚪️":   {"win": 0, "loss": 0},
        "🟡⚪️": {"win": 0, "loss": 0},
        "🔴":    {"count": 0},
    }
    all_stats = {m: empty_stats() for m in models}

    for m in models:
        cursor = conn.execute("""
            SELECT l.match_id, l.signal_type, l.confidence, l.result
            FROM bet_monitor_log_v2 l
            JOIN bet_monitor_schedule_v2 s ON l.match_id = s.match_id
            WHERE l.model_version = ? AND s.event_date = ?
            ORDER BY l.id ASC
        """, (m, date_str))
        rows = cursor.fetchall()

        deduped = {}
        for r in rows:
            deduped[r["match_id"]] = r

        for r in deduped.values():
            sig  = r["signal_type"] or ""
            conf = r["confidence"]  or 0.0
            res  = r["result"]      or ""

            if conf > 1.0:
                conf = conf / 100.0

            if "NO_BET" in sig or "BET" not in sig:
                all_stats[m]["🔴"]["count"] += 1
                continue

            is_late     = "LATE" in sig
            is_low_conf = round(conf, 4) < 0.30
            cat = ("🟡⚪️" if is_low_conf else "⚪️") if is_late else ("🟡" if is_low_conf else "🟢")

            if res in ("win", "hit"):
                all_stats[m][cat]["win"] += 1
            elif res in ("loss", "miss"):
                all_stats[m][cat]["loss"] += 1

    # -------------------------------------------------------------
    # Fusion stats calculation
    # -------------------------------------------------------------
    fusion_engine = FusionConsensusEngine()
    fusion_stats = {
        "S": {"win": 0, "loss": 0},
        "A": {"win": 0, "loss": 0},
        "B": {"win": 0, "loss": 0},
        "C": {"win": 0, "loss": 0},
        "Filtered": {"count": 0}
    }

    # -------------------------------------------------------------
    # Borrego stats calculation
    # -------------------------------------------------------------
    borrego_engine = FusionBorregoEngine()
    borrego_stats = {
        "HOME": {"win": 0, "loss": 0},
        "AWAY": {"win": 0, "loss": 0},
        "Total": {"win": 0, "loss": 0},
        "Pending": {"count": 0},
        "NO_BET": {"count": 0}
    }

    cursor = conn.execute("""
        SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league
        FROM bet_monitor_schedule_v2 s
        JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
        WHERE s.event_date = ?
    """, (date_str,))
    matches = cursor.fetchall()

    for m in matches:
        mid = m["match_id"]
        qs = conn.execute("""
            SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
            FROM quarter_scores_v2 WHERE match_id = ?
        """, (mid,)).fetchone()
        
        logs = conn.execute("""
            SELECT model_version, signal_type, picked_side, confidence, result
            FROM bet_monitor_log_v2
            WHERE match_id = ?
            ORDER BY id ASC
        """, (mid,)).fetchall()
        
        deduped_logs = {}
        for l in logs:
            deduped_logs[l["model_version"]] = l
            
        v6_pred = Prediction(model="v6_2", pick="NO BET", confidence=0.0)
        m27_pred = Prediction(model="m27_v3", pick="NO BET", confidence=0.0)
        
        v6_log = deduped_logs.get("v6_2")
        if v6_log:
            sig = v6_log["signal_type"] or ""
            picked = v6_log["picked_side"] or ""
            conf = v6_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                v6_pred = Prediction(model="v6_2", pick=picked, confidence=conf)
                
        m27_log = deduped_logs.get("m27_v3")
        if m27_log:
            sig = m27_log["signal_type"] or ""
            picked = m27_log["picked_side"] or ""
            conf = m27_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                m27_pred = Prediction(model="m27_v3", pick=picked, confidence=conf)
                
        if not v6_log and not m27_log:
            continue
            
        res_fusion = fusion_engine.evaluate(v6_pred, m27_pred)
        actual_winner = get_actual_winner(qs, deduped_logs)
        
        if res_fusion.final_pick == "NO BET" or res_fusion.recommended_stake == 0:
            fusion_stats["Filtered"]["count"] += 1
        else:
            tier = res_fusion.tier
            if not actual_winner:
                pass
            elif res_fusion.final_pick == actual_winner:
                fusion_stats[tier]["win"] += 1
            else:
                fusion_stats[tier]["loss"] += 1

        # Fusion Borrego
        borrego_pick = borrego_engine.evaluate(v6_pred, m27_pred)
        if borrego_pick == "NO BET":
            borrego_stats["NO_BET"]["count"] += 1
        else:
            if not actual_winner:
                borrego_stats["Pending"]["count"] += 1
            elif borrego_pick == actual_winner:
                borrego_stats[borrego_pick]["win"] += 1
                borrego_stats["Total"]["win"] += 1
            else:
                borrego_stats[borrego_pick]["loss"] += 1
                borrego_stats["Total"]["loss"] += 1

    header_parts = [
        pad_cell(f"v6_2 Stats", COL_W),
        pad_cell(f"m27_v3 Stats", COL_W),
        pad_cell(f"Fusion Stats", COL_W),
        pad_cell(f"Borrego Stats", COL_W)
    ]
    header = "    ".join(header_parts).rstrip()

    table_lines = [header]

    for cat_idx, cat in enumerate(BET_CATS):
        fusion_tier = FUSION_TIERS[cat_idx]
        for outcome, emoji in (("win", f"{GREEN}✅{RESET}"), ("loss", f"{RED}❌{RESET}")):
            cells = []
            for m in models:
                st  = all_stats[m][cat]
                n   = st["win"] if outcome == "win" else st["loss"]
                tot = st["win"] + st["loss"]
                
                clean_cat = cat.replace("\ufe0f", "")
                emoji_spacing = " " if len(clean_cat) > 1 else "  "
                
                if tot > 0:
                    pct  = int(round(n * 100.0 / tot))
                    cell = f"{emoji}{cat}{emoji_spacing}{n}   {pct}%"
                else:
                    cell = f"{emoji}{cat}{emoji_spacing}0   0%"
                cells.append(pad_cell(cell, COL_W))
                
            st_f = fusion_stats[fusion_tier]
            n_f = st_f["win"] if outcome == "win" else st_f["loss"]
            tot_f = st_f["win"] + st_f["loss"]
            
            tier_disp = f"{BLUE}{fusion_tier}{RESET}" if fusion_tier == "S" else (f"{GREEN}{fusion_tier}{RESET}" if fusion_tier == "A" else (f"{YELLOW}{fusion_tier}{RESET}" if fusion_tier == "B" else f"{CYAN}{fusion_tier}{RESET}"))
            
            if tot_f > 0:
                pct_f = int(round(n_f * 100.0 / tot_f))
                cell_f = f"{emoji}{tier_disp}   {n_f}   {pct_f}%"
            else:
                cell_f = f"{emoji}{tier_disp}   0   0%"
            cells.append(pad_cell(cell_f, COL_W))

            # Borrego Cell Construction
            # Determine which row we are on based on cat_idx and outcome
            row_idx = cat_idx * 2 + (0 if outcome == "win" else 1)
            if row_idx == 0: # HOME win
                h_win = borrego_stats["HOME"]["win"]
                h_tot = h_win + borrego_stats["HOME"]["loss"]
                h_pct = int(round(h_win * 100.0 / h_tot)) if h_tot > 0 else 0
                cell_b = f"{emoji}H   {h_win}   {h_pct}%"
            elif row_idx == 1: # HOME loss
                h_loss = borrego_stats["HOME"]["loss"]
                h_tot = borrego_stats["HOME"]["win"] + h_loss
                h_pct = int(round(h_loss * 100.0 / h_tot)) if h_tot > 0 else 0
                cell_b = f"{emoji}H   {h_loss}   {h_pct}%"
            elif row_idx == 2: # AWAY win
                a_win = borrego_stats["AWAY"]["win"]
                a_tot = a_win + borrego_stats["AWAY"]["loss"]
                a_pct = int(round(a_win * 100.0 / a_tot)) if a_tot > 0 else 0
                cell_b = f"{emoji}A   {a_win}   {a_pct}%"
            elif row_idx == 3: # AWAY loss
                a_loss = borrego_stats["AWAY"]["loss"]
                a_tot = borrego_stats["AWAY"]["win"] + a_loss
                a_pct = int(round(a_loss * 100.0 / a_tot)) if a_tot > 0 else 0
                cell_b = f"{emoji}A   {a_loss}   {a_pct}%"
            elif row_idx == 4: # Total win
                t_win = borrego_stats["Total"]["win"]
                t_tot = t_win + borrego_stats["Total"]["loss"]
                t_pct = int(round(t_win * 100.0 / t_tot)) if t_tot > 0 else 0
                cell_b = f"{emoji}T   {t_win}   {t_pct}%"
            elif row_idx == 5: # Total loss
                t_loss = borrego_stats["Total"]["loss"]
                t_tot = borrego_stats["Total"]["win"] + t_loss
                t_pct = int(round(t_loss * 100.0 / t_tot)) if t_tot > 0 else 0
                cell_b = f"{emoji}T   {t_loss}   {t_pct}%"
            elif row_idx == 6: # Pending count
                p_cnt = borrego_stats["Pending"]["count"]
                cell_b = f"{YELLOW}⏳{RESET}   P. {p_cnt}"
            elif row_idx == 7: # Win rate
                t_win = borrego_stats["Total"]["win"]
                t_tot = t_win + borrego_stats["Total"]["loss"]
                t_pct = int(round(t_win * 100.0 / t_tot)) if t_tot > 0 else 0
                cell_b = f"{CYAN}🏅{RESET}   E. {t_pct}%"
            
            cells.append(pad_cell(cell_b, COL_W))
            
            table_lines.append("    ".join(cells).rstrip())

    nobet_cells = []
    for m in models:
        count = all_stats[m]["🔴"]["count"]
        total_all = count
        for cat in BET_CATS:
            total_all += all_stats[m][cat]["win"] + all_stats[m][cat]["loss"]
        
        if total_all > 0:
            pct = int(round(count * 100.0 / total_all))
            cell = f"{RED}🔴{RESET} {count}   {pct}%"
        else:
            cell = f"{RED}🔴{RESET} 0   0%"
        nobet_cells.append(pad_cell(cell, COL_W))
        
    count_f = fusion_stats["Filtered"]["count"]
    total_all_f = count_f + sum(fusion_stats[t]["win"] + fusion_stats[t]["loss"] for t in FUSION_TIERS)
    if total_all_f > 0:
        pct_f = int(round(count_f * 100.0 / total_all_f))
        cell_f = f"{RED}🔴{RESET} {count_f}   {pct_f}%"
    else:
        cell_f = f"{RED}🔴{RESET} 0   0%"
    nobet_cells.append(pad_cell(cell_f, COL_W))

    # Borrego NO BET row
    count_b = borrego_stats["NO_BET"]["count"]
    t_tot_b = borrego_stats["Total"]["win"] + borrego_stats["Total"]["loss"]
    total_all_b = count_b + t_tot_b + borrego_stats["Pending"]["count"]
    if total_all_b > 0:
        pct_b = int(round(count_b * 100.0 / total_all_b))
        cell_b = f"{RED}🔴{RESET} {count_b}   {pct_b}%"
    else:
        cell_b = f"{RED}🔴{RESET} 0   0%"
    nobet_cells.append(pad_cell(cell_b, COL_W))
    
    table_lines.append("    ".join(nobet_cells).rstrip())

    return "\n".join(table_lines)

def justify_matches_for_date(conn: sqlite3.Connection, date_str: str) -> None:
    cursor = conn.execute("""
        SELECT DISTINCT s.match_id, s.home_team, s.away_team, s.league, s.scheduled_utc_ts
        FROM bet_monitor_schedule_v2 s
        JOIN bet_monitor_log_v2 l ON s.match_id = l.match_id
        WHERE s.event_date = ?
        ORDER BY s.scheduled_utc_ts ASC
    """, (date_str,))
    matches = cursor.fetchall()

    if not matches:
        print(f"\n{YELLOW}No se encontraron justificaciones de partidos para esta fecha.{RESET}")
        return

    col_widths = [7, 42, 18, 23, 23, 27, 25]
    line_w = sum(col_widths) + len(col_widths) * 3 + 1
    border_eq = "=" * line_w

    print(f"\n{BOLD}{CYAN}{border_eq}{RESET}")
    print(f"{BOLD}{CYAN}{' ' * ((line_w - 58) // 2)}JUSTIFICACIÓN DETALLADA PARTIDO A PARTIDO (VISTA RESUMIDA){RESET}")
    print(f"{BOLD}{CYAN}{border_eq}{RESET}\n")

    top_border = "   ┌" + "┬".join("─" * w for w in col_widths) + "┐"
    header_line = "   │ " + " │ ".join(pad_cell(h, w - 2) for h, w in zip(["Hora", "Partido / Liga", "Score / Q4", "v6_2 Pick | Res", "m27_v3 Pick | Res", "Fusion Consensus | Res", "Fusion Borrego | Res"], col_widths)) + " │"
    mid_border = "   ├" + "┼".join("─" * w for w in col_widths) + "┤"
    bot_border = "   └" + "┴".join("─" * w for w in col_widths) + "┘"
    
    print(top_border)
    print(header_line)
    print(mid_border)

    engine = FusionConsensusEngine()
    borrego_engine = FusionBorregoEngine()

    for m in matches:
        mid = m["match_id"]
        home = m["home_team"]
        away = m["away_team"]
        league = m["league"]
        sched_ts = m["scheduled_utc_ts"]
        
        sched_time = datetime.fromtimestamp(sched_ts, tz=timezone(timedelta(hours=-6))).strftime("%H:%M")

        qs = conn.execute("""
            SELECT q1_home, q1_away, q2_home, q2_away, q3_home, q3_away, q4_home, q4_away
            FROM quarter_scores_v2 WHERE match_id = ?
        """, (mid,)).fetchone()

        score_str = "N/D"
        if qs:
            q4h = qs["q4_home"]
            q4a = qs["q4_away"]
            if q4h is not None and q4a is not None:
                if all(qs[c] is not None for c in ["q1_home", "q1_away", "q2_home", "q2_away", "q3_home", "q3_away"]):
                    tot_h = qs["q1_home"] + qs["q2_home"] + qs["q3_home"] + q4h
                    tot_a = qs["q1_away"] + qs["q2_away"] + qs["q3_away"] + q4a
                    tot_h_col, tot_a_col = colorize_scores(tot_h, tot_a)
                    q4h_col, q4a_col = colorize_scores(q4h, q4a)
                    score_str = f"{tot_h_col}-{tot_a_col} ({q4h_col}-{q4a_col})"
                else:
                    q4h_col, q4a_col = colorize_scores(q4h, q4a)
                    score_str = f"Q4:{q4h_col}-{q4a_col}"
            else:
                score_str = "Pendiente"

        match_name = f"{home} vs {away} ({league})"
        if visual_len(match_name) > 40:
            short_league = league[:10] + ".." if len(league) > 10 else league
            h_short = home[:12] + ".." if len(home) > 12 else home
            a_short = away[:12] + ".." if len(away) > 12 else away
            match_name = f"{h_short} vs {a_short} ({short_league})"
            if visual_len(match_name) > 40:
                match_name = match_name[:37] + "..."

        logs_cursor = conn.execute("""
            SELECT model_version, signal_type, picked_side, confidence, result
            FROM bet_monitor_log_v2
            WHERE match_id = ?
            ORDER BY id ASC
        """, (mid,))
        logs = logs_cursor.fetchall()

        deduped = {}
        for l in logs:
            deduped[l["model_version"]] = l

        # Pre-compute active confidence percentages for comparison
        raw_confs = {}
        for model in ["v6_2", "m27_v3"]:
            log = deduped.get(model)
            if log:
                sig = log["signal_type"] or ""
                if "BET" in sig and "NO_BET" not in sig:
                    conf = log["confidence"] or 0.0
                    conf_pct = int(round(conf * 100 if conf <= 1.0 else conf))
                    raw_confs[model] = conf_pct
        
        big_diff = False
        if len(raw_confs) == 2:
            big_diff = abs(raw_confs["v6_2"] - raw_confs["m27_v3"]) > 40

        model_cols = []
        for model in ["v6_2", "m27_v3"]:
            log = deduped.get(model)
            if not log:
                model_str = f"{RED}🔴 N/A{RESET}"
            else:
                sig = log["signal_type"] or ""
                picked = log["picked_side"] or ""
                conf = log["confidence"] or 0.0
                res = log["result"] or "pending"
                
                if "NO_BET" in sig or "BET" not in sig:
                    model_str = f"{RED}🔴 NO BET{RESET}"
                else:
                    side_emoji = "🏠" if picked == "HOME" else "✈️"
                    conf_pct = int(round(conf * 100 if conf <= 1.0 else conf))
                    res_emoji = f"{GREEN}✅{RESET}" if res == "win" else (f"{RED}❌{RESET}" if res == "loss" else f"{YELLOW}⏳{RESET}")
                    
                    if big_diff:
                        conf_color = RED
                    elif conf_pct < 30:
                        conf_color = YELLOW
                    else:
                        conf_color = GREEN
                        
                    model_str = f"{side_emoji} ({conf_color}{conf_pct}%{RESET}) | {res_emoji}"
            model_cols.append(model_str)

        # -------------------------------------------------------------
        # Evaluate Fusion Consensus
        # -------------------------------------------------------------
        v6_pred = Prediction(model="v6_2", pick="NO BET", confidence=0.0)
        m27_pred = Prediction(model="m27_v3", pick="NO BET", confidence=0.0)
        
        v6_log = deduped.get("v6_2")
        if v6_log:
            sig = v6_log["signal_type"] or ""
            picked = v6_log["picked_side"] or ""
            conf = v6_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                v6_pred = Prediction(model="v6_2", pick=picked, confidence=conf)
                
        m27_log = deduped.get("m27_v3")
        if m27_log:
            sig = m27_log["signal_type"] or ""
            picked = m27_log["picked_side"] or ""
            conf = m27_log["confidence"] or 0.0
            if conf <= 1.0:
                conf = conf * 100.0
            if "BET" in sig and "NO_BET" not in sig and picked in ("HOME", "AWAY"):
                m27_pred = Prediction(model="m27_v3", pick=picked, confidence=conf)
                
        # Determine actual winner
        actual_winner = get_actual_winner(qs, deduped)

        if not v6_log and not m27_log:
            fusion_str = f"{RED}🔴 N/A{RESET}"
        else:
            res_fusion = engine.evaluate(v6_pred, m27_pred)
            
            if res_fusion.final_pick == "NO BET" or res_fusion.recommended_stake == 0:
                fusion_str = f"{RED}🔴 FILTERED{RESET}"
            else:
                side_emoji = "🏠" if res_fusion.final_pick == "HOME" else "✈️"
                tier_col = BLUE if res_fusion.tier == "S" else (GREEN if res_fusion.tier == "A" else (YELLOW if res_fusion.tier == "B" else (CYAN if res_fusion.tier == "C" else RESET)))
                
                if not actual_winner:
                    res_emoji = f"{YELLOW}⏳{RESET}"
                elif res_fusion.final_pick == actual_winner:
                    res_emoji = f"{GREEN}✅{RESET}"
                else:
                    res_emoji = f"{RED}❌{RESET}"
                
                fusion_str = f"{side_emoji} {tier_col}{res_fusion.tier}{RESET} ({res_fusion.recommended_stake}u) | {res_emoji}"

        # -------------------------------------------------------------
        # Evaluate Fusion Borrego
        # -------------------------------------------------------------
        borrego_pick_val = borrego_engine.evaluate(v6_pred, m27_pred)
        if not v6_log and not m27_log:
            borrego_str = f"{RED}🔴 N/A{RESET}"
        elif borrego_pick_val == "NO BET":
            borrego_str = f"{RED}🔴 NO BET{RESET}"
        else:
            side_emoji = "🏠" if borrego_pick_val == "HOME" else "✈️"
            
            if not actual_winner:
                res_emoji = f"{YELLOW}⏳{RESET}"
            elif borrego_pick_val == actual_winner:
                res_emoji = f"{GREEN}✅{RESET}"
            else:
                res_emoji = f"{RED}❌{RESET}"
            
            borrego_str = f"{side_emoji} {borrego_pick_val} | {res_emoji}"

        row_cells = [sched_time, match_name, score_str, model_cols[0], model_cols[1], fusion_str, borrego_str]
        row_line = "   │ " + " │ ".join(pad_cell(cell, w - 2) for cell, w in zip(row_cells, col_widths)) + " │"
        print(row_line)

    print(bot_border)
    print()

def main():
    if not DB_PATH.exists():
        print(f"{RED}[ERROR]{RESET} Base de datos no encontrada en {DB_PATH}")
        sys.exit(1)

    today_str = datetime.now(timezone(timedelta(hours=-6))).strftime("%Y-%m-%d")

    print(f"\n{BOLD}{YELLOW}========================================================================================================================{RESET}")
    print(f"{BOLD}{YELLOW}                               REPORTE INTERACTIVO: ESTADÍSTICAS Y RENDIMIENTO DE MODELOS{RESET}")
    print(f"{BOLD}{YELLOW}========================================================================================================================{RESET}\n")
    print(f"   {BOLD}Seleccione el tipo de análisis:{RESET}")
    print(f"   {GREEN}1){RESET} Reporte Diario (Estadísticas del día + Justificación tipo Excel + Análisis de Liga/Confianza del día)")
    print(f"   {GREEN}2){RESET} Análisis Global Histórico (Resumen por Ligas y Distribución por Confianza de todas las apuestas)")
    print(f"   {GREEN}3){RESET} Exportar Justificación de Partidos a Excel (.xlsx) con Colores Estilizados")
    print(f"   {GREEN}4){RESET} Exportar Reporte del MES Completo a Excel (.xlsx) (Una sola hoja)")
    print(f"   {GREEN}5){RESET} Exportar Reporte a TXT (formato legible para IA/humanos)")
    print(f"   {GREEN}6){RESET} Meta-Model: analisis inteligente v6_2 + m27_v3 (filtra ligas debiles / playoffs)")
    print()

    option = input("   Seleccione una opción [1-6] (Presione ENTER para 1): ").strip()
    if not option:
        option = "1"

    if option == "2":
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            print(f"\n{BOLD}{GREEN}========================================================================================================================{RESET}")
            print(f"{BOLD}{GREEN}                               ANÁLISIS GLOBAL HISTÓRICO DE MODELOS (TODAS LAS APUESTAS LIQUIDADAS){RESET}")
            print(f"{BOLD}{GREEN}========================================================================================================================{RESET}\n")

            league_stats, conf_stats = calculate_metrics_for_logs(conn)

            print(f"{BOLD}{CYAN}1. DISTRIBUCIÓN Y RENDIMIENTO POR NIVEL DE CONFIANZA (LOW < 30% vs HIGH >= 30%){RESET}\n")
            print(format_confidence_table(conf_stats))
            print("\n" + "=" * 120 + "\n")

            print(f"{BOLD}{CYAN}2. RENDIMIENTO DETALLADO POR LIGA DE BALONCESTO{RESET}\n")
            print(format_league_table(league_stats))
            print()
        finally:
            conn.close()
        return

    if option == "3":
        date_input = input(f"   Ingrese fecha (YYYY-MM-DD) [Presione ENTER para {today_str}]: ").strip()
        if not date_input:
            date_str = today_str
        else:
            try:
                datetime.strptime(date_input, "%Y-%m-%d")
                date_str = date_input
            except ValueError:
                print(f"{RED}[ERROR]{RESET} Formato de fecha invalido. Use YYYY-MM-DD.")
                sys.exit(1)

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM bet_monitor_schedule_v2 WHERE event_date = ?", (date_str,))
            sched_count = cursor.fetchone()[0]
            if sched_count == 0:
                print(f"\n{RED}[ADVERTENCIA]{RESET} No se encontraron partidos registrados para la fecha: {date_str}")
                sys.exit(0)
            
            res_msg = export_matches_to_excel(conn, date_str)
            print(f"\n   {res_msg}\n")
        finally:
            conn.close()
        return

    if option == "4":
        month_input = input(f"   Ingrese mes (YYYY-MM) [Presione ENTER para {today_str[:7]}]: ").strip()
        if not month_input:
            year_month = today_str[:7]
        else:
            try:
                datetime.strptime(month_input + "-01", "%Y-%m-%d")
                year_month = month_input
            except ValueError:
                print(f"{RED}[ERROR]{RESET} Formato de mes invalido. Use YYYY-MM.")
                sys.exit(1)

        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            res_msg = export_matches_monthly_to_excel(conn, year_month)
            print(f"\n   {res_msg}\n")
        finally:
            conn.close()
        return

    if option == "5":
        print(f"\n   {BOLD}Exportar a TXT (formato legible){RESET}")
        sub = input(f"   ¿{GREEN}D{RESET}iario (una fecha) o {GREEN}M{RESET}ensual? [D/M] (ENTER para D): ").strip().lower()
        if sub == "m":
            month_input = input(f"   Ingrese mes (YYYY-MM) [Presione ENTER para {today_str[:7]}]: ").strip()
            if not month_input:
                year_month = today_str[:7]
            else:
                try:
                    datetime.strptime(month_input + "-01", "%Y-%m-%d")
                    year_month = month_input
                except ValueError:
                    print(f"{RED}[ERROR]{RESET} Formato de mes invalido. Use YYYY-MM.")
                    sys.exit(1)
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            try:
                res_msg = export_matches_to_text_ai(conn, year_month=year_month)
                print(f"\n   {res_msg}\n")
            finally:
                conn.close()
        else:
            date_input = input(f"   Ingrese fecha (YYYY-MM-DD) [Presione ENTER para {today_str}]: ").strip()
            if not date_input:
                date_str = today_str
            else:
                try:
                    datetime.strptime(date_input, "%Y-%m-%d")
                    date_str = date_input
                except ValueError:
                    print(f"{RED}[ERROR]{RESET} Formato de fecha invalido. Use YYYY-MM-DD.")
                    sys.exit(1)
            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM bet_monitor_schedule_v2 WHERE event_date = ?", (date_str,))
                sched_count = cursor.fetchone()[0]
                if sched_count == 0:
                    print(f"\n{RED}[ADVERTENCIA]{RESET} No se encontraron partidos para: {date_str}")
                    sys.exit(0)
                res_msg = export_matches_to_text_ai(conn, date_str=date_str)
                print(f"\n   {res_msg}\n")
            finally:
                conn.close()
        return

    if option == "6":
        print(f"\n   {BOLD}Meta-Model: analisis historico + picks{RESET}")
        sub = input(f"   {GREEN}H{RESET}=solo historico, {GREEN}P{RESET}=picks del dia/mes, "
                    f"{GREEN}T{RESET}=todo historial, {GREEN}E{RESET}=exportar a .txt [H/P/T/E] (ENTER=H): ").strip().lower()
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            if sub == "p":
                sub2 = input(f"   ¿{GREEN}D{RESET}iario o {GREEN}M{RESET}ensual? [D/M] (ENTER para D): ").strip().lower()
                if sub2 == "m":
                    month_input = input(f"   Ingrese mes (YYYY-MM) [ENTER para {today_str[:7]}]: ").strip()
                    if not month_input:
                        ym = today_str[:7]
                    else:
                        datetime.strptime(month_input + "-01", "%Y-%m-%d")
                        ym = month_input
                    run_meta_model(conn, year_month=ym)
                else:
                    date_input = input(f"   Ingrese fecha (YYYY-MM-DD) [ENTER para {today_str}]: ").strip()
                    ds = date_input if date_input else today_str
                    datetime.strptime(ds, "%Y-%m-%d")
                    run_meta_model(conn, date_str=ds)
            elif sub == "t":
                run_meta_model(conn, date_str="ALL")
            elif sub == "e":
                sub_d = input(f"   {GREEN}D{RESET}=dia, {GREEN}M{RESET}=mes, {GREEN}T{RESET}=todo historial? [D/M/T] (ENTER=T): ").strip().lower()
                if sub_d == "d":
                    di = input(f"   Fecha (YYYY-MM-DD) [ENTER para {today_str}]: ").strip()
                    ds = di or today_str
                    ds_arg, ym_arg = ds, None
                elif sub_d == "m":
                    mi = input(f"   Mes (YYYY-MM) [ENTER para {today_str[:7]}]: ").strip()
                    ym_arg = mi or today_str[:7]
                    ds_arg = None
                else:
                    ds_arg, ym_arg = "ALL", None
                txt_name = f"MetaModel_{ds_arg or ym_arg}.txt"
                if ds_arg == "ALL":
                    txt_name = f"MetaModel_ALL_{today_str}.txt"
                txt_path = ROOT / txt_name
                run_meta_model_txt(conn, str(txt_path), date_str=ds_arg, year_month=ym_arg)
            else:
                run_meta_model(conn)
        finally:
            conn.close()
        return

    # Opción 1: Reporte Diario
    date_input = input(f"   Ingrese fecha (YYYY-MM-DD) [Presione ENTER para {today_str}]: ").strip()
    if not date_input:
        date_str = today_str
    else:
        try:
            datetime.strptime(date_input, "%Y-%m-%d")
            date_str = date_input
        except ValueError:
            print(f"{RED}[ERROR]{RESET} Formato de fecha invalido. Use YYYY-MM-DD.")
            sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute("SELECT COUNT(*) FROM bet_monitor_schedule_v2 WHERE event_date = ?", (date_str,))
        sched_count = cursor.fetchone()[0]
        if sched_count == 0:
            available_cursor = conn.execute("SELECT DISTINCT event_date FROM bet_monitor_schedule_v2 ORDER BY event_date DESC LIMIT 5")
            available_dates = [r[0] for r in available_cursor.fetchall()]
            print(f"\n{RED}[ADVERTENCIA]{RESET} No se encontraron partidos registrados para la fecha: {date_str}")
            if available_dates:
                print(f"Fechas con partidos disponibles en la base de datos: {', '.join(available_dates)}")
            sys.exit(0)

        # Mostrar stats de Fusion Consensus
        fusion_stats_msg = get_fusion_stats_for_date(conn, date_str)
        if fusion_stats_msg:
            print(fusion_stats_msg)
            print()

        # Mostrar stats de Fusion Borrego
        borrego_stats_msg = get_borrego_stats_for_date(conn, date_str)
        if borrego_stats_msg:
            print(borrego_stats_msg)
            print()

        # Mostrar stats diarias de los modelos individuales
        print(f"\n{BOLD}{GREEN}========================================================================================================================{RESET}")
        print(f"{BOLD}{GREEN}                                    ESTADÍSTICAS DIARIAS DEL DÍA: {date_str}{RESET}")
        print(f"{BOLD}{GREEN}========================================================================================================================{RESET}\n")
        
        stats_msg = get_stats_for_date(conn, date_str)
        print(stats_msg)
        print(f"\n{BOLD}{GREEN}========================================================================================================================{RESET}")

        # Mostrar justificaciones detalladas
        justify_matches_for_date(conn, date_str)

        # Mostrar también el análisis de ligas y confianza para ese día específico
        print(f"\n{BOLD}{CYAN}========================================================================================================================{RESET}")
        print(f"{BOLD}{CYAN}                                RENDIMIENTO POR LIGA Y CONFIANZA PARA EL DÍA: {date_str}{RESET}")
        print(f"{BOLD}{CYAN}========================================================================================================================{RESET}\n")
        
        league_stats, conf_stats = calculate_metrics_for_logs(conn, date_str)
        
        print(f"{BOLD}{CYAN}1. DISTRIBUCIÓN POR NIVEL DE CONFIANZA ({date_str}){RESET}\n")
        print(format_confidence_table(conf_stats))
        print("\n" + "-" * 120 + "\n")
        
        print(f"{BOLD}{CYAN}2. RENDIMIENTO POR LIGA ({date_str}){RESET}\n")
        print(format_league_table(league_stats))
        print()

        # Ofrecer exportar a Excel al final de la opción 1
        print(f"\n{BOLD}{CYAN}========================================================================================================================{RESET}")
        export_opt = input("   ¿Desea exportar este reporte a un archivo Excel (.xlsx)? [S/N] (Presione ENTER para N): ").strip().lower()
        if export_opt in ("s", "si", "y", "yes"):
            res_msg = export_matches_to_excel(conn, date_str)
            print(f"\n   {res_msg}\n")

    finally:
        conn.close()

if __name__ == "__main__":
    main()
