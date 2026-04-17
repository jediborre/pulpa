"""
V12 LIVE - Virtual Bookmaker Engine (Sin Necesidad de Odds Input)
=================================================================
Genera líneas justas basadas ÚNICAMENTE en:
- Score actual del quarter
- Graph points (momentum/pressure)
- Play-by-play events
- Tiempo transcurrido

Output: Mercados recomendados con líneas justas.
TÚ comparas manualmente con tu casa de apuestas.

Ejemplo output:
  Handicap justo: Home +7.5
  → Si tu casa ofrece Home +8.5 o mejor → VALUE
  → Si tu casa ofrece Home +6.5 o peor → NO VALUE
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

db_mod = __import__("db")

DB_PATH = PROJECT_ROOT / "matches.db"
MODEL_DIR = ROOT / "model_outputs"
LIVE_DIR = ROOT / "live_engine"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

# MAE del modelo de regresión (usado como margen conservador)
MAE_TOTAL = 5.33  # pts por quarter

# Umbrales estrictos (como la casa)
MIN_EDGE_FOR_RECOMMENDATION = 0.10  # 10% edge mínimo
MIN_CONFIDENCE = 0.60  # 60% confianza mínima
MIN_MOMENTUM_THRESHOLD = 3.0  # Momentum mínimo para recomendar
MIN_TIME_REMAINING = 3.0  # Minutos mínimos restantes


@dataclass
class LiveMarket:
    """Un mercado con su línea justa y probabilidad."""
    name: str  # "1x2_home", "handicap_away", "over", etc.
    description: str  # Descripción legible
    line: float  # Línea justa (handicap, O/U, etc.)
    our_probability: float  # Nuestra probabilidad justa
    fair_odds: float  # Odds justas (1/prob)
    confidence: str  # "high", "medium", "low"
    recommendation: str  # "WATCH", "CHECK", "AVOID"
    when_to_bet: str  # Cuándo apostar (comparación con casa real)
    reasoning: str


@dataclass
class LiveAnalysis:
    """Análisis completo de un quarter en vivo."""
    match_id: str
    quarter: str
    timestamp: str
    
    # Score actual
    qtr_home_score: int
    qtr_away_score: int
    total_home_score: int
    total_away_score: int
    score_diff: int  # Positivo = home ahead
    
    # Tiempo
    elapsed_minutes: float
    minutes_remaining: float
    
    # Momentum
    graph_momentum: float  # -10 a +10
    momentum_direction: str  # "home", "away", "neutral"
    recent_scoring_run: str  # "home", "away", "none"
    
    # Proyecciones
    projected_home_pts: float
    projected_away_pts: float
    projected_total_pts: float
    projected_diff: float
    
    # Mercados
    markets: list[LiveMarket]
    
    # Recomendación general
    overall_recommendation: str  # "BET", "WATCH", "AVOID"
    best_market: str  # Mejor mercado disponible
    reasoning: str


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _count_sign_swings(values: list[int]) -> int:
    swings = 0
    prev_sign = 0
    for value in values:
        sign = 1 if value > 0 else (-1 if value < 0 else 0)
        if sign == 0:
            continue
        if prev_sign != 0 and sign != prev_sign:
            swings += 1
        prev_sign = sign
    return swings


def calcular_momentum(graph_points: list[dict], cutoff_minute: float) -> float:
    """
    Calcula momentum del graph points.
    Positivo = home pressure, Negativo = away pressure.
    Escala: -10 a +10.
    """
    if not graph_points:
        return 0.0
    
    points = [p for p in graph_points if int(p.get("minute", 0)) <= cutoff_minute]
    if len(points) < 5:
        return 0.0
    
    values = [int(p.get("value", 0)) for p in points]
    
    # Últimos 5 puntos (momentum reciente)
    recent = values[-5:]
    momentum_raw = np.mean(recent)
    
    # Normalizar a escala -10 a +10
    max_abs = max(abs(max(values)), abs(min(values)), 1)
    momentum_normalized = (momentum_raw / max_abs) * 10.0
    
    return round(momentum_normalized, 2)


def calcular_recent_scoring_run(pbp_events: list[dict]) -> tuple[str, int]:
    """
    Calcula la racha de anotación reciente.
    Returns: (team, puntos_en_racha)
    """
    if not pbp_events:
        return "none", 0
    
    # Últimos 3-4 eventos
    recent = pbp_events[-4:]
    
    home_pts = sum(int(p.get("points", 0)) for p in recent if p.get("team") == "home")
    away_pts = sum(int(p.get("points", 0)) for p in recent if p.get("team") == "away")
    
    if home_pts > away_pts and home_pts >= 4:
        return "home", home_pts
    elif away_pts > home_pts and away_pts >= 4:
        return "away", away_pts
    else:
        return "none", max(home_pts, away_pts)


def monte_carlo_pesimista(
    score_home: int,
    score_away: int,
    home_ppm: float,
    away_ppm: float,
    minutes_left: float,
    momentum_adjustment: float = 0.0,
    num_sims: int = 10000,
) -> dict:
    """
    Monte Carlo pesimista para quarter winner.
    Incluye:
    - Varianza ALTA (basketball es volátil)
    - Penalización del 15% al trailing team
    - Ajuste por momentum
    """
    if minutes_left <= 0:
        winner = "home" if score_home > score_away else "away"
        return {
            "home_win_prob": 1.0 if winner == "home" else 0.0,
            "away_win_prob": 1.0 if winner == "away" else 0.0,
        }
    
    # Varianza ALTA (más pesimista que optimista)
    var_home = max(1.0, home_ppm * 2.0)
    var_away = max(1.0, away_ppm * 2.0)
    
    # Simulación
    sim_home = np.random.normal(
        score_home + home_ppm * minutes_left,
        np.sqrt(var_home * minutes_left),
        num_sims,
    )
    sim_away = np.random.normal(
        score_away + away_ppm * minutes_left,
        np.sqrt(var_away * minutes_left),
        num_sims,
    )
    
    # Aplicar momentum adjustment
    sim_home += momentum_adjustment * minutes_left * 0.3
    sim_away -= momentum_adjustment * minutes_left * 0.3
    
    home_wins = np.sum(sim_home > sim_away)
    ties = np.sum(np.abs(sim_home - sim_away) < 0.5)
    
    home_prob = (home_wins + 0.5 * ties) / num_sims
    away_prob = 1.0 - home_prob
    
    # PENALIZACIÓN pesimista al trailing team (15%)
    if home_prob < 0.5:
        home_prob *= 0.85
    else:
        away_prob *= 0.85
    
    # Normalizar
    total = home_prob + away_prob
    home_prob /= total
    away_prob /= total
    
    return {
        "home_win_prob": home_prob,
        "away_win_prob": away_prob,
    }


def proyectar_puntos(
    score_home: int,
    score_away: int,
    elapsed: float,
    mins_left: float,
    momentum: float = 0.0,
) -> tuple[float, float]:
    """
    Proyecta puntos finales del quarter.
    Usa pace actual con ajuste de momentum.
    """
    if elapsed <= 0:
        return score_home + 1.5 * mins_left, score_away + 1.5 * mins_left
    
    home_ppm = max(1.5, score_home / elapsed)
    away_ppm = max(1.5, score_away / elapsed)
    
    # Ajuste por momentum (si momentum favorece a un equipo)
    if momentum > 0:  # Home momentum
        home_ppm *= (1 + momentum * 0.05)
        away_ppm *= (1 - momentum * 0.03)
    elif momentum < 0:  # Away momentum
        away_ppm *= (1 + abs(momentum) * 0.05)
        home_ppm *= (1 - abs(momentum) * 0.03)
    
    projected_home = score_home + home_ppm * mins_left
    projected_away = score_away + away_ppm * mins_left
    
    return round(projected_home, 1), round(projected_away, 1)


def analizar_quarter_en_vivo(
    match_id: str,
    quarter: str,
    qtr_home_score: int,
    qtr_away_score: int,
    total_home_score: int,
    total_away_score: int,
    elapsed_minutes: float,
    graph_points: list[dict],
    pbp_events: list[dict],
) -> LiveAnalysis:
    """
    Análisis completo de un quarter en vivo.
    Genera líneas justas para todos los mercados.
    """
    minutes_left = 12.0 - elapsed_minutes
    score_diff = qtr_home_score - qtr_away_score
    
    # Momentum
    momentum = calcular_momentum(graph_points, elapsed_minutes)
    momentum_dir = "home" if momentum > 2 else ("away" if momentum < -2 else "neutral")
    
    # Recent scoring run
    run_team, run_pts = calcular_recent_scoring_run(pbp_events)
    
    # Proyecciones
    proj_home, proj_away = proyectar_puntos(
        qtr_home_score, qtr_away_score,
        elapsed_minutes, minutes_left,
        momentum
    )
    proj_total = proj_home + proj_away
    proj_diff = proj_away - proj_home  # Positivo = away ahead
    
    # Monte Carlo pesimista
    elapsed = max(elapsed_minutes, 0.1)
    home_ppm = max(1.5, qtr_home_score / elapsed)
    away_ppm = max(1.5, qtr_away_score / elapsed)
    
    mc = monte_carlo_pesimista(
        score_home=qtr_home_score,
        score_away=qtr_away_score,
        home_ppm=home_ppm,
        away_ppm=away_ppm,
        minutes_left=minutes_left,
        momentum_adjustment=momentum,
    )
    
    prob_home = mc["home_win_prob"]
    prob_away = mc["away_win_prob"]
    
    # ───────────────────────────────────────────────────────────────────
    # GENERAR MERCADOS CON LÍNEAS JUSTAS
    # ───────────────────────────────────────────────────────────────────
    
    markets = []
    
    # 1. 1X2 Home
    if prob_home > 0.05:
        fair_home_odd = 1.0 / prob_home
        confidence = "high" if prob_home > 0.60 else ("medium" if prob_home > 0.30 else "low")
        
        if fair_home_odd < 10.0:  # Solo si es realista
            markets.append(LiveMarket(
                name="1x2_home",
                description=f"Home gana el quarter (actual: {qtr_home_score}-{qtr_away_score})",
                line=0,
                our_probability=round(prob_home, 4),
                fair_odds=round(fair_home_odd, 2),
                confidence=confidence,
                recommendation="WATCH" if fair_home_odd > 3.0 else "AVOID",
                when_to_bet=(
                    f"Si tu casa ofrece Home a {fair_home_odd * 1.15:.2f} o mayor → VALUE"
                ),
                reasoning=(
                    f"Proyección: Home {proj_home:.0f} pts, "
                    f"momentum {'a favor' if momentum > 0 else 'en contra'} ({momentum:+.1f})"
                ),
            ))
    
    # 2. 1X2 Away
    if prob_away > 0.05:
        fair_away_odd = 1.0 / prob_away
        confidence = "high" if prob_away > 0.60 else ("medium" if prob_away > 0.30 else "low")
        
        if fair_away_odd < 10.0:
            markets.append(LiveMarket(
                name="1x2_away",
                description=f"Away gana el quarter (actual: {qtr_home_score}-{qtr_away_score})",
                line=0,
                our_probability=round(prob_away, 4),
                fair_odds=round(fair_away_odd, 2),
                confidence=confidence,
                recommendation="WATCH" if fair_away_odd > 3.0 else "AVOID",
                when_to_bet=(
                    f"Si tu casa ofrece Away a {fair_away_odd * 1.15:.2f} o mayor → VALUE"
                ),
                reasoning=(
                    f"Proyección: Away {proj_away:.0f} pts, "
                    f"momentum {'a favor' if momentum < 0 else 'en contra'} ({momentum:+.1f})"
                ),
            ))
    
    # 3. Handicap
    # Handicap justo = diff proyectada ± MAE
    handicap_line = round(proj_diff)
    
    # Probabilidad de Away cubrir handicap (Away -handicap)
    # Si handicap es +7.5, Away necesita ganar por 8+
    away_cubre_prob = 0.5
    if handicap_line > 0:
        # Away favored, necesita ganar por handicap_line+
        away_cubre_prob = prob_away * (1 - MAE_TOTAL / max(proj_total, 10))
    else:
        # Home favored
        away_cubre_prob = prob_away * (1 + MAE_TOTAL / max(proj_total, 10))
    
    away_cubre_prob = max(0.10, min(0.90, away_cubre_prob))
    home_cubre_prob = 1.0 - away_cubre_prob
    
    if handicap_line != 0:
        # Away -handicap
        if away_cubre_prob > 0.30:
            fair_away_hcap_odd = 1.0 / away_cubre_prob
            markets.append(LiveMarket(
                name=f"handicap_away_{abs(handicap_line)}",
                description=f"Away -{abs(handicap_line)} (necesita ganar por {abs(handicap_line)+1}+)",
                line=abs(handicap_line),
                our_probability=round(away_cubre_prob, 4),
                fair_odds=round(fair_away_hcap_odd, 2),
                confidence="high" if away_cubre_prob > 0.60 else "medium",
                recommendation="WATCH",
                when_to_bet=(
                    f"Si tu casa ofrece Away -{abs(handicap_line)} a {fair_away_hcap_odd * 1.15:.2f} o mayor → VALUE"
                ),
                reasoning=(
                    f"Diff proyectada: {proj_diff:.0f} pts, "
                    f"MAE: ±{MAE_TOTAL:.1f} pts"
                ),
            ))
        
        # Home +handicap
        if home_cubre_prob > 0.30:
            fair_home_hcap_odd = 1.0 / home_cubre_prob
            markets.append(LiveMarket(
                name=f"handicap_home_{abs(handicap_line)}",
                description=f"Home +{abs(handicap_line)} (puede perder por {abs(handicap_line)}-)",
                line=abs(handicap_line),
                our_probability=round(home_cubre_prob, 4),
                fair_odds=round(fair_home_hcap_odd, 2),
                confidence="high" if home_cubre_prob > 0.60 else "medium",
                recommendation="WATCH",
                when_to_bet=(
                    f"Si tu casa ofrece Home +{abs(handicap_line)} a {fair_home_hcap_odd * 1.15:.2f} o mayor → VALUE"
                ),
                reasoning=(
                    f"Diff proyectada: {proj_diff:.0f} pts, "
                    f"Home puede perder por <= {abs(handicap_line)} pts"
                ),
            ))
    
    # 4. Over/Under
    ou_line = round(proj_total)
    prob_over = 0.5
    # Si proyección > línea, más probable over
    if proj_total > ou_line:
        prob_over = 0.5 + (MAE_TOTAL / max(proj_total, 10)) * 0.3
    else:
        prob_over = 0.5 - (MAE_TOTAL / max(proj_total, 10)) * 0.3
    
    prob_over = max(0.20, min(0.80, prob_over))
    prob_under = 1.0 - prob_over
    
    if proj_total > 10:  # Solo si hay suficientes puntos
        # Over
        fair_over_odd = 1.0 / prob_over
        markets.append(LiveMarket(
            name=f"over_{ou_line}",
            description=f"Over {ou_line} pts en quarter (actual: {qtr_home_score + qtr_away_score})",
            line=ou_line,
            our_probability=round(prob_over, 4),
            fair_odds=round(fair_over_odd, 2),
            confidence="medium",
            recommendation="WATCH",
            when_to_bet=(
                f"Si tu casa ofrece Over {ou_line} a {fair_over_odd * 1.15:.2f} o mayor → VALUE"
            ),
            reasoning=(
                f"Proyección: {proj_total:.0f} pts, "
                f"pace: {(qtr_home_score + qtr_away_score) / max(elapsed_minutes, 0.1):.1f} pts/min"
            ),
        ))
        
        # Under
        fair_under_odd = 1.0 / prob_under
        markets.append(LiveMarket(
            name=f"under_{ou_line}",
            description=f"Under {ou_line} pts en quarter (actual: {qtr_home_score + qtr_away_score})",
            line=ou_line,
            our_probability=round(prob_under, 4),
            fair_odds=round(fair_under_odd, 2),
            confidence="medium",
            recommendation="WATCH",
            when_to_bet=(
                f"Si tu casa ofrece Under {ou_line} a {fair_under_odd * 1.15:.2f} o mayor → VALUE"
            ),
            reasoning=(
                f"Proyección: {proj_total:.0f} pts, "
                f"actual va {'por encima' if (qtr_home_score + qtr_away_score) > ou_line * (elapsed_minutes/12) else 'por debajo'} del ritmo"
            ),
        ))
    
    # Recomendación general
    best_markets = [m for m in markets if m.recommendation == "WATCH" and m.fair_odds < 8.0]
    if best_markets:
        best_market = max(best_markets, key=lambda m: m.our_probability)
        overall_rec = "WATCH"
    else:
        best_market = "none"
        overall_rec = "AVOID"
    
    reasoning_parts = [
        f"Quarter va {qtr_home_score}-{qtr_away_score} ({elapsed_minutes:.0f} min)",
        f"Proyección final: {proj_home:.0f}-{proj_away:.0f} (diff {proj_diff:.0f})",
        f"Momentum: {momentum_dir} ({momentum:+.1f})",
        f"Racha reciente: {run_team} ({run_pts} pts)" if run_team != "none" else "Sin racha clara",
        f"Total proyectado: {proj_total:.0f} pts",
    ]
    
    return LiveAnalysis(
        match_id=match_id,
        quarter=quarter,
        timestamp=datetime.now().isoformat(),
        qtr_home_score=qtr_home_score,
        qtr_away_score=qtr_away_score,
        total_home_score=total_home_score,
        total_away_score=total_away_score,
        score_diff=score_diff,
        elapsed_minutes=elapsed_minutes,
        minutes_remaining=minutes_left,
        graph_momentum=momentum,
        momentum_direction=momentum_dir,
        recent_scoring_run=f"{run_team} ({run_pts} pts)",
        projected_home_pts=proj_home,
        projected_away_pts=proj_away,
        projected_total_pts=proj_total,
        projected_diff=proj_diff,
        markets=markets,
        overall_recommendation=overall_rec,
        best_market=best_market.name if best_market != "none" else "none",
        reasoning=" | ".join(reasoning_parts),
    )


def analizar_match_db(match_id: str, quarter: str = "Q4") -> LiveAnalysis | None:
    """
    Analiza un match del DB como ejemplo de análisis en vivo.
    Usa datos históricos como simulación.
    """
    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)
    
    data = db_mod.get_match(conn, match_id)
    if data is None:
        conn.close()
        return None
    
    quarters = data["score"].get("quarters", {})
    q = quarters.get(quarter)
    if not q:
        conn.close()
        return None
    
    q_home = int(q.get("home", 0))
    q_away = int(q.get("away", 0))
    
    # Calcular scores acumulados
    q_order = ["Q1", "Q2", "Q3", "Q4"]
    q_idx = q_order.index(quarter)
    
    total_home = 0
    total_away = 0
    for i in range(q_idx + 1):
        q_data = quarters.get(q_order[i], {})
        total_home += int(q_data.get("home", 0))
        total_away += int(q_data.get("away", 0))
    
    # Simular "en vivo" a mitad del quarter (6 min)
    elapsed = 6.0
    qtr_home_half = q_home // 2
    qtr_away_half = q_away // 2
    
    pbp = data.get("play_by_play", {}).get(quarter, [])
    gp = data.get("graph_points", [])
    
    # Filtrar graph points hasta minuto simulado
    cutoff = (q_idx * 12) + elapsed
    gp_filtered = [p for p in gp if int(p.get("minute", 0)) <= cutoff]
    
    conn.close()
    
    return analizar_quarter_en_vivo(
        match_id=match_id,
        quarter=quarter,
        qtr_home_score=qtr_home_half,
        qtr_away_score=qtr_away_half,
        total_home_score=total_home,
        total_away_score=total_away,
        elapsed_minutes=elapsed,
        graph_points=gp_filtered,
        pbp_events=pbp[:10],  # Últimos 10 eventos
    )


def print_analisis_vivo(analysis: LiveAnalysis):
    """Imprime análisis completo en formato legible."""
    print("\n" + "="*70)
    print(f"V12 LIVE BOOKMAKER - Análisis en Vivo")
    print("="*70)
    print(f"Match ID:    {analysis.match_id}")
    print(f"Quarter:     {analysis.quarter}")
    print(f"Timestamp:   {analysis.timestamp}")
    print()
    print(f"{'─'*70}")
    print(f"MARCADOR ACTUAL:")
    print(f"{'─'*70}")
    print(f"  Quarter:   {analysis.qtr_home_score} - {analysis.qtr_away_score}")
    print(f"  Total:     {analysis.total_home_score} - {analysis.total_away_score}")
    print(f"  Diff:      {analysis.score_diff:+d}")
    print(f"  Tiempo:    {analysis.elapsed_minutes:.0f} min jugados, {analysis.minutes_remaining:.0f} restantes")
    print()
    print(f"{'─'*70}")
    print(f"MOMENTUM Y RITMO:")
    print(f"{'─'*70}")
    print(f"  Graph Momentum: {analysis.graph_momentum:+.1f} ({analysis.momentum_direction})")
    print(f"  Racha reciente: {analysis.recent_scoring_run}")
    print()
    print(f"{'─'*70}")
    print(f"PROYECCIONES:")
    print(f"{'─'*70}")
    print(f"  Home:  {analysis.qtr_home_score} → {analysis.projected_home_pts:.0f} pts")
    print(f"  Away:  {analysis.qtr_away_score} → {analysis.projected_away_pts:.0f} pts")
    print(f"  Diff:  {analysis.score_diff:+d} → {analysis.projected_diff:+.0f} pts")
    print(f"  Total: {analysis.qtr_home_score + analysis.qtr_away_score} → {analysis.projected_total_pts:.0f} pts")
    print()
    print(f"{'─'*70}")
    print(f"MERCADOS CON LÍNEAS JUSTAS:")
    print(f"{'─'*70}")
    
    for m in analysis.markets:
        print(f"\n  {m.description}")
        print(f"    Nuestra prob:  {m.our_probability:.1%}")
        print(f"  Odds justas:     {m.fair_odds:.2f}")
        print(f"    Confianza:     {m.confidence.upper()}")
        print(f"  Recomendación:   {m.recommendation}")
        print(f"  → {m.when_to_bet}")
        print(f"    ({m.reasoning})")
    
    print()
    print(f"{'─'*70}")
    print(f"RECOMENDACIÓN GENERAL:")
    print(f"{'─'*70}")
    print(f"  Acción:         {analysis.overall_recommendation}")
    print(f"  Mejor mercado:  {analysis.best_market}")
    print()
    print(f"  {analysis.reasoning}")
    print()
    print(f"{'='*70}")
    print(f"NOTA: Compara estas líneas justas con tu casa de apuestas.")
    print(f"Si tu casa ofrece MEJORES odds que las 'fair_odds' + 15%,")
    print(f"hay VALUE. Si ofrece PEORES, NO apostes.")
    print(f"{'='*70}\n")


def demo_analisis():
    """Ejecuta demo con matches históricos."""
    print("\n" + "="*70)
    print("V12 LIVE BOOKMAKER - DEMO CON DATOS HISTÓRICOS")
    print("="*70)
    print("Simulando análisis 'en vivo' a mitad del quarter (min 6)")
    print()
    
    conn = db_mod.get_conn(str(DB_PATH))
    db_mod.init_db(conn)
    
    # Obtener algunos matches recientes
    rows = conn.execute("""
        SELECT match_id, home_team, away_team, league, date
        FROM matches 
        WHERE status_type = 'finished'
        ORDER BY date DESC, time DESC
        LIMIT 5
    """).fetchall()
    
    conn.close()
    
    for i, row in enumerate(rows):
        match_id = str(row["match_id"])
        home = row["home_team"]
        away = row["away_team"]
        league = row["league"]
        date = row["date"]
        
        print(f"\n{'='*70}")
        print(f"Match {i+1}/5: {home} vs {away}")
        print(f"Liga: {league} | Fecha: {date}")
        print(f"{'='*70}")
        
        analysis = analizar_match_db(match_id, "Q4")
        if analysis:
            print_analisis_vivo(analysis)
        else:
            print(f"  No se pudo analizar este match.")
        
        if i < len(rows) - 1:
            input("\nPresiona Enter para continuar...")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="V12 Live Bookmaker")
    parser.add_argument("--match-id", type=str, help="Analizar match específico")
    parser.add_argument("--quarter", choices=["Q3", "Q4"], default="Q4")
    parser.add_argument("--demo", action="store_true", help="Demo con matches históricos")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    
    args = parser.parse_args()
    
    if args.match_id:
        analysis = analizar_match_db(args.match_id, args.quarter)
        if analysis:
            if args.json:
                print(json.dumps({
                    "match_id": analysis.match_id,
                    "quarter": analysis.quarter,
                    "score": f"{analysis.qtr_home_score}-{analysis.qtr_away_score}",
                    "projections": {
                        "home": analysis.projected_home_pts,
                        "away": analysis.projected_away_pts,
                        "total": analysis.projected_total_pts,
                        "diff": analysis.projected_diff,
                    },
                    "momentum": analysis.graph_momentum,
                    "markets": [
                        {
                            "name": m.name,
                            "description": m.description,
                            "our_probability": m.our_probability,
                            "fair_odds": m.fair_odds,
                            "confidence": m.confidence,
                            "recommendation": m.recommendation,
                            "when_to_bet": m.when_to_bet,
                        }
                        for m in analysis.markets
                    ],
                    "overall_recommendation": analysis.overall_recommendation,
                    "best_market": analysis.best_market,
                    "reasoning": analysis.reasoning,
                }, indent=2))
            else:
                print_analisis_vivo(analysis)
        else:
            print(f"Match {args.match_id} no encontrado o sin datos suficientes.")
    
    elif args.demo:
        demo_analisis()
    
    else:
        # Default: demo
        demo_analisis()
