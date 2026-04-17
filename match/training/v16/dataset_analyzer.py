"""
dataset_analyzer.py - Analisis inteligente del dataset para v16.

Objetivo: decidir QUE ligas entran al catalogo de training con una
justificacion basada en datos, no con un simple threshold de muestras.

Score por liga (0-100):
  * volume_score      (40 pts) - volumen total de muestras
  * recency_score     (30 pts) - partidos en ultimos 14/30 dias
  * consistency_score (15 pts) - varianza de partidos por semana (estabilidad)
  * gender_score      (15 pts) - penaliza ligas femeninas con poco historico

Si score >= config.LEAGUE_ACTIVATION_MIN_SCORE -> ACTIVAR
Si no -> DESACTIVAR con razon explicita.

Salida:
  - JSON con scoring por liga
  - CSV para revisar manualmente
  - Log de ligas activadas vs dormidas (con razon)
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from training.v16 import config

DB_PATH = Path(__file__).resolve().parent.parent.parent / "matches.db"
OUT_DIR = config.V16_DIR / "model_outputs"


@dataclass
class LeagueAssessment:
    league: str
    total_samples: int
    last_30_days: int
    last_7_days: int
    weeks_active: int
    samples_per_week_mean: float
    samples_per_week_std: float
    first_seen: str
    last_seen: str
    is_women: bool
    volume_score: float
    recency_score: float
    consistency_score: float
    gender_score: float
    total_score: float
    decision: str        # "activate" | "dormant" | "reevaluate"
    reason: str


def _is_women(league: str) -> bool:
    lg = (league or "").lower()
    return any(k in lg for k in config.WOMEN_KEYWORDS)


def _fetch_league_data(conn: sqlite3.Connection) -> list[tuple]:
    cur = conn.cursor()
    cur.execute("SELECT MIN(date), MAX(date) FROM matches WHERE date IS NOT NULL")
    min_d, max_d = cur.fetchone()

    cur.execute(f"""
        SELECT
            league,
            COUNT(*) AS total,
            SUM(CASE WHEN date >= date('{max_d}', '-30 day') THEN 1 ELSE 0 END) AS l30,
            SUM(CASE WHEN date >= date('{max_d}', '-7 day')  THEN 1 ELSE 0 END) AS l7,
            COUNT(DISTINCT strftime('%Y-%W', date)) AS weeks,
            MIN(date) AS first_seen,
            MAX(date) AS last_seen
        FROM matches
        WHERE date IS NOT NULL AND league IS NOT NULL AND league != ''
        GROUP BY league
        HAVING total >= 20
        ORDER BY total DESC
    """)
    return cur.fetchall()


def _compute_weekly_stats(conn: sqlite3.Connection, league: str) -> tuple[float, float]:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM matches WHERE league=? AND date IS NOT NULL "
        "GROUP BY strftime('%Y-%W', date)",
        (league,),
    )
    counts = [c[0] for c in cur.fetchall()]
    if not counts:
        return 0.0, 0.0
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / max(len(counts), 1)
    return mean, var ** 0.5


def _volume_score(total: int) -> float:
    """Score 0-40 basado en volumen total."""
    if total >= 500:
        return 40.0
    if total >= 300:
        return 35.0
    if total >= 200:
        return 28.0
    if total >= 150:
        return 22.0
    if total >= 100:
        return 15.0
    if total >= 50:
        return 8.0
    return 2.0


def _recency_score(last_30: int, last_7: int) -> float:
    """Score 0-30 basado en actividad reciente."""
    pts = 0.0
    # Por actividad en 30 dias
    if last_30 >= 50:
        pts += 18.0
    elif last_30 >= 30:
        pts += 14.0
    elif last_30 >= 15:
        pts += 10.0
    elif last_30 >= 5:
        pts += 6.0
    # Por actividad en 7 dias
    if last_7 >= 15:
        pts += 12.0
    elif last_7 >= 8:
        pts += 9.0
    elif last_7 >= 3:
        pts += 6.0
    elif last_7 >= 1:
        pts += 3.0
    return min(pts, 30.0)


def _consistency_score(mean: float, std: float, weeks: int) -> float:
    """Score 0-15 basado en estabilidad de volumen semanal."""
    if weeks < 3 or mean < 1:
        return 0.0
    # Coeficiente de variacion
    cv = std / max(mean, 0.1)
    if cv < 0.3:
        return 15.0
    if cv < 0.6:
        return 12.0
    if cv < 1.0:
        return 8.0
    if cv < 1.5:
        return 4.0
    return 1.0


def _gender_score(is_women: bool, total: int) -> float:
    """Penaliza mujeres con poco historico. Hombres pasan full."""
    if not is_women:
        return 15.0
    # Ligas femeninas: solo pasan si tienen volumen generoso
    if total >= 600:
        return 15.0
    if total >= 400:
        return 10.0
    if total >= 200:
        return 4.0
    return 0.0


def _decision_and_reason(
    assessment_partial: dict,
    total_score: float,
    total: int,
    last_30: int,
) -> tuple[str, str]:
    """Decide activate/dormant/reevaluate con razon explicita."""
    if total_score >= config.LEAGUE_ACTIVATION_MIN_SCORE:
        if last_30 < config.LEAGUE_ACTIVATION_MIN_RECENT:
            return (
                "dormant",
                f"score {total_score:.1f} suficiente pero solo {last_30} partidos "
                f"en ultimos 30 dias (minimo {config.LEAGUE_ACTIVATION_MIN_RECENT}). "
                f"Probable temporada terminada."
            )
        return (
            "activate",
            f"score {total_score:.1f}>={config.LEAGUE_ACTIVATION_MIN_SCORE}, "
            f"{total} muestras totales, {last_30} en ultimos 30 dias."
        )
    # No activate
    if total < 150:
        return (
            "dormant",
            f"volumen insuficiente ({total} muestras). "
            f"Esperar a acumular mas data."
        )
    if last_30 < 5:
        return (
            "dormant",
            f"volumen ok ({total}) pero sin actividad reciente ({last_30} "
            f"partidos ultimos 30d). Ya terminaron?"
        )
    if assessment_partial.get("is_women"):
        return (
            "dormant",
            f"liga femenina con volumen {total} (minimo sugerido 400). "
            f"Reevaluar con mas data."
        )
    return (
        "reevaluate",
        f"score {total_score:.1f} limite. Candidata a incluir en siguiente "
        f"ciclo con mas data."
    )


def assess_leagues(db_path: Path = DB_PATH) -> list[LeagueAssessment]:
    conn = sqlite3.connect(str(db_path))
    try:
        rows = _fetch_league_data(conn)
        out: list[LeagueAssessment] = []
        for lg, total, l30, l7, weeks, first, last in rows:
            mean, std = _compute_weekly_stats(conn, lg)
            is_women = _is_women(lg)
            v = _volume_score(total)
            r = _recency_score(l30, l7)
            c = _consistency_score(mean, std, weeks)
            g = _gender_score(is_women, total)
            total_score = v + r + c + g
            partial = {"is_women": is_women}
            decision, reason = _decision_and_reason(
                partial, total_score, total, l30
            )
            out.append(LeagueAssessment(
                league=lg, total_samples=total,
                last_30_days=l30, last_7_days=l7,
                weeks_active=weeks,
                samples_per_week_mean=round(mean, 2),
                samples_per_week_std=round(std, 2),
                first_seen=first or "", last_seen=last or "",
                is_women=is_women,
                volume_score=v, recency_score=r,
                consistency_score=c, gender_score=g,
                total_score=round(total_score, 1),
                decision=decision, reason=reason,
            ))
        out.sort(key=lambda a: (-a.total_score, -a.total_samples))
        return out
    finally:
        conn.close()


def save_report(
    assessments: list[LeagueAssessment],
    out_dir: Path = OUT_DIR,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "league_assessments_v16.json"
    csv_path = out_dir / "league_assessments_v16.csv"

    data = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "activation_threshold": config.LEAGUE_ACTIVATION_MIN_SCORE,
        "n_activate": sum(1 for a in assessments if a.decision == "activate"),
        "n_dormant": sum(1 for a in assessments if a.decision == "dormant"),
        "n_reevaluate": sum(1 for a in assessments if a.decision == "reevaluate"),
        "total_leagues": len(assessments),
        "assessments": [asdict(a) for a in assessments],
    }
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # CSV
    headers = [
        "league", "total", "l30", "l7", "weeks",
        "is_women", "volume_s", "recency_s", "consistency_s", "gender_s",
        "total_score", "decision", "reason",
    ]
    lines = [",".join(headers)]
    for a in assessments:
        row = [
            a.league.replace(",", ";"),
            str(a.total_samples), str(a.last_30_days),
            str(a.last_7_days), str(a.weeks_active),
            "1" if a.is_women else "0",
            f"{a.volume_score:.1f}", f"{a.recency_score:.1f}",
            f"{a.consistency_score:.1f}", f"{a.gender_score:.1f}",
            f"{a.total_score:.1f}", a.decision,
            a.reason.replace(",", ";").replace('"', "'"),
        ]
        lines.append(",".join(row))
    csv_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, csv_path


def get_active_leagues(db_path: Path = DB_PATH) -> list[str]:
    """Devuelve la lista de ligas a incluir en training (ACTIVATE)."""
    assessments = assess_leagues(db_path)
    return [a.league for a in assessments if a.decision == "activate"]


def print_report(assessments: list[LeagueAssessment], top: int = 40):
    activate = [a for a in assessments if a.decision == "activate"]
    dormant = [a for a in assessments if a.decision == "dormant"]
    reeval = [a for a in assessments if a.decision == "reevaluate"]

    print(f"\n{'='*80}")
    print(f"  ANALISIS INTELIGENTE DE LIGAS - V16")
    print(f"{'='*80}")
    print(f"  Total ligas con datos:  {len(assessments)}")
    print(f"  ACTIVATE (entran):      {len(activate)}")
    print(f"  REEVALUATE (proximas):  {len(reeval)}")
    print(f"  DORMANT (esperando):    {len(dormant)}")
    print()
    print(f"{'-'*80}")
    print(f"  {'liga':<40s} {'tot':>5s} {'30d':>4s} {'7d':>4s} {'score':>6s} {'dec':<12s}")
    print(f"{'-'*80}")
    for a in assessments[:top]:
        dec_color = {"activate": "\033[32m", "reevaluate": "\033[33m",
                     "dormant": "\033[31m"}.get(a.decision, "")
        reset = "\033[0m" if dec_color else ""
        print(
            f"  {a.league[:40]:<40s} {a.total_samples:>5d} "
            f"{a.last_30_days:>4d} {a.last_7_days:>4d} "
            f"{a.total_score:>6.1f} "
            f"{dec_color}{a.decision:<12s}{reset}"
        )
    print()
    print(f"  [ACTIVATE] {len(activate)} ligas listas para training:")
    for a in activate:
        print(f"    * {a.league} (score={a.total_score:.1f}, n={a.total_samples})")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    assessments = assess_leagues()
    json_path, csv_path = save_report(assessments)
    print_report(assessments)
    print()
    print(f"  JSON -> {json_path}")
    print(f"  CSV  -> {csv_path}")
