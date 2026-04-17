"""
NBA Bet Margin Analysis
=======================
Para cada apuesta NBA generada por v13 (señal bot_hybrid_f1):
- Muestra el pick (HOME/AWAY), quarter, marcador al inicio del quarter
- Resultado real del quarter
- Margen: +N (ganamos por N pts) / -N (perdimos por N pts)
- Handicap que hubiera necesitado para ganar

Útil para evaluar si una apuesta de handicap sería viable.
"""

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "match" / "matches.db"

TAG = "bot_hybrid_f1"
DATE_FROM = "2026-04-01"
DATE_TO   = "2026-04-15"
LEAGUE_FILTER = "nba"

def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    # Obtener columnas disponibles en eval_match_results
    cols = [r[1] for r in conn.execute("PRAGMA table_info(eval_match_results)").fetchall()]
    q3_sig_col = f"q3_signal__{TAG}"
    q4_sig_col = f"q4_signal__{TAG}"
    q3_conf_col = f"q3_confidence__{TAG}"
    q4_conf_col = f"q4_confidence__{TAG}"

    has_q3_conf = q3_conf_col in cols
    has_q4_conf = q4_conf_col in cols

    # Pull all NBA bets (Q3 + Q4)
    rows = conn.execute(f"""
        SELECT
            e.event_date,
            e.match_id,
            m.home_team,
            m.away_team,
            m.league,
            e.q3_home_score,
            e.q3_away_score,
            e.q3_winner,
            e.q4_home_score,
            e.q4_away_score,
            e.q4_winner,
            e."{q3_sig_col}" AS q3_signal,
            e."{q4_sig_col}" AS q4_signal
            {', e."' + q3_conf_col + '" AS q3_conf' if has_q3_conf else ', NULL AS q3_conf'}
            {', e."' + q4_conf_col + '" AS q4_conf' if has_q4_conf else ', NULL AS q4_conf'}
        FROM eval_match_results e
        JOIN matches m ON m.match_id = e.match_id
        WHERE LOWER(m.league) LIKE ?
          AND e.event_date BETWEEN ? AND ?
        ORDER BY e.event_date, e.match_id
    """, (f"%{LEAGUE_FILTER}%", DATE_FROM, DATE_TO)).fetchall()

    # Also get quarter_scores for cumulative scores (Q1+Q2 at Q3 start, Q1+Q2+Q3 at Q4 start)
    def get_q_scores(match_id):
        qs = {}
        for r in conn.execute(
            "SELECT quarter, home, away FROM quarter_scores WHERE match_id = ?", (match_id,)
        ).fetchall():
            qs[r["quarter"].upper()] = (r["home"], r["away"])
        return qs

    # Collect bets
    bets = []
    for row in rows:
        qs = get_q_scores(row["match_id"])

        for target in ("q3", "q4"):
            signal = row[f"{target}_signal"]
            if signal not in ("BET_HOME", "BET_AWAY", "BET"):
                continue

            # Determine pick direction
            if signal == "BET_HOME":
                pick = "home"
            elif signal == "BET_AWAY":
                pick = "away"
            else:
                # Generic BET — skip, ambiguous
                continue

            # Score of the bet's quarter (actual result)
            q_home = row[f"{target}_home_score"]
            q_away = row[f"{target}_away_score"]
            q_winner = row[f"{target}_winner"]

            if q_home is None or q_away is None:
                continue

            # Quarter margin from pick's perspective
            if pick == "home":
                q_margin = (q_home or 0) - (q_away or 0)
            else:
                q_margin = (q_away or 0) - (q_home or 0)

            won = q_margin > 0

            # Cumulative score at start of this quarter (to compute deficit)
            if target == "q3":
                # At Q3 start: after Q1+Q2
                cum_home = (qs.get("Q1", (0,0))[0] or 0) + (qs.get("Q2", (0,0))[0] or 0)
                cum_away = (qs.get("Q1", (0,0))[1] or 0) + (qs.get("Q2", (0,0))[1] or 0)
            else:
                # At Q4 start: after Q1+Q2+Q3
                cum_home = sum((qs.get(f"Q{i}", (0,0))[0] or 0) for i in range(1, 4))
                cum_away = sum((qs.get(f"Q{i}", (0,0))[1] or 0) for i in range(1, 4))

            if pick == "home":
                cum_lead = cum_home - cum_away
            else:
                cum_lead = cum_away - cum_home

            # Handicap needed: if losing, +N covers the deficit
            # handicap_needed = max(0, -cum_lead) when behind; or 0 when ahead
            # Actually: to WIN the handicap line, need q_margin + handicap > 0
            # → needed handicap = -q_margin + 0.5 (to beat the line, assume line = -q_margin)
            # More useful: what spread would have been needed to cover?
            # If q_margin = -3 → needed handicap of at least +3.5
            handicap_to_cover = (-q_margin + 0.5) if not won else None

            conf = row[f"{target}_conf"]

            bets.append({
                "date": row["event_date"],
                "match": f"{row['home_team']} vs {row['away_team']}",
                "league": row["league"],
                "target": target.upper(),
                "pick": pick.upper(),
                "cum_lead": cum_lead,         # lead at start of quarter (+= ahead, -= behind)
                "q_home": q_home,
                "q_away": q_away,
                "q_margin": q_margin,          # positive = won the quarter
                "won": won,
                "handicap_to_cover": handicap_to_cover,
                "conf": conf,
            })

    conn.close()

    if not bets:
        print(f"No NBA bets found for {DATE_FROM} → {DATE_TO}")
        return

    # ── Print detail ──────────────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"  NBA BET MARGIN ANALYSIS  |  {DATE_FROM} → {DATE_TO}  |  tag={TAG}")
    print(f"{'='*90}")
    print(f"{'Date':<12} {'Quarter':<5} {'Pick':<6} {'Lead@start':>10} {'QScore':>10} {'Margin':>8} {'Result':<7} {'HcapNeeded':>12}  Match")
    print(f"{'-'*90}")

    wins = losses = 0
    win_margins = []
    loss_margins = []
    hcap_needed = []

    for b in bets:
        result = "WIN ✓" if b["won"] else "LOSS ✗"
        lead_str = f"{b['cum_lead']:+d}"
        q_score = f"{b['q_home']}-{b['q_away']}"
        margin_str = f"{b['q_margin']:+d}"
        hcap_str = f"+{b['handicap_to_cover']:.1f}" if b["handicap_to_cover"] else "—"
        conf_str = f" (conf={b['conf']:.3f})" if b["conf"] else ""

        print(f"{b['date']:<12} {b['target']:<5} {b['pick']:<6} {lead_str:>10} {q_score:>10} {margin_str:>8} {result:<7} {hcap_str:>12}  {b['match']}{conf_str}")

        if b["won"]:
            wins += 1
            win_margins.append(b["q_margin"])
        else:
            losses += 1
            loss_margins.append(b["q_margin"])
            hcap_needed.append(b["handicap_to_cover"])

    total = wins + losses

    print(f"\n{'='*90}")
    print(f"  RESUMEN")
    print(f"{'='*90}")
    print(f"  Total bets : {total}")
    print(f"  Wins       : {wins}  ({100*wins/total:.1f}%)")
    print(f"  Losses     : {losses}  ({100*losses/total:.1f}%)")
    if win_margins:
        print(f"  Margin WIN  avg: +{sum(win_margins)/len(win_margins):.1f} pts  (min +{min(win_margins)}, max +{max(win_margins)})")
    if loss_margins:
        print(f"  Margin LOSS avg:  {sum(loss_margins)/len(loss_margins):.1f} pts  (max deficit {min(loss_margins):+d})")
    if hcap_needed:
        print(f"\n  Handicap necesario para cubrir cada pérdida:")
        for h in sorted(hcap_needed):
            print(f"    +{h:.1f}")
        print(f"  → Handicap típico para cubrir: +{sum(hcap_needed)/len(hcap_needed):.1f} pts avg")
        print(f"  → El {100*sum(1 for h in hcap_needed if h <= 6.5)/len(hcap_needed):.0f}% de pérdidas se cubría con handicap ≤ +6.5")
        print(f"  → El {100*sum(1 for h in hcap_needed if h <= 10.5)/len(hcap_needed):.0f}% de pérdidas se cubría con handicap ≤ +10.5")

    # ── Distribution of leads at bet time ────────────────────────────────────
    print(f"\n  Distribución de ventaja/desventaja al inicio del quarter apostado:")
    leads = [b["cum_lead"] for b in bets]
    buckets = [
        ("Ganando +10 o más",  lambda x: x >= 10),
        ("Ganando  +5 a  +9",  lambda x: 5 <= x <= 9),
        ("Ganando  +1 a  +4",  lambda x: 1 <= x <= 4),
        ("Empatado (0)",        lambda x: x == 0),
        ("Perdiendo -1 a -4",  lambda x: -4 <= x <= -1),
        ("Perdiendo -5 a -9",  lambda x: -9 <= x <= -5),
        ("Perdiendo -10 o más",lambda x: x <= -10),
    ]
    for label, fn in buckets:
        subset = [b for b in bets if fn(b["cum_lead"])]
        if not subset:
            continue
        sw = sum(1 for b in subset if b["won"])
        sl = len(subset) - sw
        print(f"    {label:<26}: {len(subset):2d} bets  {sw}W/{sl}L  ({100*sw/len(subset):.0f}% win rate)")

if __name__ == "__main__":
    main()
