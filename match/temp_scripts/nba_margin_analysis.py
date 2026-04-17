"""
NBA Bet Margin Analysis — roba datos directamente de la DB + inference v13
Sin depender de eval_match_results (que solo almacena señales del monitor).
"""
import sys
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "match"))

DB_PATH = ROOT / "match" / "matches.db"
DATE_FROM = "2026-04-01"
DATE_TO   = "2026-04-15"

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def main():
    conn = get_db()

    # 1. Ver qué hay en eval_match_results para NBA
    cols = [r[1] for r in conn.execute("PRAGMA table_info(eval_match_results)").fetchall()]
    sig_cols = [c for c in cols if "signal" in c]
    print(f"Signal columns in eval_match_results: {sig_cols}")
    print()

    # 2. Buscar matches NBA en la tabla matches con quarter_scores completos
    nba_matches = conn.execute("""
        SELECT m.match_id, m.date, m.home_team, m.away_team, m.league
        FROM matches m
        WHERE LOWER(m.league) LIKE '%nba%'
          AND m.date BETWEEN ? AND ?
        ORDER BY m.date
    """, (DATE_FROM, DATE_TO)).fetchall()

    print(f"NBA matches en matches table ({DATE_FROM}->{DATE_TO}): {len(nba_matches)}")

    # 3. Para cada match, obtener quarter scores
    results = []
    for m in nba_matches:
        mid = m["match_id"]
        qs_rows = conn.execute(
            "SELECT quarter, home, away FROM quarter_scores WHERE match_id = ?", (mid,)
        ).fetchall()
        qs = {r["quarter"].upper(): (r["home"], r["away"]) for r in qs_rows}

        # Solo partidos con Q4 terminado (datos completos)
        if not all(q in qs for q in ("Q1","Q2","Q3","Q4")):
            continue

        results.append({
            "match_id": mid,
            "date": m["date"],
            "home": m["home_team"],
            "away": m["away_team"],
            "league": m["league"],
            "qs": qs,
        })

    print(f"NBA matches con Q1-Q4 completos: {len(results)}")
    if not results:
        conn.close()
        return

    # 4. Correr inference v13 para cada match
    # Gates 4 (UC) y 5 (NBA direction) desactivados para ver bets potenciales
    print("\nCorriendo inference v13 (SIN Gate-3b volumen, Gate-4 UC, Gate-5 NBA — solo umbral base)...\n")
    from match.training.v13 import infer_match_v13 as inf

    # Patch: disable ALL gates except base confidence (Gate 2)
    # — UC gate (Gate 4)
    # — NBA direction gate (Gate 5)
    # — Gate 3b volume boost (NBA has 73 bets -> +0.08 elevates threshold)
    _orig_is_uc          = inf._is_ultra_conservative_league
    _orig_is_nba         = inf._is_nba_league
    _orig_count_league   = inf._count_league_bets
    inf._is_ultra_conservative_league = lambda league: False
    inf._is_nba_league                = lambda league: False
    inf._count_league_bets            = lambda league: 0   # no volume boost

    bets = []
    debug_n = 0
    for r in results:
        for target in ("q3", "q4"):
            try:
                pred = inf.run_inference(r["match_id"], target)
            except Exception as e:
                if debug_n < 3:
                    print(f"  ERROR {r['match_id']} {target}: {e}")
                    debug_n += 1
                continue
            finally:
                pass  # cleanup after loop

            if not pred.get("ok"):
                if debug_n < 3:
                    print(f"  NOT OK {r['match_id']} {target}: {pred.get('reason')}")
                    debug_n += 1
                continue

            pred_obj = pred.get("prediction")
            if pred_obj is None:
                continue

            signal   = getattr(pred_obj, "winner_signal", "") or getattr(pred_obj, "final_signal", "") or ""
            conf_val = getattr(pred_obj, "winner_confidence", 0) or 0
            reason   = getattr(pred_obj, "reasoning", "") or ""

            # Debug first 10
            if debug_n < 10:
                print(f"  [{r['date'][:10]}] {target.upper()} {r['home'][:20]} | sig={signal} conf={conf_val:.3f} | {reason[:70]}")
                debug_n += 1

            if signal not in ("BET_HOME", "BET_AWAY"):
                continue

            pick = "home" if signal == "BET_HOME" else "away"
            conf = conf_val

            qs = r["qs"]
            # Actual quarter score
            qt = target.upper()
            q_home, q_away = qs.get(qt, (None, None))
            if q_home is None:
                continue

            q_margin = (q_home - q_away) if pick == "home" else (q_away - q_home)
            won = q_margin > 0

            # Cumulative score at START of this quarter
            if target == "q3":
                cum_h = (qs.get("Q1",(0,0))[0] or 0) + (qs.get("Q2",(0,0))[0] or 0)
                cum_a = (qs.get("Q1",(0,0))[1] or 0) + (qs.get("Q2",(0,0))[1] or 0)
            else:
                cum_h = sum((qs.get(f"Q{i}",(0,0))[0] or 0) for i in range(1,4))
                cum_a = sum((qs.get(f"Q{i}",(0,0))[1] or 0) for i in range(1,4))

            cum_lead = (cum_h - cum_a) if pick == "home" else (cum_a - cum_h)

            # Handicap needed to cover (if lost)
            hcap = (-q_margin + 0.5) if not won else None

            bets.append({
                "date": r["date"][:10],
                "match": f"{r['home']} vs {r['away']}",
                "target": target.upper(),
                "pick": pick.upper(),
                "conf": conf,
                "cum_lead": cum_lead,
                "q_home": q_home,
                "q_away": q_away,
                "q_margin": q_margin,
                "won": won,
                "hcap": hcap,
                "reasoning": reason,
            })

    # Restore patches
    inf._is_ultra_conservative_league = _orig_is_uc
    inf._is_nba_league                = _orig_is_nba
    inf._count_league_bets            = _orig_count_league

    conn.close()

    if not bets:
        print("No NBA bets generados por v13.")
        return

    # ── Print ──────────────────────────────────────────────────────────────────
    W = 100
    print("=" * W)
    print(f"  NBA - V13 BET MARGIN ANALYSIS  |  {DATE_FROM} -> {DATE_TO}")
    print("=" * W)
    print(f"{'Fecha':<11} {'Q':<3} {'Pick':<5} {'Lead@ini':>9} {'QScore':>9} {'Margen':>7} {'Res':<6} {'HcapNec':>8}  Partido")
    print("-" * W)

    wins, losses = 0, 0
    win_margins, loss_margins, hcaps = [], [], []

    for b in bets:
        tag = "✓ WIN " if b["won"] else "✗ LOSS"
        lead = f"{b['cum_lead']:+d}"
        score = f"{b['q_home']}-{b['q_away']}"
        margin = f"{b['q_margin']:+d}"
        h = f"+{b['hcap']:.1f}" if b["hcap"] else "—"
        c = f"({b['conf']:.0%})"
        print(f"{b['date']:<11} {b['target']:<3} {b['pick']:<5} {lead:>9} {score:>9} {margin:>7} {tag:<6} {h:>8}  {b['match']} {c}")

        if b["won"]:
            wins += 1
            win_margins.append(b["q_margin"])
        else:
            losses += 1
            loss_margins.append(b["q_margin"])
            hcaps.append(b["hcap"])

    total = wins + losses
    print("=" * W)
    print(f"\n  Total: {total} bets  |  {wins}W / {losses}L  |  Hit rate: {100*wins/total:.1f}%  |  Necesario: 71.4% (odds 1.4)")

    if win_margins:
        print(f"  Margen promedio WIN : +{sum(win_margins)/len(win_margins):.1f} pts  (min +{min(win_margins)}, max +{max(win_margins)})")
    if loss_margins:
        print(f"  Margen promedio LOSS:  {sum(loss_margins)/len(loss_margins):.1f} pts  (mejor -{min(abs(x) for x in loss_margins)}, peor -{max(abs(x) for x in loss_margins)})")

    if hcaps:
        print(f"\n  Handicaps necesarios para cubrir cada pérdida:")
        for h in sorted(hcaps):
            print(f"    +{h:.1f}")

        avg_h = sum(hcaps) / len(hcaps)
        pct_35  = 100 * sum(1 for h in hcaps if h <= 3.5)  / len(hcaps)
        pct_65  = 100 * sum(1 for h in hcaps if h <= 6.5)  / len(hcaps)
        pct_105 = 100 * sum(1 for h in hcaps if h <= 10.5) / len(hcaps)
        print(f"\n  Avg handicap necesario : +{avg_h:.1f} pts")
        print(f"  Cubierto con ≤ +3.5   : {pct_35:.0f}% de pérdidas")
        print(f"  Cubierto con ≤ +6.5   : {pct_65:.0f}% de pérdidas")
        print(f"  Cubierto con ≤ +10.5  : {pct_105:.0f}% de pérdidas")

    # ── Por ventaja al inicio del quarter ─────────────────────────────────────
    print(f"\n  Win rate por ventaja/desventaja al inicio del quarter:")
    buckets = [
        ("Ganando +10+",      lambda x: x >= 10),
        ("Ganando  +5 a +9",  lambda x: 5 <= x <= 9),
        ("Ganando  +1 a +4",  lambda x: 1 <= x <= 4),
        ("Igual    0",         lambda x: x == 0),
        ("Perdiendo -1 a -4", lambda x: -4 <= x <= -1),
        ("Perdiendo -5 a -9", lambda x: -9 <= x <= -5),
        ("Perdiendo -10+",    lambda x: x <= -10),
    ]
    for label, fn in buckets:
        sub = [b for b in bets if fn(b["cum_lead"])]
        if not sub:
            continue
        sw = sum(1 for b in sub if b["won"])
        print(f"    {label:<22}: {len(sub):2d} bets  {sw}W/{len(sub)-sw}L  ({100*sw/len(sub):.0f}% win rate)")

    # ── Por confianza ──────────────────────────────────────────────────────────
    print(f"\n  Win rate por banda de confianza:")
    conf_buckets = [
        ("55-59%", lambda c: 0.55 <= c < 0.60),
        ("60-64%", lambda c: 0.60 <= c < 0.65),
        ("65-69%", lambda c: 0.65 <= c < 0.70),
        ("70%+",   lambda c: c >= 0.70),
    ]
    for label, fn in conf_buckets:
        sub = [b for b in bets if fn(b["conf"])]
        if not sub:
            continue
        sw = sum(1 for b in sub if b["won"])
        print(f"    {label:<10}: {len(sub):2d} bets  {sw}W/{len(sub)-sw}L  ({100*sw/len(sub):.0f}% win rate)")

if __name__ == "__main__":
    main()
