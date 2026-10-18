"""
tmp/eda_nuevo_modelo/evaluate_unseen_recent_days_and_live.py
============================================================
Propósito:
1. Descargar en memoria los partidos de los días verdaderamente NO VISTOS por `matches.db`
   ("desde antier": `2026-10-15`, `2026-10-16` y los ya finalizados de `2026-10-17`)
   directamente desde la API móvil de SofaScore, sin bloquear la base de datos.
2. Evaluar el modelo `models/m27_v4` estrictamente en el minuto 27 (`m27` para FIBA 10m y
   `m33` para NBA 12m, antes del Q4) sobre estos partidos 100% nuevos, reportando el
   Win Rate real, desglose por liga y efecto de los filtros adicionales.
3. Ejecutar `models/m27_v4` EN TIEMPO REAL sobre los partidos que están EN VIVO ahora mismo
   en Q3 / Descanso / Q4, mostrando su predicción de las 4 cabezas junto con los momios
   en vivo reales de SofaScore (`/event/{id}/odds/1/all`).
"""
import asyncio
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from match.scraper import _parse
from models.m27_v4.predict import predict
from monitor_v3.core.mobile_client import get_mobile_client


def frac_to_dec(frac_str: str) -> float:
    try:
        if "/" in str(frac_str):
            n, d = str(frac_str).split("/")
            return round(1.0 + float(n) / float(d), 3)
        return round(float(frac_str), 3)
    except Exception:
        return 0.0


async def fetch_and_parse_match_in_memory(client, ev_summary: dict, q_len: int) -> dict | None:
    mid = str(ev_summary.get("id"))
    r_inc, r_grp, r_odd = await asyncio.gather(
        client.request("GET", f"event/{mid}/incidents"),
        client.request("GET", f"event/{mid}/graph"),
        client.request("GET", f"event/{mid}/odds/1/all"),
        return_exceptions=True,
    )
    inc_json = r_inc.json() if (not isinstance(r_inc, Exception) and r_inc and r_inc.status_code == 200) else {}
    grp_json = r_grp.json() if (not isinstance(r_grp, Exception) and r_grp and r_grp.status_code == 200) else {}
    odd_json = r_odd.json() if (not isinstance(r_odd, Exception) and r_odd and r_odd.status_code == 200) else {}

    inc_list = inc_json.get("incidents", []) if isinstance(inc_json, dict) else []
    grp_list = grp_json.get("graphPoints", []) if isinstance(grp_json, dict) else []
    parsed = _parse({"event": ev_summary}, inc_list, grp_list)
    parsed["odds_markets"] = odd_json.get("markets", []) if isinstance(odd_json, dict) else []
    return parsed


async def evaluate_unseen_days_and_live():
    uri = f"file:{ROOT / 'matches.db'}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30)
    conn.row_factory = sqlite3.Row
    existing_ids = {
        str(r[0])
        for r in conn.execute("SELECT match_id FROM matches WHERE date >= '2026-10-14'").fetchall()
    }
    tax_rows = conn.execute("""
        SELECT league, quarter_duration_minutes, is_women, is_youth, is_college, competition_type
        FROM leagues_classification
    """).fetchall()

    tax_map = {
        r["league"]: {
            "q_len": int(r["quarter_duration_minutes"] or 10),
            "is_women": int(r["is_women"] or 0),
            "is_youth": int(r["is_youth"] or 0),
            "is_college": int(r["is_college"] or 0),
            "comp_type": str(r["competition_type"] or ""),
        }
        for r in tax_rows
    }

    client = get_mobile_client()
    unseen_dates = ["2026-10-15", "2026-10-16", "2026-10-17"]
    candidates = []

    print("=" * 90)
    print("1. BUSCANDO PARTIDOS TERMINADOS EN DÍAS NO VISTOS (2026-10-15, 2026-10-16, 2026-10-17)")
    print("=" * 90)

    for d_str in unseen_dates:
        evs = await client.get_all_scheduled_events_for_date(d_str)
        fin = [
            e for e in evs
            if str((e.get("status") or {}).get("type", "")).lower() == "finished"
            and str(e.get("id")) not in existing_ids
            and all(k in (e.get("homeScore") or {}) for k in ("period1", "period2", "period3", "period4"))
        ]
        print(f"  Fecha {d_str}: {len(evs)} eventos programados | {len(fin)} terminados con Q1..Q4 NO VISTOS en matches.db")
        for e in fin:
            e["_date_str"] = d_str
            candidates.append(e)

    # Filtrar cyber / 3x3 antes de descargar detalle
    clean_candidates = []
    for ev in candidates:
        cat = ((ev.get("tournament") or {}).get("category") or {}).get("name", "")
        tourn = (ev.get("tournament") or {}).get("name", "")
        lg_full = f"{cat} - {tourn}" if cat else tourn
        if any(w in lg_full.lower() for w in ["cyber", "virtual", "3x3", "ebasketball"]):
            continue
        clean_candidates.append(ev)

    print(f"\nTotal candidatos reales 100% nuevos (fuera de matches.db): {len(clean_candidates)}")
    # Evaluar una muestra amplia de 150 partidos distribuidos en los 3 días
    step = max(1, len(clean_candidates) // 150)
    sample_candidates = clean_candidates[::step][:150]
    print(f"Descargando detalle (/incidents, /graph, /odds) y evaluando m27_v4 en {len(sample_candidates)} partidos nuevos...")

    results = []
    for ev in sample_candidates:
        mid = str(ev.get("id"))
        cat = ((ev.get("tournament") or {}).get("category") or {}).get("name", "")
        tourn = (ev.get("tournament") or {}).get("name", "")
        lg_full = f"{cat} - {tourn}" if cat else tourn
        t_info = tax_map.get(lg_full) or tax_map.get(tourn) or {
            "q_len": 10,
            "is_women": int("women" in lg_full.lower()),
            "is_youth": int(any(u in lg_full.lower() for u in ["u18", "u19", "u20", "u16", "youth", "u15", "u17"])),
            "is_college": int("ncaa" in lg_full.lower()),
            "comp_type": "",
        }
        if any(k in lg_full.upper() for k in ["NBA", "CBA", "PBA", "NBL"]):
            t_info["q_len"] = 12

        parsed = await fetch_and_parse_match_in_memory(client, ev, t_info["q_len"])
        if not parsed:
            continue

        q_scores = (parsed.get("score") or {}).get("quarters") or {}
        if not all(k in q_scores for k in ("Q1", "Q2", "Q3", "Q4")):
            continue

        q4_h = int(q_scores["Q4"]["home"])
        q4_a = int(q_scores["Q4"]["away"])
        if q4_h == q4_a:
            continue

        parsed["match"]["league"] = lg_full
        snap_min = 33.0 if t_info["q_len"] == 12 else 27.0

        pred = predict(mid, target="q4", match_data=parsed, conn=conn, snapshot_minute=snap_min)
        if not pred.available:
            continue

        raw_p = pred.raw_json or {}
        pick_home = pred.pick == "HOME"
        won = int(q4_h > q4_a) if pick_home else int(q4_a > q4_h)
        real_q4_margin_fav = (q4_h - q4_a) if pick_home else (q4_a - q4_h)
        real_q4_total = q4_h + q4_a

        exp_m = float(raw_p.get("expected_q4_margin", 0.0))
        exp_m_fav = exp_m if pick_home else -exp_m
        exp_tot = float(raw_p.get("expected_q4_total", 40.0))

        q13_h = sum(int(q_scores[q]["home"]) for q in ("Q1", "Q2", "Q3"))
        q13_a = sum(int(q_scores[q]["away"]) for q in ("Q1", "Q2", "Q3"))
        q3_lead_fav = (q13_h - q13_a) if pick_home else (q13_a - q13_h)

        results.append({
            "match_id": mid,
            "date": ev.get("_date_str"),
            "league": lg_full,
            "archetype": raw_p.get("archetype", "fiba_men_10m"),
            "is_youth": t_info["is_youth"],
            "comp_type": t_info["comp_type"],
            "home": parsed["match"]["home_team"],
            "away": parsed["match"]["away_team"],
            "pick": "HOME" if pick_home else "AWAY",
            "conf": float(pred.confidence),
            "won": won,
            "exp_margin_fav": exp_m_fav,
            "real_margin_fav": real_q4_margin_fav,
            "exp_total": exp_tot,
            "real_total": real_q4_total,
            "q3_lead_fav": q3_lead_fav,
        })

    print(f"\nPartidos nuevos evaluados exitosamente en m27/m33: {len(results)}")
    if results:
        df_u = pd.DataFrame(results)
        print("\n--- RESULTADOS EN PARTIDOS 100% NO VISTOS (DESDE ANTIER 2026-10-15 A HOY 2026-10-17) ---")
        cuts = [
            ("Todos los partidos (P >= 0.50)", df_u["conf"] >= 0.50),
            ("Umbral P >= 0.65", df_u["conf"] >= 0.65),
            ("Umbral P >= 0.68 + Gate Margen >= 1.8", (df_u["conf"] >= 0.68) & (df_u["exp_margin_fav"] >= 1.8)),
            ("Umbral P >= 0.70 + Gate Margen >= 1.8 (V4.4 Estándar)", (df_u["conf"] >= 0.70) & (df_u["exp_margin_fav"] >= 1.8)),
            ("Umbral P >= 0.70 + Gate Margen >= 2.2 + Anti-Blowout (q3_lead<=14)", (df_u["conf"] >= 0.70) & (df_u["exp_margin_fav"] >= 2.2) & (df_u["q3_lead_fav"] <= 14) & (df_u["is_youth"] == 0)),
            ("Umbral P >= 0.72 + Gate Margen >= 2.5 + Anti-Blowout (Élite)", (df_u["conf"] >= 0.72) & (df_u["exp_margin_fav"] >= 2.5) & (df_u["q3_lead_fav"] <= 14) & (df_u["is_youth"] == 0)),
        ]
        for label, mask in cuts:
            sub = df_u[mask]
            if len(sub) > 0:
                wr = sub["won"].mean() * 100.0
                be = 100.0 / wr if wr > 0 else 99.0
                roi140 = (wr / 100.0 * 1.40 - 1.0) * 100.0
                print(f"  * {label:<68} -> WR: {wr:5.2f}% ({int(sub['won'].sum())}/{len(sub)}) | BreakEven: {be:.2f} | ROI@1.40: {roi140:+5.2f}%")

        print("\nDetalle de las apuestas (P >= 0.68 + Gate Margen >= 1.8) en los días no vistos:")
        sub_bets = df_u[(df_u["conf"] >= 0.68) & (df_u["exp_margin_fav"] >= 1.8)].sort_values("conf", ascending=False)
        for _, r in sub_bets.head(25).iterrows():
            mark = "✅ WIN " if r["won"] else "❌ LOSS"
            print(f"  [{r['date']}] {mark} | Conf:{r['conf']*100:5.1f}% (ExpM:+{r['exp_margin_fav']:.1f}, RealQ4:{r['real_margin_fav']:+.0f}) | {r['league'][:35]:<35} | {r['home']} vs {r['away']} -> Pick:{r['pick']}")

    # 2. EVALUAR PARTIDOS EN VIVO AHORA MISMO
    print("\n" + "=" * 90)
    print("2. INFERENCIA EN VIVO AHORA MISMO (PARTIDOS EN Q3 / DESCANSO / Q4) + MOMIOS EN VIVO")
    print("=" * 90)
    live_evs = await client.get_live_events()
    for ev in live_evs:
        st_desc = (ev.get("status") or {}).get("description", "")
        cat = ((ev.get("tournament") or {}).get("category") or {}).get("name", "")
        tourn = (ev.get("tournament") or {}).get("name", "")
        lg_full = f"{cat} - {tourn}" if cat else tourn
        if "cyber" in lg_full.lower() or "virtual" in lg_full.lower():
            continue
        parsed = await fetch_and_parse_match_in_memory(client, ev, 10)
        if not parsed:
            continue
        parsed["match"]["league"] = lg_full
        mid = str(ev.get("id"))
        pred = predict(mid, target="q4", match_data=parsed, conn=conn, snapshot_minute=27.0)
        raw_p = pred.raw_json or {}

        live_mkts = []
        for m in parsed.get("odds_markets", []):
            m_name = m.get("marketName", "")
            is_l = m.get("isLive", False)
            line = m.get("choiceGroup", "")
            ch_s = "/".join(f"{c.get('name')}:{frac_to_dec(c.get('fractionalValue'))}" for c in m.get("choices", []))
            if is_l or m_name == "Full time":
                live_mkts.append(f"[{'LIVE' if is_l else 'PRE'}] {m_name}({line})={ch_s}")

        hs = (ev.get("homeScore") or {}).get("current", 0)
        as_ = (ev.get("awayScore") or {}).get("current", 0)
        print(
            f"  * [{st_desc:<11}] {lg_full[:38]:<38} | {parsed['match']['home_team']} ({hs}) vs {parsed['match']['away_team']} ({as_})\n"
            f"    -> m27_v4 Signal: {raw_p.get('bet_signal','?')} | Pick Q4: {pred.pick} (Conf: {pred.confidence*100:.1f}% | E[dQ4]: {raw_p.get('expected_q4_margin',0):+.1f} pts | E[TotQ4]: {raw_p.get('expected_q4_total',0):.1f} [{raw_p.get('q4_over_under_signal','-')}] | P_FT_Home: {raw_p.get('p_ft_home_win',0.5)*100:.1f}%)\n"
            f"    -> Momios SofaScore ({len(parsed.get('odds_markets',[]))} mkts): {' ; '.join(live_mkts[:4]) if live_mkts else 'Sin momios en SofaScore'}"
        )

    conn.close()


if __name__ == "__main__":
    asyncio.run(evaluate_unseen_days_and_live())
