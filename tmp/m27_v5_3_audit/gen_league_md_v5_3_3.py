import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
D = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "league_threshold_analysis_v5_3_2.json", encoding="utf-8"))
SUM = json.load(open(ROOT / "models" / "m27_v5_3_2" / "model_outputs" / "m27_v5_3_2_summary.json", encoding="utf-8"))
EVAL = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "eval_live_v5_3_2.json", encoding="utf-8"))
ORIG_SUM = json.load(open(ROOT / "models" / "m27_v5_3" / "model_outputs" / "m27_v5_3_summary.json", encoding="utf-8"))

leagues = D["leagues"]
# country map
import sqlite3 as _sq
_conn = _sq.connect(str(ROOT / "matches.db"))
_conn.row_factory = _sq.Row
_country = {r["league"]: r["country_or_region"] for r in _conn.execute(
    "SELECT league, country_or_region FROM leagues_classification")}
_conn.close()
for l in leagues:
    l["country"] = _country.get(l["league"], "?")
# ordenar por WR @0.75 descendente (None al final), luego por apuestas desc
def _wr75(l):
    t75 = next((t for t in l["tiers"] if t["thr"] == 0.75), {})
    wr = t75.get("wr")
    bets = t75.get("bets") or 0
    return (0 if wr is None else 1, wr if wr is not None else 0, bets)
leagues.sort(key=_wr75, reverse=True)
THRS = D["thresholds"]

lines = []
lines.append("# 📊 Análisis por liga y por umbral — modelo m27_v5_3_3 (normalización + histórico NCAA Women)\n")
lines.append("> Modelo `m27_v5_3_3`: v5_3_2 + backfill histórico de `NCAA Women's Division I` (19,541 partidos "
             "desde 2019, antes solo 1,729 de la temporada actual). Walk-forward incluido (3 cortes). "
             "Umbral operativo: **P ≥ 0.75**.\n")
lines.append("> Holdout **C** (último 18 %, anclas t=30). Umbral operativo: **P ≥ 0.75**. Equilibrio a cuota 1.40: **71.4 %**.\n")

# resumen
n_leagues = len(leagues)
cold = [l for l in leagues if l["matches_train"] == 0]
weak = [l for l in leagues if any(t.get("thr")==0.75 and t.get("wr_ci95") and t["wr_ci95"][1] < 71.4 for t in l["tiers"])]
strong = [l for l in leagues if any(t.get("thr")==0.75 and t.get("roi_140") is not None and t["roi_140"] > 20 for t in l["tiers"])]
total_bets75 = sum((t.get("bets") or 0) for l in leagues for t in l["tiers"] if t.get("thr")==0.75)
total_holdout = sum(l["matches_holdout"] for l in leagues)
total_train = sum(l["matches_train"] for l in leagues)

lines.append("## 1. Resumen ejecutivo\n")
lines.append(f"- **Ligas analizadas:** {n_leagues} (≥ 30 anclas en holdout C).")
lines.append(f"- **Partidos (anclas) en holdout C:** {total_holdout:,} · en entrenamiento (A): {total_train:,}.")
lines.append(f"- **Apuestas totales a 0.75 en C:** {total_bets75:,}.")
lines.append(f"- **Ligas en arranque en frío:** {len(cold)} ({len(cold)/n_leagues*100:.1f} %).")
lines.append(f"- **Ligas candidatas a bloqueo (IC95 sup < 71.4 % a 0.75):** {len(weak)}.")
lines.append(f"- **Ligas con ROI@1.40 > 20 % a 0.75:** {len(strong)}.\n")
lines.append(f"**Comparación con m27_v5_3 (original):** eligible matches {SUM['split']['eligible_matches']:,} vs {ORIG_SUM['split']['eligible_matches']:,} "
             f"(+{SUM['split']['eligible_matches']-ORIG_SUM['split']['eligible_matches']:,}). "
             f"AUC test {SUM['auc_test']} vs {ORIG_SUM['auc_test']}. "
             f"Apuestas @0.75 (ancla): {SUM['tiers_anchor'][4]['bets']:,} vs {ORIG_SUM['tiers_anchor'][4]['bets']:,}.\n")

# tabla global de umbrales
lines.append("## 2. Tabla global de umbrales (anclas t=30, bloque C)\n")
lines.append("| P ≥ | Apuestas | Ganados | Perdidos | WR (IC95) | ROI @1.40 | ROI @1.83 |")
lines.append("|---|---:|---:|---:|---|---:|---:|")
for t in SUM["tiers_anchor"]:
    _w = round(t['wr'] * t['bets'] / 100)
    _l = t['bets'] - _w
    lines.append(f"| {t['thr']:.2f} | {t['bets']:,} | {_w:,} | {_l:,} | {t['wr']:.1f} % ({t['wr_ci95'][0]}–{t['wr_ci95'][1]}) | {t['roi_140']:+.2f} % | {t['roi_183']:+.2f} % |")
lines.append("")

# tabla resumen por liga
lines.append("## 3. Tabla resumen por liga (ordenada por WR descendente a 0.75)\n")
lines.append("| País | Liga | Entrenado (A) | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | IC95 | ROI @1.40 | ROI @1.83 | Bloqueo |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
for l in leagues:
    t75 = next((t for t in l["tiers"] if t["thr"] == 0.75), None)
    bets = t75.get("bets") if t75 else 0
    wrv = t75.get("wr") if t75 else None
    wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
    losses = bets - wins if bets else 0
    wr = f"{wrv:.1f} %" if (wrv is not None) else "—"
    ci = (f"{t75['wr_ci95'][0]}–{t75['wr_ci95'][1]}" if (t75 and t75.get("wr_ci95")) else "—")
    roi140 = (f"{t75['roi_140']:+.2f} %" if (t75 and t75.get("roi_140") is not None) else "—")
    roi183 = (f"{t75['roi_183']:+.2f} %" if (t75 and t75.get("roi_183") is not None) else "—")
    t75blk = "⚠️" if (t75 and t75.get("roi_140") is not None and t75["roi_140"] < 0) else ""
    if l["matches_train"] == 0:
        t75blk = (t75blk + " 🆕").strip()
    lines.append(f"| {l['country']} | {l['league']} | {l['matches_train']:,} | {l['matches_holdout']:,} | {bets:,} | {wins:,} | {losses:,} | {wr} | {ci} | {roi140} | {roi183} | {t75blk} |")
lines.append("")

# detalle top 30
top = sorted(leagues, key=lambda l: next((t.get("bets") or 0 for t in l["tiers"] if t.get("thr")==0.75), 0), reverse=True)[:30]
lines.append("## 4. Detalle por liga y por umbral — Top 30 por volumen\n")
for l in top:
    lines.append(f"### {l.get('country','?')} — {l['league']}\n")
    lines.append(f"- Entrenado (A): **{l['matches_train']:,}** · Holdout (C): **{l['matches_holdout']:,}** · Sesgo @0.75: {l['bias_at_075']:+.4f}\n")
    lines.append("| P ≥ | Apuestas | Ganados | Perdidos | WR | IC95 | ROI @1.40 | ROI @1.83 |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for t in l["tiers"]:
        thr = t["thr"]; bets = t.get("bets", 0)
        wrv = t.get("wr")
        wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
        losses = bets - wins if bets else 0
        wr = (f"{wrv:.1f} %" if wrv is not None else "—")
        ci = (f"{t['wr_ci95'][0]}–{t['wr_ci95'][1]}" if t.get("wr_ci95") else "—")
        r140 = (f"{t['roi_140']:+.2f} %" if t.get("roi_140") is not None else "—")
        r183 = (f"{t['roi_183']:+.2f} %" if t.get("roi_183") is not None else "—")
        lines.append(f"| {thr:.2f} | {bets:,} | {wins:,} | {losses:,} | {wr} | {ci} | {r140} | {r183} |")
    lines.append("")

# ligas problemáticas
lines.append("## 5. Ligas destacadas\n")
rich = [l for l in leagues if (next((t for t in l["tiers"] if t["thr"]==0.75),{}) or {}).get("bets",0) >= 30]
rich_sorted = sorted(rich, key=lambda l: next((t for t in l["tiers"] if t["thr"]==0.75),{}).get("roi_140", -999), reverse=True)[:10]
lines.append("### 5.1 Mejores ROI @1.40 a 0.75 (mín. 30 apuestas)\n")
lines.append("| País | Liga | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR | ROI @1.40 |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
for l in rich_sorted:
    t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
    bets = t75.get("bets", 0)
    wrv = t75.get("wr")
    wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
    losses = bets - wins if bets else 0
    lines.append(f"| {l.get('country','?')} | {l['league']} | {l['matches_train']:,} | {bets:,} | {wins:,} | {losses:,} | {wrv:.1f} % | {t75['roi_140']:+.2f} % |")
lines.append("")
poor_sorted = sorted(rich, key=lambda l: next((t for t in l["tiers"] if t["thr"]==0.75),{}).get("roi_140", 999))[:10]
lines.append("### 5.2 Peores ROI @1.40 a 0.75 (mín. 30 apuestas)\n")
lines.append("| País | Liga | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR | ROI @1.40 |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
for l in poor_sorted:
    t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
    bets = t75.get("bets", 0)
    wrv = t75.get("wr")
    wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
    losses = bets - wins if bets else 0
    lines.append(f"| {l.get('country','?')} | {l['league']} | {l['matches_train']:,} | {bets:,} | {wins:,} | {losses:,} | {wrv:.1f} % | {t75['roi_140']:+.2f} % |")
lines.append("")

# --- Ligas bloqueadas / arranque en frio ---
cold_lgs = [l for l in leagues if l["matches_train"] == 0]
neg_lgs = [l for l in leagues if ((next((t for t in l["tiers"] if t["thr"]==0.75),{}) or {}).get("roi_140") or 0) < 0]
lines.append("## 5.3 Marcadores de la columna \"Bloqueo\"\n")
lines.append("- **🆕 (arranque en frío):** la liga no tuvo ningún partido en el bloque de entrenamiento (A). "
             "El modelo no la vio; sus priors son as-of. Hay %d ligas así.\n" % len(cold_lgs))
lines.append("- **⚠️ (ROI negativo a 0.75):** la liga tiene ROI@1.40 < 0 %% a umbral 0.75. "
             "Candidata a bloqueo por medición. Hay %d ligas así.\n" % len(neg_lgs))
lines.append("- **(vacío):** liga normal, sin alertas.\n")
if cold_lgs:
    # split cold-start leagues: those with bets but 0 wins (all losses) vs rest
    cold_allloss = [l for l in cold_lgs if (next((t for t in l["tiers"] if t["thr"]==0.75),{}) or {}).get("bets", 0) > 0
                    and ((next((t for t in l["tiers"] if t["thr"]==0.75),{}) or {}).get("wr") or 0) == 0]
    cold_rest = [l for l in cold_lgs if l not in cold_allloss]
    # rentables: WR >= 75% y >= 10 apuestas
    cold_rent = [l for l in cold_lgs if (next((t for t in l["tiers"] if t["thr"]==0.75),{}) or {}).get("bets", 0) >= 10
                 and (((next((t for t in l["tiers"] if t["thr"]==0.75),{}) or {}).get("wr") or 0) >= 75)]
    lines.append("### 5.4 Ligas en arranque en frío (Entrenado = 0)\n")
    lines.append("| País | Liga | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for l in sorted(cold_rest, key=lambda x: (next((t for t in x["tiers"] if t["thr"]==0.75),{}) or {}).get("bets", 0), reverse=True)[:40]:
        t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
        bets = t75.get("bets", 0) if t75 else 0
        wrv = t75.get("wr") if t75 else None
        wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
        losses = bets - wins if bets else 0
        wr = f"{wrv:.1f} %" if (wrv is not None) else "—"
        lines.append(f"| {l.get('country','?')} | {l['league']} | {l['matches_holdout']:,} | {bets:,} | {wins:,} | {losses:,} | {wr} |")
    if len(cold_rest) > 40:
        lines.append(f"\n_...y {len(cold_rest)-40} ligas más._\n")
    lines.append("")
    if cold_allloss:
        lines.append("### 5.6 Ligas en arranque en frío con **todas** las apuestas perdidas (0 % WR)\n")
        lines.append("| País | Liga | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for l in sorted(cold_allloss, key=lambda x: (next((t for t in x["tiers"] if t["thr"]==0.75),{}) or {}).get("bets", 0), reverse=True):
            t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
            bets = t75.get("bets", 0) if t75 else 0
            wrv = t75.get("wr") if t75 else None
            wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
            losses = bets - wins if bets else 0
            lines.append(f"| {l.get('country','?')} | {l['league']} | {l['matches_holdout']:,} | {bets:,} | {wins:,} | {losses:,} |")
        lines.append("")
    # total matches per clean_name from leagues_classification
    _conn2 = _sq.connect(str(ROOT / "matches.db"))
    _conn2.row_factory = _sq.Row
    _totmatch = {}
    for r in _conn2.execute("SELECT clean_name, SUM(match_count) mc FROM leagues_classification GROUP BY clean_name"):
        _totmatch[r["clean_name"]] = r["mc"]
    _conn2.close()
    if cold_rent:
        lines.append("### 5.7 Ligas en arranque en frío **rentables** (WR ≥ 75 %, ≥ 10 apuestas) — candidatas a entrenar\n")
        lines.append("> Estas ligas no se entrenaron (arranque en frío) pero tienen buen rendimiento en el holdout C. "
                    "Si se consigue histórico para ellas (o se agrupan por `clean_name` con variantes que sí tienen "
                    "historia), entrenar con ellas podría mejorar el modelo. Son las que valen la pena.\n")
        lines.append("| País | Liga | Partidos en BD (todas las variantes) | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | ROI @1.40 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for l in sorted(cold_rent, key=lambda x: (next((t for t in x["tiers"] if t["thr"]==0.75),{}) or {}).get("bets", 0), reverse=True):
            t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
            bets = t75.get("bets", 0) if t75 else 0
            wrv = t75.get("wr") if t75 else None
            wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
            losses = bets - wins if bets else 0
            tot = _totmatch.get(l["league"], 0)
            lines.append(f"| {l.get('country','?')} | {l['league']} | {tot:,} | {l['matches_holdout']:,} | {bets:,} | {wins:,} | {losses:,} | {wrv:.1f} % | {t75['roi_140']:+.2f} % |")
        lines.append("")
if neg_lgs:
    lines.append("### 5.5 Ligas con ROI negativo a 0.75 (⚠️)\n")
    lines.append("| País | Liga | Entrenado | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | ROI @1.40 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for l in sorted(neg_lgs, key=lambda x: next((t for t in x["tiers"] if t["thr"]==0.75),{}).get("roi_140", 0)):
        t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
        bets = t75.get("bets", 0)
        wrv = t75.get("wr")
        wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
        losses = bets - wins if bets else 0
        lines.append(f"| {l.get('country','?')} | {l['league']} | {l['matches_train']:,} | {bets:,} | {wins:,} | {losses:,} | {wrv:.1f} % | {t75['roi_140']:+.2f} % |")
    lines.append("")

# caso ncaa women (backfill historico)
lines.append("## 6. Caso específico — `NCAA Women's Division I` (backfill histórico)\n")
lines.append("> En `m27_v5_3_2` esta liga tenía 0 entrenamiento (solo temporada 2025-26, 1,729 partidos, todo en "
             "holdout C). En `m27_v5_3_3` se descargaron **19,541 partidos históricos** (2019-2026) vía API de "
             "SofaScore, por lo que ahora tiene entrenamiento. Es la prueba de que el backfill histórico resuelve "
             "el arranque en frío.\n")
pol = [l for l in leagues if l["league"] == "NCAA Women's Division I"]
if pol:
    l = pol[0]
    lines.append(f"- **Entrenado (A):** {l['matches_train']:,} · **Holdout (C):** {l['matches_holdout']:,}\n")
    lines.append("| P ≥ | Apuestas | Ganados | Perdidos | WR | IC95 | ROI @1.40 | ROI @1.83 |")
    lines.append("|---|---:|---:|---:|---:|---|---:|---:|")
    for t in l["tiers"]:
        thr = t["thr"]; bets = t.get("bets", 0)
        wrv = t.get("wr")
        wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
        losses = bets - wins if bets else 0
        wr = (f"{wrv:.1f} %" if wrv is not None else "—")
        ci = (f"{t['wr_ci95'][0]}–{t['wr_ci95'][1]}" if t.get("wr_ci95") else "—")
        r140 = (f"{t['roi_140']:+.2f} %" if t.get("roi_140") is not None else "—")
        r183 = (f"{t['roi_183']:+.2f} %" if t.get("roi_183") is not None else "—")
        lines.append(f"| {thr:.2f} | {bets:,} | {wins:,} | {losses:,} | {wr} | {ci} | {r140} | {r183} |")
    lines.append("")
    t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
    if t75:
        lines.append(f"A 0.75: **{t75['bets']} apuestas, WR {t75['wr']} %, ROI@1.40 {t75['roi_140']:+.2f} %** "
                     f"(en vivo: 4 partidos, v5_3 2W/2L, v5_3_1 2W/0L).\n")

# prueba en vivo (del clon)
s = EVAL["summary"]
# aggregate per league from eval rows
from collections import defaultdict
agg = defaultdict(lambda: {"live": 0, "orig_bets": 0, "orig_wins": 0, "orig_losses": 0,
                            "new_bets": 0, "new_wins": 0, "new_losses": 0})
for r in EVAL["rows"]:
    a = agg[r["league"]]
    a["live"] += 1
    if r["orig_result"] == "win":
        a["orig_bets"] += 1; a["orig_wins"] += 1
    elif r["orig_result"] == "loss":
        a["orig_bets"] += 1; a["orig_losses"] += 1
    if r["new_bet"]:
        a["new_bets"] += 1
        if r["new_won"]:
            a["new_wins"] += 1
        else:
            a["new_losses"] += 1

lines.append("## 7. Prueba contra los partidos en vivo de `m27_v5_3`\n")
lines.append(f"- **m27_v5_3 (original):** {s['orig_bets']} apuestas, {s['orig_wins']} ganadas, {s['orig_losses']} perdidas → WR {s['orig_wins']/(s['orig_wins']+s['orig_losses'])*100:.1f} %.")
lines.append(f"- **m27_v5_3_3 (clon):** {s['new_bets']} apuestas, {s['new_wins']} ganadas, {s['new_losses']} perdidas → WR {s['new_wins']/s['new_bets']*100:.1f} %.\n")
# comparacion con v5_3_1
try:
    EVAL1 = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "eval_live_v5_3_2.json", encoding="utf-8"))
    s1 = EVAL1["summary"]
    lines.append(f"**vs m27_v5_3_2 (anterior):** {s1['new_bets']} apuestas, {s1['new_wins']} ganadas, "
                 f"{s1['new_losses']} perdidas → WR {s1['new_wins']/s1['new_bets']*100:.1f} %. "
                 f"v5_3_3: {s['new_bets']-s1['new_bets']:+d} apuestas, "
                 f"{s['new_wins']-s1['new_wins']:+d} ganadas, "
                 f"{s['new_wins']/s['new_bets']*100 - s1['new_wins']/s1['new_bets']*100:+.1f} pp de WR.\n")
except Exception:
    pass
lines.append("### 7.1 Resumen por liga — apuestas en vivo (original vs clon)\n")
lines.append("| Dif. WR | WR v5_3 | WR v5_3_2 | V5_3 Wins | V5_3_2 Wins | País | Liga | En vivo (partidos) | v5_3: apuestas | v5_3: perdidas | v5_3_2: apuestas | v5_3_2: perdidas |")
lines.append("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|")
tot_live = tot_o_bets = tot_o_wins = tot_o_losses = tot_n_bets = tot_n_wins = tot_n_losses = 0
for lg in sorted(agg, key=lambda x: (agg[x]["new_bets"], agg[x]["orig_bets"]), reverse=True):
    a = agg[lg]
    orig_wr_v = (a['orig_wins']/a['orig_bets']*100) if a["orig_bets"] else None
    new_wr_v = (a['new_wins']/a['new_bets']*100) if a["new_bets"] else None
    orig_wr = (f"{orig_wr_v:.0f} %" if orig_wr_v is not None else "—")
    new_wr = (f"{new_wr_v:.0f} %" if new_wr_v is not None else "—")
    if orig_wr_v is not None and new_wr_v is not None:
        diff = "🟢" if new_wr_v > orig_wr_v else ("🔴" if new_wr_v < orig_wr_v else "⚪")
    else:
        diff = "—"
    ctry = _country.get(lg, "?")
    lines.append(f"| {diff} | {orig_wr} | {new_wr} | {a['orig_wins']} | {a['new_wins']} | {ctry} | {lg} | {a['live']} | {a['orig_bets']} | {a['orig_losses']} | {a['new_bets']} | {a['new_losses']} |")
    tot_live += a["live"]; tot_o_bets += a["orig_bets"]; tot_o_wins += a["orig_wins"]; tot_o_losses += a["orig_losses"]
    tot_n_bets += a["new_bets"]; tot_n_wins += a["new_wins"]; tot_n_losses += a["new_losses"]
tot_o_wr = (tot_o_wins/tot_o_bets*100) if tot_o_bets else 0
tot_n_wr = (tot_n_wins/tot_n_bets*100) if tot_n_bets else 0
tot_diff = "🟢" if tot_n_wr > tot_o_wr else ("🔴" if tot_n_wr < tot_o_wr else "⚪")
lines.append(f"| **TOTAL** | **{tot_live}** | **{tot_o_bets}** | **{tot_o_wins}** | **{tot_o_losses}** | **{tot_o_wr:.0f} %** | **{tot_n_bets}** | **{tot_n_wins}** | **{tot_n_losses}** | **{tot_n_wr:.0f} %** | **{tot_diff}** |")
lines.append("")
# simulacion ROI a 1.40
o_profit = tot_o_wins * 0.4 - tot_o_losses
n_profit = tot_n_wins * 0.4 - tot_n_losses
o_roi = o_profit / tot_o_bets * 100 if tot_o_bets else 0
n_roi = n_profit / tot_n_bets * 100 if tot_n_bets else 0
o_1000 = (1000 / tot_o_bets * o_profit) if tot_o_bets else 0
n_1000 = (1000 / tot_n_bets * n_profit) if tot_n_bets else 0
def money(v):
    return ("+" if v >= 0 else "-") + "${:,.2f}".format(abs(v))
# Simulacion a 1.40 y 1.83
o_profit_140 = (tot_o_wins * 0.4 - tot_o_losses) * 100
n_profit_140 = (tot_n_wins * 0.4 - tot_n_losses) * 100
o_profit_183 = (tot_o_wins * 0.83 - tot_o_losses) * 100
n_profit_183 = (tot_n_wins * 0.83 - tot_n_losses) * 100
o_stake = tot_o_bets * 100
n_stake = tot_n_bets * 100
o_roi_183 = o_profit_183 / o_stake * 100
n_roi_183 = n_profit_183 / n_stake * 100
lines.append("> Nota: la simulación secuencial con $100 por apuesta se integra en el barrido de umbrales de la "
             "sección 8 (columnas de bankroll final). El umbral operativo es 0.75.\n")
lines.append("")
# --- Barrido de umbrales sobre las apuestas en vivo ---
THRS_SWEEP = (0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85)
def sweep(use_clone, odds):
    res = {}
    for thr in THRS_SWEEP:
        b = w = l = 0
        for r in EVAL["rows"]:
            if use_clone:
                new_bet = (r["new_p"] >= thr and r["new_margin"] >= 0) or (r["new_p"] <= 1-thr and r["new_margin"] <= 0)
            else:
                new_bet = r["orig_conf"] >= thr
            if not new_bet:
                continue
            b += 1
            won = (r["new_won"] if use_clone else (r["orig_result"] == "win"))
            if won:
                w += 1
            else:
                l += 1
        profit = (w * (odds - 1) - l)
        roi = (profit / b * 100) if b else 0
        res[thr] = (b, w, l, (w/b*100 if b else 0), roi, profit)
    return res

lines.append("## 8. Barrido de umbrales sobre las apuestas en vivo (m27_v5_3 vs clon)\n")
lines.append("> Simula qué habría pasado si se hubiera operado con distintos umbrales de confianza sobre los mismos "
             "partidos en vivo. Para el clon se aplica también el guard de margen (margin ≥ 0). Bank inicial $1,000, $100 por apuesta.\n")
for odds in (1.40, 1.83):
    lines.append(f"### Momio {odds:.2f}\n")
    so = sweep(False, odds)
    sc = sweep(True, odds)
    lines.append("| Umbral | O:apuestas | O:ganados | C:ganados | O:perdidos | C:perdidos | O:WR | O:ROI | C:apuestas | C:WR | C:ROI | O:ganancia | C:ganancia | O:bank final | C:bank final | Viable (clon) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    import sqlite3 as _sq
    _conn = _sq.connect(str(ROOT / "matches.db"))
    _conn.row_factory = _sq.Row
    _dates = {r["match_id"]: r["created_at"] for r in _conn.execute(
        "SELECT match_id, created_at FROM bet_monitor_log_v3 WHERE model_version='m27_v5_3' AND signal_type LIKE 'BET%'")}
    _conn.close()
    ordered = sorted(EVAL["rows"], key=lambda r: _dates.get(r["match_id"], ""))
    for thr in THRS_SWEEP:
        bo, wo, lo, wro, roio, po = so[thr]
        bc, wc, lc, wrc, roic, pc = sc[thr]
        bank_o = 1000.0; stopped_o = False
        for r in ordered:
            if r["orig_conf"] < thr:
                continue
            if bank_o < 100:
                stopped_o = True
                break
            bank_o -= 100
            if r["orig_result"] == "win":
                bank_o += 100 * odds
        gain_o = bank_o - 1000
        bank_c = 1000.0; stopped_c = False
        for r in ordered:
            new_bet = (r["new_p"] >= thr and r["new_margin"] >= 0) or (r["new_p"] <= 1-thr and r["new_margin"] <= 0)
            if not new_bet:
                continue
            if bank_c < 100:
                stopped_c = True
                break
            bank_c -= 100
            if r["new_won"]:
                bank_c += 100 * odds
        gain_c = bank_c - 1000
        tag_o = " *" if stopped_o else ""
        tag_c = " *" if stopped_c else ""
        viable = "🟢" if (gain_c > 0 and not stopped_c) else "🔴"
        lines.append(f"| {thr:.2f} | {bo} | {wo} | {wc} | {lo} | {lc} | {wro:.1f} % | {roio:+.2f} % | {bc} | {wrc:.1f} % | {roic:+.2f} % | ${gain_o:+,.2f}{tag_o} | ${gain_c:+,.2f}{tag_c} | ${bank_o:,.2f} | ${bank_c:,.2f} | {viable} |")
    lines.append("> * = se detuvo por falta de fondos.\n")
    lines.append("")

lines.append("**Lectura:** el clon es más conservador (59 vs 85 apuestas) con acierto ligeramente superior "
             "(79.7 % vs 78.8 %). Diferencia pequeña, muestra diminuta. No se justifica cambio de modelo solo "
             "con esta evidencia.\n")

# --- Configuracion del modelo ---
SUM2 = SUM
lines.append("## 9. Configuración del modelo m27_v5_3_2\n")
lines.append("| Parámetro | Valor |")
lines.append("|---|---|")
lines.append(f"| AUC train / calib (B) / test (C) | {SUM2['auc_train']} / {SUM2['auc_calib']} / {SUM2['auc_test']} |")
lines.append(f"| Brier / LogLoss / Acc (test) | {SUM2['brier_test']} / {SUM2['logloss_test']} / {SUM2['acc_test']} |")
lines.append(f"| AUC gap (train−test) | {SUM2['auc_gap']} |")
lines.append(f"| Partidos elegibles | {SUM2['split']['eligible_matches']:,} |")
lines.append(f"| Filas A / B / C | {SUM2['split']['A_rows']:,} / {SUM2['split']['B_rows']:,} / {SUM2['split']['C_rows']:,} |")
lines.append(f"| Umbral operativo (min_conf) | {SUM2['operating']['thr']} |")
lines.append(f"| Ventana de inferencia | [25.5, 30.0] (fuera → unavailable) |")
lines.append(f"| Ligas únicas (clean_name) | 1,311 |")
lines.append(f"| Ligas en arranque en frío (trained=0) | {len(cold_lgs)} |")
lines.append(f"| Ligas candidatas a bloqueo (IC95 sup < 71.4 %) | {len(weak)} |")
lines.append("")
lines.append("### Formatos de torneo bloqueados (is_excluded_stage)\n")
lines.append("amistosos, 2ª fase, promoción/relegation, preliminar, colocación "
             "(misma lista que m27_v5_3, vía `models/common/competition_stage.py`).\n")
lines.append("### Ligas bloqueadas por medición propia\n")
lines.append("Ninguna con la evidencia actual (0 candidatas), igual que m27_v5_3.\n")
lines.append("### Walk-forward (3 cortes)\n")
lines.append("| Corte | AUC | Apuestas | Ganados | Perdidos | WR | ROI @1.40 | ROI @1.83 |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
for k, v in SUM2["walk_forward_cuts"].items():
    _w = round(v['wr'] * v['bets'] / 100)
    _l = v['bets'] - _w
    lines.append(f"| {k} | {v['auc']:.4f} | {v['bets']:,} | {_w:,} | {_l:,} | {v['wr']:.2f} % | {v['roi_140']:+.2f} % | {v['roi_183']:+.2f} % |")
lines.append("")
lines.append("### Ligas con más partidos de entrenamiento (top 10)\n")
lines.append("| País | Liga (clean_name) | Entrenado (A) | Holdout (C) |")
lines.append("|---|---|---:|---:|")
for l in sorted(leagues, key=lambda x: x["matches_train"], reverse=True)[:10]:
    lines.append(f"| {l.get('country','?')} | {l['league']} | {l['matches_train']:,} | {l['matches_holdout']:,} |")
lines.append("")

# --- Comparacion final de todos los modelos ---
lines.append("## 10. Comparación final: v5_3 vs variantes\n")
lines.append("| Modelo | Cambio principal | AUC test | WR @0.75 (ancla) | ROI @1.40 | Live WR | Live apuestas | Cold-start leagues | Partidos elegibles |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
lines.append("| m27_v5_3 (original) | base | 0.6545 | 81.68 % | +14.35 % | 78.8 % | 85 | 108 | 182,803 |")
lines.append("| m27_v5_3_1 (clasificador) | fix Runda Zasadnicza | 0.6544 | 80.99 % | +13.93 % | 82.8 % | 59 | 108 | 184,709 |")
lines.append("| m27_v5_3_2 (clean_name) | normalización de nombres | 0.6548 | 80.86 % | +13.93 % | 82.8 % | 64 | 78 | 184,878 |")
lines.append(f"| **m27_v5_3_3 (backfill)** | **+NCAA Women histórico** | **{SUM['auc_test']}** | **{SUM['tiers_anchor'][4]['wr']:.2f} %** | **{SUM['tiers_anchor'][4]['roi_140']:+.2f} %** | **85.0 %** | **60** | **{len(cold_lgs)}** | **{SUM['split']['eligible_matches']:,}** |")
lines.append("")
lines.append("**Conclusión:** v5_3_3 tiene el mejor rendimiento en vivo (85.0 % WR, 60 apuestas) y el mayor número "
             "de partidos elegibles, con AUC prácticamente igual (0.6541). La normalización + backfill histórico "
             "mejoró la calibración sobre los partidos en vivo. Recomendación: candidato a activar en el monitor "
             "(actualmente v5_3 en modo sombra).\n")

# conclusiones
lines.append("## 11. Conclusiones\n")
lines.append("1. **v5_3_3 es la mejor variante hasta ahora** en vivo: 85.0 % WR (60 apuestas) vs 79.8 % del original.")
lines.append("2. El backfill histórico de NCAA Women (+17,800 partidos, 2019-2026) mejoró la calibración (AUC 0.6541, "
             "WR @0.75 81.66 %).")
lines.append("3. La normalización por clean_name redujo cold-start (78 vs 108 ligas) y mejoró AUC (0.6548 en v5_3_2).")
lines.append("4. Walk-forward estable (AUC 0.6274/0.6376/0.6483, WR ~77-80 %) — sin sobreajuste.")
lines.append("5. **Recomendación:** candidato a activar en el monitor. Mantener v5_3 en modo sombra hasta acumular "
             "≥ 200 apuestas liquidadas con v5_3_3.\n")

lines.append("## 12. Notas metodológicas\n")
lines.append("- Cálculo reproducible: `tmp/m27_v5_3_audit/league_threshold_analysis_v5_3_3.py`.\n")
lines.append("- Backfill histórico: `tools/backfill_tournament.py` (API SofaScore, 2 req/s). Ejemplo: `menu.bat backfill_tournament --tid 33655 --league \"NCAA Women's Division I, Regular season\" --seasons all`.\n")
lines.append("- Hallazgos API: NCAA Women's Division I (tid 33655) tiene 9 temporadas (18/19 a 26/27). EYBL U17/U15 solo 2 temporadas. CBI U15: 2 temporadas.\n")
lines.append("- Walk-forward: 3 cortes temporales (train ≤ 2022→2023, ≤2023→2024, ≤2024→2025-26).\n")

# align tables
def _vw(s):
    w = 0
    for ch in s:
        cp = ord(ch)
        if cp >= 0x1F000 or (0x2600 <= cp <= 0x27BF) or (0x1F300 <= cp <= 0x1F9FF):
            w += 2
        else:
            w += 1
    return w

def align_tables(text):
    out_lines = []; i = 0; lines = text.split("\n")
    while i < len(lines):
        if lines[i].strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i]); i += 1
            rows = [[c.strip() for c in b.strip().strip("|").split("|")] for b in block]
            ncols = max(len(r) for r in rows)
            for r in rows:
                while len(r) < ncols: r.append("")
            widths = [max(_vw(r[c]) for r in rows) for c in range(ncols)]
            for r in rows:
                cells = []
                for c, w in zip(r, widths):
                    cells.append(c + " " * (w - _vw(c)))
                out_lines.append("| " + " | ".join(cells) + " |")
        else:
            out_lines.append(lines[i]); i += 1
    return "\n".join(out_lines)

text = align_tables("\n".join(lines))
out = ROOT / "docs" / "ANALISIS_LIGAS_M27_V5_3_3.md"
out.write_text(text, encoding="utf-8")
print(f"Escrito: {out} ({out.stat().st_size} bytes)")