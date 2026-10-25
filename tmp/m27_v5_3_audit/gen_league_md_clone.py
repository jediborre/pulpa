import json
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
D = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "league_threshold_analysis_clone.json", encoding="utf-8"))
SUM = json.load(open(ROOT / "models" / "m27_v5_3_1" / "model_outputs" / "m27_v5_3_1_summary.json", encoding="utf-8"))
EVAL = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "eval_live_clone.json", encoding="utf-8"))
ORIG_SUM = json.load(open(ROOT / "models" / "m27_v5_3" / "model_outputs" / "m27_v5_3_summary.json", encoding="utf-8"))

leagues = D["leagues"]
# ordenar por WR @0.75 descendente (None al final), luego por apuestas desc
def _wr75(l):
    t75 = next((t for t in l["tiers"] if t["thr"] == 0.75), {})
    wr = t75.get("wr")
    bets = t75.get("bets") or 0
    return (0 if wr is None else 1, wr if wr is not None else 0, bets)
leagues.sort(key=_wr75, reverse=True)
THRS = D["thresholds"]

lines = []
lines.append("# 📊 Análisis por liga y por umbral — modelo m27_v5_3_1 (clon)\n")
lines.append("> Clon de `m27_v5_3` con la corrección del clasificador: `Runda Zasadnicza` (temporada regular polaca) "
             "dejó de clasificarse como `SEGUNDA_FASE`. Ligas añadidas al entrenamiento: "
             "`Poland 2nd Basketball League, Runda Zasadnicza` (852) y `1 Liga Kobiet, Runda Zasadnicza` (270).\n")
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
             f"(+{SUM['split']['eligible_matches']-ORIG_SUM['split']['eligible_matches']:,} por Runda Zasadnicza). "
             f"AUC test {SUM['auc_test']} vs {ORIG_SUM['auc_test']}. "
             f"Apuestas @0.75 (ancla): {SUM['tiers_anchor'][4]['bets']:,} vs {ORIG_SUM['tiers_anchor'][4]['bets']:,}.\n")

# tabla global de umbrales
lines.append("## 2. Tabla global de umbrales (anclas t=30, bloque C)\n")
lines.append("| P ≥ | Apuestas | WR (IC95) | ROI @1.40 | ROI @1.83 |")
lines.append("|---|---:|---|---:|---:|")
for t in SUM["tiers_anchor"]:
    lines.append(f"| {t['thr']:.2f} | {t['bets']:,} | {t['wr']:.1f} % ({t['wr_ci95'][0]}–{t['wr_ci95'][1]}) | {t['roi_140']:+.2f} % | {t['roi_183']:+.2f} % |")
lines.append("")

# tabla resumen por liga
lines.append("## 3. Tabla resumen por liga (ordenada por WR descendente a 0.75)\n")
lines.append("| Liga | Entrenado (A) | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 | IC95 | ROI @1.40 | ROI @1.83 | Bloqueo |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---|")
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
    lines.append(f"| {l['league']} | {l['matches_train']:,} | {l['matches_holdout']:,} | {bets:,} | {wins:,} | {losses:,} | {wr} | {ci} | {roi140} | {roi183} | {t75blk} |")
lines.append("")

# detalle top 30
top = sorted(leagues, key=lambda l: next((t.get("bets") or 0 for t in l["tiers"] if t.get("thr")==0.75), 0), reverse=True)[:30]
lines.append("## 4. Detalle por liga y por umbral — Top 30 por volumen\n")
for l in top:
    lines.append(f"### {l['league']}\n")
    lines.append(f"- Entrenado (A): **{l['matches_train']:,}** · Holdout (C): **{l['matches_holdout']:,}** · Sesgo @0.75: {l['bias_at_075']:+.4f}\n")
    lines.append("| P ≥ | Apuestas | WR | IC95 | ROI @1.40 | ROI @1.83 |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for t in l["tiers"]:
        thr = t["thr"]; bets = t.get("bets", 0)
        wr = (f"{t['wr']:.1f} %" if t.get("wr") is not None else "—")
        ci = (f"{t['wr_ci95'][0]}–{t['wr_ci95'][1]}" if t.get("wr_ci95") else "—")
        r140 = (f"{t['roi_140']:+.2f} %" if t.get("roi_140") is not None else "—")
        r183 = (f"{t['roi_183']:+.2f} %" if t.get("roi_183") is not None else "—")
        lines.append(f"| {thr:.2f} | {bets:,} | {wr} | {ci} | {r140} | {r183} |")
    lines.append("")

# ligas problemáticas
lines.append("## 5. Ligas destacadas\n")
rich = [l for l in leagues if (next((t for t in l["tiers"] if t["thr"]==0.75),{}) or {}).get("bets",0) >= 30]
rich_sorted = sorted(rich, key=lambda l: next((t for t in l["tiers"] if t["thr"]==0.75),{}).get("roi_140", -999), reverse=True)[:10]
lines.append("### 5.1 Mejores ROI @1.40 a 0.75 (mín. 30 apuestas)\n")
lines.append("| Liga | Entrenado | Apuestas @0.75 | WR | ROI @1.40 |")
lines.append("|---|---:|---:|---:|---:|")
for l in rich_sorted:
    t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
    lines.append(f"| {l['league']} | {l['matches_train']:,} | {t75['bets']:,} | {t75['wr']:.1f} % | {t75['roi_140']:+.2f} % |")
lines.append("")
poor_sorted = sorted(rich, key=lambda l: next((t for t in l["tiers"] if t["thr"]==0.75),{}).get("roi_140", 999))[:10]
lines.append("### 5.2 Peores ROI @1.40 a 0.75 (mín. 30 apuestas)\n")
lines.append("| Liga | Entrenado | Apuestas @0.75 | WR | ROI @1.40 |")
lines.append("|---|---:|---:|---:|---:|")
for l in poor_sorted:
    t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
    lines.append(f"| {l['league']} | {l['matches_train']:,} | {t75['bets']:,} | {t75['wr']:.1f} % | {t75['roi_140']:+.2f} % |")
lines.append("")

# caso polaco
lines.append("## 6. Caso específico — `Poland 2nd Basketball League, Runda Zasadnicza`\n")
lines.append("> Esta liga (852 partidos históricos, cobertura 100 %) fue **excluida** en `m27_v5_3` por el bug del "
             "clasificador (`Runda Zasadnicza` → `SEGUNDA_FASE`). En `m27_v5_3_1` se incluye.\n")
pol = [l for l in leagues if "Poland 2nd" in l["league"] and "Runda" in l["league"]]
if pol:
    l = pol[0]
    lines.append(f"- **Entrenado (A):** {l['matches_train']:,} · **Holdout (C):** {l['matches_holdout']:,}\n")
    lines.append("| P ≥ | Apuestas | WR | IC95 | ROI @1.40 | ROI @1.83 |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for t in l["tiers"]:
        thr = t["thr"]; bets = t.get("bets", 0)
        wr = (f"{t['wr']:.1f} %" if t.get("wr") is not None else "—")
        ci = (f"{t['wr_ci95'][0]}–{t['wr_ci95'][1]}" if t.get("wr_ci95") else "—")
        r140 = (f"{t['roi_140']:+.2f} %" if t.get("roi_140") is not None else "—")
        r183 = (f"{t['roi_183']:+.2f} %" if t.get("roi_183") is not None else "—")
        lines.append(f"| {thr:.2f} | {bets:,} | {wr} | {ci} | {r140} | {r183} |")
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
lines.append(f"- **m27_v5_3_1 (clon):** {s['new_bets']} apuestas, {s['new_wins']} ganadas, {s['new_losses']} perdidas → WR {s['new_wins']/s['new_bets']*100:.1f} %.\n")
lines.append("### 7.1 Resumen por liga — apuestas en vivo (original vs clon)\n")
lines.append("| Liga | En vivo (partidos) | v5_3: apuestas | v5_3: ganadas | v5_3: perdidas | v5_3: WR | v5_3_1: apuestas | v5_3_1: ganadas | v5_3_1: perdidas | v5_3_1: WR | Dif. WR |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
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
    lines.append(f"| {lg} | {a['live']} | {a['orig_bets']} | {a['orig_wins']} | {a['orig_losses']} | {orig_wr} | {a['new_bets']} | {a['new_wins']} | {a['new_losses']} | {new_wr} | {diff} |")
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
lines.append("**Simulación a cuota 1.40 (apuesta flat):**")
lines.append(f"- `m27_v5_3` (original): {tot_o_bets} apuestas → ganancia **{o_profit:+.2f} u.** (ROI {o_roi:+.2f} %). "
             f"Equivalente con 1000 u. de stake total: **{o_1000:+.2f} u.**")
lines.append(f"- `m27_v5_3_1` (clon): {tot_n_bets} apuestas → ganancia **{n_profit:+.2f} u.** (ROI {n_roi:+.2f} %). "
             f"Equivalente con 1000 u. de stake total: **{n_1000:+.2f} u.**")
lines.append("")
lines.append("**Lectura:** el clon es más conservador (59 vs 85 apuestas) con acierto ligeramente superior "
             "(79.7 % vs 78.8 %). Diferencia pequeña, muestra diminuta. No se justifica cambio de modelo solo "
             "con esta evidencia.\n")

# conclusiones
lines.append("## 8. Conclusiones\n")
lines.append("1. La inclusión de las 2 ligas polacas (Runda Zasadnicza) **no cambia materialmente** las métricas globales "
             "(AUC 0.6544 vs 0.6545, +1906 partidos elegibles).")
lines.append("2. En vivo, el clon es más conservador y ligeramente más preciso, pero la muestra es diminuta (94 partidos).")
lines.append("3. El caso polaco es prometedor: evitó las 2 apuestas perdedoras (2/2 = 100 % en 4 partidos).")
lines.append("4. **Recomendación:** mantener `m27_v5_3` en producción (modo sombra) y seguir acumulando evidencia. "
             "El clon puede re-evaluarse cuando haya ≥ 200 apuestas liquidadas.\n")

lines.append("## 9. Notas metodológicas\n")
lines.append("- Cálculo reproducible: `tmp/m27_v5_3_audit/league_threshold_analysis_clone.py`.\n")

# align tables
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
            widths = [max(len(r[c]) for r in rows) for c in range(ncols)]
            for r in rows:
                out_lines.append("| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |")
        else:
            out_lines.append(lines[i]); i += 1
    return "\n".join(out_lines)

text = align_tables("\n".join(lines))
out = ROOT / "docs" / "ANALISIS_LIGAS_M27_V5_3_1.md"
out.write_text(text, encoding="utf-8")
print(f"Escrito: {out} ({out.stat().st_size} bytes)")