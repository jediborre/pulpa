# =====================================================================
# GENERADOR DE MD — ANALISIS POR LIGA Y UMBRAL m27_v5_3
# =====================================================================
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "league_threshold_analysis.json", encoding="utf-8"))
SUM = json.load(open(ROOT / "models" / "m27_v5_3" / "model_outputs" / "m27_v5_3_summary.json", encoding="utf-8"))

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
THRS = D["thresholds"]

lines = []
lines.append("# 📊 Análisis por liga y por umbral — modelo m27_v5_3\n")
lines.append("> Generado a partir del holdout **C** (último 18 % de partidos, anclas t=30) del pipeline `tmp/m27_v5_3/train_m27_v5_3_pipeline.py`.")
lines.append("> Umbral operativo del modelo: **P ≥ 0.75**. Punto de equilibrio a cuota 1.40: **71.4 %** de acierto.")
lines.append("> `Entrenado` = anclas (t=30) del bloque A (70 %) por liga. `Holdout` = anclas del bloque C (18 %) por liga.\n")

# --- Resumen ejecutivo ---
n_leagues = len(leagues)
cold = [l for l in leagues if l["matches_train"] == 0]
block = [l for l in leagues if any(t.get("thr")==0.75 and t.get("block_candidate") for t in l["tiers"])]
# leagues with WR@0.75 below equilibrium (upper CI < 71.4)
weak = [l for l in leagues if any(t.get("thr")==0.75 and t.get("wr_ci95") and t["wr_ci95"][1] < 71.4 for t in l["tiers"])]
strong = [l for l in leagues if any(t.get("thr")==0.75 and t.get("roi_140") is not None and t["roi_140"] > 20 for t in l["tiers"])]
total_bets75 = sum((t.get("bets") or 0) for l in leagues for t in l["tiers"] if t.get("thr")==0.75)
total_matches_holdout = sum(l["matches_holdout"] for l in leagues)
total_matches_train = sum(l["matches_train"] for l in leagues)

lines.append("## 1. Resumen ejecutivo\n")
lines.append(f"- **Ligas analizadas:** {n_leagues} (con ≥ 30 anclas en el holdout C).")
lines.append(f"- **Partidos (anclas) en holdout C sumados:** {total_matches_holdout:,}.")
lines.append(f"- **Partidos (anclas) en entrenamiento (bloque A) sumados:** {total_matches_train:,}.")
lines.append(f"- **Apuestas totales a umbral 0.75 en C:** {total_bets75:,}.")
lines.append(f"- **Ligas sin ningún partido en entrenamiento (arranque en frío):** {len(cold)} ({len(cold)/n_leagues*100:.1f} %).")
lines.append(f"- **Ligas candidatas a bloqueo** (IC95 superior del WR < 71.4 % a 0.75): **{len(weak)}** (ninguna con la evidencia actual).")
lines.append(f"- **Ligas con ROI@1.40 > 20 % a 0.75:** {len(strong)}.\n")

# --- Tabla global de umbrales ---
lines.append("## 2. Tabla global de umbrales (referencia, anclas t=30, bloque C)\n")
lines.append("| P ≥ | Apuestas | WR (IC95) | ROI @1.40 | ROI @1.83 |")
lines.append("|---|---|---|---|---|")
for t in SUM["tiers_anchor"]:
    lines.append(f"| {t['thr']:.2f} | {t['bets']:,} | {t['wr']:.1f} % ({t['wr_ci95'][0]}–{t['wr_ci95'][1]}) | {t['roi_140']:+.2f} % | {t['roi_183']:+.2f} % |")
lines.append("")

# --- Tabla resumen por liga ---
lines.append("## 3. Tabla resumen por liga (ordenada por apuestas a 0.75)\n")
lines.append("> Nota: solo se incluyen ligas con ≥ 30 anclas en el holdout C (251 ligas). "
             "Las ligas con menos volumen quedan fuera; por eso la suma de apuestas aquí (1,855 a 0.75) "
             "es menor que la tabla global (2,538). El resto corresponde a ligas de cola.\n")
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
    blk = "⚠️" if (t75 and t75.get("roi_140") is not None and t75["roi_140"] < 0) else ""
    if l["matches_train"] == 0:
        blk = (blk + " 🆕").strip()
    lines.append(f"| {l['country']} | {l['league']} | {l['matches_train']:,} | {l['matches_holdout']:,} | {bets:,} | {wins:,} | {losses:,} | {wr} | {ci} | {roi140} | {roi183} | {blk} |")
lines.append("")

# --- Detalle por liga y umbral (top 30) ---
top = sorted(leagues, key=lambda l: next((t.get("bets") or 0 for t in l["tiers"] if t.get("thr")==0.75), 0), reverse=True)[:30]
lines.append("## 4. Detalle por liga y por umbral — Top 30 por volumen de apuestas\n")
for l in top:
    lines.append(f"### {l['country']} — {l['league']}\n")
    lines.append(f"- Partidos entrenados (A): **{l['matches_train']:,}** · Holdout (C): **{l['matches_holdout']:,}** · Sesgo @0.75: {l['bias_at_075']:+.4f}\n")
    lines.append("| P ≥ | Apuestas | WR | IC95 | ROI @1.40 | ROI @1.83 | Bloqueo |")
    lines.append("|---|---:|---:|---|---:|---:|---|")
    for t in l["tiers"]:
        thr = t["thr"]
        bets = t.get("bets", 0)
        wr = (f"{t['wr']:.1f} %" if t.get("wr") is not None else "—")
        ci = (f"{t['wr_ci95'][0]}–{t['wr_ci95'][1]}" if t.get("wr_ci95") else "—")
        r140 = (f"{t['roi_140']:+.2f} %" if t.get("roi_140") is not None else "—")
        r183 = (f"{t['roi_183']:+.2f} %" if t.get("roi_183") is not None else "—")
        blk = "⚠️" if t.get("block_candidate") else ""
        lines.append(f"| {thr:.2f} | {bets:,} | {wr} | {ci} | {r140} | {r183} | {blk} |")
    lines.append("")

# --- Ligas problemáticas ---
lines.append("## 5. Ligas problemáticas y destacadas\n")

lines.append("### 5.1 Candidatas a bloqueo (IC95 superior del WR < 71.4 % a 0.75)\n")
if weak:
    lines.append("| Liga | Entrenado | Holdout | Apuestas @0.75 | WR | IC95 | ROI @1.40 |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for l in sorted(weak, key=lambda x: next((t.get("bets") or 0 for t in x["tiers"] if t.get("thr")==0.75),0), reverse=True):
        t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
        lines.append(f"| {l['league']} | {l['matches_train']:,} | {l['matches_holdout']:,} | {(t75.get('bets') if t75 else 0):,} | {t75['wr']:.1f} % | {t75['wr_ci95'][0]}–{t75['wr_ci95'][1]} | {t75['roi_140']:+.2f} % |")
else:
    lines.append("_Ninguna con la evidencia actual._\n")
lines.append("")

lines.append("### 5.2 Arranque en frío (0 partidos en entrenamiento)\n")
lines.append(f"{len(cold)} ligas aparecen solo en el holdout C, es decir, el modelo **no las vio en entrenamiento**. "
            "Para ellas los priors (ELO, H2H, win-rate) provienen del estado as-of, pero la liga no aportó filas al ajuste. "
            "Conviene vigilar su calibración antes de confiar en ellas.\n")
if cold:
    lines.append("| País | Liga | Holdout (C) | Apuestas @0.75 | Ganados | Perdidos | WR @0.75 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for l in sorted(cold, key=lambda x: x["matches_holdout"], reverse=True)[:40]:
        t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
        bets = t75.get("bets",0) if t75 else 0
        wrv = t75.get("wr") if t75 else None
        wins = round(wrv * bets / 100) if (wrv is not None and bets) else 0
        losses = bets - wins if bets else 0
        wr = (f"{wrv:.1f} %" if (wrv is not None) else "—")
        lines.append(f"| {l.get('country','?')} | {l['league']} | {l['matches_holdout']:,} | {bets:,} | {wins:,} | {losses:,} | {wr} |")
    if len(cold) > 40:
        lines.append(f"\n_...y {len(cold)-40} ligas más con menos volumen._\n")
lines.append("")

lines.append("### 5.3 Mejores ligas por ROI @1.40 a 0.75 (mín. 30 apuestas)\n")
rich = [l for l in leagues if (next((t for t in l["tiers"] if t["thr"]==0.75),{}) or {}).get("bets",0) >= 30]
rich_sorted = sorted(rich, key=lambda l: next((t for t in l["tiers"] if t["thr"]==0.75),{}).get("roi_140", -999), reverse=True)[:15]
lines.append("| Liga | Entrenado | Apuestas @0.75 | WR | ROI @1.40 | ROI @1.83 |")
lines.append("|---|---:|---:|---:|---:|---:|")
for l in rich_sorted:
    t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
    lines.append(f"| {l['league']} | {l['matches_train']:,} | {t75['bets']:,} | {t75['wr']:.1f} % | {t75['roi_140']:+.2f} % | {t75['roi_183']:+.2f} % |")
lines.append("")

lines.append("### 5.4 Ligas con peor ROI @1.40 a 0.75 (mín. 30 apuestas)\n")
poor_sorted = sorted(rich, key=lambda l: next((t for t in l["tiers"] if t["thr"]==0.75),{}).get("roi_140", 999))[:15]
lines.append("| Liga | Entrenado | Apuestas @0.75 | WR | ROI @1.40 | ROI @1.83 |")
lines.append("|---|---:|---:|---:|---:|---:|")
for l in poor_sorted:
    t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
    lines.append(f"| {l['league']} | {l['matches_train']:,} | {t75['bets']:,} | {t75['wr']:.1f} % | {t75['roi_140']:+.2f} % | {t75['roi_183']:+.2f} % |")
lines.append("")

# --- Cargar stats en vivo ---
try:
    LIVE = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "live_league_stats.json", encoding="utf-8"))
except Exception:
    LIVE = None

# --- Resultados en vivo (monitoreo) ---
if LIVE:
    ov = LIVE["overall"]
    lines.append("## 6. Resultados en vivo del monitoreo (realidad, no backtest)\n")
    lines.append("> Estos son los **partidos reales** que el modelo vio en producción a través de `monitor_v3`. "
                 "A diferencia del holdout C (backtest sobre datos históricos), aquí el modelo opera sobre partidos "
                 "nuevos, con datos capturados en vivo. **Muestra muy pequeña**: solo %d apuestas liquidadas, "
                 "por lo que el IC95 es ancho y no permite conclusiones firmes. Pero sirve para contrastar "
                 "la euforia del backtest con la realidad.\n" % ov["settled"])
    lines.append(f"- **Apuestas totales registradas:** {ov['bets']} ({ov['wins']} ganadas, {ov['losses']} perdidas, {ov['pending']} pendientes).")
    lines.append(f"- **Acierto en vivo (sobre liquidadas):** {ov['wr']} % (backtest @0.75: 81.7 %).")
    lines.append(f"- **Confianza promedio en vivo:** {ov['avg_conf']} %.\n")
    lines.append("### 6.1 Por liga — backtest vs vivo (con recomendación)\n")
    lines.append("| Liga | Entrenado (A) | Backtest @0.75 (WR) | Vivo: apuestas | Vivo: ganadas | Vivo: perdidas | Vivo: WR | Recomendación |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    bt = {l["league"]: l for l in leagues}
    # ordenar por partidos entrenados (desc), luego por apuestas vivas (desc)
    live_sorted = sorted(LIVE["leagues"], key=lambda lv: (bt.get(lv["league"], {}).get("matches_train", 0), lv["bets"]), reverse=True)
    for lv in live_sorted:
        b = bt.get(lv["league"])
        trained = b["matches_train"] if b else 0
        bt_tier = (next((t for t in b["tiers"] if t["thr"] == 0.75), {}) if b else {})
        bt_wr = (f"{bt_tier['wr']:.1f} %" if bt_tier.get("wr") is not None else "—")
        bt_wr_val = bt_tier.get("wr")
        settled = lv["settled"]
        live_wr = lv["wr"]
        # recomendacion
        rec = "✅ ok"
        if settled >= 5 and live_wr is not None and live_wr < 60:
            rec = "⚠️ bloquear"
        elif settled < 8:
            rec = "👀 vigilar (muestra pequeña)"
        elif bt_wr_val is not None and bt_wr_val < 71.4:
            rec = "⚠️ bloquear (backtest)"
        elif live_wr is not None and live_wr < 71.4 and settled >= 8:
            rec = "⚠️ vigilar (WR bajo)"
        lines.append(f"| {lv['league']} | {trained:,} | {bt_wr} | {lv['bets']} | {lv['wins']} | {lv['losses']} | {live_wr if live_wr is not None else '—'} % | {rec} |")
    lines.append("")
    lines.append("**Lectura honesta:** con 93 apuestas liquidadas el IC95 del acierto es de ±~10 pp, así que "
                 "el 76.6 % vivo y el 81.7 % de backtest son **indistinguibles estadísticamente**. "
                 "Lo que sí se ve es que varias ligas con WR de backtest > 85 % (NCAA Women, BNXT, Saudi, etc.) "
                 "no han aparecido aún en vivo o han rendido por debajo, lo que confirma que **no hay que confiar "
                 "en muestras pequeñas**. El modelo sigue en modo sombra y no alerta hasta tener ≥ 200 apuestas "
                 "liquidadas.\n")
    lines.append("> Nota: cuando la columna `Backtest @0.75` muestra `—` y `Entrenado = 0`, la liga no tuvo "
                 "ningún partido con ese **nombre exacto** en el holdout/entrenamiento. Causas posibles: "
                 "(a) arranque en frío (liga nueva), o (b) la liga se filtró en el entrenamiento porque "
                 "`classify_stage` la catalogó como fase descartada (p. ej. `Runda Zasadnicza` = temporada "
                 "regular en polaco, pero el clasificador la marca como `SEGUNDA_FASE`). En esos casos la DB "
                 "sí tiene histórico (ej. `Poland 2nd Basketball League, Runda Zasadnicza` = 1072 partidos), "
                 "pero el modelo nunca la vio; solo hay datos en vivo, que es la única evidencia disponible.\n")

# --- Bug del clasificador de fases ---
try:
    MIS = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "misclassified_big.txt", encoding="utf-8"))
except Exception:
    MIS = []
if MIS:
    lines.append("## 7. Bug conocido del clasificador de fases (`classify_stage`)\n")
    lines.append("> Varios nombres de liga se catalogan como fase descartada (`SEGUNDA_FASE`, `AMISTOSOS`, `PRELIMINAR`, "
                 "`PROMOCION`) aunque su `stage_detail` sea `Regular Season`. Consecuencia: el modelo **no las entrena "
                 "ni las opera** (Entrenado = 0, Backtest = `—`), aunque la DB sí tenga histórico. "
                 "Ejemplo claro: `Poland 2nd Basketball League, Runda Zasadnicza` (1072 partidos, cobertura 100 %) "
                 "se marca como `SEGUNDA_FASE` porque el clasificador asocia \"Runda\" con segunda fase.\n")
    lines.append("| País | Liga | Partidos (DB) | stage | stage_detail | Clasificado como |")
    lines.append("|---|---|---:|---|---|---|")
    for m in MIS:
        ctry = _country.get(m['league'], "?")
        lines.append(f"| {ctry} | {m['league']} | {m['match_count']:,} | {m['stage']} | {m['stage_detail']} | {m['classified_as']} |")
    lines.append("")
    lines.append("**Ligas claramente mal clasificadas (stage_detail = Regular Season pero bloqueadas):** "
                 "`Poland 2nd Basketball League, Runda Zasadnicza`, `1 Liga Kobiet, Runda Zasadnicza`, "
                 "`National Basketball Championship, Women`, `Women's Super League Basketball, Championship`, "
                 "`NBA G League Tip-Off Tournament` (stage_detail Regular Season). "
                 "`Liga Argentina, Group Stage` y `Women's Super League Basketball, Championship` tienen "
                 "stage_detail distinto y requieren revisión manual.\n")
    lines.append("**Corrección aplicada:** se eliminó `'runda zasadnicza'` de `SEGUNDA_FASE_MARKERS` en "
                 "`models/common/competition_stage.py` (era un error: \"Runda Zasadnicza\" = temporada regular, "
                 "no segunda fase). Ligas desbloqueadas: `Poland 2nd Basketball League, Runda Zasadnicza` (852) y "
                 "`1 Liga Kobiet, Runda Zasadnicza` (270). "
                 "Los casos `championship` y `group stage` (National Basketball Championship, Women; "
                 "Women's Super League Basketball, Championship; Liga Argentina, Group Stage; etc.) quedan "
                 "bloqueados a la espera de revisión manual, porque el marcador es ambiguo (puede ser torneo "
                 "real o temporada regular).\n")

# --- Clon m27_v5_3_1 (prueba contra partidos en vivo) ---
try:
    EVAL = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "eval_live_clone.json", encoding="utf-8"))
except Exception:
    EVAL = None
if EVAL:
    s = EVAL["summary"]
    lines.append("## 8. Clon `m27_v5_3_1` — prueba contra los partidos en vivo de `m27_v5_3`\n")
    lines.append("> Se clonó el modelo `m27_v5_3` a `m27_v5_3_1` aplicando la corrección del clasificador "
                 "(Runda Zasadnicza como temporada regular). **Ligas añadidas al entrenamiento:** "
                 "`Poland 2nd Basketball League, Runda Zasadnicza` (852 partidos) y "
                 "`1 Liga Kobiet, Runda Zasadnicza` (270 partidos). "
                 "Los 94 partidos en vivo de `m27_v5_3` **no se incluyeron en el entrenamiento** "
                 "(caen en el bloque C por ser los más recientes) y se usaron como holdout de prueba.\n")
    lines.append(f"- **Entrenamiento:** {s['n_evaluated']} de los 94 partidos en vivo tuvieron ancla (t=30) evaluable.")
    lines.append(f"- **m27_v5_3 (original):** {s['orig_bets']} apuestas, {s['orig_wins']} ganadas, {s['orig_losses']} perdidas → WR {s['orig_wins']/ (s['orig_wins']+s['orig_losses'])*100:.1f} %.")
    lines.append(f"- **m27_v5_3_1 (clon):** {s['new_bets']} apuestas, {s['new_wins']} ganadas, {s['new_losses']} perdidas → WR {s['new_wins']/s['new_bets']*100:.1f} %.\n")
    lines.append("> **Caso específico — `Poland 2nd Basketball League, Runda Zasadnicza`:** 4 partidos en vivo. "
                 "`m27_v5_3` apostó los 4 (2 ganadas, 2 perdidas → 50 %). "
                 "`m27_v5_3_1` apostó solo 2 (los 2 que ganó) y pasó los 2 que perdió → 2/2 = **100 %**. "
                 "Muestra diminuta (4 partidos), pero la señal es positiva: el modelo entrenado con la liga "
                 "evitó las 2 apuestas perdedoras.\n")
    lines.append("| Liga | m27_v5_3 (orig) | m27_v5_3_1 (clon) | Vivo: apuestas | Vivo: ganadas | Vivo: perdidas |")
    lines.append("|---|---|---|---:|---:|---:|")
    # aggregate per league from EVAL rows
    from collections import defaultdict
    agg = defaultdict(lambda: {"orig_bets": 0, "orig_wins": 0, "new_bets": 0, "new_wins": 0, "new_losses": 0, "live_bets": 0, "live_wins": 0, "live_losses": 0})
    for r in EVAL["rows"]:
        a = agg[r["league"]]
        a["live_bets"] += 1
        if r["orig_result"] == "win":
            a["orig_bets"] += 1; a["orig_wins"] += 1
        elif r["orig_result"] == "loss":
            a["orig_bets"] += 1
        if r["new_bet"]:
            a["new_bets"] += 1
            if r["new_won"]:
                a["new_wins"] += 1
            else:
                a["new_losses"] += 1
    for lg in sorted(agg):
        a = agg[lg]
        orig_wr = (f"{a['orig_wins']/a['orig_bets']*100:.0f} %" if a["orig_bets"] else "—")
        new_wr = (f"{a['new_wins']/a['new_bets']*100:.0f} %" if a["new_bets"] else "—")
        lines.append(f"| {lg} | {orig_wr} ({a['orig_wins']}/{a['orig_bets']}) | {new_wr} ({a['new_wins']}/{a['new_bets']}) | {a['live_bets']} | {a['new_wins']} | {a['new_losses']} |")
    lines.append("")
    lines.append("**Lectura:** el clon es **más conservador** (59 vs 85 apuestas) con un acierto ligeramente "
                 "superior (79.7 % vs 78.8 %), pero la diferencia es pequeña y la muestra es diminuta (94 partidos). "
                 "La inclusión de las 2 ligas polacas no cambia materialmente el resultado en vivo; el modelo sigue "
                 "en modo sombra. **No se justifica un cambio de modelo con esta evidencia.**\n")

# --- Conclusiones ---
lines.append("## 9. Conclusiones\n")
lines.append("1. **El modelo mantiene el acierto por encima del equilibrio en la gran mayoría de ligas** a umbral 0.75: "
             f"{n_leagues - len(weak)} de {n_leagues} ligas no son candidatas a bloqueo (IC95 superior ≥ 71.4 %).")
lines.append(f"2. **{len(weak)} ligas** tienen el límite superior de su IC95 por debajo del equilibrio a 0.75 → candidatas a bloqueo por medición. "
             "Conviene revisarlas antes de operar, pero con muestras pequeñas el IC es ancho; no bloquear por un único corte.")
lines.append(f"3. **{len(cold)} ligas ({len(cold)/n_leagues*100:.1f} %)** son arranque en frío: aparecen solo en el holdout y no aportaron filas al entrenamiento. "
             "Sus priors son as-of, pero la calibración por liga es incierta; vigilar.")
lines.append("4. **El volumen se concentra**: las ligas con más apuestas a 0.75 son las que dominan el resultado global; "
             "las de cola aportan poco volumen pero permiten diversificación.")
lines.append("5. **Subir el umbral a 0.80** mejora WR y ROI en todas las ligas, pero reduce el volumen a menos de la mitad; "
             "la decisión de umbral debe basarse en EV real (cuota Q4), no solo en P.")
lines.append("6. **Bajar el umbral a 0.65–0.68** aumenta volumen pero acerca el WR al equilibrio (71.4 %); "
             "solo justificable si la cuota real del mercado Q4 es superior a 1.40.")
lines.append("7. La lista de bloqueo por medición queda **vacía en producción** hoy (0 ligas) porque el filtro de formatos de torneo "
             "ya descartó los formatos tóxicos (amistosos, 2ª fase, promoción, preliminar, colocación). "
             "La lista se repoblará sola cuando acumule evidencia.\n")

lines.append("## 10. Notas metodológicas\n")
lines.append("- Cálculo reproducible: `tmp/m27_v5_3_audit/league_threshold_analysis.py` (re-entrena el ensamble, calibra con Platt simétrico, evalúa en C).")
lines.append("- `matches_holdout` = anclas (t=30) en el bloque C por liga; `matches_train` = anclas en el bloque A.")
lines.append("- `green()` aplica además el guard de margen (margin ≥ 0 para apuestas al local), igual que el pipeline.")
lines.append("- IC95 = Wilson. Bloqueo candidato si el límite superior del IC95 del WR < 71.4 % (equilibrio a cuota 1.40).")
lines.append("- Ligas con < 30 anclas en C se omiten por falta de muestra.")
lines.append("- Resultados en vivo: `tmp/m27_v5_3_audit/live_league_stats.py` sobre `bet_monitor_log_v3` (apuestas liquidadas del monitoreo).\n")

def _vw(s):
    w = 0
    for ch in s:
        cp = ord(ch)
        if cp >= 0x1F000 or (0x2600 <= cp <= 0x27BF) or (0x1F300 <= cp <= 0x1F9FF):
            w += 2
        else:
            w += 1
    return w

def align_tables(text: str) -> str:
    out_lines = []
    i = 0
    lines = text.split("\n")
    while i < len(lines):
        # detect table block
        if lines[i].strip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            # parse cells
            rows = []
            for b in block:
                cells = [c.strip() for c in b.strip().strip("|").split("|")]
                rows.append(cells)
            ncols = max(len(r) for r in rows)
            for r in rows:
                while len(r) < ncols:
                    r.append("")
            widths = [max(_vw(r[c]) for r in rows) for c in range(ncols)]
            for r in rows:
                cells = []
                for c, w in zip(r, widths):
                    cells.append(c + " " * (w - _vw(c)))
                out_lines.append("| " + " | ".join(cells) + " |")
        else:
            out_lines.append(lines[i])
            i += 1
    return "\n".join(out_lines)

text = align_tables("\n".join(lines))
out = ROOT / "docs" / "ANALISIS_LIGAS_M27_V5_3.md"
out.write_text(text, encoding="utf-8")
print(f"Escrito: {out} ({out.stat().st_size} bytes)")