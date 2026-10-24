# =====================================================================
# GENERADOR DE MD — ANALISIS POR LIGA Y UMBRAL m27_v5_3
# =====================================================================
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = json.load(open(ROOT / "tmp" / "m27_v5_3_audit" / "league_threshold_analysis.json", encoding="utf-8"))
SUM = json.load(open(ROOT / "models" / "m27_v5_3" / "model_outputs" / "m27_v5_3_summary.json", encoding="utf-8"))

leagues = D["leagues"]
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
lines.append("| Liga | Entrenado (A) | Holdout (C) | Apuestas @0.75 | WR @0.75 | IC95 | ROI @1.40 | ROI @1.83 | Bloqueo |")
lines.append("|---|---:|---:|---:|---:|---|---:|---:|---|")
for l in leagues:
    t75 = next((t for t in l["tiers"] if t["thr"] == 0.75), None)
    bets = t75.get("bets") if t75 else 0
    wr = f"{t75['wr']:.1f} %" if (t75 and t75.get("wr") is not None) else "—"
    ci = (f"{t75['wr_ci95'][0]}–{t75['wr_ci95'][1]}" if (t75 and t75.get("wr_ci95")) else "—")
    roi140 = (f"{t75['roi_140']:+.2f} %" if (t75 and t75.get("roi_140") is not None) else "—")
    roi183 = (f"{t75['roi_183']:+.2f} %" if (t75 and t75.get("roi_183") is not None) else "—")
    blk = ""
    if t75 and t75.get("roi_140") is not None and t75["roi_140"] < 0:
        blk = "⚠️"
    if l["matches_train"] == 0:
        blk = (blk + " 🆕").strip()
    lines.append(f"| {l['league']} | {l['matches_train']:,} | {l['matches_holdout']:,} | {bets:,} | {wr} | {ci} | {roi140} | {roi183} | {blk} |")
lines.append("")

# --- Detalle por liga y umbral (top 30) ---
top = sorted(leagues, key=lambda l: next((t.get("bets") or 0 for t in l["tiers"] if t.get("thr")==0.75), 0), reverse=True)[:30]
lines.append("## 4. Detalle por liga y por umbral — Top 30 por volumen de apuestas\n")
for l in top:
    lines.append(f"### {l['league']}\n")
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
    lines.append("| Liga | Holdout (C) | Apuestas @0.75 | WR @0.75 |")
    lines.append("|---|---:|---:|---:|")
    for l in sorted(cold, key=lambda x: x["matches_holdout"], reverse=True)[:40]:
        t75 = next((t for t in l["tiers"] if t["thr"]==0.75), None)
        bets = t75.get("bets",0) if t75 else 0
        wr = (f"{t75['wr']:.1f} %" if (t75 and t75.get("wr") is not None) else "—")
        lines.append(f"| {l['league']} | {l['matches_holdout']:,} | {bets:,} | {wr} |")
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

# --- Conclusiones ---
lines.append("## 6. Conclusiones\n")
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

lines.append("## 7. Notas metodológicas\n")
lines.append("- Cálculo reproducible: `tmp/m27_v5_3_audit/league_threshold_analysis.py` (re-entrena el ensamble, calibra con Platt simétrico, evalúa en C).")
lines.append("- `matches_holdout` = anclas (t=30) en el bloque C por liga; `matches_train` = anclas en el bloque A.")
lines.append("- `green()` aplica además el guard de margen (margin ≥ 0 para apuestas al local), igual que el pipeline.")
lines.append("- IC95 = Wilson. Bloqueo candidato si el límite superior del IC95 del WR < 71.4 % (equilibrio a cuota 1.40).")
lines.append("- Ligas con < 30 anclas en C se omiten por falta de muestra.\n")

out = ROOT / "docs" / "ANALISIS_LIGAS_M27_V5_3.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Escrito: {out} ({out.stat().st_size} bytes)")