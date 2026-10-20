# =====================================================================
# CLASIFICADOR MULTILINGUE DE FASE vs stage_detail + LISTA DEFINITIVA
# ---------------------------------------------------------------------
# QUÉ HACE:
#   1. Aplica el clasificador multilingue (models/common/competition_stage.py)
#      a las 3.758 variantes de liga y lo contrasta con `stage_detail`:
#      los DESACUERDOS son los agujeros de la clasificacion existente.
#   2. Lista las ligas "championship" con su categoria y muestra el agujero.
#   3. Audita de nuevo las ligas de 12 minutos (marcadas vs firma por Q4).
#   4. Emite la LISTA DEFINITIVA por categoria con el volumen que cada una
#      representa dentro del dataset de entrenamiento de m27_v5_3.
#
# USO:
#   .venv\Scripts\python.exe tmp/m27_v5_3_audit/stage_classifier_audit.py
# =====================================================================

import pickle
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.common.competition_stage import classify_stage, classify_stage_detail  # noqa: E402

DB = ROOT / "matches.db"


def block(t: str) -> None:
    print(f"\n{'=' * 100}\n{t}\n{'=' * 100}")


def main() -> None:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    vol = {r["league"]: r["n"] for r in conn.execute(
        "SELECT league, COUNT(*) n FROM matches WHERE status_type='finished' "
        "AND league IS NOT NULL AND league!='' GROUP BY league")}
    total = sum(vol.values())

    en_train = set()
    ds = ROOT / "tmp" / "m27_v5_3" / "dataset_v5_3.pkl"
    if ds.exists():
        en_train = {m["lg"] for m in pickle.load(open(ds, "rb"))["meta"]}

    filas = [dict(r) for r in conn.execute(
        "SELECT league, stage, stage_detail, competition_type, is_youth, is_college, "
        "is_tournament, is_playoffs, is_final, quarter_duration_minutes, match_count, "
        "avg_total_points, avg_q4_total_points FROM leagues_classification")]

    block("1. CLASIFICADOR MULTILINGUE vs stage_detail (acuerdos y agujeros)")
    cats = Counter()
    desacuerdos = []
    for f in filas:
        cat, motivo = classify_stage(f["league"], f["stage"] or "")
        cats[cat] += 1
        det = classify_stage_detail(f["stage_detail"] or "")
        if det != cat and f["league"] in vol:
            desacuerdos.append((f["league"], f["stage"], cat, det, vol[f["league"]]))
    total_lig = sum(cats.values())
    print(f"  {'categoria (multilingue)':<28}{'ligas':>8}{'%':>7}   stage_detail equivalente")
    equiv = {"ELIMINATORIA": "Playoffs / Quarterfinal / Final / Semifinal",
             "COPA": "Cup Stage", "SEGUNDA_FASE": "Group Stage",
             "PROMOCION": "Relegation / Playout", "PUESTO": "Bronze / 3rd Place",
             "PRELIMINAR": "Play-in", "AMISTOSOS": "friendly/tournament",
             "REGULAR": "Regular Season"}
    for cat, n in cats.most_common():
        print(f"  {cat:<28}{n:>8,}{n * 100 / total_lig:>6.1f}%   {equiv.get(cat, '')}")
    print(f"\n  DESACUERDOS con stage_detail: {len(desacuerdos)} liga(s) con partidos")
    for lg, stage, cat, det, n in sorted(desacuerdos, key=lambda x: -x[4])[:35]:
        print(f"    {lg[:46]:<48}stage='{str(stage)[:24]:<26}mio={cat:<14}stage_detail={det:<20}{n:>6,} p.")

    block("2. LAS 'CHAMPIONSHIP' CON SU CATEGORIA (el agujero que viste)")
    champ = [f for f in filas if "championship" in (f["league"] + " " + (f["stage"] or "")).lower()
             and f["league"] in vol]
    por_cat = Counter()
    for f in champ:
        cat, _ = classify_stage(f["league"], f["stage"] or "")
        por_cat[cat] += 1
    print(f"  ligas con 'championship': {len(champ)}  ->  {dict(por_cat)}")
    print(f"  {'liga':<46}{'stage':<26}{'categoria':<16}{'part.':>8}")
    for f in sorted(champ, key=lambda x: -vol.get(x["league"], 0))[:28]:
        cat, _ = classify_stage(f["league"], f["stage"] or "")
        print(f"  {f['league'][:44]:<46}{str(f['stage'])[:24]:<26}{cat:<16}"
              f"{vol.get(f['league'], 0):>8,}")

    block("3. AUDITORIA DE 12 MINUTOS (marcadas + candidatas por firma de Q4)")
    doce = [f for f in filas if f["quarter_duration_minutes"] == 12 and f["league"] in vol]
    print(f"  marcadas q_dur=12: {len(doce)} ligas | "
          f"{sum(vol.get(f['league'], 0) for f in doce):,} partidos")
    print("  base (todas):")
    for f in sorted(doce, key=lambda x: -vol.get(x["league"], 0))[:12]:
        print(f"    {f['league'][:52]:<54}{vol.get(f['league'], 0):>8,} part.")
    print("\n  candidatas NO marcadas (Q4 medio >= 47 y >= 40 partidos): revisables a mano")
    for f in sorted(filas, key=lambda x: -(x["avg_q4_total_points"] or 0)):
        if f["league"] not in vol or vol[f["league"]] < 40:
            continue
        if f["quarter_duration_minutes"] == 12:
            continue
        if (f["avg_q4_total_points"] or 0) < 47:
            continue
        cat, _ = classify_stage(f["league"], f["stage"] or "")
        print(f"    {f['league'][:52]:<54}{vol[f['league']]:>7,} part.  "
              f"pts={(f['avg_total_points'] or 0):>6.1f} Q4={(f['avg_q4_total_points'] or 0):>5.1f} "
              f"cat={cat}")

    block("4. LISTA DEFINITIVA POR CATEGORIA (con presencia en el train de v5_3)")
    grupos = defaultdict(lambda: {"lig": 0, "n": 0, "lig_tr": 0, "n_tr": 0, "ej": []})
    for f in filas:
        lg = f["league"]
        if lg not in vol:
            continue
        cat, motivo = classify_stage(lg, f["stage"] or "")
        g = grupos[cat]
        g["lig"] += 1
        g["n"] += vol[lg]
        g["ej"].append((lg, vol[lg], f["stage"], motivo))
        if lg in en_train:
            g["lig_tr"] += 1
            g["n_tr"] += vol[lg]
    print(f"  {'categoria':<16}{'ligas':>7}{'part.':>10}{'% base':>8}"
          f"{'en train':>10}{'part. train':>13}   top ligas")
    for cat, g in sorted(grupos.items(), key=lambda kv: -kv[1]["n"]):
        top = ", ".join(x[0][:26] for x in sorted(g["ej"], key=lambda y: -y[1])[:3])
        print(f"  {cat:<16}{g['lig']:>7,}{g['n']:>10,}{g['n'] * 100 / total:>7.1f}%"
              f"{g['lig_tr']:>10,}{g['n_tr']:>13,}   {top}")
    print(f"\n  TOTAL: {sum(g['n'] for g in grupos.values()):,} partidos | "
          f"en train: {sum(g['n_tr'] for g in grupos.values()):,}")

    block("5. MOTIVOS DE DESCARTE (lo que dira el modelo)")
    for cat, g in sorted(grupos.items(), key=lambda kv: -kv[1]["n"]):
        if cat == "REGULAR":
            continue
        motivos = Counter(x[3] for x in g["ej"])
        print(f"  {cat:<16} {motivos.most_common(1)[0][0] if motivos else ''}")
    conn.close()


if __name__ == "__main__":
    main()
