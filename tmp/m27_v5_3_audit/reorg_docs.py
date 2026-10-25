import os, shutil, re
from pathlib import Path

ROOT = Path(r"C:\Users\App\Desktop\pulpa")
DOCS = ROOT / "docs"

# Category mapping: filename -> subfolder
CAT = {
 # modelos / analisis de modelos
 "ANALISIS_LIGAS_M27_V5_3.md": "modelos",
 "ANALISIS_LIGAS_M27_V5_3_1.md": "modelos",
 "ANALISIS_LIGAS_M27_V5_3_2.md": "modelos",
 "ANALISIS_LIGAS_M27_V5_3_3.md": "modelos",
 "ANALISIS_LIGAS_M27_V5_3_4.md": "modelos",
 "ANALISIS_LIGAS_M27_V5_3_5.md": "modelos",
 "ANALISIS_LIVE_M27_V5_3.md": "modelos",
 "ANALISIS_POR_MINUTO.md": "modelos",
 "CONCLUSION_M27_V5_3.md": "modelos",
 "DIAGNOSTICO_DIA_M27_V5_3.md": "modelos",
 "FINDINGS_M27_V4_ITERACIONES.md": "modelos",
 "findings_m27_2.md": "modelos",
 "findings_m27_v5_3.md": "modelos",
 "M27_FEATURES_COMPARISON.md": "modelos",
 "M27_V1_ROADMAP.md": "modelos",
 "M27_V2_ROADMAP.md": "modelos",
 "M30_V1_FEATURE_PRUNING_NOTES.md": "modelos",
 "MATRIZ_FORTALEZA_Y_CUARTOS.md": "modelos",
 "modelos.md": "modelos",
 "PROPUESTA_NUEVO_MODELO_M27_V4.md": "modelos",
 "RESUMEN_LIVE_M27_V5_3.md": "modelos",
 "RESUMEN_RENDIMIENTO_MODELOS_2026-04-18.md": "modelos",
 "AUDITORIA_M27_V5_3_PERDIDAS.md": "modelos",
 "ARREGLO_CLASIFICADOR_FASES.md": "modelos",
 "ARQUITECTURA_MULTI_MODELO_Y_BACKFILL.md": "modelos",
 "M27_V1_LEAGUE_TIERS.md": "modelos",
 "V6_2_REPORT.md": "modelos",
 "V6_3_REPORT.md": "modelos",
 "roadmap_m30_v1.md": "modelos",
 "roadmap.md": "modelos",
 # monitor
 "monitoreo_v2.md": "monitor",
 "monitor_arquitectura.md": "monitor",
 "PLAN_MONITOR_V3.md": "monitor",
 "ROBUSTEZ_INFERENCIA_LIVE_Y_MONITOR_V3.md": "monitor",
 "FORMATO_LOGS_V3.md": "monitor",
 "IMAGEN_APUESTA_Q4.md": "monitor",
 # scraping
 "ANALISIS_BROWSERS_ANTI_BOT.md": "scraping",
 "HALLAZGOS_403_SOFASCORE.md": "scraping",
 "REVERSE_ENGINEERING_SOFASCORE.md": "scraping",
 "OBSCURA_FIX.md": "scraping",
 "OBSCURA_LIMITATIONS.md": "scraping",
 "OBSCURA_VS_SOFASCORE.md": "scraping",
 # base de datos
 "SCHEMA_DATABASE.md": "base_datos",
 "EXPLORACION_GLOBAL_DATOS.md": "base_datos",
 # ligas
 "ligas_10min.md": "ligas",
 "ligas_12min.md": "ligas",
 "CLASIFICACION_LIGAS.md": "ligas",
 "FILTROS_LIGAS.md": "ligas",
 "COMPORTAMIENTO_LIGAS_Y_GUIA_ENTRENAMIENTO.md": "ligas",
 "MATRIZ_FORTALEZA_Y_CUARTOS.md": "ligas",
 # planificacion
 "ESTADO_DEL_PROYECTO.md": "planificacion",
 "PLAN_DESCARGAS_HISTORICAS_Y_BACKFILL.md": "planificacion",
 # findings
 "findings.md": "findings",
 "findings_m27_2.md": "findings",
 # telegram / misc
 "glosario.md": "telegram",
 # v5_4
}

# Move files
moved = {}
for fname, cat in CAT.items():
    src = DOCS / fname
    if not src.exists():
        # already moved or missing
        # check if it's in a subfolder
        for sub in (DOCS).iterdir():
            if sub.is_dir() and (sub / fname).exists():
                moved[fname] = sub.name
        continue
    if src.is_dir():
        continue
    dst_dir = DOCS / cat
    dst_dir.mkdir(exist_ok=True)
    dst = dst_dir / fname
    shutil.move(str(src), str(dst))
    moved[fname] = cat
    print("moved", fname, "->", cat)

# Save mapping
import json
(ROOT / "tmp" / "m27_v5_3_audit" / "reorg_map.json").write_text(json.dumps(moved, indent=2, ensure_ascii=False), encoding="utf-8")
print("total moved:", len(moved))