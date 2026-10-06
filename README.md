# 🏀 Pulpa — Sistema de Apuestas Deportivas NBA y Basketball FIBA

Sistema automatizado cuantitativo de monitoreo en tiempo real, predicción estadística y gestión de apuestas en partidos de basketball (NBA y más de 1,200 ligas FIBA internacionales), usando modelos de Machine Learning propios integrados con un bot de Telegram, un dashboard web y una suite de analítica avanzada.

---

## 📚 Índice de Documentación (`docs/`)

Toda la documentación técnica, científica y operativa del proyecto ha sido consolidada en la carpeta [`docs/`](file:///C:/Users/App/Desktop/pulpa/docs):

### 🌟 Visión General, Estado y Reglas
| Archivo en `docs/` | Ubicación Original | Contenido Principal |
|---|---|---|
| [`docs/ESTADO_DEL_PROYECTO.md`](file:///C:/Users/App/Desktop/pulpa/docs/ESTADO_DEL_PROYECTO.md) | `ESTADO_DEL_PROYECTO.md (Raíz)` | Visión integral del sistema, arquitectura completa, catálogo de features F0-F15 y G1-G9, comparativa de los 24 modelos, investigación de scraping, últimos avances y guía rápida. |
| [`docs/SCHEMA_DATABASE.md`](file:///C:/Users/App/Desktop/pulpa/docs/SCHEMA_DATABASE.md) | `docs/SCHEMA_DATABASE.md` | Especificación exhaustiva y formal del esquema relacional de `matches.db` (ERD, 14 tablas catalogadas campo por campo, PKs/FKs e índices). |
| [`docs/EXPLORACION_GLOBAL_DATOS.md`](file:///C:/Users/App/Desktop/pulpa/docs/EXPLORACION_GLOBAL_DATOS.md) | `docs/EXPLORACION_GLOBAL_DATOS.md` | Balance e inventario global de datos (48,623 partidos, 1,958 ligas, cobertura por componente, estacionalidad y queries para EDAs). |
| [`docs/modelos.md`](file:///C:/Users/App/Desktop/pulpa/docs/modelos.md) | `findings/modelos.md` | Documento enciclopédico de todos los modelos (V1 a V17, m27_v1/v2/v3, m30_v1), hiperparámetros, snapshots de corte, métricas ROC AUC/Accuracy/Yield y catálogo técnico de features. *(Disponible en raíz como [`modelos.md`](file:///C:/Users/App/Desktop/pulpa/modelos.md))*. |
| [`docs/AGENTS.md`](file:///C:/Users/App/Desktop/pulpa/docs/AGENTS.md) | `AGENTS.md (Raíz)` | Reglas operativas obligatorias para agentes de IA (commit descriptivo en español y push inmediato tras cada cambio) y mapa del repositorio. *(Disponible en raíz como [`AGENTS.md`](file:///C:/Users/App/Desktop/pulpa/AGENTS.md))*. |
| [`docs/findings.md`](file:///C:/Users/App/Desktop/pulpa/docs/findings.md) | `findings/findings.md` | Resumen ejecutivo de hallazgos estadísticos clave: por qué el snapshot 27 supera al 30, impacto cuantitativo del H2H (+0.122 AUC), correlaciones y feature importance. |

### 🧠 Modelado Predictivo, Roadmaps y Features
| Archivo en `docs/` | Ubicación Original | Contenido Principal |
|---|---|---|
| [`docs/M27_V1_ROADMAP.md`](file:///C:/Users/App/Desktop/pulpa/docs/M27_V1_ROADMAP.md) | `findings/M27_V1_ROADMAP.md` | Hoja de ruta, hipótesis y validación del modelo predictivo fijado al minuto 27 (`m27_v1`). |
| [`docs/M27_V2_ROADMAP.md`](file:///C:/Users/App/Desktop/pulpa/docs/M27_V2_ROADMAP.md) | `findings/M27_V2_ROADMAP.md` | Experimentos de `m27_v2`, análisis de feature importance y demostración del peso del 35.9% en ventanas recientes. |
| [`docs/M27_FEATURES_COMPARISON.md`](file:///C:/Users/App/Desktop/pulpa/docs/M27_FEATURES_COMPARISON.md) | `findings/M27_FEATURES_COMPARISON.md` | Comparación minuciosa de variables entre las versiones preliminares y avanzadas de la serie M27. |
| [`docs/M27_V1_LEAGUE_TIERS.md`](file:///C:/Users/App/Desktop/pulpa/docs/M27_V1_LEAGUE_TIERS.md) | `findings/M27_V1_LEAGUE_TIERS.md` | Clasificación de ligas por tiers de fiabilidad y lista negra para el modelo m27_v1. |
| [`docs/roadmap_m30_v1.md`](file:///C:/Users/App/Desktop/pulpa/docs/roadmap_m30_v1.md) | `findings/roadmap_m30_v1.md` | Roadmap inicial de experimentos para modelos evaluados al minuto 30. |
| [`docs/M30_V1_FEATURE_PRUNING_NOTES.md`](file:///C:/Users/App/Desktop/pulpa/docs/M30_V1_FEATURE_PRUNING_NOTES.md) | `findings/M30_V1_FEATURE_PRUNING_NOTES.md` | Análisis de poda de features en m30 y demostración de por qué los minutos 28-30 agregan ruido. |
| [`docs/FILTROS_LIGAS.md`](file:///C:/Users/App/Desktop/pulpa/docs/FILTROS_LIGAS.md) | `match/FILTROS_LIGAS.md` | Reglas de filtrado, bandas de confianza y restricciones por liga para el modelo V6. |
| [`docs/V6_2_REPORT.md`](file:///C:/Users/App/Desktop/pulpa/docs/V6_2_REPORT.md) | `match/training/.../V6_2_REPORT.md` | Reporte de validación, métricas y poda automática de ligas del modelo campeón `v6_2`. |
| [`docs/V6_3_REPORT.md`](file:///C:/Users/App/Desktop/pulpa/docs/V6_3_REPORT.md) | `match/training/.../V6_3_REPORT.md` | Reporte técnico del modelo de doble ventana dinámica `v6_3`. |
| [`docs/RESUMEN_RENDIMIENTO_MODELOS_2026-04-18.md`](file:///C:/Users/App/Desktop/pulpa/docs/RESUMEN_RENDIMIENTO_MODELOS_2026-04-18.md) | `api_cache/...` | Snapshot histórico y balance económico de modelos evaluados en abril 2026. |

### 🏀 Clasificación Reglamentaria de Ligas
| Archivo en `docs/` | Ubicación Original | Contenido Principal |
|---|---|---|
| [`docs/ligas_10min.md`](file:///C:/Users/App/Desktop/pulpa/docs/ligas_10min.md) | `findings/ligas_10min.md` | Listado oficial de **1,185 ligas** con cuartos reglamentarios de 10 minutos (FIBA / Europa / Latinoamérica). |
| [`docs/ligas_12min.md`](file:///C:/Users/App/Desktop/pulpa/docs/ligas_12min.md) | `findings/ligas_12min.md` | Listado de **16 ligas** con cuartos de 12 minutos (NBA, CBA, PBA, etc.). |

### ⚡ Monitoreo en Vivo (`monitor_v2` y `monitor_v1`)
| Archivo en `docs/` | Ubicación Original | Contenido Principal |
|---|---|---|
| [`docs/monitoreo_v2.md`](file:///C:/Users/App/Desktop/pulpa/docs/monitoreo_v2.md) | `monitoreo_v2.md (Raíz)` | Especificación formal del daemon asíncrono `monitor_v2`, arquitectura desacoplada, control de red y ciclos de vida. |
| [`docs/monitor_arquitectura.md`](file:///C:/Users/App/Desktop/pulpa/docs/monitor_arquitectura.md) | `findings/monitor_arquitectura.md` | Diagramas de flujo y arquitectura interna del sistema de monitoreo en tiempo real. |
| [`docs/glosario.md`](file:///C:/Users/App/Desktop/pulpa/docs/glosario.md) | `glosario.md (Raíz)` | Glosario de mensajes de log coloreados (`[SYSTEM]`, `[WATCHER]`, `[PROBE]`, `[LIVE]`, `[EVAL]`, `[FT]`). |

### 🛡️ Evasión Anti-Bot y Scraping (Obscura vs. Chrome)
| Archivo en `docs/` | Ubicación Original | Contenido Principal |
|---|---|---|
| [`docs/OBSCURA_VS_SOFASCORE.md`](file:///C:/Users/App/Desktop/pulpa/docs/OBSCURA_VS_SOFASCORE.md) | `scratch/OBSCURA_VS_SOFASCORE.md` | Comparativa técnica profunda entre Obscura (Rust) y Chrome Headless para SofaScore, fundamentando el stack de producción. |
| [`docs/OBSCURA_FIX.md`](file:///C:/Users/App/Desktop/pulpa/docs/OBSCURA_FIX.md) | `scratch/OBSCURA_FIX.md` | Documentación del fix aplicado al código fuente Rust de Obscura (`ops.rs`) para enviar cookies cross-origin en `fetch()`. |
| [`docs/OBSCURA_LIMITATIONS.md`](file:///C:/Users/App/Desktop/pulpa/docs/OBSCURA_LIMITATIONS.md) | `scratch/OBSCURA_LIMITATIONS.md` | Análisis de limitaciones de Obscura ante Single-Page Applications construidas en React/Next.js. |

---

## 🤖 Guía Obligatoria para Agentes de IA

Si eres un asistente de IA trabajando en este repositorio, consulta obligatoriamente [`AGENTS.md`](file:///C:/Users/App/Desktop/pulpa/AGENTS.md) (y [`docs/AGENTS.md`](file:///C:/Users/App/Desktop/pulpa/docs/AGENTS.md)):
- **Regla mandatoria:** Cada cambio debe terminar con un `git commit -m "mensaje en español"` y un `git push` inmediato a `origin/main`.
- **Mapa de navegación:** Ubicación de la base de datos `matches.db`, scrapers, evaluadores, menú de control y carpeta `docs/`.

---

## 🏗️ Estructura del Proyecto

```text
pulpa/
├── AGENTS.md               # Reglas obligatorias para agentes de IA y mapa de navegación
├── README.md               # Este archivo — índice y guía principal (solo en raíz)
├── modelos.md              # Documento maestro con stats y features de todos los modelos
├── menu.bat                # Centro de control principal (5 bloques operativos y CLI integrado)
├── api.py                  # API REST en FastAPI para servir inferencias al dashboard
│
├── docs/                   # 📚 Hub central de documentación (22 archivos .md)
│
├── monitor_v2/             # Daemon asíncrono modular de monitoreo en tiempo real
│   ├── config/             # Constantes y leagues.yaml (filtros de ligas)
│   ├── database/           # Capa de datos SQLite (tablas _v2)
│   ├── scrapers/           # Clientes Playwright / Chrome CDP
│   ├── models/             # Evaluador de inferencia en tiempo real (v6_2 y m27_v3)
│   ├── notifications/      # Bot despachador de alertas y resultados a Telegram
│   └── main.py             # Event loop principal con asyncio
│
├── monitor_v1/             # Monitor original / Telegram Bot interactivo
│   ├── telegram_bot.py     # Bot interactivo de Telegram con teclado y reportes
│   ├── bet_monitor.py      # Daemon de monitoreo en hilo
│   └── main.py             # Punto de entrada ejecutable
│
├── match/                  # Almacenamiento histórico, scraping e inferencia
│   ├── scraper.py          # Scraper con CDP e inyección JS (_js_fetch)
│   ├── cli.py              # CLI para descarga de fechas y reentrenamiento
│   └── training/           # Scripts de entrenamiento de modelos (m27_v3, v6_2, etc.)
│
├── matches.db              # Base de datos SQLite central (~737 MB, excluida en .gitignore)
│
├── tools/                  # Suite analítica y consenso
│   ├── stats_cli.py        # Herramienta analítica (3,144 líneas) con FusionConsensusEngine
│   └── obscura-src/        # Código fuente en Rust del navegador stealth Obscura
│
├── dashboard/              # Frontend Web SPA (React + Vite + TypeScript)
├── tmp/                    # 🗄️ Repositorio central de scripts temporales, pruebas y diagnóstico
└── api_cache/              # Caché local de itinerarios y respuestas API
```

---

## ⚡ Requisitos e Instalación

- **Python 3.11+**
- **Node.js 18+** (para el dashboard web)
- Google Chrome instalado
- Cuenta y Token de Telegram (`match/.env`)

```bat
# Instalar / reparar entorno y dependencias (.venv, pip, playwright, npm)
menu.bat   # Seleccionar la opción 25
```

---

## 🚀 Uso Rápido — Menú Principal

Ejecuta el menú interactivo para acceder a todas las funciones organizadas en 5 bloques operativos:

```bat
menu.bat
```

| Bloque | Opciones Clave | Descripción |
|:---|:---:|---|
| **[1] Operación en Vivo** | `1`-`7` | Monitor V2 Chrome sin proxy (`1`), Monitor V1 (`2`), Monitor V2 CDP (`3`), API FastAPI (`4`), Dashboard Web (`5`) y All-in-One (`6` con V2, `7` con V1). |
| **[2] Análisis y Consenso** | `8`, `9` | Estadísticas de Modelos / Fusion Consensus / Excel (`tools/stats_cli.py`, `8`) y Reporte ROI M27_V3 (`9`). |
| **[3] Ingesta y Backfill** | `10`-`13` | Descarga de fechas faltantes (`10`), backfill general (`11`), backfill masivo H2H SofaScore (`12`) y comparador de scrapers (`13`). |
| **[4] Modelos ML** | `14`-`23` | Entrenamiento y reportes ROI para la serie M27 (v1, v2, v3) y V6 (v6.2, v6.3, base v2/v6). |
| **[5] Mantenimiento** | `24`, `25` | Control integrado de Obscura (`24`) e Instalador / Reparador de dependencias (`25`). |

