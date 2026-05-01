# Pulpa — Sistema de Apuestas Deportivas NBA/Basketball

Sistema automatizado de monitoreo, predicción y gestión de apuestas en partidos de basketball, usando modelos ML propios integrados con un bot de Telegram y un dashboard web.

---

## Estructura del proyecto

```
pulpa/
├── api.py                  # API FastAPI — sirve predicciones al dashboard
├── match/
│   ├── telegram_bot.py     # Bot de Telegram (menú, alertas, reportes)
│   ├── bet_monitor.py      # Monitor en vivo — scraping + inferencia + alertas
│   ├── scraper.py          # Scraper de SofaScore via Playwright
│   ├── db.py               # Acceso a SQLite (matches.db)
│   ├── ml_tools.py         # Utilidades ML compartidas
│   ├── cli.py              # CLI interactivo (fetch, ingest, eval, retrain)
│   └── training/           # Scripts de entrenamiento por versión de modelo
│       ├── v12/ … v17/     # Módulos de inferencia por versión
│       ├── train_q3_q4_models_v2.py
│       ├── train_q3_q4_models_v6.py
│       └── ...
├── v13_dashboard/          # Frontend React/Vite
│   └── src/
├── api_cache/              # Caché JSON de inferencias (local, no versionado)
├── instalar.bat            # Instalación de dependencias
└── menu.bat                # Menú principal de arranque
```

---

## Requisitos

- **Python 3.11+**
- **Node.js 18+** (para el dashboard)
- Cuenta de Telegram y token de bot (`BOT_TOKEN` en `match/.env`)

---

## Instalación

```bat
instalar.bat
```

Esto:
1. Crea el entorno virtual `.venv`
2. Instala dependencias Python (`match/requirements.txt`)
3. Instala navegadores Playwright (`chromium`)
4. Ejecuta `npm install` en `v13_dashboard/`

---

## Uso — Menú principal

```bat
menu.bat
```

| Opción | Acción |
|--------|--------|
| 1 | Iniciar Bot de Telegram |
| 2 | Iniciar API Backend (FastAPI) |
| 3 | Iniciar Dashboard (API + Vite) |
| 4 | Iniciar todo (Bot + API + Dashboard) |
| 5 | Traer fecha nueva — CLI interactivo, seleccionar opción 15 |
| 6 | Entrenar modelo V2 |
| 7 | Entrenar modelo V6 |
| 8 | Entrenar V2 + V6 en orden |
| 9 | Instalar / actualizar dependencias |

---

## Variables de entorno

Crear `match/.env` con:

```env
BOT_TOKEN=tu_token_de_telegram
CHAT_ID=tu_chat_id
DB_PATH=matches.db        # opcional, default: match/matches.db
```

---

## Modelos de predicción

El sistema usa un stack de modelos ML para predecir el ganador del Q3 y Q4 de cada partido:

| Versión | Tipo | Uso |
|---------|------|-----|
| v2 | Clasificador (Q3) | Filtro por liga/confianza |
| v6 | Clasificador (Q4) | Filtro activo con reglas por liga |
| v9 | Clasificador ensamble | Referencia |
| v13–v17 | Modelos avanzados (LightGBM/CatBoost) | Inferencia en vivo |

### Filtros v6 (`_v6_pick_filter`)

Las reglas de aceptación/rechazo para Q4 con el modelo v6 están centralizadas en `match/bet_monitor.py → _v6_pick_filter()`:

- **Hard ban**: ligas con historial negativo (Brazil NBB, Germany BBL, Israeli National, SLB, etc.)
- **Trusted-35**: ligas confiables desde 35% (Euroleague, France Pro A, Puerto Rico BSN, etc.)
- **Trusted-38**: Bulgaria NBL — mínimo 38%
- **Min-40+**: Segunda FEB, 1. A SKL — requieren >40%
- **Reglas específicas**: Colombia LPB y Korean Basketball League exigen >50% para picks home
- **Global**: bloqueo de rango 70–80% para ligas no trusted

---

## Bot de Telegram — Funcionalidades

- **Alertas de apuesta** en tiempo real al llegar al minuto de análisis (Q3/Q4)
- **Historial de liga** en cada alerta: win% por pick y por banda de confianza
- **Menú de reply keyboard**: Stats, Señales hoy, Reporte mensual, Buscar por ID
- **Reporte Excel diario** con filtros v6 aplicados y links a SofaScore
- **Reporte Excel mensual** con hoja Resumen dinámica (Q3/Q4 o ambos)
- **Vista de señales** del día con pick, confianza, resultado y proyección

---

## Base de datos

SQLite en `match/matches.db`. Tablas principales:

| Tabla | Contenido |
|-------|-----------|
| `matches` | Partidos scrapeados (slugs, equipos, liga, fecha) |
| `quarter_scores` | Marcadores por cuarto |
| `bet_monitor_log` | Historial de señales BET/NO BET con resultado |
| `bet_monitor_schedule` | Partidos programados para monitoreo |
| `inference_debug_log` | Detalle de cada inferencia |

---

## Entrenamiento

```bat
# Desde menu.bat
6) Entrenar V2
7) Entrenar V6
8) Entrenar V2 + V6

# O directamente
.venv\Scripts\activate
python match\training\train_q3_q4_models_v6.py
```

---

## CLI avanzado

```bat
.venv\Scripts\activate
python match\cli.py menu
```

Opciones relevantes:
- `15` — Traer fecha nueva (detecta automáticamente días faltantes)
- `8` — Inferencia por match ID
- `13` — Reentrenamiento de modelos / calibrar gate
- `14` — Reporte comparación de modelos (Excel)
