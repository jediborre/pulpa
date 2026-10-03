# 🤖 Instrucciones Obligatorias para Agentes de IA (AGENTS.md)

> **PROPÓSITO:** Esta guía establece las **reglas mandatorias de flujo de trabajo** para cualquier agente de inteligencia artificial que trabaje en el repositorio `pulpa`, así como el mapa completo para localizar archivos, datos y documentación.

---

## ⚠️ REGLAS MANDATORIAS DE FLUJO DE TRABAJO

### 1. REGLA DE ORO DE GIT (COMMIT Y PUSH INMEDIATOS)
Cada vez que realices cualquier cambio, corrección, refactorización o adición en el código o documentación:
1. **COMMIT INMEDIATO:** Debes realizar un `git commit` con un **mensaje corto, claro y descriptivo en ESPAÑOL**.
   - *Ejemplos válidos:* 
     - `git commit -m "agrega filtros de ligas en monitor v2"`
     - `git commit -m "corrige timeout en scraper de cdp"`
     - `git commit -m "actualiza documentacion en docs"`
2. **PUSH INMEDIATO:** Justo después del commit, debes realizar un `git push` a `origin/main` para sincronizar los cambios de forma remota:
   ```bash
   git push
   ```

### 2. USO OBLIGATORIO DEL ENTORNO VIRTUAL (.venv)
- **SIEMPRE** se debe utilizar el entorno virtual de Python ubicado en `.venv`.
- Al ejecutar comandos de Python en consola, activa antes el entorno con `.venv\Scripts\activate` o invoca directamente el ejecutable del entorno:
  ```powershell
  .venv\Scripts\python.exe <script.py>
  ```
- **NUNCA** ejecutes scripts ni instales librerías usando el Python global del sistema para evitar incompatibilidades de dependencias o rotura de paquetes.
- Si se añaden dependencias con pip, hazlo exclusivamente dentro del entorno virtual (`.venv\Scripts\pip.exe install ...`).

### 3. UBICACIÓN OBLIGATORIA DE LA BASE DE DATOS (matches.db)
- **UBICACIÓN ÚNICA Y EXCLUSIVA:** La base de datos SQLite histórica y en vivo de todo el sistema se encuentra **estrictamente en la raíz del proyecto**:
  ```
  matches.db
  ```
  *(Ruta relativa desde la raíz del proyecto: `matches.db`)*.
- **EXCLUSIÓN DE GIT:** El archivo `matches.db` (junto con sus auxiliares `-journal`, `-shm`, `-wal` y copias de respaldo) se encuentra estrictamente ignorado en `.gitignore` para salvaguardar el tamaño del repositorio.
- **PROHIBIDO** instanciar, buscar o crear bases de datos `matches.db` dentro de `match/` o en cualquier otro subdirectorio.
- Cualquier script, consulta SQLite, endpoint de API, modelo de inferencia o tarea de backfill debe conectarse obligatoriamente a esta ruta canónica en la raíz:
  ```python
  from pathlib import Path
  DB_PATH = Path(__file__).resolve().parents[...] / "matches.db"
  # O si se ejecuta directamente desde la raíz:
  DB_PATH = Path("matches.db")
  ```

### 4. CREACIÓN DE ARCHIVOS AUXILIARES Y EXPERIMENTALES (ESTRICTAMENTE EN /tmp)
- **UBICACIÓN MANDATORIA:** Cada vez que vayas a crear un archivo auxiliar, script de apoyo, prueba de concepto (PoC), análisis puntual o diagnóstico temporal, **DEBE crearse obligatoriamente dentro de la carpeta `tmp/` en la raíz**.
- **PROHIBIDO:** Crear scripts de prueba, archivos temporales, logs o volcados sueltos en la raíz (`./`), en `match/`, en `tools/` o en cualquier otra carpeta de producción del proyecto.
- **SUBDIRECTORIOS TEMÁTICOS POR TAREA:** Para mantener el orden y la trazabilidad dentro de `tmp/`, se debe crear una subcarpeta interna alusiva a la funcionalidad, modelo o tema en el que se esté trabajando:
  - *Ejemplo nuevo modelo:* Si se trabaja en el modelo 100, la ruta debe ser `tmp/modelo_100/` y allí van todos sus scripts temporales, datos JSON y pruebas auxiliares.
  - *Ejemplo mejoras de scraping:* Si se prueba una variante de extracción, usar `tmp/scraper_cdp/`.
  - *Ejemplo auditoría o backfill:* Si se auditan enfrentamientos directos, usar `tmp/h2h_audit/`.
- **CABECERA DOCUMENTADA OBLIGATORIA:** Todo archivo que se cree en `tmp/` debe incluir al inicio un comentario/docstring explicando claramente qué hace, para qué se usa y qué hipótesis o problema aborda.

### 5. CLARIFICACIÓN OBLIGATORIA DE REQUERIMIENTOS (PREGUNTAR ANTES DE ASUMIR)
- **CERO ASUNCIONES:** Si una solicitud o instrucción del usuario no está 100% clara, es ambigua, incompleta o deja dudas sobre su alcance o implementación, **ESTÁ ESTRICTAMENTE PROHIBIDO ASUMIR** o adivinar lo que el usuario quiso decir.
- **PREGUNTAS PROACTIVAS:** El agente debe formular al usuario todas las preguntas necesarias, estructuradas y precisas para clarificar y elaborar correctamente la solicitud antes de actuar o tomar decisiones de diseño por su cuenta.

---

## 🗺️ Mapa del Repositorio: Dónde Encontrar Todo

Si necesitas investigar o modificar alguna parte del sistema, consulta esta guía:

### 1. Documentación Central (`docs/` y Raíz)
Toda la documentación conceptual, histórica y técnica del proyecto ha sido consolidada:
- **`README.md` (Raíz):** Índice general del repositorio, guía de instalación y mapa de documentación en `docs/`.
- **`modelos.md` (Raíz y `docs/`):** El documento maestro de modelos de ML. Contiene el catálogo de features (F0-F15 y G1-G9), hiperparámetros, snapshots y estadísticas de todas las versiones (V1 a V17, m27, m30).
- **`docs/ESTADO_DEL_PROYECTO.md`:** Visión general completa del repositorio, arquitectura, base de datos, hallazgos y guía rápida.
- **`docs/findings.md`:** Hallazgos cuantitativos (por qué el minuto 27 supera al 30, impacto del H2H, correlaciones).
- **`docs/monitoreo_v2.md` y `docs/glosario.md`:** Especificación formal del daemon asíncrono y glosario de logs coloreados.
- **`docs/OBSCURA_*.md`:** Investigación de evasión anti-bot, bugs corregidos en Rust y comparativa de scraping.
- **`docs/ligas_10min.md` y `docs/ligas_12min.md`:** Clasificación de ligas por duración de cuartos reglamentarios (FIBA vs NBA).

### 2. Monitoreo en Vivo (`monitor_v2/` y `monitor_v1/`)
- **`monitor_v2/` (Versión Moderna y Asíncrona):** Daemon modular de alto rendimiento basado en `asyncio`:
  - **`monitor_v2/main.py`:** Event loop de `asyncio`. Controla la sonda pre-partido, el bucle en vivo de Q4 y la liquidación final FT.
  - **`monitor_v2/config/constants.py` y `leagues.yaml`:** Configuración declarativa de umbrales y filtrado de ligas (excluidas vs. `ft_only`).
  - **`monitor_v2/database/repository.py`:** Transacciones SQLite para tablas `_v2`.
  - **`monitor_v2/models/evaluator.py`:** Ejecución de inferencias cargando en caché `v6_2` y `m27_v3`.
  - **`monitor_v2/scrapers/browser_client.py`:** Conexión CDP a Google Chrome para extracción robusta sin bloqueos.
  - **`monitor_v2/notifications/telegram_bot.py`:** Despacho de mensajes y señales operables a Telegram.
- **`monitor_v1/` (Versión Original / Telegram):** Monitor interactivo guiado por bot de Telegram:
  - **`monitor_v1/telegram_bot.py`:** Bot interactivo con teclado inline, selección de modelos (V1 a V11) y reportes.
  - **`monitor_v1/bet_monitor.py`:** Daemon en hilo secundario para sondeo y alertas en vivo.
  - **`monitor_v1/main.py`:** Punto de entrada ejecutable.

### 3. Base de Datos Central (`matches.db` en Raíz)
Base de datos SQLite (~737 MB, excluida en `.gitignore`) con almacenamiento histórico masivo:
- `matches`: Metadatos de ~40,000 partidos (equipos, liga, fecha).
- `quarter_scores` y `quarter_scores_v2`: Marcadores individuales por cuarto (Q1 a Q4).
- `play_by_play`: Casi 3 millones de eventos jugada a jugada.
- `graph_points`: Curva de momentum y presión (1.36 millones de registros).
- `match_h2h`: Historial cara a cara entre equipos (354k+ registros).
- `bet_monitor_log_v2`: Registro auditado de cada predicción en vivo (señal, confianza, resultado win/loss).

### 4. Subsistema Unificado de Modelos de Machine Learning (`models/`)
Toda la suite de ML se organiza bajo el directorio raíz `models/`:
- **`models/registry.py` e `__init__.py`:** Interfaz universal (`predict`, `predict_all`, `get_available_models`) que desacopla el ML de los monitores.
- **`models/common/`:** Módulos transversales compartidos (`schema.py` para `PredictionResult`, `data_loader.py` conectado a la raíz, `persistence.py` para `eval_match_results_v2`, `pbp_utils.py`, `time_clip.py`, `h2h.py`, `monte_carlo.py`).
- **`models/m27_v3/`:** Directorio del modelo **campeón actual (`m27_v3`)** con `train.py`, `predict.py`, `evaluate.py`, `features.py` y `model_outputs/`.
- **`models/v6_2/`:** Directorio del modelo **campeón Q3/Q4 (`v6_2`)** con poda de ligas.
- **`models/v6_3/`:** Modelo Q4 con blacklist manual y snapshots tempranos.
- **`models/v1/` a `models/v17/`:** Versiones históricas y experimentales homologadas.
- **`match/training/infer_match.py`:** Adaptador retrocompatible preservado para llamadas heredadas.

### 5. Suite Analítica y Motor de Consenso (`tools/`)
- **`tools/stats_cli.py` (3,144 líneas):** Contiene el `FusionConsensusEngine` que combina `v6_2` y `m27_v3`, genera reportes a Excel multi-hoja y exporta resúmenes sintéticos (`MetaModel_ALL.txt`).
- **`tools/obscura-src/`:** Código fuente en Rust del navegador headless Obscura con parches para cookies cross-origin y tiempos de promesa.

### 6. Interfaz y Operación
- **`menu.bat`:** Centro de control unificado y único script por lotes del sistema. Organizado en 5 bloques temáticos (1-25 opciones y soporte CLI): Operación en Vivo, Análisis/Consenso, Ingesta/Backfill, Modelos ML y Mantenimiento (incluyendo control total de Obscura).
- **`api.py`:** Backend en FastAPI.
- **`dashboard/`:** Frontend en React + Vite + TypeScript.

### 7. Scripts Temporales y Experimentales (`tmp/`)
- Todos los scripts auxiliares de diagnóstico, pruebas de concepto, inspecciones de base de datos y experimentos puntuales han sido consolidados en `tmp/` en la raíz.
- Cada archivo en `tmp/` cuenta con un encabezado descriptivo indicando su propósito y ubicación original.
- **Creación de nuevas pruebas:** Deben ubicarse obligatoriamente en `tmp/<nombre_tarea>/` (por ejemplo: `tmp/modelo_100/`) y documentar en la cabecera del código su propósito y qué problema resuelven.

---

## 📌 Resumen de Directrices Técnicas
- **Archivos Temporales y Auxiliares:** Crear siempre dentro de `tmp/<nombre_tema>/` (ej. `tmp/modelo_100/`). PROHIBIDO dejar scripts o archivos sueltos en la raíz o en los paquetes de producción.
- **Base de Datos matches.db:** Se ubica estrictamente en la raíz (`matches.db`) y está excluida por `.gitignore`. PROHIBIDO buscarla o crear copias en `match/` u otros subdirectorios.
- **Clarificación de Requerimientos:** Ante solicitudes ambiguas, incompletas o dudosas, formular de inmediato preguntas estructuradas al usuario; nunca asumir intenciones.
- **Entorno Virtual (.venv):** Obligatorio. Siempre activar `.venv\Scripts\activate` o ejecutar `.venv\Scripts\python.exe`. NUNCA usar Python global.
- **Codificación en Windows:** Al imprimir a consola, configurar salida UTF-8 (`sys.stdout.reconfigure(encoding='utf-8')`).
- **Scraping:** Usar siempre Google Chrome Headless vía CDP e inyectar llamadas con `page.evaluate(fetch(...))` para respetar la sesión y evitar baneos Cloudflare (HTTP 403).
- **Al finalizar cualquier tarea:** ¡No olvides realizar el **commit con mensaje corto en español** y el **git push**!
