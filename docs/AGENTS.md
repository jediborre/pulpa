> **Ubicación original:** `AGENTS.md (Raíz)`

---

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
- **UBICACIÓN ÚNICA Y EXCLUSIVA:** La base de datos SQLite histórica y en vivo de todo el sistema se encuentra **estrictamente en**:
  ```
  match/matches.db
  ```
  *(Ruta relativa desde la raíz del proyecto: `match/matches.db`)*.
- **PROHIBIDO** buscar, instanciar o crear bases de datos `matches.db` en la raíz (`./matches.db`) o en cualquier otro subdirectorio.
- Cualquier script, consulta SQLite, endpoint de API, modelo de inferencia o tarea de backfill debe conectarse obligatoriamente a esta ruta canónica:
  ```python
  from pathlib import Path
  DB_PATH = Path(__file__).resolve().parents[...] / "match" / "matches.db"
  ```

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

### 2. Monitoreo en Vivo (`bet_monitor_v2/`)
La versión modular moderna del monitor en tiempo real:
- **`bet_monitor_v2/main.py`:** Event loop de `asyncio`. Controla la sonda pre-partido, el bucle en vivo de Q4 y la liquidación final FT.
- **`bet_monitor_v2/config/constants.py` y `leagues.yaml`:** Configuración declarativa de umbrales y filtrado de ligas (excluidas vs. `ft_only`).
- **`bet_monitor_v2/database/repository.py`:** Transacciones SQLite para tablas `_v2`.
- **`bet_monitor_v2/models/evaluator.py`:** Ejecución de inferencias cargando en caché `v6_2` y `m27_v3`.
- **`bet_monitor_v2/scrapers/browser_client.py`:** Conexión CDP a Google Chrome para extracción robusta sin bloqueos.
- **`bet_monitor_v2/notifications/telegram_bot.py`:** Despacho de mensajes y señales operables a Telegram.

### 3. Base de Datos Central (`match/matches.db`)
Base de datos SQLite (~737 MB) con almacenamiento histórico masivo:
- `matches`: Metadatos de ~40,000 partidos (equipos, liga, fecha).
- `quarter_scores` y `quarter_scores_v2`: Marcadores individuales por cuarto (Q1 a Q4).
- `play_by_play`: Casi 3 millones de eventos jugada a jugada.
- `graph_points`: Curva de momentum y presión (1.36 millones de registros).
- `match_h2h`: Historial cara a cara entre equipos (354k+ registros).
- `bet_monitor_log_v2`: Registro auditado de cada predicción en vivo (señal, confianza, resultado win/loss).

### 4. Pipeline de Machine Learning (`match/training/`)
- **`train_q4_m27_v3.py`:** Script de entrenamiento del modelo **campeón actual (`m27_v3`)** con features de Head-to-Head (AUC 0.789, Yield +13% a +29%).
- **`train_q4_m27_v1.py` y `train_q4_m27_v2.py`:** Versiones baseline del snapshot 27.
- **`train_q3_q4_models_v6_2.py`:** Modelo champion v6 con poda de ligas.
- **`report_m_v1_roi.py`:** Simulador de rentabilidad y ROI con criterio de Kelly.
- **`infer_match.py`:** Inferencia en vivo reutilizada por el monitor y el bot.

### 5. Suite Analítica y Motor de Consenso (`tools/`)
- **`tools/stats_cli.py` (3,144 líneas):** Contiene el `FusionConsensusEngine` que combina `v6_2` y `m27_v3`, genera reportes a Excel multi-hoja y exporta resúmenes sintéticos (`MetaModel_ALL.txt`).
- **`tools/obscura-src/`:** Código fuente en Rust del navegador headless Obscura con parches para cookies cross-origin y tiempos de promesa.

### 6. Interfaz y Operación
- **`menu.bat`:** Centro de control unificado y único script por lotes del sistema. Organizado en 5 bloques temáticos (1-24 opciones y soporte CLI): Operación en Vivo, Análisis/Consenso, Ingesta/Backfill, Modelos ML y Mantenimiento (incluyendo control total de Obscura).
- **`api.py`:** Backend en FastAPI.
- **`dashboard/`:** Frontend en React + Vite + TypeScript.

### 7. Scripts Temporales y Experimentales (`tmp/`)
- Todos los scripts auxiliares de diagnóstico, pruebas de concepto, inspecciones de base de datos y experimentos puntuales han sido consolidados en `tmp/` en la raíz.
- Cada archivo en `tmp/` cuenta con un encabezado descriptivo indicando su propósito y ubicación original.

---

## 📌 Resumen de Directrices Técnicas
- **Base de Datos matches.db:** Se ubica estrictamente en `match/matches.db`. PROHIBIDO crear copias o buscarla en la raíz.
- **Entorno Virtual (.venv):** Obligatorio. Siempre activar `.venv\Scripts\activate` o ejecutar `.venv\Scripts\python.exe`. NUNCA usar Python global.
- **Codificación en Windows:** Al imprimir a consola, configurar salida UTF-8 (`sys.stdout.reconfigure(encoding='utf-8')`).
- **Scraping:** Usar siempre Google Chrome Headless vía CDP e inyectar llamadas con `page.evaluate(fetch(...))` para respetar la sesión y evitar baneos Cloudflare (HTTP 403).
- **Al finalizar cualquier tarea:** ¡No olvides realizar el **commit con mensaje corto en español** y el **git push**!
