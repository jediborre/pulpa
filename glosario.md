# Glosario de Monitoreo - Bet Monitor V2 🏀

Este documento explica de forma clara y concisa el significado de cada tipo de mensaje de log generado en tiempo real por el daemon de monitoreo modularizado **Bet Monitor V2**.

---

## 🏗️ 1. Estructura de un Mensaje de Log Estándar

Cada línea de log sigue este formato estricto y colorizado en consola:
```text
{Fecha Hora} [NIVEL] [COMPONENTE] [CATEGORÍA] Mensaje descriptivo
```

*   **Niveles (`[NIVEL]`)**:
    *   `[INFO]` (Azul): Flujo operativo normal e información de seguimiento.
    *   `[WARNING]` (Amarillo): Desvíos temporales, ventanas superadas o alertas de reintentos seguros.
    *   `[ERROR]` (Rojo): Fallos críticos o bloqueos de red (imprime `HTTP 403/404` en rojo brillante).
*   **Componentes (`[COMPONENTE]`)**:
    *   `[SYSTEM]`: Sistema central, carga del itinerario o configuración global.
    *   `[MONITOREO]`: Control y flujo del ciclo de vida de los partidos individuales.
    *   `[EVALUACION]`: Ejecución de los modelos de inferencia y toma de decisiones.
    *   `[DESCARGA]`: Extracción de datos robustos finales por Playwright (Chrome Headless).

---

## 🏷️ 2. Glosario de Prefijos de Logs

Para hacer los logs increíblemente legibles y compactos, cada acción utiliza un tag o prefijo temático de categoría entre corchetes:

### ⚙️ `[SYSTEM]` - Ciclo de Vida del Daemon
*   `SYSTEM Iniciando bet_monitor_v2...`
    *   *Significado:* El daemon se está iniciando e inicializando las tablas SQLite `_v2`, sincronizando la configuración y limpiando pendientes.
*   `SYSTEM Descargando itinerario: YYYY-MM-DD | YYYY-MM-DD`
    *   *Significado:* Conexión al backend de Sofascore para descargar el listado de partidos programados del día de hoy y de mañana.
*   `SYSTEM Itinerario YYYY-MM-DD guardado: N partidos`
    *   *Significado:* Partidos programados insertados con éxito en la base de datos local `bet_monitor_schedule_v2` con estatus `pending`.
*   `SYSTEM Excluido: Equipo A vs Equipo B (liga: Nombre de Liga)`
    *   *Significado:* Partido ignorado automáticamente porque su liga está declarada en la sección `excluded_leagues` de `config/leagues.yaml`.

### 👁️ `[WATCHER]` - Orquestación de Tareas
*   `MONITOREO [WATCHER] Start: {match_display} | ft_only=True/False`
    *   *Significado:* Se ha creado una corrutina asíncrona independiente dedicada a vigilar y controlar en exclusiva este partido.
    *   `ft_only=True`: Flujo simplificado. No consume recursos evaluando, solo guardará el resultado final (FT) al acabar.
    *   `ft_only=False`: Monitoreo en vivo completo de Q4 con motores neuronales.

### 🛡️ `[PROBE]` - Modo Sonda Ligera (Pre-Partido)
*   `MONITOREO [PROBE] En Vivo: {match_display}`
    *   *Significado:* La sonda ligera de bajo consumo detecta que el partido ha comenzado a jugarse. Saliendo de modo espera a modo activo.
*   `MONITOREO [PROBE] Finalizado prematuro: {match_display}`
    *   *Significado:* El partido se canceló, se pospuso o terminó antes de que se activara el bucle activo de Q4. El watcher se cierra limpiamente.
*   `MONITOREO [PROBE] Drift Exit: {match_display}`
    *   *Significado:* Medida de seguridad. Ha pasado suficiente tiempo desde la hora teórica de inicio sin recibir estatus "En Vivo", por lo que se asume que el juego ya comenzó y se avanza al bucle activo.
*   `MONITOREO [PROBE] Timeout 6h: {match_display}`
    *   *Significado:* El partido lleva más de 6 horas en limbo o pospuesto sin arrancar. Se descarta automáticamente para evitar colgar el daemon.

### ⏱️ `[FT-ONLY]` - Modo Degradado Simple
*   `MONITOREO [FT-ONLY] Espera pasiva: {tiempo} | {match_display}`
    *   *Significado:* El partido pertenece a una liga de la categoría `ft_only_leagues`. El watcher entra en modo dormido pasivo hasta que teóricamente el juego termine para hacer la descarga robusta final.

### 🟠 `Q4 🟠 [LIVE]` - Monitoreo Activo en Vivo
*   `MONITOREO Q4 🟠 {match_display} [{Periodo} | min~{Minuto}] Score: X - Y`
    *   *Significado:* **Log en vivo principal**. Indica que el partido está en juego.
    *   Muestra el cuarto exacto (ej: `4th quarter`, `Halftime`, `3rd quarter`) y el minuto estimado del partido de forma clara.
    *   Utiliza los marcadores reales actualizados al segundo.
*   `MONITOREO [LIVE] Fin detectado -> Derivando a FT | {match_display}`
    *   *Significado:* El partido ha concluido su tiempo reglamentario. Se detiene el bucle de Q4 y se envía a la cola de persistencia final Playwright (FT).
*   `MONITOREO [LIVE] Ventana Q4 superada (min {minuto}) | {match_display}`
    *   *Significado:* El partido cruzó el minuto límite operable (minuto 38) sin dar oportunidades de apuesta. Se detiene la corrutina para ahorrar ancho de banda.
*   `MONITOREO [LIVE] Q4 Concluido. Esperando FT | {match_display}`
    *   *Significado:* Ambos modelos (`v6_2` y `m27_v3`) ya tomaron una decisión definitiva (operaron una señal o declararon un `NO_BET` definitivo). Se entra en modo de reposo pasivo hasta el cierre del partido.

### 🧠 `[EVAL]` - Motor de Inferencia Neuronal
*   `EVALUACION Q4 🟠 [EVAL] Señal operable [{sig}] | Model: {model} | {match_display}`
    *   *Significado:* **¡Alerta Operable Detectada!** El modelo neuronal (`v6_2` o `m27_v3`) ha encontrado un desvío matemático de cuota de valor operable y emite la señal.
    *   *Señales operables:* `HOME_BET_Q4`, `AWAY_BET_Q4`.
    *   *Acción automática:* Se registra en la BD local `bet_monitor_log_v2` y se despacha de inmediato una **alerta con emoji verde 🟢 a Telegram**.
*   `EVALUACION Q4 🟠 [EVAL] NO_BET definitivo | Model: {model} | {match_display}`
    *   *Significado:* El modelo ha analizado el partido dentro de la ventana de Q4 (minutos 27-36) y ha determinado matemáticamente que no hay valor suficiente. Se registra la señal `NO_BET` definitiva para liberar el partido.

### 📥 `[DESCARGA]` o `[FT]` - Persistencia Final
*   `DESCARGA [FT] Final y liquidación de apuestas persistidos | Equipo A vs Equipo B (gp=N)`
    *   *Significado:* El partido se ha descargado completamente usando el motor robusto Playwright Headless.
    *   Se guardan los scores de cada cuarto de manera atómica en `quarter_scores_v2`.
    *   Se cotejan los resultados finales con las apuestas guardadas y se actualizan los estatus a **Ganada (`win`)** o **Perdida (`loss`)** en `bet_monitor_log_v2`.
    *   Se envía un mensaje de **cierre y confirmación final (`✅` o `❌`) a Telegram**.

---

## 🎨 3. Formato y Colores Estándar en Pantalla

*   **Horarios del Partido**: Se muestran siempre en **Amarillo** (ej. `00:00`).
*   **IDs de Partido**: Se muestran siempre en **Azul** (ej. `15935079`).
*   **Prefijo Live de Cuarto Final**: Usa siempre la etiqueta destacada con emoji `Q4 🟠` para fácil reconocimiento visual.
*   **Visualización Estándar**:
    ```text
    18:30 15935079 Cleveland Cavaliers vs New York Knicks
    ```
