# 🖼️ Imagen de Apuesta Q4 (`monitor_v3`)

> **Ámbito:** `monitor_v3` en producción. Genera la **imagen que acompaña a los avisos de Telegram** de apuestas Q4 y de resultados FT.
> **Implementación:** `monitor_v3/notifications/bet_image.py` (dibujo) + `monitor_v3/notifications/telegram_bot.py` (envío).
> **Pruebas:** `tmp/q4_bet_image/` (`render_samples.py`, `test_edge_cases.py`, `send_to_chat.py`).

---

## 1. Dos modos de imagen

| Modo | Cuándo | Contenido | Flag |
| --- | --- | --- | --- |
| **Apuesta en vivo** | `send_bet_alert`, `send_combined_bet_alert` | **Solo tarjeta**: cabecera + panel de apuesta (sin gráfica, sin resultado) | `include_chart=False` |
| **Resultado FT** | `send_final_confirmation`, `send_combined_final_confirmation` | Tarjeta + **gráfica de momentum completa** del partido | `is_result=True, include_chart=True` |

La imagen sustituye al mensaje de texto (se envía como **foto con caption**); el texto del aviso se recorta a 1024 caracteres y viaja como pie de foto. **Si la imagen falla, el aviso cae automáticamente a texto plano** (nunca se pierde la alerta).

---

## 2. Layout

```text
┌─ 1080 px ─────────────────────────────────────────────────────┐
│ ARGENTINA | LIGA ARGENTINA, CONFERENCE SOUTH                  │  país gris · liga cian
│ #12  Deportivo Viedma  vs  PFC Básquet                        │  nº del día · local verde · visitante violeta
│                                                               │  (ganador en negrita)
│ 00:46 · FT · MIN~40 · MARCADOR 86-76                          │  FT verde · 86 verde · 76 rojo
│ Q1 29-18   Q2 24-19   Q3 12-22   Q4 21-17                     │  Q cian · ganador verde · perdedor rojo
│ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │
│ ┃ RESULTADO Q4                        (o APUESTA Q4)        ┃ │  barra de acento lateral
│ ┃ BET     m27_v5_3 78%  ->  🏠 Deportivo Viedma    GANADA   ┃ │
│ ┃ Cuota 1.85 (-118) · $500 -> $925.00 (+$425.00)            ┃ │
│ ┃ NO BET  m27_v5_2 51%  ->  ✈️ PFC Básquet    HUBIERA GANADO┃ │
│ ┃ no operó (umbral 75%)                                     ┃ │
│ ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛ │
├───────────────────────────────────────────────────────────────┤  (solo en FT)
│ Q1        Q2        Q3        Q4                              │  etiquetas de cuarto destacadas
│ ░░░░░░░═════ gráfica de momentum estilo SofaScore ═════░░░░░░ │  verde local / añil visitante
│ 00:46                                              FT         │  inicio verde · FT verde
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Paleta

### 3.1 Tarjeta

| Constante | RGB | Uso |
| --- | --- | --- |
| `COLOR_BG` | `#0C141B` | Fondo general |
| `COLOR_PANEL_ALT` | `#132031` | Panel de apuesta/resultado |
| `COLOR_BORDER` | `#2B3C4F` | Bordes |
| `COLOR_TEXT` / `COLOR_MUTED` | `#E2E8F0` / `#8FA0B4` | Texto principal / secundario |
| `COLOR_HOME` / `COLOR_AWAY` | `#4CD05E` / `#7882EB` | Nombre del local / visitante |
| `COLOR_ACCENT` | `#4FD3FF` | Liga, etiquetas `Q`, acento neutro |
| `COLOR_ALERT` | `#FF4D57` | Apuesta tardía, alertas |
| `SS_BET` / `SS_NO_BET` | `#3FCB5C` / `#FF4D57` | Palabra `BET` / `NO BET` |
| `SS_SCORE_WIN` / `SS_SCORE_LOSS` | `#3FCB5C` / `#FF4D57` | Marcador del ganador / perdedor |
| `SS_FT` | `#3FCB5C` | Token `FT` destacado |
| `SS_CONF_BET` / `SS_CONF_NEAR` / `SS_CONF_FAR` | `#3FCB5C` / `#FFD24A` / `#FF4D57` | Semáforo de confianza |

### 3.2 Gráfica de momentum (estilo oficial SofaScore dark)

| Constante | RGB | Uso |
| --- | --- | --- |
| `SS_ZONE_HOME` | `#163226` | Tinte de la mitad local |
| `SS_ZONE_AWAY` | `#1F2840` | Tinte de la mitad visitante |
| `SS_FILL_HOME` / `SS_FILL_AWAY` | `#3FCB5C` / `#8C9CF7` | Relleno sobre / bajo la línea base |
| `SS_BASELINE` | `#D0D6DE` | Línea base (0) |
| `SS_SEPARATOR` | `#000000` | **Separadores de cuarto en negro** |
| `SS_Q_LABEL` | `#4FD3FF` | Etiquetas `Q1..Q4` sobre la gráfica |
| `SS_Y_LABEL` / `SS_Y_ZERO` | `#4CD05E` / `#C8D0DA` | Escala derecha (máx/máx y 0) |
| `SS_X_LABEL` / `SS_X_FT` | `#4CD05E` / `#3FCB5C` | Hora de inicio y `FT` (ambos verdes) |
| `SS_LIVE` | `#FF4D57` | Línea/punto del minuto actual y cuarto en curso |

---

## 4. Anatomía de la línea por modelo

```text
BET     m27_v5_3 78%  ->  🏠 Deportivo Viedma              GANADA
└──┬──┘ └──┬───┘ └┬┘  └┬┘ └──┬───┘ └──────┬──────┘        └──┬──┘
   │        │      │    │     │             │                 │
   │        │      │    │     │             └── equipo (nombre del pick)
   │        │      │    │     └── emoji del lado (🏠 local / ✈️ visitante)
   │        │      │    └── separador
   │        │      └── porcentaje con SEMÁFORO (§4.2)
   │        └── modelo
   └── BET verde / NO BET rojo (§4.1)
                                etiqueta de resultado solo en FT (§4.4)
```

**Línea 2** (`Cuota …` / `no operó (umbral …%)`):
- Modelo que apostó: `Cuota <decimal> (<americano>) · $500 -> <total> (+<ganancia>)`; en FT refleja el resultado real: `$500 -> $0.00 (-$500.00)` si perdió, `$500 -> $500.00 ($0.00)` si push.
- Modelo que no apostó: `no operó (umbral <X>%)`, con el **umbral real** del modelo.

### 4.1 `BET` / `NO BET`
- Apuesta operable (señal con `BET` y sin `NO_BET`) → **`BET` en verde** `#3FCB5C`.
- No operable → **`NO BET` en rojo** `#FF4D57`.

### 4.2 Semáforo del porcentaje
- **Apostó → verde.**
- **No apostó → amarillo→rojo** según cuánto le faltó, comparando contra el **umbral real**:
  `ratio = confianza / umbral`; ≥ 90 % del umbral → amarillo `#FFD24A`; ≤ 45 % → rojo `#FF4D57`; interpolación lineal en medio.

Umbrales reales (leídos de `inference_json.raw_json`, con respaldo por familia):

| Modelo | Clave en `raw_json` | Valor real |
| --- | --- | --- |
| `m27_v5_3` | `min_conf_threshold` | **0.72** |
| `m27_v5_2` / `m27_v5_1` | `min_conf_threshold` | **0.75** |
| `v6_2` | `threshold_bet` | **0.14** (q4) / 0.18 (q3) |
| `m27_v3` | no lo expone | 0.68 (respaldo) |
| cualquiera | no lo expone | 0.65 (respaldo) |

`_threshold_pct()` en `telegram_bot.py` usa el mismo valor para el texto `[NO BET (&lt;72%)]` de los mensajes de Telegram.

### 4.3 Lado apostado
Emoji en **color real** con `seguiemj.ttf` (`embedded_color=True`): 🏠 local, ✈️ visitante. Si el sistema no tuviera fuente de emoji, cae a `LOCAL` / `VISITANTE`.

### 4.4 Etiquetas de resultado (solo FT)

| Situación | Etiqueta | Color |
| --- | --- | --- |
| Apostó y ganó | `GANADA` | verde |
| Apostó y perdió | `PERDIDA` | rojo |
| **No apostó** y su pick habría ganado | `HUBIERA GANADO` | verde |
| **No apostó** y su pick habría perdido | `HUBIERA PERDIDO` | rojo |
| Push | `PUSH` | gris |
| Otro | `PENDIENTE` | gris |

El resultado hipotético se calcula con el **parcial del cuarto objetivo** (Q4 por defecto), que es el mercado real de la apuesta: `reconcile_pending_results()` liquida comparando `quarter_scores_v2.q4_home` vs `q4_away`, **no** el marcador final del partido. Por eso un modelo que pickeó al local puede mostrar `HUBIERA PERDIDO` aunque el local haya ganado el partido, si perdió el Q4. Se omite la hipótesis si el cuarto está empatado o no hay datos del cuarto. La **barra de acento** del panel resume el conjunto: verde si todo ganó, roja si todo perdió, cian si está mezclado.

### 4.5 Número del partido dentro del día (`#N`)
Cada partido del itinerario recibe un número secuencial **por fecha** (columna `daily_seq` de `bet_monitor_schedule_v3`, ordenado por hora de inicio, reiniciando en 1 cada día). Se muestra como distintivo `#12` en cian antes de los equipos, tanto en la imagen como en el texto del aviso, para poder seguir un partido sin ambigüedad.

---

## 5. Marcadores y cuartos

- **Marcador global**: `MARCADOR ` en gris + ganador en **verde negrita** + `-` gris + perdedor en **rojo negrita** (empate sin color).
- **Marcador por cuarto**: `Q1 29-18  Q2 24-19 …`, con la etiqueta `Q` en **cian negrita** y cada parcial coloreado igual que el global. La línea se auto-ajusta de tamaño si hay muchas entradas (prórrogas `OT`).
- **Nombres de equipos**: el ganador del partido va en **negrita**; el perdedor en regular.

---

## 6. Gráfica de momentum

Réplica del gráfico de presión de la web oficial de SofaScore (modo dark), dibujada **con Pillow** en el mismo lienzo de la tarjeta:

1. Fondo oscuro con **tinte verde** en la mitad local y **añil** en la visitante.
2. **Relleno verde** (`#3FCB5C`) para valores ≥ 0 y **añil** (`#8C9CF7`) para < 0, con **interpolación exacta en los cruces por cero** (`_split_runs`).
3. **Separadores de cuarto en negro**, sin etiquetas dentro del gráfico.
4. **Etiquetas `Q1..Q4` sobre la gráfica**, en cian; el **cuarto en curso en rojo** (en partidos terminados no se resalta ninguno).
5. Etiquetas de escala a la derecha (`máx / 0 / máx`) en **verde**.
6. Hora de inicio abajo-izquierda en **verde** con el punto de origen, y **`FT` también en verde** abajo-derecha.
7. **Marcador de tiempo real** (línea + punto rojos) y etiqueta del periodo en rojo (`4th quarter`), que cae al último dato si el API no expone el reloj.

Duración del eje (`_total_minutes`): `period_length` del API (10 min FIBA / 12 min NBA), contemplando partidos de 2 mitades; el eje nunca es menor que el último minuto con datos.

---

## 7. Detalles técnicos

| Aspecto | Regla |
| --- | --- |
| Motor | **Pillow** puro (sin matplotlib): la gráfica se dibuja en el mismo lienzo, sin archivos temporales ni candados de hilo |
| Supersampling | `CHART_SS = 2` (se dibuja al doble y se reduce con LANCZOS para bordes suaves) |
| Formato | **JPEG** `quality=92`, construido en memoria (`io.BytesIO`) como `bytes`; no se escribe a disco |
| Fuentes | **Segoe UI** (`segoeui.ttf` / `seguisb.ttf`) con respaldo Arial, DejaVu Sans y, en último caso, `load_default()` |
| Emoji | **Segoe UI Emoji** (`seguiemj.ttf`) con `embedded_color=True`; respaldo monocromo y, si no existe, texto |
| Tamaño | 1080 px de ancho; alto dinámico (tarjeta ≈ 300-420 px + panel por modelo + gráfica 320 px) |
| Sin emojis en texto | Las fuentes de texto no traen glifos emoji: los emojis van solo por la fuente de color |
| Hilos | La generación se ejecuta con `asyncio.to_thread` para no bloquear el event loop del daemon |
| Envío | `sendPhoto` por **multipart/form-data** (`requests`), con `reply_markup` del botón 📱 Sofascore |
| Caption | Recortado a 1024 caracteres (`_photo_caption`) |
| Degradación | Cualquier fallo (dibujo o envío) → el aviso se manda en **texto plano** |

---

## 8. Pruebas y utilidades (`tmp/q4_bet_image/`)

| Script | Qué hace |
| --- | --- |
| `render_samples.py` | Genera las variantes (apuesta, combinada, tardía, resultado) con datos reales. `--live` usa un partido en vivo; `--match-id <id>` renderiza uno concreto; `--send` las envía por Telegram |
| `test_edge_cases.py` | Casos límite sin red: sin datos → texto plano, sin momentum → tarjeta sola, nombres largos, cuotas, umbrales, semáforo, resultado hipotético y recorte de caption |
| `send_to_chat.py` | Resuelve el nombre del destinatario con `getChat` y envía una muestra usando el camino real de producción (`--dry-run` para listar) |

```powershell
.venv\Scripts\python.exe tmp\q4_bet_image\test_edge_cases.py
.venv\Scripts\python.exe tmp\q4_bet_image\render_samples.py --live
.venv\Scripts\python.exe tmp\q4_bet_image\send_to_chat.py --dry-run
```

---

## 9. Límites conocidos

- Sin `graph_points` (ligas menores) la confirmación FT envía **solo la tarjeta**.
- El marcador por cuarto se colorea con el parcial de ese cuarto, no con el acumulado.
- Los emojis requieren una fuente de emoji en color; en Linux sin `NotoColorEmoji` se usa el texto `LOCAL`/`VISITANTE`.
- El resultado hipotético de un `NO_BET` es informativo: la apuesta no se registró en `bet_monitor_log_v3` como operable.
