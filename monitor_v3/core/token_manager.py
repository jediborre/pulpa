# =====================================================================
# REGLAS DE ARQUITECTURA, FORMATEO Y LOGGING DE PRODUCCIÓN (MANTENER):
# 1. ESTRUCTURA: Estrictamente modularizado (config, core, database, scrapers, 
#    models, notifications, utils) coordinados asíncronamente por main.py.
# 2. INFRAESTRUCTURA DB: El archivo base SQLite se localiza exclusivamente en 
#    matches.db (en la raíz del proyecto) y todas las tablas de V3 finalizan
#    con el sufijo '_v3'.
# 3. TABLA DE LOGS: 'bet_monitor_log_v3' se particiona por modelo y contiene 
#    obligatoriamente los campos 'raw_json' (TEXT), 'inference_minute' (INT), 
#    y 'graph_points_count' (INT) junto con marcadores reales del juego.
# 4. CONFIGURACIÓN DE LIGAS: Consumida declarativamente desde config/leagues.yaml.
# 5. Formato Log: {Fecha Hora} [INFO/WARNING/ERROR] [COMPONENTE]
#    - Colores ANSI: INFO=Azul, WARNING=Amarillo, ERROR=Rojo.
# 6. Formato Matches en Log: {horario_match} {match_id} {home} vs {away}
# 7. Errores de red críticos: Imprimir explícitamente "HTTP 403/404" en ROJO.
# 8. Monitoreo avanzado en progreso de cuarto final: Usar obligatoriamente "Q4 🟠".
# 9. Telegram Prefijos de Apuestas: 🟢 (Bettable), 🟡 (No Bettable), ⚪ (Tardía).
# 10. Telegram Resultados FT: Prefijar con ✅ (Ganada) o ❌ (Perdida).
# 11. Conversión de tiempos siempre legibles en formato humano.
# =====================================================================

import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
import httpx

from monitor_v3.config.constants import (
    TOKENS_CONFIG_PATH,
    TOKEN_ROTATION_POOL_SIZE,
    TOKEN_MAX_FAILURES,
    SOFASCORE_PROXY_URL,
    SOFASCORE_CERT_PATH,
    SOFASCORE_MOBILE_UA,
)
from monitor_v3.utils.logger import log_info, log_warning, log_error

@dataclass
class TokenItem:
    token: str
    created_at: float
    device_uuid: str
    advertising_id: str
    failures: int = 0
    last_used: float = 0.0

class TokenPool:
    """
    Gestor dinámico de un pool rotativo de tokens JWT para la API móvil de SofaScore.
    Mantiene múltiples tokens concurrentes generados bajo demanda, realiza rotación
    Round-Robin y reemplaza de forma auto-regenerativa cualquier token que falle.
    """
    def __init__(self, pool_size: int = TOKEN_ROTATION_POOL_SIZE):
        self.pool_size = pool_size
        self._current_index = 0
        self._config_path = Path(TOKENS_CONFIG_PATH)
        self._matches_on_current_token = 0
        self.tokens: list[TokenItem] = self._load_from_disk()

    @property
    def lock(self) -> asyncio.Lock:
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None
        if not hasattr(self, "_lock_loop") or self._lock_loop != current_loop:
            self._lock_loop = current_loop
            self._lock_obj = asyncio.Lock()
        return self._lock_obj

    def _load_from_disk(self) -> list[TokenItem]:
        if not self._config_path.exists():
            return []
        try:
            with open(self._config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = []
            for d in data.get("tokens", []):
                items.append(TokenItem(
                    token=d["token"],
                    created_at=d.get("created_at", time.time()),
                    device_uuid=d.get("device_uuid", str(uuid.uuid4())),
                    advertising_id=d.get("advertising_id", str(uuid.uuid4())),
                    failures=d.get("failures", 0),
                    last_used=d.get("last_used", 0.0),
                ))
            return items
        except Exception as e:
            log_warning("TOKEN_POOL", f"No se pudo cargar tokens.json: {e}")
            return []

    def _save_to_disk(self) -> None:
        try:
            self._config_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self._config_path.with_suffix(".tmp")
            payload = {
                "updated_at": time.time(),
                "count": len(self.tokens),
                "tokens": [asdict(t) for t in self.tokens]
            }
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            temp_path.replace(self._config_path)
        except Exception as e:
            log_warning("TOKEN_POOL", f"Error guardando tokens.json: {e}")

    async def _mint_token(self) -> TokenItem | None:
        """Fallback: Emite un nuevo JWT móvil llamando a /api/v1/token/init vía proxy HTTP Toolkit."""
        dev_uuid = str(uuid.uuid4())
        ad_id = str(uuid.uuid4())

        headers = {
            'User-Agent': SOFASCORE_MOBILE_UA,
            'x-timestamp': str(int(time.time() * 1000)),
            'Content-Type': 'application/json; charset=UTF8',
            'Accept-Encoding': 'gzip',
            'Connection': 'Keep-Alive',
            'Host': 'api.sofascore.com',
        }

        payload = {
            "deviceType": "android",
            "version": 260921,
            "sdk": 29,
            "language": "en",
            "country": "MX",
            "timezone": -18000,
            "advertisingId": ad_id,
            "uuid": dev_uuid
        }

        url = "https://api.sofascore.com/api/v1/token/init"
        verify_cert = SOFASCORE_CERT_PATH if os.path.exists(SOFASCORE_CERT_PATH) else True
        proxy = SOFASCORE_PROXY_URL if SOFASCORE_PROXY_URL else None

        try:
            async with httpx.AsyncClient(proxy=proxy, verify=verify_cert, timeout=12.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 200:
                    token_str = res.json().get("token")
                    if token_str:
                        return TokenItem(
                            token=token_str,
                            created_at=time.time(),
                            device_uuid=dev_uuid,
                            advertising_id=ad_id,
                            failures=0,
                            last_used=time.time()
                        )
                else:
                    log_error("TOKEN_POOL", f"Error en /token/init: HTTP {res.status_code} - {res.text[:150]}")
        except Exception as e:
            log_error("TOKEN_POOL", f"Fallo al emitir nuevo token: {e}")
        return None

    async def initialize(self) -> None:
        """Inicializa el pool cargando de disco los tokens legítimos válidos."""
        async with self.lock:
            existing = self._load_from_disk()
            self.tokens = [t for t in existing if t.failures < TOKEN_MAX_FAILURES]
            self._save_to_disk()
            log_info("TOKEN_POOL", f"Pool activo listo con {len(self.tokens)} tokens en rotación.")

    def notify_match_done(self) -> None:
        """
        Notifica que se completó la descarga de un partido.
        Cada 10 partidos consecutivos rota automáticamente al siguiente token
        del pool para alternar la sesión y distribuir la carga entre usuarios.
        """
        self._matches_on_current_token += 1
        if len(self.tokens) > 1 and self._matches_on_current_token >= 10:
            self._matches_on_current_token = 0
            self._current_index = (self._current_index + 1) % len(self.tokens)
            curr = self.tokens[self._current_index]
            log_info(
                "TOKEN_POOL",
                f"🔄 Switcheando de sesión (10 partidos completados). "
                f"Ahora usando token #{self._current_index + 1} de {len(self.tokens)} (...{curr.token[-12:]})"
            )

    async def get_token(self) -> str:
        """Retorna el token actualmente activo en el pool."""
        async with self.lock:
            if not self.tokens:
                raise RuntimeError(
                    "El pool de tokens está vacío o todos han sido desafiados. "
                    "Por favor captura o sincroniza nuevos tokens desde la App en menu.bat."
                )

            self._current_index = self._current_index % len(self.tokens)
            selected = self.tokens[self._current_index]
            selected.last_used = time.time()
            return selected.token

    async def mark_failure(self, token_str: str, status_code: int = 0) -> None:
        """
        Registra un fallo. Si el token recibe HTTP 401/403 ('challenge' o revocado)
        o excede el umbral de fallos, se elimina de inmediato de la lista de tokens válidos.
        """
        async with self.lock:
            for i, t in enumerate(self.tokens):
                if t.token == token_str:
                    t.failures += 1
                    log_warning(
                        "TOKEN_POOL",
                        f"Fallo registrado en token ...{token_str[-12:]} "
                        f"(Fallos: {t.failures}/{TOKEN_MAX_FAILURES} | Código: HTTP {status_code})"
                    )

                    if t.failures >= TOKEN_MAX_FAILURES or status_code in (401, 403):
                        log_warning(
                            "TOKEN_POOL",
                            f"❌ Token ...{token_str[-12:]} desafiado o caducado (HTTP {status_code}). "
                            f"Eliminado permanentemente de la lista de tokens válidos."
                        )
                        self.tokens.pop(i)
                        self._save_to_disk()
                        self._matches_on_current_token = 0
                        if self.tokens:
                            self._current_index = self._current_index % len(self.tokens)
                            next_tok = self.tokens[self._current_index]
                            log_info(
                                "TOKEN_POOL",
                                f"➡️ Switcheando inmediatamente al siguiente token disponible: "
                                f"...{next_tok.token[-12:]} ({len(self.tokens)} restantes en pool)"
                            )
                        else:
                            log_error(
                                "TOKEN_POOL",
                                "⚠️ Se han agotado todos los tokens válidos del pool. "
                                "Por favor captura o sincroniza nuevos tokens desde la App en menu.bat."
                            )
                        break

    def get_pool_status(self) -> dict:
        """Retorna estadísticas descriptivas del estado del pool."""
        return {
            "total_tokens": len(self.tokens),
            "pool_size": self.pool_size,
            "tokens": [
                {
                    "prefix": f"{t.token[:10]}...{t.token[-10:]}",
                    "failures": t.failures,
                    "age_hours": round((time.time() - t.created_at) / 3600, 1),
                }
                for t in self.tokens
            ]
        }

# Singleton global del pool de tokens
_GLOBAL_TOKEN_POOL = TokenPool()

def get_token_pool() -> TokenPool:
    return _GLOBAL_TOKEN_POOL
