"""
v6_1_league_filter.py - Post-filter para rechazar ligas de baja calidad en v6_1.

Lee los CSVs de soporte (league_bucket_support_q3.csv, league_bucket_support_q4.csv)
y aplica criterios configurables para filtrar predicciones de ligas que:
- Tienen muy pocos registros históricos
- No tienen datos de test (no se pueden validar)
- Muestran posible overfitting (confianza extrema)
- Tienen muy baja capacidad predictiva

Uso:
    from training.v6_1_league_filter import LeagueFilter
    
    league_filter = LeagueFilter.load()
    
    # Actualizar configuración
    league_filter.config.min_total_rows = 50
    league_filter.config.max_confidence = 0.75
    
    # Verificar si una liga debería filtrarse
    is_filtered, reason = league_filter.should_filter(
        league_bucket="NBA",
        target="q4",
        model_confidence=0.78,
    )
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Path to support files
MODEL_DIR_V6_1 = Path(__file__).parent / "model_outputs_v6_1"
SUPPORT_Q3_PATH = MODEL_DIR_V6_1 / "league_bucket_support_q3.csv"
SUPPORT_Q4_PATH = MODEL_DIR_V6_1 / "league_bucket_support_q4.csv"
CONFIG_PATH = MODEL_DIR_V6_1 / "league_filter_config.json"


@dataclass
class LeagueFilterConfig:
    """Configuración del filtro de ligas."""
    
    # Master switch: activar/desactivar todo el filtro
    enabled: bool = True
    
    # Mínimo de registros históricos totales para aceptar una liga
    # - Si total_rows < min_total_rows => FILTRADA
    min_total_rows: int = 50
    
    # Mínimo de registros en el test set (validación en datos no vistos)
    # - Si test_rows == 0 => FILTRADA (no se puede validar)
    # - Si test_rows < min_test_rows => WARNING pero no filtra
    min_test_rows_hard: int = 1  # Hard requirement: al menos 1
    min_test_rows_soft: int = 10  # Soft: si es < 10, log warning
    
    # Ratio mínimo de test_rows respecto a total_rows
    # - Si test_rows / total_rows < min_test_ratio => FILTRADA
    min_test_ratio: float = 0.05  # Al menos 5% de datos para test
    
    # Confianza extrema sugiere overfitting
    # - Si abs(target_positive_rate - 0.5) > confidence_threshold => FILTRADA
    # - Valores cercanos a 0.5 son neutrales, valores extremos sugieren problemas
    max_confidence_for_extreme_rate: float = 0.25  # >75% o <25% => sospechoso
    
    # Rango "seguro" de positive_rate
    # Si está fuera, requiere más datos para confiar
    min_safe_positive_rate: float = 0.40
    max_safe_positive_rate: float = 0.60
    
    # Ligas blancas (nunca filtrar)
    whitelist_leagues: set[str] = field(default_factory=lambda: {
        "NBA",
        "Euroleague",
        "China CBA",
    })
    
    # Ligas negras (siempre filtrar)
    blacklist_leagues: set[str] = field(default_factory=lambda: set())


@dataclass
class LeagueMetrics:
    """Métricas extraídas de los CSVs de soporte."""
    
    league_bucket: str
    target: str
    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    positive_rows: int
    target_positive_rate: float
    first_date: str
    last_date: str
    test_first_date: Optional[str] = None
    test_last_date: Optional[str] = None
    
    def test_ratio(self) -> float:
        """Porcentaje de datos en test set."""
        if self.total_rows == 0:
            return 0.0
        return self.test_rows / self.total_rows
    
    def has_extreme_rate(self, threshold: float = 0.25) -> bool:
        """¿Esta liga tiene un positive_rate extremo (posible overfitting)?"""
        return abs(self.target_positive_rate - 0.5) > threshold
    
    def is_in_safe_range(self) -> bool:
        """¿El positive_rate está en rango 'normal' (no extremo)?"""
        return 0.40 <= self.target_positive_rate <= 0.60


@dataclass
class FilterDecision:
    """Resultado de la decisión de filtrado."""
    
    should_filter: bool
    reason: str  # Por qué se filtró (o "", si se aceptó)
    severity: str  # "HARD" (definitivamente filtrar) o "SOFT" (advertencia)
    metrics: Optional[LeagueMetrics] = None


class LeagueFilter:
    """Post-filtro para v6_1 basado en métricas de calidad de liga."""
    
    def __init__(self, config: Optional[LeagueFilterConfig] = None):
        self.config = config or LeagueFilterConfig()
        self.metrics_by_target: dict[str, dict[str, LeagueMetrics]] = {
            "q3": {},
            "q4": {},
        }
        self._loaded = False
    
    @classmethod
    def load(cls, config: Optional[LeagueFilterConfig] = None) -> LeagueFilter:
        """Crear una instancia y cargar CSVs de soporte y configuración."""
        # Si no se proporciona config, intenta cargar desde JSON
        if config is None:
            config = cls._load_config_from_json()
        
        instance = cls(config)
        instance._load_support_files()
        return instance
    
    @classmethod
    def _load_config_from_json(cls) -> LeagueFilterConfig:
        """Cargar configuración desde league_filter_config.json si existe."""
        if not CONFIG_PATH.exists():
            logger.debug(f"Config file not found at {CONFIG_PATH}, using defaults")
            return LeagueFilterConfig()
        
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not data.get("enabled", True):
                logger.info("v6_1 league filter is disabled in config")
                config = LeagueFilterConfig(enabled=False)
                return config
            
            settings = data.get("settings", {})
            
            config = LeagueFilterConfig(
                enabled=data.get("enabled", True),
                min_total_rows=int(settings.get("min_total_rows", 50)),
                min_test_rows_hard=int(settings.get("min_test_rows_hard", 1)),
                min_test_rows_soft=int(settings.get("min_test_rows_soft", 10)),
                min_test_ratio=float(settings.get("min_test_ratio", 0.05)),
                max_confidence_for_extreme_rate=float(
                    settings.get("max_confidence_for_extreme_rate", 0.25)
                ),
                min_safe_positive_rate=float(
                    settings.get("min_safe_positive_rate", 0.40)
                ),
                max_safe_positive_rate=float(
                    settings.get("max_safe_positive_rate", 0.60)
                ),
                whitelist_leagues=set(settings.get("whitelist_leagues", ["NBA", "Euroleague"])),
                blacklist_leagues=set(settings.get("blacklist_leagues", [])),
            )
            
            logger.info(
                f"Loaded v6_1 league filter config from {CONFIG_PATH}: "
                f"enabled={config.enabled}, "
                f"min_total_rows={config.min_total_rows}, "
                f"whitelisted={len(config.whitelist_leagues)} leagues"
            )
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config from {CONFIG_PATH}: {e}")
            logger.info("Using default configuration")
            return LeagueFilterConfig()
    
    def _load_support_files(self) -> None:
        """Cargar CSVs de soporte para Q3 y Q4."""
        if not SUPPORT_Q3_PATH.exists() or not SUPPORT_Q4_PATH.exists():
            logger.warning(
                f"Support files not found: q3={SUPPORT_Q3_PATH.exists()}, "
                f"q4={SUPPORT_Q4_PATH.exists()}"
            )
            return
        
        for target, path in [("q3", SUPPORT_Q3_PATH), ("q4", SUPPORT_Q4_PATH)]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        league_bucket = row["league_bucket"]
                        metrics = LeagueMetrics(
                            league_bucket=league_bucket,
                            target=target,
                            total_rows=int(row["total_rows"]),
                            train_rows=int(row["train_rows"]),
                            validation_rows=int(row["validation_rows"]),
                            test_rows=int(row.get("test_rows", 0) or 0),
                            positive_rows=int(row["positive_rows"]),
                            target_positive_rate=float(row["target_positive_rate"]),
                            first_date=row["first_date"],
                            last_date=row["last_date"],
                            test_first_date=row.get("test_first_date") or None,
                            test_last_date=row.get("test_last_date") or None,
                        )
                        self.metrics_by_target[target][league_bucket] = metrics
                logger.info(
                    f"Loaded {len(self.metrics_by_target[target])} "
                    f"leagues from {target} support file"
                )
            except Exception as e:
                logger.error(f"Error loading {target} support file: {e}")
        
        self._loaded = True
    
    def should_filter(
        self,
        league_bucket: str,
        target: str,
        model_confidence: float = 0.5,
    ) -> tuple[bool, str]:
        """
        Determinar si una predicción de v6_1 debe filtrarse.
        
        Args:
            league_bucket: Nombre de la liga
            target: "q3" o "q4"
            model_confidence: Confianza del modelo en la predicción
        
        Returns:
            (should_filter, reason)
            - should_filter: True si debe rechazarse la predicción
            - reason: Explicación de por qué
        """
        if not self.config.enabled:
            return False, ""
        
        if not self._loaded:
            logger.warning("LeagueFilter not loaded, cannot filter")
            return False, ""
        
        # Whitelist: nunca filtrar
        if league_bucket in self.config.whitelist_leagues:
            return False, ""
        
        # Blacklist: siempre filtrar
        if league_bucket in self.config.blacklist_leagues:
            return True, f"League in blacklist: {league_bucket}"
        
        # Obtener métricas
        metrics = self.metrics_by_target.get(target, {}).get(league_bucket)
        if metrics is None:
            # Liga no en datos de entrenamiento = FILTRADA
            return True, f"League not in training data: {league_bucket}"
        
        # HARD FILTERS: criterios que siempre filtran
        
        # 1. Muy pocos registros históricos
        if metrics.total_rows < self.config.min_total_rows:
            return (
                True,
                f"Insufficient historical data: {metrics.total_rows} < {self.config.min_total_rows}",
            )
        
        # 2. Sin validación (no hay test set)
        if metrics.test_rows < self.config.min_test_rows_hard:
            return (
                True,
                f"No test set validation data: test_rows={metrics.test_rows}",
            )
        
        # 3. Muy bajo ratio de test
        test_ratio = metrics.test_ratio()
        if test_ratio < self.config.min_test_ratio:
            return (
                True,
                f"Insufficient test ratio: {test_ratio:.1%} < {self.config.min_test_ratio:.1%}",
            )
        
        # 4. Confianza extrema (posible overfitting)
        if metrics.has_extreme_rate(self.config.max_confidence_for_extreme_rate):
            # Si tiene confianza extrema en los datos de training, es sospechoso
            # A menos que tenga MUCHOS datos (para justificar la confianza)
            if metrics.total_rows < 200 or metrics.test_rows < 30:
                return (
                    True,
                    f"Extreme win rate ({metrics.target_positive_rate:.1%}) "
                    f"with insufficient validation data "
                    f"(total={metrics.total_rows}, test={metrics.test_rows}) "
                    f"— potential overfitting",
                )
        
        # SOFT FILTERS: criterios que generan warning pero no filtran
        
        if metrics.test_rows < self.config.min_test_rows_soft:
            logger.warning(
                f"[v6_1] Low test set for {league_bucket} ({target}): "
                f"{metrics.test_rows} < {self.config.min_test_rows_soft} — "
                f"predictions may be unreliable"
            )
        
        if not metrics.is_in_safe_range():
            logger.warning(
                f"[v6_1] Extreme win rate for {league_bucket} ({target}): "
                f"{metrics.target_positive_rate:.1%} (possible edge league)"
            )
        
        # Pasó todos los filtros
        return False, ""
    
    def apply_to_prediction(
        self,
        league_bucket: str,
        target: str,
        prediction_dict: dict,
        model_confidence: float = 0.5,
    ) -> dict:
        """
        Aplicar filtro a una predicción. Si se filtra, marca como unavailable.
        
        Args:
            league_bucket: Nombre de la liga
            target: "q3" o "q4"
            prediction_dict: Dict de predicción de infer_match.py
            model_confidence: Confianza del modelo
        
        Returns:
            Dict modificado (si se filtra, tendrá available=False)
        """
        should_filter, reason = self.should_filter(league_bucket, target, model_confidence)
        
        if should_filter:
            prediction_dict = dict(prediction_dict)  # copia
            prediction_dict["available"] = False
            prediction_dict["reason"] = f"v6_1_league_filter: {reason}"
            logger.info(f"[v6_1] Filtered {league_bucket} ({target}): {reason}")
        
        return prediction_dict
