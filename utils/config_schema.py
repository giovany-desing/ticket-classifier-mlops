"""
Schemas Pydantic para validación de configuración.

Valida config.yaml al cargar, asegurando tipos correctos y valores válidos.

Autor: Sistema MLOps
Fecha: 2024
"""

from pydantic import BaseModel, Field, validator, ValidationError
from typing import List, Optional
import logging
import re

logger = logging.getLogger(__name__)


class DataConfig(BaseModel):
    """Configuración de datos"""
    raw_path: str = Field(..., description="Path al directorio de datos raw")
    dataset_name: str = Field(..., description="Nombre del archivo de dataset")
    target_col: str = Field(..., description="Nombre de la columna objetivo")
    
    @validator('raw_path')
    def validate_raw_path(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("raw_path no puede estar vacío")
        return v.strip()


class MLflowTrackingConfig(BaseModel):
    """Configuración de MLflow"""
    experiment_name: str = Field(..., description="Nombre del experimento MLflow")
    
    @validator('experiment_name')
    def validate_experiment_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("experiment_name no puede estar vacío")
        return v.strip()


class PreprocessingConfig(BaseModel):
    """Configuración de preprocesamiento"""
    custom_stopwords: List[str] = Field(default_factory=list, description="Stopwords personalizadas")
    language: str = Field(default="spanish", description="Idioma para stemming")
    apply_stemming: bool = Field(default=True, description="Aplicar stemming")
    
    @validator('language')
    def validate_language(cls, v):
        valid_languages = ['spanish', 'english', 'portuguese', 'french', 'german']
        if v.lower() not in valid_languages:
            raise ValueError(f"language debe ser uno de: {valid_languages}")
        return v.lower()


class MonitoringConfig(BaseModel):
    """Configuración de monitoreo"""
    drift_threshold: float = Field(default=0.05, ge=0, le=1, description="Umbral de drift (p-value)")
    drift_score_threshold: float = Field(default=0.5, ge=0, le=1, description="Score de drift para reentrenamiento")
    min_predictions_for_drift: int = Field(default=100, ge=1, description="Mínimo de predicciones para analizar drift")
    performance_drop_threshold: float = Field(default=0.05, ge=0, le=1, description="Caída de performance para trigger")
    
    min_labeled_predictions: int = Field(default=50, ge=1, description="Mínimo de predicciones etiquetadas")
    evaluation_window_hours: int = Field(default=48, ge=1, description="Ventana de tiempo para evaluación")
    
    retrain_on_drift: bool = Field(default=True, description="Reentrenar automáticamente si hay drift")
    retrain_on_performance_drop: bool = Field(default=True, description="Reentrenar si performance baja")
    min_improvement_for_deploy: float = Field(default=0.01, ge=0, le=1, description="Mejora mínima para deploy")
    
    @validator('drift_threshold', 'performance_drop_threshold', 'min_improvement_for_deploy')
    def validate_threshold_range(cls, v, field):
        if not 0 <= v <= 1:
            raise ValueError(f"{field.name} debe estar entre 0 y 1, recibido: {v}")
        return v


class ThresholdsConfig(BaseModel):
    """Configuración de umbrales avanzados"""
    # Drift detection
    drift_score_threshold: float = Field(default=0.5, ge=0, le=1)
    ks_test_threshold: float = Field(default=0.05, ge=0, le=1)
    chi2_test_threshold: float = Field(default=0.05, ge=0, le=1)
    vocab_growth_threshold: float = Field(default=1.2, ge=1.0)
    
    # Performance
    f1_score_minimum: float = Field(default=0.85, ge=0, le=1)
    performance_drop_threshold: float = Field(default=0.05, ge=0, le=1)
    min_improvement_for_deploy: float = Field(default=0.01, ge=0, le=1)
    
    # Confidence
    low_confidence_threshold: float = Field(default=0.6, ge=0, le=1)
    low_confidence_ratio_max: float = Field(default=0.3, ge=0, le=1)
    
    # Data requirements
    min_samples_training: int = Field(default=100, ge=1)
    min_predictions_for_drift: int = Field(default=50, ge=1)
    min_labeled_predictions: int = Field(default=50, ge=1)
    
    # Retention
    metrics_retention_days: int = Field(default=30, ge=1)
    backup_retention_count: int = Field(default=5, ge=1)


class TrainingConfig(BaseModel):
    """Configuración de entrenamiento"""
    test_size: float = Field(default=0.2, ge=0.1, le=0.5, description="Tamaño del conjunto de test")
    random_seed: int = Field(default=42, ge=0, description="Seed para reproducibilidad")
    n_trials_default: int = Field(default=20, ge=1, description="Trials de Optuna (default)")
    n_trials_ci: int = Field(default=10, ge=1, description="Trials de Optuna (CI/CD)")
    cv_folds_default: int = Field(default=3, ge=2, description="Folds de CV (default)")
    cv_folds_ci: int = Field(default=2, ge=2, description="Folds de CV (CI/CD)")
    max_features: int = Field(default=5000, ge=100, description="Features máximas TF-IDF")
    
    @validator('test_size')
    def validate_test_size(cls, v):
        if not 0.1 <= v <= 0.5:
            raise ValueError(f"test_size debe estar entre 0.1 y 0.5, recibido: {v}")
        return v


class TimeoutsConfig(BaseModel):
    """Configuración de timeouts"""
    api_health_check: int = Field(default=5, ge=1, description="Timeout health check (segundos)")
    api_drift_check: int = Field(default=10, ge=1, description="Timeout drift check (segundos)")
    training_max: int = Field(default=3600, ge=60, description="Timeout máximo entrenamiento (segundos)")
    dvc_operation: int = Field(default=300, ge=10, description="Timeout operaciones DVC (segundos)")
    
    @validator('training_max')
    def validate_training_max(cls, v):
        if v > 7200:  # 2 horas
            logger.warning(f"training_max es muy alto: {v}s (>2 horas), considera reducirlo")
        return v


class APIConfig(BaseModel):
    """Configuración de API"""
    host: str = Field(default="0.0.0.0", description="Host de la API")
    port: int = Field(default=8000, ge=1024, le=65535, description="Puerto de la API")
    log_dir: str = Field(default="monitoring/logs", description="Directorio de logs")
    
    @validator('port')
    def validate_port(cls, v):
        if v < 1024 or v > 65535:
            raise ValueError(f"port debe estar entre 1024 y 65535, recibido: {v}")
        return v


class DatabaseConfig(BaseModel):
    """Configuración de base de datos"""
    table_name: str = Field(default="tickets_fiducia", description="Nombre de la tabla")
    auto_update: bool = Field(default=True, description="Actualizar BD automáticamente al predecir")
    
    @validator('table_name')
    def validate_table_name(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError("table_name no puede estar vacío")
        # Validar caracteres válidos para nombre de tabla SQL
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', v):
            raise ValueError(f"table_name inválido: {v}. Debe contener solo letras, números y guiones bajos")
        return v.strip()


class Config(BaseModel):
    """Configuración completa del sistema"""
    data: DataConfig
    mlflow_tracking: MLflowTrackingConfig
    preprocessing: PreprocessingConfig
    monitoring: MonitoringConfig
    thresholds: ThresholdsConfig
    training: TrainingConfig
    timeouts: TimeoutsConfig
    api: APIConfig
    database: DatabaseConfig
    
    class Config:
        """Configuración de Pydantic"""
        validate_assignment = True  # Validar también al asignar valores
        extra = 'forbid'  # No permitir campos extras no definidos


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def validate_config(config_dict: dict) -> Config:
    """
    Valida un diccionario de configuración contra el schema.
    
    Args:
        config_dict: Diccionario con la configuración
        
    Returns:
        Objeto Config validado
        
    Raises:
        ValidationError: Si la configuración es inválida
    """
    try:
        config = Config(**config_dict)
        logger.info("✅ Configuración validada exitosamente")
        return config
    except ValidationError as e:
        logger.error("❌ Error en validación de configuración:")
        for error in e.errors():
            field = ' → '.join(str(x) for x in error['loc'])
            message = error['msg']
            value = error.get('input', 'N/A')
            logger.error(f"   Campo: {field}")
            logger.error(f"   Error: {message}")
            logger.error(f"   Valor: {value}")
        raise


def print_config_summary(config: Config):
    """
    Imprime un resumen de la configuración validada.
    
    Args:
        config: Objeto Config validado
    """
    print("=" * 80)
    print("CONFIGURACIÓN VALIDADA")
    print("=" * 80)
    print(f"\n📊 Datos:")
    print(f"   Dataset: {config.data.dataset_name}")
    print(f"   Path: {config.data.raw_path}")
    print(f"   Target: {config.data.target_col}")
    
    print(f"\n🔬 Entrenamiento:")
    print(f"   Test size: {config.training.test_size}")
    print(f"   Random seed: {config.training.random_seed}")
    print(f"   Optuna trials: {config.training.n_trials_default}")
    print(f"   CV folds: {config.training.cv_folds_default}")
    
    print(f"\n⚠️  Monitoreo:")
    print(f"   Drift threshold: {config.monitoring.drift_threshold}")
    print(f"   Performance drop: {config.monitoring.performance_drop_threshold}")
    print(f"   Min improvement: {config.monitoring.min_improvement_for_deploy}")
    
    print(f"\n🌐 API:")
    print(f"   Host: {config.api.host}")
    print(f"   Port: {config.api.port}")
    
    print(f"\n🗄️  Base de Datos:")
    print(f"   Tabla: {config.database.table_name}")
    print(f"   Auto-update: {config.database.auto_update}")
    
    print("=" * 80)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    import yaml
    from pathlib import Path
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 Testing validación de configuración\n")
    
    # Cargar config.yaml
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        try:
            config = validate_config(config_dict)
            print_config_summary(config)
            print("\n✅ Todas las validaciones pasaron correctamente")
        except ValidationError:
            print("\n❌ Errores en la configuración")
    else:
        print(f"❌ No se encontró {config_path}")

