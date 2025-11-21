import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import requests
import time

import pandas as pd
import numpy as np
import joblib

# Agregar raíz del proyecto al path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.monitoring import PredictionLogger, DriftDetector
from utils.preprocessing_data import load_config

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuración
API_URL = os.getenv("API_URL", "http://localhost:8000")
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))  # Score de drift para trigger
MIN_PREDICTIONS_FOR_DRIFT = int(os.getenv("MIN_PREDICTIONS_FOR_DRIFT", "100"))
PERFORMANCE_DROP_THRESHOLD = float(os.getenv("PERFORMANCE_DROP_THRESHOLD", "0.05"))  # 5% de caída

# ============================================================================
# FUNCIONES DE MONITOREO
# ============================================================================

def check_api_health() -> bool:
    """Verifica que la API esté funcionando"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"API no disponible: {e}")
        return False

def check_drift() -> Dict[str, Any]:
    """Verifica drift usando el endpoint de la API"""
    try:
        response = requests.get(f"{API_URL}/monitoring/drift", timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error en check de drift: {response.status_code}")
            return {"drift_detected": False, "error": True}
    except Exception as e:
        logger.error(f"Error verificando drift: {e}")
        return {"drift_detected": False, "error": True}

def get_recent_metrics() -> Dict[str, Any]:
    """Obtiene métricas recientes de la API"""
    try:
        response = requests.get(f"{API_URL}/monitoring/metrics", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {}
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        return {}

def evaluate_model_performance() -> Optional[Dict[str, Any]]:
    """
    Evalúa el modelo actual usando datos recientes con labels.
    Retorna None si no hay suficientes datos etiquetados.
    """
    try:
        # Obtener predicciones recientes con labels verdaderos
        logger_path = project_root / "monitoring" / "logs" / "predictions.jsonl"
        
        if not logger_path.exists():
            return None
        
        # Leer predicciones recientes (últimas 48 horas)
        cutoff_time = datetime.now() - timedelta(hours=48)
        predictions = []
        
        with open(logger_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    if entry.get('true_label') and entry.get('timestamp'):
                        entry_time = datetime.fromisoformat(entry['timestamp'])
                        if entry_time >= cutoff_time:
                            predictions.append(entry)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON inválido en línea {line_number}: {e}")
                    continue
                except (KeyError, ValueError) as e:
                    logger.warning(f"Datos inválidos en línea {line_number}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error inesperado en línea {line_number}: {e}")
                    continue
        
        if len(predictions) < 50:  # Mínimo de predicciones etiquetadas
            logger.info(f"Solo {len(predictions)} predicciones etiquetadas, necesitamos al menos 50")
            return None
        
        # Calcular métricas
        correct = sum(1 for p in predictions if p.get('correct') == True)
        total = len(predictions)
        accuracy = correct / total if total > 0 else 0.0
        
        # Calcular F1 por clase
        from sklearn.metrics import f1_score, precision_score, recall_score
        
        y_true = [p['true_label'] for p in predictions]
        y_pred = [p['prediction'] for p in predictions]
        
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'total_predictions': total,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error evaluando modelo: {e}")
        return None

def compare_models(current_metrics: Dict[str, Any], new_metrics: Dict[str, Any]) -> bool:
    """
    Compara dos modelos y determina si el nuevo es mejor.
    
    Args:
        current_metrics: Métricas del modelo actual
        new_metrics: Métricas del nuevo modelo
        
    Returns:
        True si el nuevo modelo es mejor
    """
    # Comparar principalmente por F1-score
    current_f1 = current_metrics.get('f1_score', 0.0)
    new_f1 = new_metrics.get('f1_score', 0.0)
    
    # El nuevo modelo debe ser al menos 1% mejor
    improvement = new_f1 - current_f1
    
    if improvement > 0.01:
        logger.info(f"✅ Nuevo modelo es mejor: F1 {current_f1:.4f} → {new_f1:.4f} (+{improvement:.4f})")
        return True
    elif improvement < -0.01:
        logger.info(f"❌ Nuevo modelo es peor: F1 {current_f1:.4f} → {new_f1:.4f} ({improvement:.4f})")
        return False
    else:
        # Si son similares, comparar por accuracy
        current_acc = current_metrics.get('accuracy', 0.0)
        new_acc = new_metrics.get('accuracy', 0.0)
        
        if new_acc > current_acc + 0.005:  # 0.5% mejor
            logger.info(f"✅ Nuevo modelo ligeramente mejor en accuracy")
            return True
        else:
            logger.info(f"⚠️ Modelos similares, manteniendo el actual")
            return False

# ============================================================================
# FUNCIONES DE REENTRENAMIENTO Y DEPLOY
# ============================================================================

def trigger_retraining() -> bool:
    """
    Dispara el proceso de reentrenamiento.
    
    Returns:
        True si el entrenamiento fue exitoso
    """
    logger.info("🚀 Disparando reentrenamiento...")
    
    try:
        # Ejecutar script de entrenamiento
        train_script = project_root / "scripts" / "train_model.py"
        
        result = subprocess.run(
            [sys.executable, str(train_script)],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hora máximo
        )
        
        if result.returncode == 0:
            logger.info("✅ Reentrenamiento completado exitosamente")
            return True
        else:
            logger.error(f"❌ Error en reentrenamiento:")
            logger.error(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Reentrenamiento excedió tiempo límite")
        return False
    except Exception as e:
        logger.error(f"❌ Error disparando reentrenamiento: {e}")
        return False

def load_model_metrics(model_path: Path) -> Optional[Dict[str, Any]]:
    """Carga métricas de un modelo desde su metadata"""
    metadata_path = model_path.parent / "best_model_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return {
            'f1_score': metadata.get('f1_score', 0.0),
            'accuracy': metadata.get('all_results', {}).get(metadata.get('model_name', ''), {}).get('accuracy', 0.0),
            'model_name': metadata.get('model_name'),
            'timestamp': metadata.get('timestamp')
        }
    except Exception as e:
        logger.error(f"Error cargando métricas: {e}")
        return None

def deploy_model() -> bool:
    """
    Hace deploy del nuevo modelo.
    En producción, esto podría:
    - Reiniciar la API con el nuevo modelo
    - Actualizar un servicio de ML
    - Notificar a sistemas downstream
    
    Por ahora, solo verifica que el modelo existe.
    """
    logger.info("📦 Haciendo deploy del nuevo modelo...")
    
    model_path = project_root / "models" / "best_model.pkl"
    
    if not model_path.exists():
        logger.error("❌ Modelo no encontrado para deploy")
        return False
    
    # En producción, aquí harías:
    # - Copiar modelo a directorio de producción
    # - Reiniciar servicio/API
    # - Notificar sistemas downstream
    # - Actualizar versiones en MLflow
    
    logger.info("✅ Modelo listo para deploy")
    logger.info("⚠️  Nota: En producción, implementar lógica de deploy real")
    
    return True

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de monitoreo y reentrenamiento"""
    
    logger.info("=" * 80)
    logger.info("🔍 SISTEMA DE MONITOREO Y REENTRENAMIENTO AUTOMÁTICO")
    logger.info("=" * 80)
    
    # 1. Verificar salud de la API
    logger.info("\n📡 Verificando API...")
    if not check_api_health():
        logger.error("❌ API no está disponible. Abortando.")
        return 1
    
    logger.info("✅ API está funcionando")
    
    # 2. Verificar drift
    logger.info("\n🔍 Verificando drift...")
    drift_results = check_drift()
    
    if drift_results.get('error'):
        logger.warning("⚠️  No se pudo verificar drift, continuando...")
        drift_detected = False
    else:
        drift_detected = drift_results.get('drift_detected', False)
        drift_score = drift_results.get('drift_score', 0.0)
        
        logger.info(f"Drift detectado: {drift_detected}")
        logger.info(f"Drift score: {drift_score:.4f}")
        
        if drift_detected:
            logger.warning("⚠️  DRIFT DETECTADO - Considerando reentrenamiento")
    
    # 3. Evaluar performance actual
    logger.info("\n📊 Evaluando performance del modelo actual...")
    current_performance = evaluate_model_performance()
    
    if current_performance:
        logger.info(f"Performance actual:")
        logger.info(f"  - Accuracy: {current_performance['accuracy']:.4f}")
        logger.info(f"  - F1-Score: {current_performance['f1_score']:.4f}")
        logger.info(f"  - Precision: {current_performance['precision']:.4f}")
        logger.info(f"  - Recall: {current_performance['recall']:.4f}")
    else:
        logger.info("⚠️  No hay suficientes datos etiquetados para evaluación")
        current_performance = None
    
    # 4. Decidir si reentrenar
    should_retrain = False
    retrain_reason = []
    
    if drift_detected and drift_results.get('drift_score', 0) > DRIFT_THRESHOLD:
        should_retrain = True
        retrain_reason.append(f"Data drift detectado (score: {drift_results.get('drift_score', 0):.4f})")
    
    if current_performance:
        # Cargar métricas del modelo entrenado
        model_metadata_path = project_root / "models" / "best_model_metadata.json"
        if model_metadata_path.exists():
            with open(model_metadata_path, 'r', encoding='utf-8') as f:
                model_metadata = json.load(f)
            
            trained_f1 = model_metadata.get('f1_score', 1.0)
            current_f1 = current_performance['f1_score']
            performance_drop = trained_f1 - current_f1
            
            if performance_drop > PERFORMANCE_DROP_THRESHOLD:
                should_retrain = True
                retrain_reason.append(
                    f"Performance degradada: F1 {trained_f1:.4f} → {current_f1:.4f} "
                    f"(drop: {performance_drop:.4f})"
                )
    
    # 5. Reentrenar si es necesario
    if should_retrain:
        logger.info("\n" + "=" * 80)
        logger.info("🔄 REENTRENAMIENTO NECESARIO")
        logger.info("=" * 80)
        logger.info("Razones:")
        for reason in retrain_reason:
            logger.info(f"  - {reason}")
        
        # Guardar métricas del modelo actual antes de reentrenar
        current_model_metrics = None
        if model_metadata_path.exists():
            current_model_metrics = load_model_metrics(project_root / "models" / "best_model.pkl")
        
        # Reentrenar
        retrain_success = trigger_retraining()
        
        if retrain_success:
            # Cargar métricas del nuevo modelo
            new_model_metrics = load_model_metrics(project_root / "models" / "best_model.pkl")
            
            if new_model_metrics and current_model_metrics:
                # Comparar modelos
                if compare_models(current_model_metrics, new_model_metrics):
                    # Hacer deploy del nuevo modelo
                    deploy_success = deploy_model()
                    
                    if deploy_success:
                        logger.info("\n✅ Nuevo modelo desplegado exitosamente")
                        logger.info("⚠️  Nota: Reiniciar API manualmente o implementar auto-restart")
                    else:
                        logger.error("\n❌ Error en deploy")
                else:
                    logger.info("\n⚠️  Nuevo modelo no es mejor, manteniendo el actual")
            else:
                logger.warning("\n⚠️  No se pudieron comparar modelos, haciendo deploy de todos modos")
                deploy_model()
        else:
            logger.error("\n❌ Reentrenamiento falló")
            return 1
    else:
        logger.info("\n✅ No se requiere reentrenamiento en este momento")
        logger.info("   El modelo está funcionando correctamente")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ MONITOREO COMPLETADO")
    logger.info("=" * 80)
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

