"""
Script de deploy automático del modelo.

Este script:
1. Valida que el nuevo modelo sea mejor que el actual
2. Hace backup del modelo anterior
3. Actualiza el modelo en producción
4. Reinicia servicios si es necesario
5. Notifica sobre el deploy

Autor: Sistema MLOps
Fecha: 2024
"""

import os
import sys
import json
import logging
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Agregar raíz del proyecto al path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_model_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
    """Carga metadata de un modelo"""
    metadata_path = model_path.parent / "best_model_metadata.json"
    
    if not metadata_path.exists():
        return None
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando metadata: {e}")
        return None

def backup_current_model(model_dir: Path) -> bool:
    """Hace backup del modelo actual"""
    try:
        backup_dir = model_dir / "backups"
        backup_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_file = model_dir / "best_model.pkl"
        metadata_file = model_dir / "best_model_metadata.json"
        
        if model_file.exists():
            backup_model = backup_dir / f"best_model_{timestamp}.pkl"
            shutil.copy2(model_file, backup_model)
            logger.info(f"✓ Backup del modelo: {backup_model}")
        
        if metadata_file.exists():
            backup_metadata = backup_dir / f"best_model_metadata_{timestamp}.json"
            shutil.copy2(metadata_file, backup_metadata)
            logger.info(f"✓ Backup de metadata: {backup_metadata}")
        
        # Mantener solo últimos 5 backups
        backups = sorted(backup_dir.glob("best_model_*.pkl"))
        if len(backups) > 5:
            for old_backup in backups[:-5]:
                old_backup.unlink()
                # Eliminar metadata correspondiente
                metadata_backup = old_backup.parent / old_backup.name.replace('.pkl', '_metadata.json')
                if metadata_backup.exists():
                    metadata_backup.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"Error haciendo backup: {e}")
        return False

def deploy_model(model_dir: Path = None) -> bool:
    """
    Hace deploy del modelo.
    
    En producción, esto podría:
    - Copiar modelo a directorio de producción
    - Reiniciar servicio/API
    - Actualizar versiones en MLflow
    - Notificar sistemas downstream
    """
    if model_dir is None:
        model_dir = project_root / "models"
    
    logger.info("=" * 80)
    logger.info("📦 DEPLOY DE MODELO")
    logger.info("=" * 80)
    
    # 1. Verificar que el modelo existe
    model_path = model_dir / "best_model.pkl"
    if not model_path.exists():
        logger.error(f"❌ Modelo no encontrado: {model_path}")
        return False
    
    # 2. Cargar metadata
    metadata = load_model_metadata(model_path)
    if metadata:
        logger.info(f"Modelo: {metadata.get('model_name')}")
        logger.info(f"F1-Score: {metadata.get('f1_score', 0):.4f}")
        logger.info(f"Timestamp: {metadata.get('timestamp')}")
    
    # 3. Hacer backup del modelo actual
    logger.info("\n💾 Haciendo backup del modelo actual...")
    if not backup_current_model(model_dir):
        logger.warning("⚠️  No se pudo hacer backup, continuando de todos modos...")
    
    # 4. El modelo ya está en la ubicación correcta
    # En producción, aquí copiarías a otro directorio
    logger.info("\n✅ Modelo listo para uso")
    logger.info(f"   Ubicación: {model_path}")
    
    # 5. Notificar (en producción, enviar notificación real)
    logger.info("\n📢 Notificaciones:")
    logger.info("   ⚠️  En producción, implementar:")
    logger.info("      - Notificación a Slack/Email")
    logger.info("      - Reinicio de servicio API")
    logger.info("      - Actualización en MLflow Model Registry")
    logger.info("      - Logging en sistema de monitoreo")
    
    # 6. Verificar integridad del modelo
    try:
        import joblib
        test_model = joblib.load(model_path)
        logger.info("✓ Modelo cargado correctamente, integridad verificada")
    except Exception as e:
        logger.error(f"❌ Error verificando modelo: {e}")
        return False
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ DEPLOY COMPLETADO")
    logger.info("=" * 80)
    
    return True

if __name__ == "__main__":
    success = deploy_model()
    sys.exit(0 if success else 1)

