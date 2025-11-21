"""
Script para descargar modelo desde S3 usando DVC.

Este script se ejecuta durante el build en Render para descargar
el modelo que no está trackeado en Git.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dvc_installed():
    """Verifica si DVC está instalado"""
    try:
        result = subprocess.run(
            ["dvc", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.info(f"✓ DVC instalado: {result.stdout.strip()}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return False

def install_dvc():
    """Instala DVC y dvc-s3"""
    logger.info("📦 Instalando DVC y dvc-s3...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "dvc", "dvc-s3"],
            check=True,
            timeout=120
        )
        logger.info("✓ DVC instalado exitosamente")
        return True
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
        logger.error(f"❌ Error instalando DVC: {e}")
        return False

def check_aws_credentials():
    """Verifica que las credenciales AWS estén configuradas"""
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if not access_key or not secret_key:
        logger.warning("⚠️ AWS_ACCESS_KEY_ID o AWS_SECRET_ACCESS_KEY no configuradas")
        return False
    
    logger.info("✓ Credenciales AWS configuradas")
    return True

def download_model():
    """Descarga el modelo desde S3 usando DVC"""
    model_path = Path("models/best_model.pkl")
    dvc_file = Path("models/best_model.pkl.dvc")
    
    # Verificar si el modelo ya existe
    if model_path.exists():
        logger.info(f"✓ Modelo ya existe: {model_path}")
        logger.info(f"  Tamaño: {model_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    
    # Verificar que el archivo .dvc existe
    if not dvc_file.exists():
        logger.error(f"❌ Archivo DVC no encontrado: {dvc_file}")
        logger.error("  El archivo models/best_model.pkl.dvc debe estar en el repositorio")
        return False
    
    # Verificar credenciales AWS
    if not check_aws_credentials():
        logger.error("❌ No se pueden descargar modelos sin credenciales AWS")
        return False
    
    # Instalar DVC si no está instalado
    if not check_dvc_installed():
        if not install_dvc():
            logger.error("❌ No se pudo instalar DVC")
            return False
    
    # Descargar modelo
    logger.info("📥 Descargando modelo desde S3...")
    try:
        result = subprocess.run(
            ["dvc", "pull", str(dvc_file)],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutos máximo
            env=os.environ.copy()
        )
        
        if result.returncode == 0:
            if model_path.exists():
                logger.info(f"✅ Modelo descargado exitosamente: {model_path}")
                logger.info(f"  Tamaño: {model_path.stat().st_size / (1024*1024):.2f} MB")
                return True
            else:
                logger.warning("⚠️ DVC pull exitoso pero modelo no encontrado")
                return False
        else:
            logger.error(f"❌ Error en dvc pull:")
            logger.error(f"  stdout: {result.stdout}")
            logger.error(f"  stderr: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Timeout descargando modelo (más de 5 minutos)")
        return False
    except Exception as e:
        logger.error(f"❌ Error inesperado: {e}")
        return False

def main():
    """Función principal"""
    logger.info("=" * 60)
    logger.info("📥 DESCARGA DE MODELO DESDE S3")
    logger.info("=" * 60)
    
    # Crear directorio models si no existe
    Path("models").mkdir(exist_ok=True)
    
    success = download_model()
    
    if success:
        logger.info("=" * 60)
        logger.info("✅ PROCESO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 60)
        return 0
    else:
        logger.warning("=" * 60)
        logger.warning("⚠️ NO SE PUDO DESCARGAR EL MODELO")
        logger.warning("  La API funcionará pero requerirá entrenar un modelo primero")
        logger.warning("=" * 60)
        # No fallar el build, solo advertir
        return 0

if __name__ == "__main__":
    sys.exit(main())












