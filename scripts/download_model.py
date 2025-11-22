"""
Script para descargar modelo desde S3 usando DVC o boto3 como fallback.

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

def check_dvc_config():
    """Verifica la configuración de DVC"""
    dvc_config = Path(".dvc/config")
    if not dvc_config.exists():
        logger.warning("⚠️ Archivo .dvc/config no encontrado")
        return False
    
    logger.info("✓ Archivo .dvc/config encontrado")
    
    # Intentar verificar que DVC puede leer la configuración
    try:
        result = subprocess.run(
            ["dvc", "remote", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            env=os.environ.copy()
        )
        if result.returncode == 0:
            logger.info(f"✓ DVC puede leer configuración. Remotes: {result.stdout.strip()}")
        else:
            logger.warning(f"⚠️ DVC no pudo listar remotes: {result.stderr}")
    except Exception as e:
        logger.warning(f"⚠️ No se pudo verificar configuración de DVC: {e}")
    
    return True

def download_with_dvc():
    """Intenta descargar el modelo usando DVC"""
    model_path = Path("models/best_model.pkl")
    dvc_file = Path("models/best_model.pkl.dvc")
    
    # Verificar que el archivo .dvc existe
    if not dvc_file.exists():
        logger.error(f"❌ Archivo DVC no encontrado: {dvc_file}")
        return False
    
    # Verificar configuración de DVC
    if not check_dvc_config():
        logger.warning("⚠️ Configuración de DVC no encontrada")
        return False
    
    # Instalar DVC si no está instalado
    if not check_dvc_installed():
        if not install_dvc():
            logger.warning("⚠️ No se pudo instalar DVC")
            return False
    
    logger.info("📥 Intentando descargar modelo con DVC...")
    
    # Intentar diferentes formas de ejecutar dvc pull
    commands_to_try = [
        # Forma 1: dvc pull con el archivo específico
        ["dvc", "pull", str(dvc_file)],
        # Forma 2: dvc pull sin especificar archivo (descarga todo)
        ["dvc", "pull"],
        # Forma 3: dvc pull con ruta relativa
        ["dvc", "pull", "models/best_model.pkl.dvc"],
    ]
    
    for cmd in commands_to_try:
        try:
            logger.info(f"  Ejecutando: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutos máximo
                env=os.environ.copy()
            )
            
            logger.info(f"  Return code: {result.returncode}")
            if result.stdout:
                logger.info(f"  stdout: {result.stdout[:500]}")  # Primeros 500 chars
            if result.stderr:
                logger.warning(f"  stderr: {result.stderr[:500]}")  # Primeros 500 chars
            
            if result.returncode == 0:
                if model_path.exists():
                    logger.info(f"✅ Modelo descargado exitosamente con DVC: {model_path}")
                    logger.info(f"  Tamaño: {model_path.stat().st_size / (1024*1024):.2f} MB")
                    return True
                else:
                    logger.warning("⚠️ DVC pull exitoso pero modelo no encontrado")
                    continue  # Intentar siguiente comando
            else:
                logger.warning(f"⚠️ Comando falló: {' '.join(cmd)}")
                continue  # Intentar siguiente comando
                
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ Timeout ejecutando: {' '.join(cmd)}")
            continue
        except Exception as e:
            logger.warning(f"⚠️ Error ejecutando {' '.join(cmd)}: {e}")
            continue
    
    logger.warning("⚠️ Todos los intentos con DVC fallaron")
    return False

def download_with_boto3():
    """Descarga el modelo usando boto3 directamente (más robusto)"""
    model_path = Path("models/best_model.pkl")
    
    logger.info("📥 Intentando descargar modelo con boto3...")
    
    # Verificar credenciales AWS
    if not check_aws_credentials():
        logger.error("❌ No se pueden descargar modelos sin credenciales AWS")
        return False
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Leer el hash MD5 del archivo .dvc
        dvc_file = Path("models/best_model.pkl.dvc")
        if not dvc_file.exists():
            logger.error("❌ Archivo .dvc no encontrado para obtener hash MD5")
            return False
        
        # Parsear el archivo .dvc para obtener el hash MD5
        md5_hash = None
        try:
            import yaml
            with open(dvc_file, 'r') as f:
                dvc_data = yaml.safe_load(f)
                if 'outs' in dvc_data and len(dvc_data['outs']) > 0:
                    md5_hash = dvc_data['outs'][0].get('md5')
        except Exception as e:
            logger.warning(f"⚠️ No se pudo leer hash MD5 del archivo .dvc: {e}")
            # Usar hash conocido del código
            md5_hash = "204e6d0e36a8d6658f0bde0039761267"
        
        if not md5_hash:
            logger.error("❌ No se pudo obtener hash MD5 del modelo")
            return False
        
        # Configuración S3 (basada en .dvc/config)
        bucket = "ticketsfidudavivienda"
        # DVC guarda archivos usando MD5: primeros 2 chars como directorio
        s3_key = f"dvc-storage/models/{md5_hash[:2]}/{md5_hash[2:]}"
        
        logger.info(f"  Bucket: {bucket}")
        logger.info(f"  S3 Key: {s3_key}")
        
        # Crear cliente S3
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        )
        
        # Crear directorio models si no existe
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Descargar archivo
        logger.info("  Descargando archivo desde S3...")
        s3_client.download_file(bucket, s3_key, str(model_path))
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"✅ Modelo descargado exitosamente con boto3: {model_path} ({size_mb:.2f} MB)")
            return True
        else:
            logger.error("❌ Descarga completa pero archivo no encontrado")
            return False
            
    except ImportError:
        logger.error("❌ boto3 no instalado. Agregalo a requirements.txt")
        return False
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"❌ Error de AWS S3 ({error_code}): {e}")
        if error_code == 'NoSuchKey':
            logger.error(f"  El archivo no existe en S3: s3://{bucket}/{s3_key}")
        elif error_code == 'AccessDenied':
            logger.error("  Verifica que las credenciales AWS tengan permisos de lectura en S3")
        return False
    except Exception as e:
        logger.error(f"❌ Error descargando modelo con boto3: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def download_model():
    """Descarga el modelo desde S3 usando DVC o boto3 como fallback"""
    model_path = Path("models/best_model.pkl")
    
    # Verificar si el modelo ya existe
    if model_path.exists():
        logger.info(f"✓ Modelo ya existe: {model_path}")
        logger.info(f"  Tamaño: {model_path.stat().st_size / (1024*1024):.2f} MB")
        return True
    
    # Verificar credenciales AWS
    if not check_aws_credentials():
        logger.error("❌ No se pueden descargar modelos sin credenciales AWS")
        return False
    
    # Intentar primero con DVC
    logger.info("=" * 60)
    logger.info("MÉTODO 1: Intentando descargar con DVC...")
    logger.info("=" * 60)
    if download_with_dvc():
        return True
    
    # Si DVC falla, intentar con boto3
    logger.info("=" * 60)
    logger.info("MÉTODO 2: Intentando descargar con boto3 (fallback)...")
    logger.info("=" * 60)
    if download_with_boto3():
        return True
    
    # Si ambos fallan
    logger.error("=" * 60)
    logger.error("❌ Todos los métodos de descarga fallaron")
    logger.error("=" * 60)
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












