"""
Módulo de notificaciones para CI/CD y alertas del sistema.

Soporta múltiples canales:
- Slack (webhook)
- Email (SMTP)
- Discord (webhook)
- Telegram (bot)

Autor: Sistema MLOps
Fecha: 2024
"""

import os
import logging
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class NotificationLevel(Enum):
    """Niveles de notificación"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    SUCCESS = "success"


class NotificationChannel(Enum):
    """Canales de notificación disponibles"""
    SLACK = "slack"
    EMAIL = "email"
    DISCORD = "discord"
    TELEGRAM = "telegram"


# ============================================================================
# CONFIGURACIÓN DESDE VARIABLES DE ENTORNO
# ============================================================================

# Slack
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "#ml ops")

# Email
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER)
EMAIL_TO = os.getenv("EMAIL_TO", "").split(",") if os.getenv("EMAIL_TO") else []

# Discord
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Telegram
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ============================================================================
# FUNCIONES DE NOTIFICACIÓN POR CANAL
# ============================================================================

def send_slack_notification(
    message: str,
    level: NotificationLevel = NotificationLevel.INFO,
    title: Optional[str] = None,
    fields: Optional[Dict[str, str]] = None
) -> bool:
    """
    Envía notificación a Slack.
    
    Args:
        message: Mensaje a enviar
        level: Nivel de la notificación
        title: Título opcional
        fields: Campos adicionales (key-value)
        
    Returns:
        True si se envió exitosamente
    """
    if not SLACK_WEBHOOK_URL:
        logger.debug("SLACK_WEBHOOK_URL no configurada, saltando notificación")
        return False
    
    # Emoji según nivel
    emoji_map = {
        NotificationLevel.INFO: ":information_source:",
        NotificationLevel.WARNING: ":warning:",
        NotificationLevel.ERROR: ":x:",
        NotificationLevel.CRITICAL: ":rotating_light:",
        NotificationLevel.SUCCESS: ":white_check_mark:"
    }
    
    # Color según nivel
    color_map = {
        NotificationLevel.INFO: "#36a64f",  # Verde
        NotificationLevel.WARNING: "#ff9900",  # Naranja
        NotificationLevel.ERROR: "#ff0000",  # Rojo
        NotificationLevel.CRITICAL: "#8B0000",  # Rojo oscuro
        NotificationLevel.SUCCESS: "#00ff00"  # Verde brillante
    }
    
    emoji = emoji_map.get(level, ":bell:")
    color = color_map.get(level, "#808080")
    
    # Construir payload
    payload = {
        "channel": SLACK_CHANNEL,
        "username": "MLOps Bot",
        "icon_emoji": ":robot_face:",
        "attachments": [
            {
                "color": color,
                "title": f"{emoji} {title or level.value.upper()}",
                "text": message,
                "footer": "MLOps Pipeline",
                "ts": int(datetime.now().timestamp())
            }
        ]
    }
    
    # Agregar campos si existen
    if fields:
        payload["attachments"][0]["fields"] = [
            {"title": k, "value": v, "short": True}
            for k, v in fields.items()
        ]
    
    try:
        response = requests.post(
            SLACK_WEBHOOK_URL,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            logger.info("✅ Notificación Slack enviada")
            return True
        else:
            logger.error(f"❌ Error enviando Slack: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error enviando notificación Slack: {e}")
        return False


def send_email_notification(
    subject: str,
    message: str,
    level: NotificationLevel = NotificationLevel.INFO,
    html: bool = False
) -> bool:
    """
    Envía notificación por email.
    
    Args:
        subject: Asunto del email
        message: Mensaje (texto plano o HTML)
        level: Nivel de la notificación
        html: Si el mensaje es HTML
        
    Returns:
        True si se envió exitosamente
    """
    if not SMTP_USER or not SMTP_PASSWORD or not EMAIL_TO:
        logger.debug("Credenciales SMTP no configuradas, saltando email")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_FROM
        msg['To'] = ', '.join(EMAIL_TO)
        msg['Subject'] = f"[MLOps {level.value.upper()}] {subject}"
        
        # Agregar timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"{message}\n\n---\nEnviado: {timestamp}"
        
        if html:
            msg.attach(MIMEText(full_message, 'html'))
        else:
            msg.attach(MIMEText(full_message, 'plain'))
        
        # Conectar y enviar
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        logger.info(f"✅ Email enviado a {', '.join(EMAIL_TO)}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error enviando email: {e}")
        return False


def send_discord_notification(
    message: str,
    level: NotificationLevel = NotificationLevel.INFO,
    title: Optional[str] = None
) -> bool:
    """
    Envía notificación a Discord.
    
    Args:
        message: Mensaje a enviar
        level: Nivel de la notificación
        title: Título opcional
        
    Returns:
        True si se envió exitosamente
    """
    if not DISCORD_WEBHOOK_URL:
        logger.debug("DISCORD_WEBHOOK_URL no configurada, saltando notificación")
        return False
    
    # Color según nivel (formato decimal)
    color_map = {
        NotificationLevel.INFO: 3447003,  # Azul
        NotificationLevel.WARNING: 16776960,  # Amarillo
        NotificationLevel.ERROR: 15158332,  # Rojo
        NotificationLevel.CRITICAL: 9109504,  # Rojo oscuro
        NotificationLevel.SUCCESS: 3066993  # Verde
    }
    
    color = color_map.get(level, 8421504)  # Gris por defecto
    
    payload = {
        "embeds": [
            {
                "title": title or f"MLOps {level.value.upper()}",
                "description": message,
                "color": color,
                "footer": {
                    "text": "MLOps Pipeline"
                },
                "timestamp": datetime.now().isoformat()
            }
        ]
    }
    
    try:
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 204:
            logger.info("✅ Notificación Discord enviada")
            return True
        else:
            logger.error(f"❌ Error enviando Discord: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error enviando notificación Discord: {e}")
        return False


def send_telegram_notification(
    message: str,
    level: NotificationLevel = NotificationLevel.INFO
) -> bool:
    """
    Envía notificación a Telegram.
    
    Args:
        message: Mensaje a enviar
        level: Nivel de la notificación
        
    Returns:
        True si se envió exitosamente
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Credenciales Telegram no configuradas, saltando notificación")
        return False
    
    # Emoji según nivel
    emoji_map = {
        NotificationLevel.INFO: "ℹ️",
        NotificationLevel.WARNING: "⚠️",
        NotificationLevel.ERROR: "❌",
        NotificationLevel.CRITICAL: "🚨",
        NotificationLevel.SUCCESS: "✅"
    }
    
    emoji = emoji_map.get(level, "🔔")
    full_message = f"{emoji} *{level.value.upper()}*\n\n{message}"
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": full_message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info("✅ Notificación Telegram enviada")
            return True
        else:
            logger.error(f"❌ Error enviando Telegram: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error enviando notificación Telegram: {e}")
        return False


# ============================================================================
# FUNCIÓN PRINCIPAL DE NOTIFICACIÓN
# ============================================================================

def send_notification(
    message: str,
    level: NotificationLevel = NotificationLevel.INFO,
    title: Optional[str] = None,
    channels: Optional[List[NotificationChannel]] = None,
    fields: Optional[Dict[str, str]] = None
) -> Dict[str, bool]:
    """
    Envía notificación a múltiples canales.
    
    Args:
        message: Mensaje a enviar
        level: Nivel de la notificación
        title: Título opcional
        channels: Canales específicos (si None, intenta todos los configurados)
        fields: Campos adicionales para Slack
        
    Returns:
        Dict con resultado por canal {canal: éxito}
    """
    results = {}
    
    # Si no se especifican canales, intentar todos los configurados
    if channels is None:
        channels = []
        if SLACK_WEBHOOK_URL:
            channels.append(NotificationChannel.SLACK)
        if SMTP_USER and EMAIL_TO:
            channels.append(NotificationChannel.EMAIL)
        if DISCORD_WEBHOOK_URL:
            channels.append(NotificationChannel.DISCORD)
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            channels.append(NotificationChannel.TELEGRAM)
    
    # Enviar a cada canal
    for channel in channels:
        if channel == NotificationChannel.SLACK:
            results['slack'] = send_slack_notification(message, level, title, fields)
        elif channel == NotificationChannel.EMAIL:
            results['email'] = send_email_notification(title or "Notificación MLOps", message, level)
        elif channel == NotificationChannel.DISCORD:
            results['discord'] = send_discord_notification(message, level, title)
        elif channel == NotificationChannel.TELEGRAM:
            results['telegram'] = send_telegram_notification(message, level)
    
    # Log resumen
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    logger.info(f"Notificaciones enviadas: {successful}/{total} exitosas")
    
    return results


# ============================================================================
# NOTIFICACIONES PRE-DEFINIDAS PARA EVENTOS COMUNES
# ============================================================================

def notify_training_started(model_name: str, reason: str = "programado"):
    """Notifica que el entrenamiento ha comenzado"""
    message = f"Iniciando entrenamiento de modelo\n\nModelo: {model_name}\nRazón: {reason}"
    return send_notification(
        message=message,
        level=NotificationLevel.INFO,
        title="🚀 Entrenamiento Iniciado",
        fields={"Modelo": model_name, "Razón": reason}
    )


def notify_training_completed(
    model_name: str,
    f1_score: float,
    improvement: float
):
    """Notifica que el entrenamiento finalizó exitosamente"""
    emoji = "🎉" if improvement > 0 else "📊"
    message = (
        f"{emoji} Entrenamiento completado exitosamente\n\n"
        f"Modelo: {model_name}\n"
        f"F1 Score: {f1_score:.4f}\n"
        f"Mejora: {improvement:+.4f}"
    )
    
    level = NotificationLevel.SUCCESS if improvement >= 0.01 else NotificationLevel.INFO
    
    return send_notification(
        message=message,
        level=level,
        title="Entrenamiento Completado",
        fields={
            "Modelo": model_name,
            "F1 Score": f"{f1_score:.4f}",
            "Mejora": f"{improvement:+.4f}"
        }
    )


def notify_training_failed(error: str):
    """Notifica que el entrenamiento falló"""
    message = f"❌ El entrenamiento ha fallado\n\nError: {error}"
    return send_notification(
        message=message,
        level=NotificationLevel.ERROR,
        title="Entrenamiento Fallido"
    )


def notify_deploy_completed(model_name: str, f1_score: float):
    """Notifica que el deploy finalizó"""
    message = (
        f"✅ Modelo desplegado en producción\n\n"
        f"Modelo: {model_name}\n"
        f"F1 Score: {f1_score:.4f}"
    )
    
    return send_notification(
        message=message,
        level=NotificationLevel.SUCCESS,
        title="Deploy Completado",
        fields={"Modelo": model_name, "F1 Score": f"{f1_score:.4f}"}
    )


def notify_drift_detected(drift_score: float, drift_type: str = "data"):
    """Notifica que se detectó drift"""
    message = (
        f"⚠️ Drift detectado en el modelo\n\n"
        f"Tipo: {drift_type}\n"
        f"Score: {drift_score:.4f}\n"
        f"Acción: Reentrenamiento programado"
    )
    
    return send_notification(
        message=message,
        level=NotificationLevel.WARNING,
        title="Drift Detectado",
        fields={"Tipo": drift_type, "Score": f"{drift_score:.4f}"}
    )


def notify_api_error(error: str, endpoint: str = "unknown"):
    """Notifica error en la API"""
    message = f"❌ Error en la API\n\nEndpoint: {endpoint}\nError: {error}"
    
    return send_notification(
        message=message,
        level=NotificationLevel.ERROR,
        title="Error en API"
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("TESTING NOTIFICACIONES")
    print("=" * 80)
    
    # Test notificación simple
    print("\n1. Test notificación de info:")
    send_notification(
        message="Este es un mensaje de prueba del sistema MLOps",
        level=NotificationLevel.INFO,
        title="Test Notificación"
    )
    
    # Test notificación de éxito
    print("\n2. Test notificación de éxito:")
    notify_training_completed("XGBoost", 0.9234, 0.0123)
    
    # Test notificación de error
    print("\n3. Test notificación de error:")
    notify_training_failed("Out of memory error")
    
    print("\n" + "=" * 80)
    print("✅ Testing completado")
    print("\nSi no recibiste notificaciones, verifica:")
    print("  - Variables de entorno configuradas (SLACK_WEBHOOK_URL, etc.)")
    print("  - Conectividad de red")
    print("  - URLs de webhooks válidas")

