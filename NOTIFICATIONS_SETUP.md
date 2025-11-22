# 🔔 Configuración de Notificaciones

## 📋 Resumen

El sistema MLOps ahora incluye notificaciones automáticas para:
- ✅ Entrenamiento iniciado/completado/fallido
- ✅ Deploy exitoso/fallido  
- ⚠️  Drift detectado
- ❌ Errores en API
- 📊 Resultados de CI/CD Pipeline

## 🎯 Canales Soportados

### 1. Slack (Recomendado)
- Notificaciones ricas con colores y campos
- Menciones a equipos
- Threading de conversaciones

### 2. Email  
- Alertas directas a inbox
- Formato HTML/texto plano
- Múltiples destinatarios

### 3. Discord
- Embeds con colores
- Bots personalizados
- Integración con servidores

### 4. Telegram
- Mensajes instantáneos
- Bot personal o grupal
- Markdown formatting

---

## 🔧 Configuración

### Paso 1: Obtener Webhooks/Tokens

#### Slack
1. Ve a https://api.slack.com/messaging/webhooks
2. Crea una Incoming Webhook
3. Selecciona el canal (#mlops recomendado)
4. Copia el Webhook URL

```
https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX
```

#### Discord
1. Ve a tu servidor → Configuración del servidor
2. Integraciones → Webhooks
3. Nuevo Webhook
4. Selecciona canal y copia URL

```
https://discord.com/api/webhooks/123456789/XXXX-XXXX
```

#### Telegram
1. Habla con @BotFather en Telegram
2. Crea un nuevo bot: `/newbot`
3. Copia el Token
4. Agrega el bot a tu grupo
5. Obtén Chat ID enviando mensaje y visitando:
```
https://api.telegram.org/bot<TOKEN>/getUpdates
```

#### Email (Gmail ejemplo)
1. Habilita verificación en 2 pasos
2. Ve a https://myaccount.google.com/apppasswords
3. Genera una "App Password"
4. Usa esa contraseña (no tu contraseña normal)

---

### Paso 2: Configurar Variables de Entorno

#### Para Airflow (local)

Edita `.env` en el directorio de Airflow:

```bash
# Slack
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
SLACK_CHANNEL=#mlops

# Email
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=tu-email@gmail.com
SMTP_PASSWORD=tu-app-password
EMAIL_FROM=MLOps Bot <tu-email@gmail.com>
EMAIL_TO=team@empresa.com,admin@empresa.com

# Discord
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

# Telegram
TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=-1001234567890
```

#### Para Render.com (API)

1. Ve a Render Dashboard → Tu servicio
2. Environment
3. Agrega las mismas variables

#### Para GitHub Actions

1. Ve a tu repo → Settings → Secrets and variables → Actions
2. New repository secret
3. Agrega:
   - `SLACK_WEBHOOK_URL`
   - `DISCORD_WEBHOOK_URL`
   - `TELEGRAM_BOT_TOKEN`
   - `TELEGRAM_CHAT_ID`

---

### Paso 3: Verificar Configuración

```bash
# Test local
cd /path/to/proyecto
python utils/notifications.py

# Test desde Airflow
python -c "
from utils.notifications import send_notification, NotificationLevel
send_notification(
    message='Probando notificaciones desde Airflow',
    level=NotificationLevel.SUCCESS,
    title='Test Exitoso'
)
"
```

---

## 📊 Tipos de Notificaciones

### Entrenamiento

**Inicio:**
```
🚀 Entrenamiento Iniciado
Modelo: XGBoost
Razón: drift detectado
```

**Completado:**
```
🎉 Entrenamiento Completado
Modelo: XGBoost
F1 Score: 0.9234
Mejora: +0.0123
```

**Fallido:**
```
❌ Entrenamiento Fallido
Error: Out of memory
```

### Deploy

**Exitoso:**
```
✅ Modelo Desplegado en Producción
Modelo: XGBoost
F1 Score: 0.9234
```

**Fallido:**
```
❌ Deploy Fallido
Error: API no responde
```

### Drift

```
⚠️ Drift Detectado
Tipo: data
Score: 0.7234
Acción: Reentrenamiento programado
```

### CI/CD

```
✅ CI/CD Pipeline SUCCESS
Branch: main
Commit: a1b2c3d
Validate: success
Deploy: success
```

---

## 🎨 Personalización

### Cambiar Emojis

Edita `utils/notifications.py`:

```python
emoji_map = {
    NotificationLevel.INFO: ":custom_emoji:",
    NotificationLevel.SUCCESS: ":party_popper:"
}
```

### Agregar Campos Personalizados

```python
from utils.notifications import send_notification, NotificationLevel

send_notification(
    message="Entrenamiento completado",
    level=NotificationLevel.SUCCESS,
    title="Gran noticia",
    fields={
        "Modelo": "XGBoost",
        "Accuracy": "95.6%",
        "Tiempo": "2h 15min"
    }
)
```

### Notificaciones Condicionales

```python
# Solo notificar si mejora > 1%
if improvement > 0.01:
    notify_training_completed(model_name, f1_score, improvement)
```

---

## 🚨 Niveles de Alerta

| Nivel | Uso | Color | Emoji |
|-------|-----|-------|-------|
| `INFO` | Informativo | Azul | ℹ️ |
| `SUCCESS` | Operación exitosa | Verde | ✅ |
| `WARNING` | Atención requerida | Naranja | ⚠️ |
| `ERROR` | Error recuperable | Rojo | ❌ |
| `CRITICAL` | Error crítico | Rojo oscuro | 🚨 |

---

## 🧪 Pruebas

### Test Rápido

```bash
# Slack
curl -X POST $SLACK_WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{"text":"Test desde terminal"}'

# Discord
curl -X POST $DISCORD_WEBHOOK_URL \
  -H "Content-Type: application/json" \
  -d '{"content":"Test desde terminal"}'

# Telegram
curl "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage?chat_id=$TELEGRAM_CHAT_ID&text=Test"
```

### Test Desde Python

```python
from utils.notifications import (
    notify_training_completed,
    notify_deploy_completed,
    notify_drift_detected,
    send_notification,
    NotificationLevel
)

# Test diferentes tipos
notify_training_completed("XGBoost", 0.92, 0.01)
notify_deploy_completed("XGBoost", 0.92)
notify_drift_detected(0.75, "data")
send_notification("Test custom", NotificationLevel.WARNING)
```

---

## 🔧 Troubleshooting

### No recibo notificaciones

1. ✅ Verifica que las variables de entorno estén configuradas:
```bash
echo $SLACK_WEBHOOK_URL
```

2. ✅ Verifica conectividad de red:
```bash
curl -I https://hooks.slack.com
```

3. ✅ Verifica logs de la aplicación:
```bash
grep "notificación" logs/airflow.log
```

4. ✅ Test directo con curl (ver arriba)

### Webhooks expirados

- **Slack**: Regenera webhook en api.slack.com
- **Discord**: Crea nuevo webhook
- **Telegram**: Token nunca expira (excepto si revocas)

### Rate Limiting

- **Slack**: 1 mensaje/segundo
- **Discord**: 5 mensajes/5 segundos por canal
- **Telegram**: 30 mensajes/segundo

Si excedes, las notificaciones se ignoran. Usa `time.sleep()` entre envíos.

---

## 📈 Mejores Prácticas

### 1. Diferentes canales para diferentes niveles

```python
# INFO/SUCCESS → Slack (no molesta)
# ERROR/CRITICAL → Email + Slack (asegura que se vea)
# WARNING → Slack (para monitoreo)
```

### 2. Agrupar notificaciones

En lugar de enviar 100 notificaciones para 100 tickets predichos, envía un resumen:

```python
# ❌ Malo
for ticket in tickets:
    notify(f"Ticket {ticket.id} procesado")

# ✅ Bueno
notify(f"Procesados {len(tickets)} tickets exitosamente")
```

### 3. Rate limiting manual

```python
import time

for event in events:
    notify(event)
    time.sleep(1)  # Evitar rate limiting
```

### 4. Priorizar según impacto

```python
# Deploy fallido → CRITICAL (email + slack + telegram)
if deploy_failed:
    send_notification(..., level=NotificationLevel.CRITICAL)

# Drift detectado → WARNING (solo slack)
if drift_detected:
    send_notification(..., level=NotificationLevel.WARNING)
```

---

## ✅ Checklist de Configuración

- [ ] Webhooks/tokens obtenidos
- [ ] Variables de entorno configuradas en Airflow
- [ ] Variables de entorno configuradas en Render
- [ ] Secrets configurados en GitHub Actions
- [ ] Test de notificación enviado y recibido
- [ ] Canales de Slack/Discord configurados
- [ ] Equipo notificado sobre nuevo sistema de alertas

---

## 🆘 Soporte

Si tienes problemas:

1. Revisa logs: `grep -i notification logs/*.log`
2. Verifica variables: `env | grep -E "(SLACK|DISCORD|TELEGRAM|SMTP)"`
3. Test manual con curl
4. Revisa firewall/proxy

---

## 📚 Referencias

- [Slack Webhooks](https://api.slack.com/messaging/webhooks)
- [Discord Webhooks](https://discord.com/developers/docs/resources/webhook)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Gmail App Passwords](https://support.google.com/accounts/answer/185833)

---

**Estado:** ✅ Production Ready  
**Última actualización:** 2024-11-22

