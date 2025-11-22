# 🔧 Configuración de Variables de Entorno

## 📋 Guía Rápida

```bash
# 1. Copiar archivo de ejemplo
cp .env.example .env

# 2. Editar con tus credenciales
nano .env  # o usar tu editor favorito

# 3. Verificar que está en .gitignore
cat .gitignore | grep "\.env$"
```

---

## 📝 Variables Obligatorias

Estas variables **DEBEN** estar configuradas para que funcione:

### Supabase (Base de Datos)
```bash
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

**Cómo obtenerlas:**
1. Ve a [Supabase Dashboard](https://app.supabase.com)
2. Selecciona tu proyecto
3. Settings → API
4. Copia "Project URL" → `SUPABASE_URL`
5. Copia "anon public" key → `SUPABASE_KEY`

---

## 🔐 Variables de Seguridad (Recomendadas)

### API Keys
```bash
# Generar con: openssl rand -hex 32
API_KEY=tu-api-key-secreta-2024
ADMIN_API_KEY=tu-admin-api-key-ultra-secreta-2024
```

Si no configuras estas keys, la API funcionará en modo desarrollo (sin autenticación).

---

## ☁️ AWS S3 (Para DVC - Opcional)

```bash
AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
AWS_DEFAULT_REGION=us-east-1
```

**Solo necesario si usas DVC para versionar modelos en S3.**

---

## 🔔 Notificaciones (Opcional)

### Slack
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00/B00/XXX
SLACK_CHANNEL=#mlops
```

**Cómo configurar:**
1. Ve a https://api.slack.com/messaging/webhooks
2. Crea Incoming Webhook
3. Selecciona canal
4. Copia URL

### Email (Gmail)
```bash
SMTP_USER=tu-email@gmail.com
SMTP_PASSWORD=tu-app-password
EMAIL_TO=equipo@empresa.com,admin@empresa.com
```

**Cómo configurar:**
1. Habilita verificación 2 pasos en Google
2. Ve a https://myaccount.google.com/apppasswords
3. Genera "App Password"
4. Usa esa contraseña (NO tu contraseña normal)

---

## 🔄 Airflow (Desarrollo Local)

```bash
AIRFLOW_WWW_USER_USERNAME=admin
AIRFLOW_WWW_USER_PASSWORD=tu-password-segura-2024

# Generar Fernet key:
# python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
AIRFLOW_FERNET_KEY=tu-fernet-key-aqui

# Generar Secret key:
# openssl rand -hex 32
AIRFLOW_SECRET_KEY=tu-secret-key-aqui
```

---

## 🌍 Entornos

### Desarrollo Local

1. Copia `.env.example` a `.env`
2. Completa solo variables obligatorias
3. Ejecuta: `python api/inference.py`

### Producción (Render.com)

1. Ve a Render Dashboard → Tu servicio → Environment
2. Agrega variables una por una
3. **NO** uses archivos .env en producción

### CI/CD (GitHub Actions)

1. Ve a tu repo → Settings → Secrets and variables → Actions
2. New repository secret
3. Agrega:
   - `SLACK_WEBHOOK_URL`
   - `RENDER_API_KEY`
   - etc.

---

## ✅ Validación

### Check que .env existe y tiene variables

```bash
# Verificar archivo
ls -la .env

# Contar variables configuradas
grep -E "^[A-Z_]+=" .env | wc -l

# Ver variables (sin valores)
grep -E "^[A-Z_]+=" .env | cut -d'=' -f1
```

### Test de conexión

```python
# Test Supabase
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_KEY')

if url and key:
    print('✅ Supabase configurado')
else:
    print('❌ Falta configurar Supabase')
"
```

---

## 🚨 Errores Comunes

### Error: "SUPABASE_URL no configurada"

**Causa:** Archivo .env no existe o está vacío

**Solución:**
```bash
cp .env.example .env
nano .env  # Completar SUPABASE_URL
```

### Error: "API Key inválida"

**Causa:** API_KEY en .env no coincide con el request

**Solución:**
```bash
# Ver key actual
grep API_KEY= .env

# Usar esa key en requests
curl -H "X-API-Key: $(grep API_KEY= .env | cut -d'=' -f2)" ...
```

### Error: ".env not found" en Docker

**Causa:** Docker no tiene acceso al .env

**Solución:**
```yaml
# docker-compose.yml
services:
  api:
    env_file:
      - .env
```

---

## 🔒 Seguridad

### ✅ HACER:
- ✅ Usar `.env` solo en desarrollo local
- ✅ Agregar `.env` al `.gitignore`
- ✅ Usar variables de entorno del hosting en producción
- ✅ Generar keys aleatorias fuertes
- ✅ Rotar keys periódicamente

### ❌ NO HACER:
- ❌ Commitear `.env` a git
- ❌ Compartir `.env` por email/Slack
- ❌ Usar credenciales default
- ❌ Reutilizar mismo password en múltiples servicios
- ❌ Hardcodear credentials en el código

---

## 📚 Recursos

- [python-dotenv docs](https://pypi.org/project/python-dotenv/)
- [12 Factor App - Config](https://12factor.net/config)
- [Supabase Auth](https://supabase.com/docs/guides/auth)
- [AWS IAM](https://docs.aws.amazon.com/IAM/)
- [Slack Webhooks](https://api.slack.com/messaging/webhooks)

---

## 🆘 Ayuda

Si tienes problemas:

1. Verifica que `.env` existe: `ls .env`
2. Verifica formato: `cat .env | head -5`
3. Test carga: `python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('SUPABASE_URL'))"`
4. Revisa logs de la aplicación

---

**Última actualización:** 2024-11-22

