# 🚀 Apache Airflow para MLOps Pipeline

Sistema completo de orquestación con Apache Airflow para el pipeline de MLOps.

## 📋 Características

- ✅ **Monitoreo automático** cada 6 horas
- ✅ **Detección de drift** (data drift y concept drift)
- ✅ **Reentrenamiento automático** cuando es necesario
- ✅ **Deploy automático** del mejor modelo
- ✅ **Push automático a S3** con DVC
- ✅ **Interfaz visual** para monitoreo y control

## 🏗️ DAGs Incluidos

### 1. `mlops_ticket_classifier_pipeline` (Principal)

**Pipeline completo end-to-end:**
- Monitoreo → Reentrenamiento → Deploy

**Schedule:** Cada 6 horas

**Flujo:**
```
Start
  ↓
Monitoring Group
  ├─ Check API Health
  ├─ Check Drift
  ├─ Evaluate Performance
  └─ Decide Retraining
  ↓
Should Retrain? (ShortCircuit)
  ↓ (Sí)
Retraining Group
  ├─ Train Model
  └─ Compare Models
  ↓
Should Deploy? (ShortCircuit)
  ↓ (Sí)
Deploy Group
  ├─ Deploy Model
  └─ Push to S3
  ↓
End
```

### 2. `train_model_manual`

**Entrenamiento manual:**
- Útil para reentrenamientos forzados
- Se ejecuta solo manualmente desde la UI

### 3. `monitor_only`

**Solo monitoreo:**
- Monitorea sin disparar reentrenamiento
- Útil para verificar estado sin cambios
- Schedule: Cada hora

## 🚀 Instalación y Configuración

### Opción 1: Docker Compose (Recomendado)

```bash
# Ir al directorio de airflow
cd airflow

# Configurar variables de entorno
export AIRFLOW_UID=$(id -u)
export API_URL=http://host.docker.internal:8000
export AWS_ACCESS_KEY_ID=tu_key
export AWS_SECRET_ACCESS_KEY=tu_secret

# Inicializar Airflow
docker-compose up airflow-init

# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f
```

**Acceder a la UI:**
- URL: http://localhost:8080
- Usuario: `airflow`
- Password: `airflow` (cambiar en producción)

### Opción 2: Instalación Local

```bash
# Instalar Airflow
pip install apache-airflow

# Inicializar base de datos
airflow db init

# Crear usuario
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Configurar variables
airflow variables set API_URL http://localhost:8000
airflow variables set DRIFT_THRESHOLD 0.5
airflow variables set PERFORMANCE_DROP_THRESHOLD 0.05
airflow variables set MIN_IMPROVEMENT_FOR_DEPLOY 0.01

# Iniciar webserver
airflow webserver --port 8080

# En otra terminal, iniciar scheduler
airflow scheduler
```

## ⚙️ Configuración de Variables

Configura estas variables en Airflow UI (Admin → Variables):

| Variable | Descripción | Default |
|----------|-------------|---------|
| `API_URL` | URL de la API de inferencia | `http://localhost:8000` |
| `DRIFT_THRESHOLD` | Umbral de drift para reentrenamiento | `0.5` |
| `PERFORMANCE_DROP_THRESHOLD` | Caída de performance para trigger | `0.05` |
| `MIN_IMPROVEMENT_FOR_DEPLOY` | Mejora mínima para deploy | `0.01` |

## 📊 Uso de la UI

### 1. Ver DAGs

1. Abre http://localhost:8080
2. Verás los 3 DAGs listados
3. Activa/desactiva con el toggle

### 2. Ejecutar DAG Manualmente

1. Click en el DAG
2. Click en "Trigger DAG"
3. Monitorea la ejecución en "Graph View"

### 3. Ver Logs

1. Click en una tarea
2. Click en "Log"
3. Ver logs detallados

### 4. Ver XComs (datos compartidos)

1. Click en una tarea
2. Click en "XCom"
3. Ver datos compartidos entre tareas

## 🔍 Monitoreo del Pipeline

### Ver Estado de Ejecuciones

```bash
# Listar ejecuciones
airflow dags list-runs -d mlops_ticket_classifier_pipeline

# Ver logs de una tarea específica
airflow tasks logs mlops_ticket_classifier_pipeline check_drift 2024-01-01
```

### Alertas por Email

Configura SMTP en `airflow.cfg`:

```ini
[smtp]
smtp_host = smtp.gmail.com
smtp_starttls = True
smtp_ssl = False
smtp_user = tu_email@gmail.com
smtp_password = tu_password
smtp_port = 587
smtp_mail_from = airflow@example.com
```

## 🎯 Flujo de Ejecución

### Escenario 1: Sin Drift

```
Monitoring → No drift detected → End
```

### Escenario 2: Drift Detectado

```
Monitoring → Drift detected → Retrain → Compare → Deploy (if better) → End
```

### Escenario 3: Performance Degradada

```
Monitoring → Performance drop → Retrain → Compare → Deploy (if better) → End
```

## 🐛 Troubleshooting

### DAG no aparece

```bash
# Verificar que los DAGs están en el directorio correcto
ls -la airflow/dags/

# Verificar sintaxis
python airflow/dags/mlops_pipeline.py

# Ver logs del scheduler
docker-compose logs airflow-scheduler
```

### Tareas fallan

1. Verificar logs en la UI
2. Verificar que la API está corriendo (si es necesario)
3. Verificar credenciales de AWS para DVC

### Variables no se cargan

```bash
# Verificar variables
airflow variables list

# Setear manualmente
airflow variables set API_URL http://localhost:8000
```

## 📝 Personalización

### Cambiar Schedule

Edita `schedule_interval` en el DAG:

```python
schedule_interval=timedelta(hours=6),  # Cada 6 horas
# O
schedule_interval='0 */6 * * *',  # Cron expression
```

### Agregar Notificaciones

```python
from airflow.operators.email import EmailOperator

send_email = EmailOperator(
    task_id='send_notification',
    to='team@example.com',
    subject='Model Retrained',
    html_content='New model deployed successfully'
)
```

### Integrar con Slack

```python
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator

slack_notification = SlackWebhookOperator(
    task_id='slack_notification',
    http_conn_id='slack',
    message='Model retrained successfully'
)
```

## 🔐 Seguridad

### Cambiar Credenciales por Defecto

```bash
# En docker-compose.yml, cambiar:
_AIRFLOW_WWW_USER_USERNAME: admin
_AIRFLOW_WWW_USER_PASSWORD: tu_password_seguro
```

### Configurar SSL/TLS

Edita `docker-compose.yml` para agregar certificados SSL.

## 📚 Recursos

- [Documentación de Airflow](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [XComs Documentation](https://airflow.apache.org/docs/apache-airflow/stable/concepts/xcoms.html)

## ✅ Checklist

- [ ] Airflow instalado y corriendo
- [ ] Variables configuradas
- [ ] DAGs visibles en la UI
- [ ] API de inferencia accesible
- [ ] Credenciales AWS configuradas
- [ ] Primer DAG ejecutado exitosamente





