# 🎫 Sistema de Clasificación de Tickets - MLOps

Sistema completo de MLOps para clasificación automática de tickets de soporte con monitoreo, detección de drift y reentrenamiento automático.

## 📋 Características

- ✅ **Preprocesamiento robusto** de texto en español
- ✅ **Entrenamiento automático** de 7 modelos con optimización de hiperparámetros (Optuna)
- ✅ **Tracking de experimentos** con MLflow
- ✅ **Versionamiento de modelos** con DVC y S3
- ✅ **API de inferencia** con FastAPI
- ✅ **Monitoreo en tiempo real** de predicciones
- ✅ **Detección automática de drift** (data drift y concept drift)
- ✅ **Reentrenamiento automático** cuando se detectan problemas
- ✅ **Deploy automático** del mejor modelo
- ✅ **CI/CD completo** con GitHub Actions

## 🏗️ Arquitectura

```
┌─────────────────┐
│   Datos (S3)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Preprocesamiento│
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Entrenamiento  │─────▶│    MLflow     │
│   (7 modelos)    │      │  Tracking     │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Mejor Modelo   │─────▶│  DVC + S3    │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  API Inference  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   Monitoreo     │─────▶│  Drift Det.  │
│   (Logs)        │      └──────┬───────┘
└────────┬────────┘             │
         │                      │
         │              ┌───────▼───────┐
         │              │ ¿Drift?       │
         │              └───────┬───────┘
         │                      │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────┐
         │ Apache Airflow    │
         │   Orquestación    │
         └────────┬──────────┘
                  │
                  ▼
         ┌──────────────────┐
         │ Reentrenamiento  │
         │   Automático     │
         └──────────────────┘
```

## 📁 Estructura del Proyecto

```
ticket-classifier-mlops/
├── api/
│   ├── __init__.py
│   └── inference.py          # API FastAPI para predicciones
├── scripts/
│   ├── train_model.py        # Script de entrenamiento
│   ├── monitor_and_retrain.py # Monitoreo y reentrenamiento automático
│   └── deploy_model.py       # Script de deploy
├── utils/
│   ├── preprocessing_data.py # Preprocesamiento de texto
│   └── monitoring.py         # Sistema de monitoreo y drift detection
├── data-tickets-train/
│   └── dataset_tickets.csv   # Dataset de entrenamiento
├── models/
│   ├── best_model.pkl        # Mejor modelo entrenado
│   └── best_model_metadata.json
├── monitoring/
│   └── logs/                 # Logs de predicciones
├── .github/
│   └── workflows/
│       ├── train_model.yml   # CI/CD para entrenamiento
│       └── monitor_and_retrain.yml # CI/CD para monitoreo
├── config.yaml               # Configuración del proyecto
└── requirements.txt          # Dependencias
```

## 🚀 Inicio Rápido

### Opción 1: Setup Automático (Recomendado)

```bash
# Clonar repositorio
git clone <repo-url>
cd ticket-classifier-mlops

# Ejecutar script de setup
./setup.sh
```

### Opción 2: Setup Manual

```bash
# Clonar repositorio
git clone <repo-url>
cd ticket-classifier-mlops

# Crear entorno virtual
python3.9 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias (versiones fijas para reproducibilidad)
pip install --upgrade pip
pip install -r requirements-lock.txt

# Descargar recursos NLTK
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
```

**📚 Ver `SETUP.md` para guía completa de setup y reproducibilidad.**

### 2. Configurar DVC y S3

```bash
# Configurar DVC (si aún no está configurado)
dvc remote add -d s3remote s3://tu-bucket/dvc-storage
dvc remote modify s3remote endpointurl https://s3.amazonaws.com

# Pull de datos y modelo
dvc pull
```

### 3. Entrenar Modelo

```bash
# Entrenar modelo localmente
python scripts/train_model.py
```

### 4. Iniciar API de Inferencia

```bash
# Iniciar API
python api/inference.py

# O con uvicorn
uvicorn api.inference:app --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000`

### 5. Usar la API

```python
import requests

# Predicción individual
response = requests.post("http://localhost:8000/predict", json={
    "short_description": "Tengo un problema con mi cuenta",
    "close_notes": "El cliente reporta error al iniciar sesión"
})

result = response.json()
print(f"Predicción: {result['prediction']}")
print(f"Probabilidad: {result['probability']:.4f}")
```

## 📊 Monitoreo y Drift Detection

### Sistema de Monitoreo

El sistema monitorea automáticamente:

1. **Data Drift**: Cambios en la distribución de datos de entrada
   - Distribución de longitud de texto
   - Distribución de clases
   - Cambios en vocabulario

2. **Concept Drift**: Degradación de performance del modelo
   - Accuracy, F1-Score, Precision, Recall
   - Confianza en predicciones

### Ejecutar Monitoreo Manualmente

```bash
# Verificar drift y reentrenar si es necesario
python scripts/monitor_and_retrain.py
```

### Endpoints de Monitoreo

```bash
# Verificar drift
curl http://localhost:8000/monitoring/drift

# Obtener métricas
curl http://localhost:8000/monitoring/metrics

# Guardar métricas diarias
curl -X POST http://localhost:8000/monitoring/save-metrics
```

## 🔄 Reentrenamiento Automático

El sistema reentrena automáticamente cuando:

1. **Data Drift detectado**: Score de drift > 0.5
2. **Performance degradada**: F1-Score baja > 5% vs modelo entrenado
3. **Baja confianza**: > 30% de predicciones con confianza < 0.5

### Flujo de Reentrenamiento

1. Monitoreo detecta problema
2. Se dispara reentrenamiento automático
3. Se entrena nuevo modelo con todos los algoritmos
4. Se compara con modelo actual
5. Si el nuevo es mejor (>1% mejora), se hace deploy automático

### Configurar Reentrenamiento Automático

El workflow de GitHub Actions ejecuta monitoreo cada 6 horas:

```yaml
# .github/workflows/monitor_and_retrain.yml
schedule:
  - cron: '0 */6 * * *'  # Cada 6 horas
```

También se puede ejecutar manualmente desde GitHub Actions.

## 🚢 Deploy

### Deploy Automático

El deploy se ejecuta automáticamente después de reentrenamiento exitoso:

```bash
# Deploy manual
python scripts/deploy_model.py
```

### Deploy en Producción

Para producción, actualizar `scripts/deploy_model.py` para:

1. Copiar modelo a directorio de producción
2. Reiniciar servicio API
3. Actualizar MLflow Model Registry
4. Enviar notificaciones (Slack, Email, etc.)

## ⚙️ Configuración

Editar `config.yaml` para personalizar:

```yaml
monitoring:
  drift_threshold: 0.05
  drift_score_threshold: 0.5
  performance_drop_threshold: 0.05
  min_improvement_for_deploy: 0.01
```

## 📈 Modelos Disponibles

El sistema entrena y compara 7 modelos:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. SVM
5. LightGBM
6. Gradient Boosting ⭐ (actualmente el mejor)
7. Extra Trees

## 🔍 Endpoints de la API

### `GET /`
Health check básico

### `GET /health`
Health check detallado

### `POST /predict`
Predicción individual
```json
{
  "short_description": "texto",
  "close_notes": "texto opcional",
  "true_label": "label opcional (para evaluación)"
}
```

### `POST /predict/batch`
Predicción en batch (múltiples tickets)

### `GET /monitoring/drift`
Verifica drift en datos recientes

### `GET /monitoring/metrics`
Obtiene métricas de monitoreo

### `POST /monitoring/save-metrics`
Guarda métricas diarias

## 🧪 Testing

```bash
# Test de la API
curl http://localhost:8000/health

# Test de predicción
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "short_description": "Problema con login",
    "close_notes": "Usuario no puede acceder"
  }'
```

## 📝 Logs y Monitoreo

Los logs de predicciones se guardan en:
- `monitoring/logs/predictions.jsonl` - Predicciones individuales
- `monitoring/logs/daily_metrics.json` - Métricas diarias agregadas

## 🔐 Variables de Entorno

```bash
# API
export API_URL=http://localhost:8000

# Monitoreo
export DRIFT_THRESHOLD=0.5
export MIN_PREDICTIONS_FOR_DRIFT=100
export PERFORMANCE_DROP_THRESHOLD=0.05

# AWS (para DVC)
export AWS_ACCESS_KEY_ID=tu_key
export AWS_SECRET_ACCESS_KEY=tu_secret
```

## 🐛 Troubleshooting

### API no inicia
- Verificar que el modelo existe: `models/best_model.pkl`
- Verificar que las dependencias están instaladas

### Drift detection no funciona
- Verificar que hay suficientes predicciones (>100)
- Verificar que la API está corriendo y accesible

### Reentrenamiento falla
- Verificar que hay datos en `data-tickets-train/dataset_tickets.csv`
- Verificar permisos de AWS para DVC push

## 🚀 Orquestación con Apache Airflow

El sistema incluye orquestación completa con Apache Airflow:

### DAGs Disponibles

1. **`mlops_ticket_classifier_pipeline`** - Pipeline completo E2E
   - Monitoreo → Reentrenamiento → Deploy
   - Schedule: Cada 6 horas

2. **`train_model_manual`** - Entrenamiento manual
   - Útil para reentrenamientos forzados

3. **`monitor_only`** - Solo monitoreo
   - Schedule: Cada hora

### Inicio Rápido con Airflow

```bash
cd airflow
docker-compose up -d
```

Accede a la UI en: http://localhost:8080

Ver documentación completa en: `airflow/README.md`

## 📚 Próximos Pasos

- [x] ✅ Orquestación con Apache Airflow
- [ ] Implementar notificaciones (Slack/Email)
- [ ] Dashboard de monitoreo (Grafana/Dash)
- [ ] A/B testing de modelos
- [ ] Feature store para datos
- [ ] Model explainability (SHAP)

## 📄 Licencia

[Tu licencia aquí]

## 👥 Autores

Sistema MLOps - 2024

