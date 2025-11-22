# 📝 Sistema de Rotación de Logs

## 🎯 Problema Resuelto

El archivo `monitoring/logs/predictions.jsonl` crecía indefinidamente, causando:
- ❌ Disco lleno en Render (plan gratuito con límites)
- ❌ Lecturas lentas al analizar métricas
- ❌ Potencial crash de la API

## ✅ Solución Implementada

### Rotación Automática de Logs

El sistema ahora:
1. **Monitorea** el tamaño del archivo `predictions.jsonl`
2. **Rota** cuando alcanza 10MB (configurable)
3. **Comprime** archivos antiguos con gzip (ahorra ~90% de espacio)
4. **Mantiene** máximo 5 backups (configurable)
5. **Elimina** automáticamente archivos más antiguos

### Estructura de Archivos

```
monitoring/logs/
├── predictions.jsonl          # Archivo actual (< 10MB)
├── predictions.jsonl.1.gz     # Backup más reciente (comprimido)
├── predictions.jsonl.2.gz     # 2do backup más antiguo
├── predictions.jsonl.3.gz     # 3er backup
├── predictions.jsonl.4.gz     # 4to backup
├── predictions.jsonl.5.gz     # Backup más antiguo
└── daily_metrics.json         # Métricas agregadas
```

### Ciclo de Vida

```
predictions.jsonl (9MB)
    ↓ Nueva predicción (10.1MB)
    ↓ ¡Trigger de rotación!
    
1. predictions.jsonl.4.gz → predictions.jsonl.5.gz
2. predictions.jsonl.3.gz → predictions.jsonl.4.gz
3. predictions.jsonl.2.gz → predictions.jsonl.3.gz
4. predictions.jsonl.1.gz → predictions.jsonl.2.gz
5. predictions.jsonl → predictions.jsonl.1.gz (comprimir)
6. predictions.jsonl (nuevo, vacío)

¡Espacio liberado! ~45MB → ~5MB comprimido
```

## 🔧 Configuración

### Parámetros por Defecto

```python
from utils.monitoring import PredictionLogger

# Configuración por defecto
logger = PredictionLogger(
    log_dir="monitoring/logs",
    max_bytes=10 * 1024 * 1024,  # 10 MB
    backup_count=5                # 5 backups
)
```

### Personalizar Configuración

```python
# Para alta carga (más predicciones)
logger = PredictionLogger(
    log_dir="monitoring/logs",
    max_bytes=20 * 1024 * 1024,  # 20 MB
    backup_count=10               # 10 backups
)

# Para bajo storage (plan gratuito)
logger = PredictionLogger(
    log_dir="monitoring/logs",
    max_bytes=5 * 1024 * 1024,   # 5 MB
    backup_count=3                # 3 backups
)
```

## 📊 Monitoreo

### Ver Estado Actual

```bash
# Ver tamaño de archivos
ls -lh monitoring/logs/

# Total de espacio usado
du -sh monitoring/logs/

# Contar predicciones en archivo actual
wc -l monitoring/logs/predictions.jsonl

# Ver predicciones en backup (descomprimir)
zcat monitoring/logs/predictions.jsonl.1.gz | head -n 10
```

### Logs de Rotación

La API registra en logs cuando rota archivos:

```
2025-11-22 01:45:30,123 - utils.monitoring - INFO - ✅ Logs rotados exitosamente. Archivo comprimido: predictions.jsonl.1.gz
```

## 🔍 Lectura de Logs Históricos

El sistema **lee automáticamente** archivos rotados cuando consultas predicciones recientes:

```python
# Obtiene predicciones de las últimas 24 horas
# Incluye datos de predictions.jsonl + archivos rotados si son recientes
df = logger.get_recent_predictions(hours=24)
```

## 💾 Estimación de Almacenamiento

### Sin Rotación (ANTES)
- 1000 predicciones/día × 365 días = **~150MB/año**
- Render Free Plan: 512MB → **Disco lleno en ~4 meses** ❌

### Con Rotación (AHORA)
- Máximo: 10MB actual + 5 backups × 10MB comprimido = **~15MB total** ✅
- Compresión gzip: ~90% → 50MB → 5MB comprimido
- **Nunca llena el disco** ✅

## 🧪 Testing

```bash
# Probar rotación manualmente
python -c "
from utils.monitoring import PredictionLogger
logger = PredictionLogger(max_bytes=1024)  # 1KB para testing rápido

# Generar muchas predicciones
for i in range(100):
    logger.log_prediction(
        text=f'Test {i}' * 100,
        prediction='Test',
        probability=0.95
    )

print('✅ Test completado. Verifica monitoring/logs/')
"

# Verificar archivos creados
ls -lh monitoring/logs/
```

## 🚀 Deploy en Producción

Los cambios son **automáticos** y **retrocompatibles**:

1. ✅ Código ya desplegado en `main`
2. ✅ No requiere cambios en Render
3. ✅ No rompe logs existentes
4. ✅ Empieza a rotar en la próxima predicción

### Verificar en Render

1. Ve a Render Dashboard → Logs
2. Busca: `"Logs rotados exitosamente"`
3. Verifica espacio en disco: Settings → Metrics

## 📈 Beneficios

| Aspecto | Antes | Después |
|---------|-------|---------|
| **Espacio Máximo** | Ilimitado (hasta llenar disco) | ~15MB fijo |
| **Riesgo Disco Lleno** | ❌ Alto | ✅ Ninguno |
| **Performance Lectura** | ❌ Lento (archivos grandes) | ✅ Rápido (archivos pequeños) |
| **Costo Storage** | ❌ Crece linealmente | ✅ Constante |
| **Historial** | ✅ Completo | ✅ Últimos ~50K predicciones |
| **Backup Automático** | ❌ No | ✅ 5 versiones comprimidas |

## 🔄 Mantenimiento

### Archivos Comprimidos

Para analizar logs antiguos:

```bash
# Ver contenido de un backup
zcat monitoring/logs/predictions.jsonl.1.gz | jq .

# Buscar predicciones específicas
zcat monitoring/logs/predictions.jsonl.*.gz | grep "prediction_id"

# Descomprimir permanentemente (si necesitas)
gunzip monitoring/logs/predictions.jsonl.1.gz
```

### Limpiar Manualmente

```bash
# Eliminar todos los backups (mantener solo actual)
rm monitoring/logs/predictions.jsonl.*.gz

# Resetear logs completamente
rm monitoring/logs/predictions.jsonl*
```

## ⚙️ Integración con API

No requiere cambios en el código de la API. El logger se usa igual:

```python
# En api/inference.py - NO CAMBIA
PREDICTION_LOGGER = PredictionLogger(log_dir=str(project_root / "monitoring" / "logs"))

# La rotación es automática
PREDICTION_LOGGER.log_prediction(
    text=combined_text,
    prediction=str(prediction),
    probability=max_proba,
    true_label=None
)
```

## 🎯 Próximos Pasos (Opcional)

Para mejorar aún más:

1. **Backup a S3** - Subir archivos rotados a S3 para historial infinito
2. **Análisis Batch** - Procesar logs rotados para tendencias históricas
3. **Alertas** - Notificar si la rotación falla
4. **Dashboard** - Visualizar distribución de predicciones en Grafana

---

## ✅ Checklist

- [x] Rotación automática implementada
- [x] Compresión gzip habilitada
- [x] Lectura de archivos rotados funcional
- [x] Documentación completa
- [x] Retrocompatible con logs existentes
- [x] Deploy en producción

**Estado:** ✅ Producción Ready

