# 🚀 Guía de Setup y Reproducibilidad

Esta guía asegura que cualquier desarrollador pueda recrear el entorno exacto del proyecto.

## 📋 Prerequisitos del Sistema

- **Python:** 3.9.x (recomendado 3.9.18)
- **Git:** Para clonar el repositorio
- **Docker:** (Opcional) Para Airflow
- **AWS CLI:** (Opcional) Para DVC con S3

### Verificar Versión de Python

```bash
python --version
# Debe mostrar: Python 3.9.x
```

Si no tienes Python 3.9, instálalo con:
- **pyenv** (recomendado): `pyenv install 3.9.18`
- **conda**: `conda create -n ticket-classifier python=3.9.18`

---

## 🔧 Setup Paso a Paso

### Paso 1: Clonar Repositorio

```bash
git clone <repo-url>
cd ticket-classifier-mlops
```

### Paso 2: Crear Entorno Virtual

```bash
# Opción A: venv (recomendado)
python3.9 -m venv venv

# Opción B: conda
conda create -n ticket-classifier python=3.9.18
conda activate ticket-classifier

# Activar entorno
source venv/bin/activate  # Linux/Mac
# O en Windows:
venv\Scripts\activate
```

### Paso 3: Instalar Dependencias

#### Opción A: Versiones Fijas (Recomendado para Reproducibilidad)

```bash
# Instalar exactamente las mismas versiones
pip install --upgrade pip
pip install -r requirements-lock.txt
```

#### Opción B: Versiones Flexibles (Desarrollo)

```bash
# Instalar con versiones mínimas
pip install --upgrade pip
pip install -r requirements.txt
```

### Paso 4: Descargar Recursos NLTK

```bash
python -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
print('✓ NLTK resources downloaded')
"
```

### Paso 5: Configurar DVC (si usas S3)

```bash
# Configurar remoto S3
dvc remote add -d s3remote s3://tu-bucket/dvc-storage
dvc remote modify s3remote endpointurl https://s3.amazonaws.com

# Descargar datos y modelo
dvc pull
```

### Paso 6: Verificar Instalación

```bash
# Verificar que todo está instalado
python -c "
import pandas as pd
import numpy as np
import sklearn
import mlflow
import dvc
import fastapi
print('✅ Todas las dependencias están instaladas')
"
```

---

## 🔒 Garantizar Reproducibilidad

### 1. Usar Versiones Fijas

El archivo `requirements-lock.txt` contiene versiones exactas:

```bash
# Generar nuevo lock file (si actualizas dependencias)
pip freeze > requirements-lock.txt

# Instalar desde lock file
pip install -r requirements-lock.txt
```

### 2. Random Seeds Configurados

El proyecto usa `random_state=42` en:
- `train_test_split`
- Todos los modelos de ML
- Optuna studies

Esto garantiza resultados reproducibles.

### 3. Configuración Externalizada

Toda la configuración está en `config.yaml`, no hardcodeada.

### 4. Versionamiento de Datos y Modelos

- **Datos:** Versionados con DVC
- **Modelos:** Versionados con DVC + MLflow

---

## 🧪 Verificar Reproducibilidad

### Test 1: Entrenar Modelo

```bash
# Entrenar modelo
python scripts/train_model.py

# Verificar que se genera el mismo modelo
# Compara best_model_metadata.json con el del repo
```

### Test 2: Preprocesamiento

```bash
# Probar preprocesamiento
python -c "
from utils.preprocessing_data import preprocess_text
result = preprocess_text('Hola mundo')
print(f'Resultado: {result}')
# Debe ser consistente entre ejecuciones
"
```

### Test 3: API

```bash
# Iniciar API
python api/inference.py

# En otra terminal, probar
python test_api.py
```

---

## 🐛 Troubleshooting

### Error: "Python version mismatch"

**Solución:**
```bash
# Usar pyenv para instalar Python 3.9
pyenv install 3.9.18
pyenv local 3.9.18
```

### Error: "Package version conflict"

**Solución:**
```bash
# Recrear entorno desde cero
rm -rf venv
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-lock.txt
```

### Error: "NLTK data not found"

**Solución:**
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Error: "DVC pull fails"

**Solución:**
- Verifica credenciales AWS
- O descarga datos manualmente si no usas S3

---

## 📦 Estructura de Dependencias

```
requirements.txt          # Versiones flexibles (desarrollo)
requirements-lock.txt     # Versiones fijas (producción/reproducibilidad)
.python-version          # Versión de Python requerida
```

---

## ✅ Checklist de Setup

- [ ] Python 3.9.x instalado
- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas desde `requirements-lock.txt`
- [ ] Recursos NLTK descargados
- [ ] DVC configurado (si aplica)
- [ ] Datos descargados (`dvc pull`)
- [ ] Modelo descargado (`dvc pull models/best_model.pkl.dvc`)
- [ ] Tests pasan
- [ ] API funciona

---

## 🔄 Actualizar Dependencias

Si necesitas actualizar dependencias:

```bash
# 1. Actualizar requirements.txt con nuevas versiones
# 2. Instalar nuevas versiones
pip install -r requirements.txt

# 3. Generar nuevo lock file
pip freeze > requirements-lock.txt

# 4. Commit ambos archivos
git add requirements.txt requirements-lock.txt
git commit -m "Update dependencies"
```

---

## 📝 Notas Importantes

1. **Siempre usa `requirements-lock.txt` en CI/CD** para garantizar reproducibilidad
2. **Commit `requirements-lock.txt`** al repositorio
3. **Documenta cambios** en versiones de dependencias críticas
4. **Usa el mismo Python version** en todos los entornos

---

## 🎯 Para Nuevos Desarrolladores

1. Clonar repo
2. Seguir esta guía paso a paso
3. Si algo falla, revisar Troubleshooting
4. Verificar con los tests que todo funciona

---

## 🔐 Variables de Entorno

Crea un archivo `.env` (no commitear):

```bash
# AWS (para DVC)
AWS_ACCESS_KEY_ID=tu_key
AWS_SECRET_ACCESS_KEY=tu_secret
AWS_DEFAULT_REGION=us-east-1

# API (opcional)
API_URL=http://localhost:8000
```

---

## 📚 Recursos Adicionales

- [Python Virtual Environments](https://docs.python.org/3/tutorial/venv.html)
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
















