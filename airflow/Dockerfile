FROM apache/airflow:2.7.0

USER root

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    dvc \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Instalar dependencias de Python para MLOps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Configurar DVC (si es necesario)
# RUN dvc remote add -d s3remote s3://tu-bucket/dvc-storage || true





















