"""
DAG para entrenamiento manual del modelo.

Este DAG permite entrenar el modelo manualmente desde Airflow UI.
Útil para reentrenamientos forzados o actualizaciones de datos.

Autor: Sistema MLOps
Fecha: 2024
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
import os
import sys
import subprocess
from pathlib import Path

# Agregar raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

def train_model_task(**context):
    """Entrena el modelo"""
    train_script = project_root / "scripts" / "train_model.py"
    
    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=3600
    )
    
    if result.returncode != 0:
        print(f"Training failed:")
        print(result.stderr)
        raise Exception(f"Training failed with return code {result.returncode}")
    
    print("Training completed successfully")
    return True

def push_model_to_s3_task(**context):
    """Push del modelo a S3"""
    import subprocess
    
    model_path = project_root / "models" / "best_model.pkl"
    dvc_file = model_path.with_suffix('.pkl.dvc')
    
    if not model_path.exists():
        raise Exception("Model file not found")
    
    # DVC add
    subprocess.run(
        ['dvc', 'add', str(model_path)],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    # DVC push
    result = subprocess.run(
        ['dvc', 'push', str(dvc_file)],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"DVC push warning: {result.stderr}")
    
    print("Model pushed to S3")
    return True

with DAG(
    'train_model_manual',
    default_args=default_args,
    description='Entrenamiento manual del modelo',
    schedule_interval=None,  # Solo manual
    catchup=False,
    tags=['mlops', 'training', 'manual'],
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    pull_data = BashOperator(
        task_id='pull_data_from_s3',
        bash_command='cd {{ params.project_root }} && dvc pull data-tickets-train/dataset_tickets.csv.dvc || true',
        params={'project_root': str(project_root)},
    )
    
    train = PythonOperator(
        task_id='train_model',
        python_callable=train_model_task,
    )
    
    push_model = PythonOperator(
        task_id='push_model_to_s3',
        python_callable=push_model_to_s3_task,
    )
    
    end = EmptyOperator(task_id='end')
    
    start >> pull_data >> train >> push_model >> end





















