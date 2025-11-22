"""
DAG para solo monitoreo (sin reentrenamiento).

Útil para monitorear el modelo sin disparar reentrenamiento automático.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
import requests
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

API_URL = Variable.get("API_URL", default_var="http://localhost:8000")

def check_drift(**context):
    """Verifica drift"""
    try:
        response = requests.get(f"{API_URL}/monitoring/drift", timeout=30)
        if response.status_code == 200:
            drift_data = response.json()
            print(f"Drift detected: {drift_data.get('drift_detected', False)}")
            print(f"Drift score: {drift_data.get('drift_score', 0.0)}")
            return drift_data
        return {"status": "api_unavailable"}
    except Exception as e:
        print(f"Warning: {e}")
        return {"status": "error"}

def get_metrics(**context):
    """Obtiene métricas"""
    try:
        response = requests.get(f"{API_URL}/monitoring/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            print(f"Total predictions: {metrics.get('total_predictions', 0)}")
            print(f"Average confidence: {metrics.get('average_confidence', 0.0):.4f}")
            return metrics
        return {}
    except Exception as e:
        print(f"Warning: {e}")
        return {}

def save_metrics(**context):
    """Guarda métricas diarias"""
    try:
        response = requests.post(f"{API_URL}/monitoring/save-metrics", timeout=10)
        if response.status_code == 200:
            print("Metrics saved successfully")
            return response.json()
        return {}
    except Exception as e:
        print(f"Warning: {e}")
        return {}

with DAG(
    'monitor_only',
    default_args=default_args,
    description='Solo monitoreo del modelo (sin reentrenamiento)',
    schedule_interval=timedelta(hours=1),  # Cada hora
    catchup=False,
    tags=['mlops', 'monitoring'],
) as dag:
    
    start = EmptyOperator(task_id='start')
    
    check_drift_task = PythonOperator(
        task_id='check_drift',
        python_callable=check_drift,
    )
    
    get_metrics_task = PythonOperator(
        task_id='get_metrics',
        python_callable=get_metrics,
    )
    
    save_metrics_task = PythonOperator(
        task_id='save_metrics',
        python_callable=save_metrics,
    )
    
    end = EmptyOperator(task_id='end')
    
    start >> [check_drift_task, get_metrics_task] >> save_metrics_task >> end





















