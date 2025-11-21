"""
DAG principal de MLOps para clasificación de tickets.

Este DAG orquesta:
1. Monitoreo de modelo en producción
2. Detección de drift (data drift y concept drift)
3. Reentrenamiento automático si es necesario
4. Evaluación y comparación de modelos
5. Deploy automático del mejor modelo

Autor: Sistema MLOps
Fecha: 2024
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
import os
import sys
from pathlib import Path

# Agregar raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}

# Variables de Airflow (configurables desde la UI)
API_URL = Variable.get("API_URL", default_var="http://localhost:8000")
DRIFT_THRESHOLD = float(Variable.get("DRIFT_THRESHOLD", default_var="0.5"))
PERFORMANCE_DROP_THRESHOLD = float(Variable.get("PERFORMANCE_DROP_THRESHOLD", default_var="0.05"))

# ============================================================================
# FUNCIONES DE TAREAS
# ============================================================================

def check_api_health(**context):
    """Verifica que la API esté funcionando"""
    import requests
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            context['ti'].xcom_push(key='api_healthy', value=True)
            context['ti'].xcom_push(key='model_loaded', value=health_data.get('model_loaded', False))
            return health_data
        else:
            raise Exception(f"API returned status {response.status_code}")
    except Exception as e:
        context['ti'].xcom_push(key='api_healthy', value=False)
        raise Exception(f"API health check failed: {e}")

def check_drift(**context):
    """Verifica drift en datos y modelo"""
    import requests
    
    try:
        response = requests.get(f"{API_URL}/monitoring/drift", timeout=30)
        if response.status_code == 200:
            drift_data = response.json()
            
            drift_detected = drift_data.get('drift_detected', False)
            drift_score = drift_data.get('drift_score', 0.0)
            
            context['ti'].xcom_push(key='drift_detected', value=drift_detected)
            context['ti'].xcom_push(key='drift_score', value=drift_score)
            context['ti'].xcom_push(key='drift_details', value=drift_data)
            
            return drift_data
        else:
            # Si la API no está disponible, asumir que no hay drift crítico
            context['ti'].xcom_push(key='drift_detected', value=False)
            context['ti'].xcom_push(key='drift_score', value=0.0)
            return {"status": "api_unavailable", "drift_detected": False}
    except Exception as e:
        # Si falla, no bloquear el pipeline
        print(f"Warning: Drift check failed: {e}")
        context['ti'].xcom_push(key='drift_detected', value=False)
        return {"status": "error", "drift_detected": False}

def evaluate_model_performance(**context):
    """Evalúa el performance del modelo actual"""
    import requests
    import json
    from pathlib import Path
    
    try:
        # Obtener métricas de la API
        response = requests.get(f"{API_URL}/monitoring/metrics", timeout=10)
        if response.status_code == 200:
            metrics = response.json()
            context['ti'].xcom_push(key='current_metrics', value=metrics)
            return metrics
        
        # Si la API no está disponible, intentar leer logs directamente
        logger_path = project_root / "monitoring" / "logs" / "predictions.jsonl"
        if logger_path.exists():
            # Leer predicciones recientes con labels
            predictions = []
            with open(logger_path, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get('true_label') and entry.get('correct') is not None:
                            predictions.append(entry)
                    except:
                        continue
            
            if len(predictions) >= 50:
                correct = sum(1 for p in predictions if p.get('correct') == True)
                accuracy = correct / len(predictions) if predictions else 0.0
                
                metrics = {
                    'accuracy': accuracy,
                    'total_predictions': len(predictions),
                    'correct_predictions': correct
                }
                context['ti'].xcom_push(key='current_metrics', value=metrics)
                return metrics
        
        # Si no hay datos, retornar None
        context['ti'].xcom_push(key='current_metrics', value=None)
        return None
        
    except Exception as e:
        print(f"Warning: Performance evaluation failed: {e}")
        context['ti'].xcom_push(key='current_metrics', value=None)
        return None

def decide_retraining(**context):
    """Decide si se necesita reentrenamiento basado en drift y performance"""
    # Obtener resultados de drift
    drift_detected = context['ti'].xcom_pull(key='drift_detected', task_ids='check_drift')
    drift_score = context['ti'].xcom_pull(key='drift_score', task_ids='check_drift')
    
    # Obtener métricas actuales
    current_metrics = context['ti'].xcom_pull(key='current_metrics', task_ids='evaluate_performance')
    
    # Obtener métricas del modelo entrenado
    metadata_path = project_root / "models" / "best_model_metadata.json"
    trained_f1 = None
    if metadata_path.exists():
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            trained_f1 = metadata.get('f1_score', None)
    
    should_retrain = False
    reasons = []
    
    # Check 1: Data drift
    if drift_detected and drift_score > DRIFT_THRESHOLD:
        should_retrain = True
        reasons.append(f"Data drift detectado (score: {drift_score:.4f})")
    
    # Check 2: Performance drop
    if current_metrics and trained_f1:
        current_f1 = current_metrics.get('f1_score') or current_metrics.get('accuracy', 0.0)
        performance_drop = trained_f1 - current_f1
        
        if performance_drop > PERFORMANCE_DROP_THRESHOLD:
            should_retrain = True
            reasons.append(
                f"Performance degradada: F1 {trained_f1:.4f} → {current_f1:.4f} "
                f"(drop: {performance_drop:.4f})"
            )
    
    context['ti'].xcom_push(key='should_retrain', value=should_retrain)
    context['ti'].xcom_push(key='retrain_reasons', value=reasons)
    
    print(f"Should retrain: {should_retrain}")
    if reasons:
        print("Reasons:")
        for reason in reasons:
            print(f"  - {reason}")
    
    return should_retrain

def train_model(**context):
    """Entrena el modelo"""
    import subprocess
    
    train_script = project_root / "scripts" / "train_model.py"
    
    result = subprocess.run(
        [sys.executable, str(train_script)],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=3600  # 1 hora máximo
    )
    
    if result.returncode != 0:
        print(f"Training failed:")
        print(result.stderr)
        raise Exception(f"Training failed with return code {result.returncode}")
    
    print("Training completed successfully")
    return True

def compare_models(**context):
    """Compara el nuevo modelo con el actual"""
    import json
    
    # Cargar métricas del modelo actual (antes del entrenamiento)
    old_metadata_path = project_root / "models" / "best_model_metadata.json"
    old_f1 = None
    if old_metadata_path.exists():
        with open(old_metadata_path, 'r') as f:
            old_metadata = json.load(f)
            old_f1 = old_metadata.get('f1_score', 0.0)
    
    # Cargar métricas del nuevo modelo
    new_metadata_path = project_root / "models" / "best_model_metadata.json"
    if not new_metadata_path.exists():
        raise Exception("New model metadata not found")
    
    with open(new_metadata_path, 'r') as f:
        new_metadata = json.load(f)
        new_f1 = new_metadata.get('f1_score', 0.0)
    
    improvement = new_f1 - old_f1 if old_f1 else 0.0
    min_improvement = float(Variable.get("MIN_IMPROVEMENT_FOR_DEPLOY", default_var="0.01"))
    
    should_deploy = improvement > min_improvement
    
    context['ti'].xcom_push(key='new_f1', value=new_f1)
    context['ti'].xcom_push(key='old_f1', value=old_f1)
    context['ti'].xcom_push(key='improvement', value=improvement)
    context['ti'].xcom_push(key='should_deploy', value=should_deploy)
    
    print(f"Old F1: {old_f1:.4f}")
    print(f"New F1: {new_f1:.4f}")
    print(f"Improvement: {improvement:.4f}")
    print(f"Should deploy: {should_deploy}")
    
    return should_deploy

def deploy_model(**context):
    """Hace deploy del modelo"""
    import subprocess
    
    deploy_script = project_root / "scripts" / "deploy_model.py"
    
    result = subprocess.run(
        [sys.executable, str(deploy_script)],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Deploy failed:")
        print(result.stderr)
        raise Exception(f"Deploy failed with return code {result.returncode}")
    
    print("Deploy completed successfully")
    return True

def push_model_to_s3(**context):
    """Push del modelo a S3 usando DVC"""
    import subprocess
    
    model_path = project_root / "models" / "best_model.pkl"
    dvc_file = model_path.with_suffix('.pkl.dvc')
    
    if not model_path.exists():
        raise Exception("Model file not found")
    
    # DVC add
    result = subprocess.run(
        ['dvc', 'add', str(model_path)],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"DVC add warning: {result.stderr}")
    
    # DVC push
    result = subprocess.run(
        ['dvc', 'push', str(dvc_file)],
        cwd=project_root,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"DVC push warning: {result.stderr}")
        # No fallar si DVC push falla (puede ser problema de credenciales)
    
    print("Model pushed to S3")
    return True

# ============================================================================
# DEFINICIÓN DEL DAG
# ============================================================================

with DAG(
    'mlops_ticket_classifier_pipeline',
    default_args=default_args,
    description='Pipeline completo de MLOps: Monitoreo → Reentrenamiento → Deploy',
    schedule_interval=timedelta(hours=6),  # Cada 6 horas
    catchup=False,
    tags=['mlops', 'ticket-classifier', 'monitoring', 'retraining'],
) as dag:
    
    # ========================================================================
    # START
    # ========================================================================
    
    start = EmptyOperator(
        task_id='start',
        doc_md="Inicio del pipeline de MLOps"
    )
    
    # ========================================================================
    # MONITORING GROUP
    # ========================================================================
    
    with TaskGroup('monitoring', tooltip="Monitoreo del modelo en producción") as monitoring_group:
        
        check_health = PythonOperator(
            task_id='check_api_health',
            python_callable=check_api_health,
            doc_md="Verifica que la API esté funcionando"
        )
        
        check_drift_task = PythonOperator(
            task_id='check_drift',
            python_callable=check_drift,
            doc_md="Detecta drift en datos y modelo"
        )
        
        evaluate_performance = PythonOperator(
            task_id='evaluate_performance',
            python_callable=evaluate_model_performance,
            doc_md="Evalúa el performance del modelo actual"
        )
        
        decide_retrain = PythonOperator(
            task_id='decide_retraining',
            python_callable=decide_retraining,
            doc_md="Decide si se necesita reentrenamiento"
        )
        
        check_health >> [check_drift_task, evaluate_performance] >> decide_retrain
    
    # ========================================================================
    # RETRAINING GROUP (condicional)
    # ========================================================================
    
    with TaskGroup('retraining', tooltip="Reentrenamiento del modelo") as retraining_group:
        
        train_new_model = PythonOperator(
            task_id='train_model',
            python_callable=train_model,
            doc_md="Entrena nuevo modelo con todos los algoritmos"
        )
        
        compare_models_task = PythonOperator(
            task_id='compare_models',
            python_callable=compare_models,
            doc_md="Compara nuevo modelo con el actual"
        )
        
        train_new_model >> compare_models_task
    
    # ========================================================================
    # DEPLOY GROUP (condicional)
    # ========================================================================
    
    with TaskGroup('deploy', tooltip="Deploy del modelo") as deploy_group:
        
        deploy_new_model = PythonOperator(
            task_id='deploy_model',
            python_callable=deploy_model,
            doc_md="Hace deploy del nuevo modelo"
        )
        
        push_to_s3 = PythonOperator(
            task_id='push_to_s3',
            python_callable=push_model_to_s3,
            doc_md="Push del modelo a S3 con DVC"
        )
        
        deploy_new_model >> push_to_s3
    
    # ========================================================================
    # END
    # ========================================================================
    
    end = EmptyOperator(
        task_id='end',
        doc_md="Fin del pipeline"
    )
    
    # ========================================================================
    # DEPENDENCIAS CON CONDICIONES
    # ========================================================================
    
    # Monitoreo siempre se ejecuta
    start >> monitoring_group
    
    # Reentrenamiento solo si decide_retrain retorna True
    from airflow.operators.python import ShortCircuitOperator
    
    def should_retrain_func(**context):
        """Verifica si se debe reentrenar"""
        should_retrain = context['ti'].xcom_pull(
            key='should_retrain',
            task_ids='monitoring.decide_retraining'
        )
        return bool(should_retrain) if should_retrain is not None else False
    
    should_retrain_check = ShortCircuitOperator(
        task_id='should_retrain_check',
        python_callable=should_retrain_func,
        doc_md="Verifica si se debe reentrenar"
    )
    
    # Deploy solo si el nuevo modelo es mejor
    def should_deploy_func(**context):
        """Verifica si se debe hacer deploy"""
        should_deploy = context['ti'].xcom_pull(
            key='should_deploy',
            task_ids='retraining.compare_models'
        )
        return bool(should_deploy) if should_deploy is not None else False
    
    should_deploy_check = ShortCircuitOperator(
        task_id='should_deploy_check',
        python_callable=should_deploy_func,
        doc_md="Verifica si se debe hacer deploy"
    )
    
    # Flujo completo
    monitoring_group >> should_retrain_check
    
    should_retrain_check >> retraining_group >> should_deploy_check
    
    should_deploy_check >> deploy_group >> end
    
    # Si no se reentrena o no se hace deploy, ir directo al end
    should_retrain_check >> end
    should_deploy_check >> end

