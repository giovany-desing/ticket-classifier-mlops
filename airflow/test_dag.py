#!/usr/bin/env python3
"""
Script para probar DAGs de Airflow localmente sin ejecutar el scheduler.

Útil para verificar sintaxis y lógica antes de ejecutar en Airflow.
"""

import sys
from pathlib import Path

# Agregar raíz del proyecto al path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

def test_dag_syntax(dag_file):
    """Prueba la sintaxis de un DAG"""
    print(f"🔍 Probando sintaxis de {dag_file}...")
    
    try:
        # Intentar importar el DAG
        import importlib.util
        spec = importlib.util.spec_from_file_location("dag_module", dag_file)
        dag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dag_module)
        
        # Verificar que hay un DAG definido
        dags = [obj for obj in dag_module.__dict__.values() 
                if hasattr(obj, 'dag_id')]
        
        if dags:
            for dag in dags:
                print(f"✅ DAG '{dag.dag_id}' cargado correctamente")
                print(f"   - Schedule: {dag.schedule_interval}")
                print(f"   - Tasks: {len(dag.tasks)}")
                print(f"   - Tags: {dag.tags}")
        else:
            print("⚠️  No se encontró ningún DAG en el archivo")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Error de sintaxis: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dag_tasks(dag_file):
    """Prueba que las tareas del DAG sean válidas"""
    print(f"\n🔍 Probando tareas de {dag_file}...")
    
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dag_module", dag_file)
        dag_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dag_module)
        
        dags = [obj for obj in dag_module.__dict__.values() 
                if hasattr(obj, 'dag_id')]
        
        for dag in dags:
            print(f"\n📋 DAG: {dag.dag_id}")
            print(f"   Tareas ({len(dag.tasks)}):")
            
            for task in dag.tasks:
                print(f"   - {task.task_id} ({task.__class__.__name__})")
                
                # Verificar dependencias
                upstream = list(task.upstream_task_ids)
                downstream = list(task.downstream_task_ids)
                
                if upstream:
                    print(f"     ↑ Depende de: {', '.join(upstream)}")
                if downstream:
                    print(f"     ↓ Sigue: {', '.join(downstream)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("🧪 PRUEBA DE DAGs DE AIRFLOW")
    print("=" * 80)
    print()
    
    dags_dir = Path(__file__).parent / "dags"
    
    if not dags_dir.exists():
        print(f"❌ Directorio de DAGs no encontrado: {dags_dir}")
        sys.exit(1)
    
    dag_files = list(dags_dir.glob("*.py"))
    
    if not dag_files:
        print(f"⚠️  No se encontraron archivos DAG en {dags_dir}")
        sys.exit(1)
    
    print(f"📁 Directorio: {dags_dir}")
    print(f"📄 Archivos encontrados: {len(dag_files)}")
    print()
    
    all_passed = True
    
    for dag_file in dag_files:
        if dag_file.name == "__init__.py":
            continue
            
        print(f"\n{'=' * 80}")
        print(f"📄 {dag_file.name}")
        print('=' * 80)
        
        # Probar sintaxis
        syntax_ok = test_dag_syntax(dag_file)
        
        if syntax_ok:
            # Probar tareas
            tasks_ok = test_dag_tasks(dag_file)
            all_passed = all_passed and tasks_ok
        else:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ TODAS LAS PRUEBAS PASARON")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON")
        print("=" * 80)
        sys.exit(1)
















