"""
Script de entrenamiento con optimización de hiperparámetros usando Optuna.
Optimizado para funcionar en local y CI/CD (GitHub Actions).
Incluye versionamiento automático con DVC y push a S3.

Modelos incluidos:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. SVM
5. LightGBM
6. Gradient Boosting
7. Extra Trees

Autor: Tu nombre
Fecha: 2024
Versión: 2.0 (con DVC integration)
"""

import os
import sys
import warnings
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np
import joblib

import mlflow
import mlflow.sklearn

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna

# ============================================================================
# CONFIGURACIÓN DE NLTK (CRÍTICO PARA CI/CD)
# ============================================================================

import nltk

def setup_nltk():
    """Descargar recursos de NLTK necesarios para preprocesamiento"""
    required_packages = ['punkt', 'punkt_tab', 'stopwords']
    
    for package in required_packages:
        try:
            if package == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
            elif package == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif package == 'stopwords':
                nltk.data.find('corpora/stopwords')
        except LookupError:
            print(f"📥 Descargando {package}...")
            nltk.download(package, quiet=True)
    
    print("✓ Recursos de NLTK verificados")

# Ejecutar setup de NLTK
setup_nltk()

# ============================================================================
# SETUP DEL PROYECTO
# ============================================================================

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.preprocessing_data import preprocess_text, load_config

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Silenciar logs verbosos de Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# DETECCIÓN DE ENTORNO CI/CD
# ============================================================================

IS_CI = os.getenv('CI', 'false').lower() == 'true'
IS_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', 'false').lower() == 'true'
IS_CI_ENV = IS_CI or IS_GITHUB_ACTIONS

# Configuración adaptativa según el entorno
if IS_CI_ENV:
    logger.info("=" * 80)
    logger.info("🤖 EJECUTANDO EN ENTORNO CI/CD (GitHub Actions)")
    logger.info("=" * 80)
    CONFIG = {
        'n_trials': 10,           # Menos trials para rapidez
        'cv_folds': 2,            # Menos folds
        'show_progress': False,   # Sin progress bars (causan errores en CI)
        'n_jobs': 2,              # GitHub Actions tiene 2 cores
        'max_features': 5000,
        'verbose': False
    }
else:
    logger.info("=" * 80)
    logger.info("💻 EJECUTANDO EN ENTORNO LOCAL")
    logger.info("=" * 80)
    CONFIG = {
        'n_trials': 20,
        'cv_folds': 3,
        'show_progress': True,
        'n_jobs': -1,             # Usar todos los cores disponibles
        'max_features': 5000,
        'verbose': True
    }

logger.info("Configuración de entrenamiento:")
logger.info("  - Trials por modelo: %d", CONFIG['n_trials'])
logger.info("  - Cross-validation folds: %d", CONFIG['cv_folds'])
logger.info("  - Mostrar progress bars: %s", CONFIG['show_progress'])
logger.info("  - Jobs paralelos: %s", CONFIG['n_jobs'])
logger.info("  - Max features TF-IDF: %d", CONFIG['max_features'])
logger.info("=" * 80)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def main():
    """Función principal de entrenamiento"""
    
    try:
        logger.info("\n🚀 PIPELINE DE ENTRENAMIENTO CON OPTIMIZACIÓN OPTUNA")
        logger.info("Modelos a entrenar: 7")
        logger.info("  1. Logistic Regression")
        logger.info("  2. Random Forest")
        logger.info("  3. XGBoost")
        logger.info("  4. SVM")
        logger.info("  5. LightGBM")
        logger.info("  6. Gradient Boosting")
        logger.info("  7. Extra Trees")
        logger.info("=" * 80)
        
        # ====================================================================
        # CARGA DE CONFIGURACIÓN Y SETUP DE DIRECTORIOS
        # ====================================================================
        
        logger.info("\n📋 Configurando proyecto...")
        config = load_config()
        
        # Directorios
        project_root_path = Path(__file__).resolve().parent.parent
        mlruns_dir = project_root_path / "mlruns"
        models_dir = project_root_path / "models"
        
        mlruns_dir.mkdir(exist_ok=True)
        models_dir.mkdir(exist_ok=True)
        
        logger.info("✓ Directorios configurados:")
        logger.info("  - Project root: %s", project_root_path)
        logger.info("  - MLruns: %s", mlruns_dir)
        logger.info("  - Models: %s", models_dir)
        
        # ====================================================================
        # CARGA Y PREPROCESAMIENTO DE DATOS
        # ====================================================================
        
        logger.info("\n📂 Cargando y procesando datos...")
        
        # Construir ruta al dataset
        raw_path = project_root_path / config["data"]["raw_path"]
        dataset_file = raw_path / "dataset_tickets.csv"
        
        # Verificar que el dataset existe
        if not dataset_file.exists():
            logger.error("❌ Dataset no encontrado en: %s", dataset_file)
            logger.error("Rutas buscadas:")
            logger.error("  - %s", raw_path)
            logger.error("  - %s", dataset_file)
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_file}")
        
        data = pd.read_csv(dataset_file)
        logger.info("✓ Dataset cargado: %d filas, %d columnas", data.shape[0], data.shape[1])
        
        # Verificar columnas necesarias
        required_cols = ['short_description', 'close_notes', 'etiqueta']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            logger.error("❌ Columnas faltantes: %s", missing_cols)
            logger.error("Columnas disponibles: %s", data.columns.tolist())
            raise ValueError(f"Columnas faltantes en el dataset: {missing_cols}")
        
        # Preprocesamiento
        logger.info("🔄 Preprocesando texto...")
        logger.info("  Procesando 'short_description'...")
        data["clean_short_description_stem"] = data["short_description"].apply(preprocess_text)
        
        logger.info("  Procesando 'close_notes'...")
        data["clean_close_notes_stem"] = data["close_notes"].apply(preprocess_text)
        
        # Combinar texto
        X = data["clean_short_description_stem"] + " " + data["clean_close_notes_stem"]
        y = data["etiqueta"]
        
        # Eliminar textos vacíos
        mask = X.str.strip() != ""
        X = X[mask]
        y = y[mask]
        logger.info("✓ Textos válidos después de limpieza: %d", len(X))
        
        # Verificar que hay suficientes datos
        if len(X) < 100:
            raise ValueError(f"Datos insuficientes para entrenamiento: {len(X)} muestras")
        
        # Split train/test
        logger.info("\n📊 Dividiendo datos...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        logger.info("✓ Train: %d muestras | Test: %d muestras", len(X_train), len(X_test))
        logger.info("✓ Distribución de clases en entrenamiento:")
        for clase, count in y_train.value_counts().items():
            logger.info("    %s: %d (%.1f%%)", clase, count, (count/len(y_train))*100)
        
        # ====================================================================
        # CONFIGURACIÓN DE MLFLOW
        # ====================================================================
        
        logger.info("\n🔬 Configurando MLflow...")

        
        mlruns_path = "./mlruns"
        Path(mlruns_path).mkdir(exist_ok=True)

       
        tracking_uri = f"file://{os.path.abspath(mlruns_path)}"
        mlflow.set_tracking_uri(tracking_uri)

        
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

        experiment_name = config["mlflow_tracking"]["experiment_name"]
        experiment = mlflow.set_experiment(experiment_name)

        logger.info("✓ Tracking URI: %s", mlflow.get_tracking_uri())
        logger.info("✓ Experiment ID: %s", experiment.experiment_id)
        logger.info("✓ Experimento: %s", experiment_name)
        
        ####
        logger.info("✓ Tracking URI: %s", mlflow.get_tracking_uri())
        logger.info("✓ Experimento: %s", experiment_name)
        
        # ====================================================================
        # VARIABLES GLOBALES PARA TRACKING
        # ====================================================================
        
        mejor_modelo = None
        mejor_nombre = None
        mejor_f1 = 0.0
        mejor_run_id = None
        mejor_params = None
        resultados_modelos = {}
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 INICIANDO OPTIMIZACIÓN DE MODELOS")
        logger.info("=" * 80)
        
        # ====================================================================
        # MODELO 1: LOGISTIC REGRESSION
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MODELO 1/7: Logistic Regression")
        logger.info("=" * 80)
        
        def objective_lr(trial):
            params = {
                'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'max_iter': 1000,
                'random_state': 42
            }
            
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
                ('clf', LogisticRegression(**params))
            ])
            
            scores = cross_val_score(
                pipe, X_train, y_train, 
                cv=CONFIG['cv_folds'], 
                scoring='f1_weighted', 
                n_jobs=CONFIG['n_jobs']
            )
            return scores.mean()
        
        study_lr = optuna.create_study(direction='maximize', study_name='LogisticRegression')
        study_lr.optimize(objective_lr, n_trials=CONFIG['n_trials'], show_progress_bar=CONFIG['show_progress'])
        
        logger.info("✓ Mejores hiperparámetros: %s", study_lr.best_params)
        logger.info("✓ Mejor F1 (CV): %.4f", study_lr.best_value)
        
        best_lr_params = study_lr.best_params.copy()
        best_lr_params['max_iter'] = 1000
        best_lr_params['random_state'] = 42
        
        pipe_lr = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
            ('clf', LogisticRegression(**best_lr_params))
        ])
        
        with mlflow.start_run(run_name="Logistic_Regression_Optimized") as run:
            pipe_lr.fit(X_train, y_train)
            y_pred = pipe_lr.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted")
            
            logger.info("📊 Resultados en Test Set:")
            logger.info("   Accuracy:  %.4f", acc)
            logger.info("   F1-Score:  %.4f", f1)
            logger.info("   Precision: %.4f", prec)
            logger.info("   Recall:    %.4f", rec)
            
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": prec,
                "recall_weighted": rec
            })
            mlflow.log_params(best_lr_params)
            mlflow.sklearn.log_model(
                pipe_lr,
                artifact_path="model",
                registered_model_name="LogisticRegression_Optimized"
            )
            
            resultados_modelos["Logistic_Regression"] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "params": best_lr_params,
                "run_id": run.info.run_id
            }
            
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_modelo = pipe_lr
                mejor_nombre = "Logistic_Regression"
                mejor_run_id = run.info.run_id
                mejor_params = best_lr_params
                logger.info("🏆 ¡Nuevo mejor modelo!")
        
        # ====================================================================
        # MODELO 2: RANDOM FOREST
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MODELO 2/7: Random Forest")
        logger.info("=" * 80)
        
        def objective_rf(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42
            }
            
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
                ('clf', RandomForestClassifier(**params))
            ])
            
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=CONFIG['cv_folds'],
                scoring='f1_weighted',
                n_jobs=CONFIG['n_jobs']
            )
            return scores.mean()
        
        study_rf = optuna.create_study(direction='maximize', study_name='RandomForest')
        study_rf.optimize(objective_rf, n_trials=CONFIG['n_trials'], show_progress_bar=CONFIG['show_progress'])
        
        logger.info("✓ Mejores hiperparámetros: %s", study_rf.best_params)
        logger.info("✓ Mejor F1 (CV): %.4f", study_rf.best_value)
        
        best_rf_params = study_rf.best_params
        pipe_rf = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
            ('clf', RandomForestClassifier(**best_rf_params))
        ])
        
        with mlflow.start_run(run_name="Random_Forest_Optimized") as run:
            pipe_rf.fit(X_train, y_train)
            y_pred = pipe_rf.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted")
            
            logger.info("📊 Resultados en Test Set:")
            logger.info("   Accuracy:  %.4f", acc)
            logger.info("   F1-Score:  %.4f", f1)
            logger.info("   Precision: %.4f", prec)
            logger.info("   Recall:    %.4f", rec)
            
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": prec,
                "recall_weighted": rec
            })
            mlflow.log_params(best_rf_params)
            mlflow.sklearn.log_model(
                pipe_rf,
                artifact_path="model",
                registered_model_name="RandomForest_Optimized"
            )
            
            resultados_modelos["Random_Forest"] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "params": best_rf_params,
                "run_id": run.info.run_id
            }
            
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_modelo = pipe_rf
                mejor_nombre = "Random_Forest"
                mejor_run_id = run.info.run_id
                mejor_params = best_rf_params
                logger.info("🏆 ¡Nuevo mejor modelo!")
        
        # ====================================================================
        # MODELO 3: XGBOOST
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MODELO 3/7: XGBoost")
        logger.info("=" * 80)
        
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
                ('clf', XGBClassifier(**params))
            ])
            
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=CONFIG['cv_folds'],
                scoring='f1_weighted',
                n_jobs=CONFIG['n_jobs']
            )
            return scores.mean()
        
        study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost')
        study_xgb.optimize(objective_xgb, n_trials=CONFIG['n_trials'], show_progress_bar=CONFIG['show_progress'])
        
        logger.info("✓ Mejores hiperparámetros: %s", study_xgb.best_params)
        logger.info("✓ Mejor F1 (CV): %.4f", study_xgb.best_value)
        
        best_xgb_params = study_xgb.best_params
        pipe_xgb = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
            ('clf', XGBClassifier(**best_xgb_params))
        ])
        
        with mlflow.start_run(run_name="XGBoost_Optimized") as run:
            pipe_xgb.fit(X_train, y_train)
            y_pred = pipe_xgb.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted")
            
            logger.info("📊 Resultados en Test Set:")
            logger.info("   Accuracy:  %.4f", acc)
            logger.info("   F1-Score:  %.4f", f1)
            logger.info("   Precision: %.4f", prec)
            logger.info("   Recall:    %.4f", rec)
            
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": prec,
                "recall_weighted": rec
            })
            mlflow.log_params(best_xgb_params)
            mlflow.sklearn.log_model(
                pipe_xgb,
                artifact_path="model",
                registered_model_name="XGBoost_Optimized"
            )
            
            resultados_modelos["XGBoost"] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "params": best_xgb_params,
                "run_id": run.info.run_id
            }
            
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_modelo = pipe_xgb
                mejor_nombre = "XGBoost"
                mejor_run_id = run.info.run_id
                mejor_params = best_xgb_params
                logger.info("🏆 ¡Nuevo mejor modelo!")
        
        # ====================================================================
        # MODELO 4: SVM
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MODELO 4/7: SVM (Support Vector Machine)")
        logger.info("=" * 80)
        
        def objective_svm(trial):
            params = {
                'C': trial.suggest_float('C', 0.1, 100.0, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'linear']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'probability': True,
                'random_state': 42
            }
            
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
                ('clf', SVC(**params))
            ])
            
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=CONFIG['cv_folds'],
                scoring='f1_weighted',
                n_jobs=CONFIG['n_jobs']
            )
            return scores.mean()
        
        study_svm = optuna.create_study(direction='maximize', study_name='SVM')
        study_svm.optimize(objective_svm, n_trials=15, show_progress_bar=CONFIG['show_progress'])
        
        logger.info("✓ Mejores hiperparámetros: %s", study_svm.best_params)
        logger.info("✓ Mejor F1 (CV): %.4f", study_svm.best_value)
        
        best_svm_params = study_svm.best_params
        pipe_svm = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
            ('clf', SVC(**best_svm_params))
        ])
        
        with mlflow.start_run(run_name="SVM_Optimized") as run:
            pipe_svm.fit(X_train, y_train)
            y_pred = pipe_svm.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted")
            
            logger.info("📊 Resultados en Test Set:")
            logger.info("   Accuracy:  %.4f", acc)
            logger.info("   F1-Score:  %.4f", f1)
            logger.info("   Precision: %.4f", prec)
            logger.info("   Recall:    %.4f", rec)
            
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": prec,
                "recall_weighted": rec
            })
            mlflow.log_params(best_svm_params)
            mlflow.sklearn.log_model(
                pipe_svm,
                artifact_path="model",
                registered_model_name="SVM_Optimized"
            )
            
            resultados_modelos["SVM"] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "params": best_svm_params,
                "run_id": run.info.run_id
            }
            
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_modelo = pipe_svm
                mejor_nombre = "SVM"
                mejor_run_id = run.info.run_id
                mejor_params = best_svm_params
                logger.info("🏆 ¡Nuevo mejor modelo!")
        
        # ====================================================================
        # MODELO 5: LIGHTGBM
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MODELO 5/7: LightGBM")
        logger.info("=" * 80)
        
        def objective_lgbm(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
                ('clf', LGBMClassifier(**params))
            ])
            
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=CONFIG['cv_folds'],
                scoring='f1_weighted',
                n_jobs=CONFIG['n_jobs']
            )
            return scores.mean()
        
        study_lgbm = optuna.create_study(direction='maximize', study_name='LightGBM')
        study_lgbm.optimize(objective_lgbm, n_trials=CONFIG['n_trials'], show_progress_bar=CONFIG['show_progress'])
        
        logger.info("✓ Mejores hiperparámetros: %s", study_lgbm.best_params)
        logger.info("✓ Mejor F1 (CV): %.4f", study_lgbm.best_value)
        
        best_lgbm_params = study_lgbm.best_params
        pipe_lgbm = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
            ('clf', LGBMClassifier(**best_lgbm_params))
        ])
        
        with mlflow.start_run(run_name="LightGBM_Optimized") as run:
            pipe_lgbm.fit(X_train, y_train)
            y_pred = pipe_lgbm.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted")
            
            logger.info("📊 Resultados en Test Set:")
            logger.info("   Accuracy:  %.4f", acc)
            logger.info("   F1-Score:  %.4f", f1)
            logger.info("   Precision: %.4f", prec)
            logger.info("   Recall:    %.4f", rec)
            
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": prec,
                "recall_weighted": rec
            })
            mlflow.log_params(best_lgbm_params)
            mlflow.sklearn.log_model(
                pipe_lgbm,
                artifact_path="model",
                registered_model_name="LightGBM_Optimized"
            )
            
            resultados_modelos["LightGBM"] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "params": best_lgbm_params,
                "run_id": run.info.run_id
            }
            
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_modelo = pipe_lgbm
                mejor_nombre = "LightGBM"
                mejor_run_id = run.info.run_id
                mejor_params = best_lgbm_params
                logger.info("🏆 ¡Nuevo mejor modelo!")
        
        # ====================================================================
        # MODELO 6: GRADIENT BOOSTING
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MODELO 6/7: Gradient Boosting")
        logger.info("=" * 80)
        
        def objective_gb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': 42
            }
            
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
                ('clf', GradientBoostingClassifier(**params))
            ])
            
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=CONFIG['cv_folds'],
                scoring='f1_weighted',
                n_jobs=CONFIG['n_jobs']
            )
            return scores.mean()
        
        study_gb = optuna.create_study(direction='maximize', study_name='GradientBoosting')
        study_gb.optimize(objective_gb, n_trials=CONFIG['n_trials'], show_progress_bar=CONFIG['show_progress'])
        
        logger.info("✓ Mejores hiperparámetros: %s", study_gb.best_params)
        logger.info("✓ Mejor F1 (CV): %.4f", study_gb.best_value)
        
        best_gb_params = study_gb.best_params
        pipe_gb = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
            ('clf', GradientBoostingClassifier(**best_gb_params))
        ])
        
        with mlflow.start_run(run_name="GradientBoosting_Optimized") as run:
            pipe_gb.fit(X_train, y_train)
            y_pred = pipe_gb.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted")
            
            logger.info("📊 Resultados en Test Set:")
            logger.info("   Accuracy:  %.4f", acc)
            logger.info("   F1-Score:  %.4f", f1)
            logger.info("   Precision: %.4f", prec)
            logger.info("   Recall:    %.4f", rec)
            
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": prec,
                "recall_weighted": rec
            })
            mlflow.log_params(best_gb_params)
            mlflow.sklearn.log_model(
                pipe_gb,
                artifact_path="model",
                registered_model_name="GradientBoosting_Optimized"
            )
            
            resultados_modelos["Gradient_Boosting"] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "params": best_gb_params,
                "run_id": run.info.run_id
            }
            
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_modelo = pipe_gb
                mejor_nombre = "Gradient_Boosting"
                mejor_run_id = run.info.run_id
                mejor_params = best_gb_params
                logger.info("🏆 ¡Nuevo mejor modelo!")
        
        # ====================================================================
        # MODELO 7: EXTRA TREES
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("🔍 MODELO 7/7: Extra Trees")
        logger.info("=" * 80)
        
        def objective_et(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42
            }
            
            pipe = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
                ('clf', ExtraTreesClassifier(**params))
            ])
            
            scores = cross_val_score(
                pipe, X_train, y_train,
                cv=CONFIG['cv_folds'],
                scoring='f1_weighted',
                n_jobs=CONFIG['n_jobs']
            )
            return scores.mean()
        
        study_et = optuna.create_study(direction='maximize', study_name='ExtraTrees')
        study_et.optimize(objective_et, n_trials=CONFIG['n_trials'], show_progress_bar=CONFIG['show_progress'])
        
        logger.info("✓ Mejores hiperparámetros: %s", study_et.best_params)
        logger.info("✓ Mejor F1 (CV): %.4f", study_et.best_value)
        
        best_et_params = study_et.best_params
        pipe_et = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=CONFIG['max_features'])),
            ('clf', ExtraTreesClassifier(**best_et_params))
        ])
        
        with mlflow.start_run(run_name="ExtraTrees_Optimized") as run:
            pipe_et.fit(X_train, y_train)
            y_pred = pipe_et.predict(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="weighted")
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec = recall_score(y_test, y_pred, average="weighted")
            
            logger.info("📊 Resultados en Test Set:")
            logger.info("   Accuracy:  %.4f", acc)
            logger.info("   F1-Score:  %.4f", f1)
            logger.info("   Precision: %.4f", prec)
            logger.info("   Recall:    %.4f", rec)
            
            mlflow.log_metrics({
                "accuracy": acc,
                "f1_weighted": f1,
                "precision_weighted": prec,
                "recall_weighted": rec
            })
            mlflow.log_params(best_et_params)
            mlflow.sklearn.log_model(
                pipe_et,
                artifact_path="model",
                registered_model_name="ExtraTrees_Optimized"
            )
            
            resultados_modelos["Extra_Trees"] = {
                "accuracy": float(acc),
                "f1_score": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "params": best_et_params,
                "run_id": run.info.run_id
            }
            
            if f1 > mejor_f1:
                mejor_f1 = f1
                mejor_modelo = pipe_et
                mejor_nombre = "Extra_Trees"
                mejor_run_id = run.info.run_id
                mejor_params = best_et_params
                logger.info("🏆 ¡Nuevo mejor modelo!")
        
        # ====================================================================
        # GUARDAR SOLO EL MEJOR MODELO
        # ====================================================================
        
        if mejor_modelo is not None:
            logger.info("\n" + "=" * 80)
            logger.info("💾 GUARDANDO MEJOR MODELO")
            logger.info("=" * 80)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Guardar el modelo
            modelo_path = models_dir / "best_model.pkl"
            joblib.dump(mejor_modelo, modelo_path)
            logger.info("✓ Modelo guardado en: %s", modelo_path)
            
            # Guardar metadata completa
            metadata = {
                "model_name": mejor_nombre,
                "f1_score": float(mejor_f1),
                "run_id": mejor_run_id,
                "timestamp": timestamp,
                "environment": "CI/CD" if IS_CI_ENV else "Local",
                "hyperparameters": mejor_params,
                "training_config": {
                    "n_trials": CONFIG['n_trials'],
                    "cv_folds": CONFIG['cv_folds'],
                    "max_features": CONFIG['max_features']
                },
                "training_samples": int(len(X_train)),
                "test_samples": int(len(X_test)),
                "all_results": resultados_modelos
            }
            
            metadata_path = models_dir / "best_model_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info("✓ Metadata guardada en: %s", metadata_path)
            
            # ================================================================
            # TRACKEAR CON DVC Y PUSH A S3 (SOLO EN CI)
            # ================================================================
            
            if IS_CI_ENV:
                logger.info("\n" + "=" * 80)
                logger.info("📤 VERSIONANDO MODELO CON DVC Y S3")
                logger.info("=" * 80)
                
                try:
                    # Verificar si el archivo ya está trackeado
                    dvc_file = modelo_path.with_suffix('.pkl.dvc')
                    
                    if not dvc_file.exists():
                        logger.info("🆕 Modelo no trackeado, agregando a DVC...")
                        result = subprocess.run(
                            ['dvc', 'add', str(modelo_path)],
                            cwd=project_root_path,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        logger.info("✓ Modelo agregado a DVC")
                        if result.stdout.strip():
                            logger.info("Output: %s", result.stdout.strip())
                    else:
                        logger.info("✓ Modelo ya trackeado, actualizando...")
                        result = subprocess.run(
                            ['dvc', 'add', str(modelo_path)],
                            cwd=project_root_path,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        logger.info("✓ Modelo actualizado en DVC")
                    
                    # Push a S3
                    logger.info("☁️  Pusheando modelo a S3 vía DVC...")
                    result = subprocess.run(
                        ['dvc', 'push', str(dvc_file)],
                        cwd=project_root_path,
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    logger.info("✓ Modelo versionado y pusheado exitosamente a S3")
                    if result.stdout.strip():
                        logger.info("DVC push output: %s", result.stdout.strip())
                    
                except subprocess.CalledProcessError as e:
                    logger.error("❌ Error al ejecutar comando DVC:")
                    logger.error("   stdout: %s", e.stdout)
                    logger.error("   stderr: %s", e.stderr)
                    logger.warning("⚠️  Continuando sin DVC push...")
                except FileNotFoundError:
                    logger.error("❌ DVC no está instalado o no está en el PATH")
                    logger.warning("⚠️  Continuando sin DVC push...")
                except Exception as e:
                    logger.error("❌ Error inesperado con DVC: %s", str(e))
                    logger.warning("⚠️  Continuando sin DVC push...")
            else:
                logger.info("\n⏭️  Saltando DVC push (entorno local)")
                logger.info("   Para pushear manualmente:")
                logger.info("   1. dvc add models/best_model.pkl")
                logger.info("   2. dvc push models/best_model.pkl.dvc")
            
            # ================================================================
            # RESUMEN FINAL
            # ================================================================
            
            logger.info("\n" + "=" * 80)
            logger.info("🎯 RESUMEN FINAL DE ENTRENAMIENTO")
            logger.info("=" * 80)
            logger.info("\n🏆 MEJOR MODELO:")
            logger.info("  Nombre:       %s", mejor_nombre)
            logger.info("  F1-Score:     %.4f", mejor_f1)
            logger.info("  Run ID:       %s", mejor_run_id)
            logger.info("  Archivo:      %s", modelo_path.name)
            logger.info("  Entorno:      %s", "CI/CD (GitHub Actions)" if IS_CI_ENV else "Local")
            
            logger.info("\n📊 COMPARACIÓN DE TODOS LOS MODELOS:")
            logger.info("-" * 80)
            logger.info("%-25s | %10s | %10s | %10s | %10s" % 
                       ("Modelo", "Accuracy", "F1-Score", "Precision", "Recall"))
            logger.info("-" * 80)
            
            # Ordenar por F1-Score
            sorted_results = sorted(
                resultados_modelos.items(),
                key=lambda x: x[1]['f1_score'],
                reverse=True
            )
            
            for rank, (nombre, metricas) in enumerate(sorted_results, 1):
                prefix = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
                logger.info("%s %-22s | %10.4f | %10.4f | %10.4f | %10.4f" % 
                           (prefix, nombre, 
                            metricas['accuracy'], 
                            metricas['f1_score'],
                            metricas['precision'],
                            metricas['recall']))
            
            logger.info("-" * 80)
            
            # Diferencia con segundo mejor
            if len(sorted_results) > 1:
                segundo_mejor_f1 = sorted_results[1][1]['f1_score']
                mejora = ((mejor_f1 - segundo_mejor_f1) / segundo_mejor_f1) * 100
                logger.info("\n📈 Mejora vs segundo mejor: +%.2f%%", mejora)
            
            logger.info("\n✅ Entrenamiento completado exitosamente!")
            logger.info("=" * 80)
            
            # Exit code 0 para CI/CD
            return 0
            
        else:
            logger.error("\n" + "=" * 80)
            logger.error("❌ ERROR: No se pudo entrenar ningún modelo correctamente")
            logger.error("=" * 80)
            return 1
    
    except FileNotFoundError as e:
        logger.error("\n" + "=" * 80)
        logger.error("❌ Error: Archivo no encontrado")
        logger.error("=" * 80)
        logger.error("%s", str(e))
        logger.error("\nVerifica que:")
        logger.error("  1. El dataset existe en la ruta configurada")
        logger.error("  2. El archivo config.yaml está en la raíz del proyecto")
        logger.error("  3. DVC pull se ejecutó correctamente (si usas DVC)")
        return 1
        
    except ValueError as e:
        logger.error("\n" + "=" * 80)
        logger.error("❌ Error: Datos inválidos")
        logger.error("=" * 80)
        logger.error("%s", str(e))
        logger.error("\nVerifica que:")
        logger.error("  1. El dataset tiene las columnas requeridas")
        logger.error("  2. Hay suficientes datos para entrenar")
        logger.error("  3. Las etiquetas son válidas")
        return 1
    
    except Exception as e:
        logger.exception("\n" + "=" * 80)
        logger.exception("❌ Error crítico durante el entrenamiento")
        logger.exception("=" * 80)
        logger.exception("%s", str(e))
        return 1

# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)