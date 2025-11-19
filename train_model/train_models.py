import os
import warnings
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import json
from datetime import datetime
import sys
import joblib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import optuna
from optuna.integration.mlflow import MLflowCallback

# ============================================================================
# SETUP DEL PROYECTO
# ============================================================================

# Agregar path del proyecto
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

# Silenciar logs de Optuna para mayor claridad
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

try:
    logger.info("=" * 80)
    logger.info("🚀 PIPELINE DE ENTRENAMIENTO CON OPTIMIZACIÓN OPTUNA")
    logger.info("=" * 80)
    logger.info("Modelos a entrenar: 7")
    logger.info("  1. Logistic Regression")
    logger.info("  2. Random Forest")
    logger.info("  3. XGBoost")
    logger.info("  4. SVM")
    logger.info("  5. LightGBM")
    logger.info("  6. Gradient Boosting")
    logger.info("  7. Extra Trees")
    logger.info("=" * 80)
    
    # ========================================================================
    # CARGA DE CONFIGURACIÓN Y SETUP DE DIRECTORIOS
    # ========================================================================
    
    logger.info("\n📋 Configurando proyecto...")
    config = load_config()
    
    # Directorios
    project_root = Path(__file__).resolve().parent.parent
    mlruns_dir = project_root / "mlruns"
    models_dir = project_root / "models"
    
    mlruns_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    
    logger.info("✓ Directorios configurados:")
    logger.info("  - Project root: %s", project_root)
    logger.info("  - MLruns: %s", mlruns_dir)
    logger.info("  - Models: %s", models_dir)
    
    # ========================================================================
    # CARGA Y PREPROCESAMIENTO DE DATOS
    # ========================================================================
    
    logger.info("\n📂 Cargando y procesando datos...")
    raw_path = project_root / config["data"]["raw_path"]
    
    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {raw_path}")
    
    data = pd.read_csv(raw_path)
    logger.info("✓ Dataset cargado: %d filas, %d columnas", data.shape[0], data.shape[1])
    
    # Verificar columnas necesarias
    required_cols = ['short_description', 'close_notes', 'etiqueta']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el dataset: {missing_cols}")
    
    # Preprocesamiento
    logger.info("🔄 Preprocesando texto...")
    data["clean_short_description_stem"] = data["short_description"].apply(preprocess_text)
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
    logger.info("✓ Clases en entrenamiento: %s", y_train.value_counts().to_dict())
    
    # ========================================================================
    # CONFIGURACIÓN DE MLFLOW
    # ========================================================================
    
    logger.info("\n🔬 Configurando MLflow...")
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    experiment_name = config["mlflow_tracking"]["experiment_name"]
    mlflow.set_experiment(experiment_name)
    
    logger.info("✓ Tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("✓ Experimento: %s", experiment_name)
    
    # ========================================================================
    # VARIABLES GLOBALES PARA TRACKING
    # ========================================================================
    
    mejor_modelo = None
    mejor_nombre = None
    mejor_f1 = 0.0
    mejor_run_id = None
    mejor_params = None
    resultados_modelos = {}
    
    logger.info("\n" + "=" * 80)
    logger.info("🎯 INICIANDO OPTIMIZACIÓN DE MODELOS")
    logger.info("=" * 80)
    
    # ========================================================================
    # MODELO 1: LOGISTIC REGRESSION
    # ========================================================================
    
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
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LogisticRegression(**params))
        ])
        
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    study_lr = optuna.create_study(direction='maximize', study_name='LogisticRegression')
    study_lr.optimize(objective_lr, n_trials=20, show_progress_bar=True)
    
    logger.info("✓ Mejores hiperparámetros: %s", study_lr.best_params)
    logger.info("✓ Mejor F1 (CV): %.4f", study_lr.best_value)
    
    # Entrenar con mejores parámetros
    best_lr_params = study_lr.best_params.copy()
    best_lr_params['max_iter'] = 1000
    best_lr_params['random_state'] = 42
    
    pipe_lr = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
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
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
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
    
    # ========================================================================
    # MODELO 2: RANDOM FOREST
    # ========================================================================
    
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
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', RandomForestClassifier(**params))
        ])
        
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    study_rf = optuna.create_study(direction='maximize', study_name='RandomForest')
    study_rf.optimize(objective_rf, n_trials=20, show_progress_bar=True)
    
    logger.info("✓ Mejores hiperparámetros: %s", study_rf.best_params)
    logger.info("✓ Mejor F1 (CV): %.4f", study_rf.best_value)
    
    best_rf_params = study_rf.best_params
    pipe_rf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
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
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
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
    
    # ========================================================================
    # MODELO 3: XGBOOST
    # ========================================================================
    
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
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', XGBClassifier(**params))
        ])
        
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    study_xgb = optuna.create_study(direction='maximize', study_name='XGBoost')
    study_xgb.optimize(objective_xgb, n_trials=20, show_progress_bar=True)
    
    logger.info("✓ Mejores hiperparámetros: %s", study_xgb.best_params)
    logger.info("✓ Mejor F1 (CV): %.4f", study_xgb.best_value)
    
    best_xgb_params = study_xgb.best_params
    pipe_xgb = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
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
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
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
    
    # ========================================================================
    # MODELO 4: SVM (SUPPORT VECTOR MACHINE)
    # ========================================================================
    
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
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', SVC(**params))
        ])
        
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    study_svm = optuna.create_study(direction='maximize', study_name='SVM')
    study_svm.optimize(objective_svm, n_trials=15, show_progress_bar=True)
    
    logger.info("✓ Mejores hiperparámetros: %s", study_svm.best_params)
    logger.info("✓ Mejor F1 (CV): %.4f", study_svm.best_value)
    
    best_svm_params = study_svm.best_params
    pipe_svm = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
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
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
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
    
    # ========================================================================
    # MODELO 5: LIGHTGBM
    # ========================================================================
    
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
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', LGBMClassifier(**params))
        ])
        
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    study_lgbm = optuna.create_study(direction='maximize', study_name='LightGBM')
    study_lgbm.optimize(objective_lgbm, n_trials=20, show_progress_bar=True)
    
    logger.info("✓ Mejores hiperparámetros: %s", study_lgbm.best_params)
    logger.info("✓ Mejor F1 (CV): %.4f", study_lgbm.best_value)
    
    best_lgbm_params = study_lgbm.best_params
    pipe_lgbm = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
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
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
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
    
    # ========================================================================
    # MODELO 6: GRADIENT BOOSTING
    # ========================================================================
    
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
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', GradientBoostingClassifier(**params))
        ])
        
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    study_gb = optuna.create_study(direction='maximize', study_name='GradientBoosting')
    study_gb.optimize(objective_gb, n_trials=20, show_progress_bar=True)
    
    logger.info("✓ Mejores hiperparámetros: %s", study_gb.best_params)
    logger.info("✓ Mejor F1 (CV): %.4f", study_gb.best_value)
    
    best_gb_params = study_gb.best_params
    pipe_gb = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
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
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
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
    
    # ========================================================================
    # MODELO 7: EXTRA TREES
    # ========================================================================
    
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
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', ExtraTreesClassifier(**params))
        ])
        
        scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_weighted', n_jobs=-1)
        return scores.mean()
    
    study_et = optuna.create_study(direction='maximize', study_name='ExtraTrees')
    study_et.optimize(objective_et, n_trials=20, show_progress_bar=True)
    
    logger.info("✓ Mejores hiperparámetros: %s", study_et.best_params)
    logger.info("✓ Mejor F1 (CV): %.4f", study_et.best_value)
    
    best_et_params = study_et.best_params
    pipe_et = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
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
            "accuracy": acc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
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
    
    # ========================================================================
    # GUARDAR SOLO EL MEJOR MODELO
    # ========================================================================
    
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
            "hyperparameters": mejor_params,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "all_results": resultados_modelos
        }
        
        metadata_path = models_dir / "best_model_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info("✓ Metadata guardada en: %s", metadata_path)
        
        # ====================================================================
        # RESUMEN FINAL
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info("🎯 RESUMEN FINAL DE ENTRENAMIENTO")
        logger.info("=" * 80)
        logger.info("\n🏆 MEJOR MODELO:")
        logger.info("  Nombre:       %s", mejor_nombre)
        logger.info("  F1-Score:     %.4f", mejor_f1)
        logger.info("  Run ID:       %s", mejor_run_id)
        logger.info("  Archivo:      %s", modelo_path.name)
        
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
        
    else:
        logger.error("\n" + "=" * 80)
        logger.error("❌ ERROR: No se pudo entrenar ningún modelo correctamente")
        logger.error("=" * 80)

except FileNotFoundError as e:
    logger.error("\n❌ Error: Archivo no encontrado")
    logger.error("   %s", str(e))
    logger.error("\nVerifica que:")
    logger.error("  1. El dataset existe en la ruta configurada")
    logger.error("  2. El archivo config.yaml está en la raíz del proyecto")
    
except ValueError as e:
    logger.error("\n❌ Error: Datos inválidos")
    logger.error("   %s", str(e))
    logger.error("\nVerifica que:")
    logger.error("  1. El dataset tiene las columnas requeridas")
    logger.error("  2. Hay suficientes datos para entrenar")

except Exception as e:
    logger.exception("\n❌ Error crítico durante el entrenamiento:")
    logger.exception("   %s", str(e))