"""
API de inferencia para el modelo de clasificación de tickets.

Esta API:
- Expone endpoint para hacer predicciones
- Registra todas las predicciones para monitoreo
- Permite evaluación con labels verdaderos
- Integra con sistema de monitoreo

Autor: Sistema MLOps
Fecha: 2024
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Agregar raíz del proyecto al path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.preprocessing_data import preprocess_text, load_config
from utils.monitoring import PredictionLogger, DriftDetector

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ticket Classifier API",
    description="API para clasificación de tickets con monitoreo",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CARGA DE MODELO Y RECURSOS
# ============================================================================

MODEL_PATH = project_root / "models" / "best_model.pkl"
MODEL_METADATA_PATH = project_root / "models" / "best_model_metadata.json"
MODEL = None
MODEL_METADATA = None
PREDICTION_LOGGER = PredictionLogger(log_dir=str(project_root / "monitoring" / "logs"))
REFERENCE_DATA = None
DRIFT_DETECTOR = None

def load_model():
    """Carga el modelo y recursos necesarios"""
    global MODEL, MODEL_METADATA, REFERENCE_DATA, DRIFT_DETECTOR
    
    try:
        # Cargar modelo
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {MODEL_PATH}")
        
        MODEL = joblib.load(MODEL_PATH)
        logger.info(f"✓ Modelo cargado desde: {MODEL_PATH}")
        
        # Cargar metadata
        if MODEL_METADATA_PATH.exists():
            with open(MODEL_METADATA_PATH, 'r', encoding='utf-8') as f:
                MODEL_METADATA = json.load(f)
            logger.info(f"✓ Metadata cargada: {MODEL_METADATA.get('model_name')}")
        
        # Cargar datos de referencia para drift detection
        try:
            config = load_config()
            dataset_path = project_root / config["data"]["raw_path"] / config["data"]["dataset_name"]
            
            if dataset_path.exists():
                ref_data = pd.read_csv(dataset_path)
                
                # Preprocesar datos de referencia
                ref_data['clean_short'] = ref_data['short_description'].apply(preprocess_text)
                ref_data['clean_close'] = ref_data['close_notes'].apply(preprocess_text)
                ref_data['processed_text'] = ref_data['clean_short'] + " " + ref_data['clean_close']
                ref_data['text_length'] = ref_data['processed_text'].str.len()
                ref_data['label'] = ref_data['etiqueta']
                
                REFERENCE_DATA = ref_data
                DRIFT_DETECTOR = DriftDetector(REFERENCE_DATA)
                logger.info("✓ Drift detector inicializado con datos de referencia")
        except Exception as e:
            logger.warning(f"No se pudo inicializar drift detector: {e}")
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {e}")
        raise

# Cargar modelo al iniciar
@app.on_event("startup")
async def startup_event():
    load_model()

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class PredictionRequest(BaseModel):
    """Request para predicción individual"""
    short_description: str = Field(..., description="Descripción corta del ticket")
    close_notes: Optional[str] = Field(None, description="Notas de cierre del ticket")
    true_label: Optional[str] = Field(None, description="Label verdadero (para evaluación)")

class BatchPredictionRequest(BaseModel):
    """Request para predicción en batch"""
    tickets: List[PredictionRequest] = Field(..., description="Lista de tickets a predecir")

class PredictionResponse(BaseModel):
    """Response de predicción"""
    prediction: str = Field(..., description="Clase predicha")
    probability: float = Field(..., description="Probabilidad de la predicción")
    probabilities: Dict[str, float] = Field(..., description="Probabilidades por clase")
    timestamp: str = Field(..., description="Timestamp de la predicción")

# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check"""
    return {
        "status": "online",
        "model_loaded": MODEL is not None,
        "model_name": MODEL_METADATA.get("model_name") if MODEL_METADATA else None,
        "model_f1_score": MODEL_METADATA.get("f1_score") if MODEL_METADATA else None
    }

@app.get("/health")
async def health():
    """Health check detallado"""
    return {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "drift_detector_ready": DRIFT_DETECTOR is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """
    Predice la clase de un ticket individual.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preprocesar texto
        clean_short = preprocess_text(request.short_description)
        clean_close = preprocess_text(request.close_notes or "")
        combined_text = clean_short + " " + clean_close
        
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="Texto vacío después de preprocesamiento")
        
        # Predecir
        prediction = MODEL.predict([combined_text])[0]
        
        # Obtener probabilidades
        if hasattr(MODEL, 'predict_proba'):
            probas = MODEL.predict_proba([combined_text])[0]
            classes = MODEL.classes_ if hasattr(MODEL, 'classes_') else None
            
            if classes is not None:
                proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, probas)}
                max_proba = float(np.max(probas))
            else:
                proba_dict = {prediction: 1.0}
                max_proba = 1.0
        else:
            proba_dict = {prediction: 1.0}
            max_proba = 1.0
        
        response = PredictionResponse(
            prediction=str(prediction),
            probability=max_proba,
            probabilities=proba_dict,
            timestamp=datetime.now().isoformat()
        )
        
        # Log en background
        background_tasks.add_task(
            PREDICTION_LOGGER.log_prediction,
            text=combined_text,
            prediction=str(prediction),
            probability=max_proba,
            true_label=request.true_label
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Predice múltiples tickets en batch.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        results = []
        
        for ticket in request.tickets:
            clean_short = preprocess_text(ticket.short_description)
            clean_close = preprocess_text(ticket.close_notes or "")
            combined_text = clean_short + " " + clean_close
            
            if not combined_text.strip():
                results.append({
                    "error": "Texto vacío",
                    "prediction": None
                })
                continue
            
            prediction = MODEL.predict([combined_text])[0]
            
            if hasattr(MODEL, 'predict_proba'):
                probas = MODEL.predict_proba([combined_text])[0]
                max_proba = float(np.max(probas))
            else:
                max_proba = 1.0
            
            results.append({
                "prediction": str(prediction),
                "probability": max_proba,
                "true_label": ticket.true_label
            })
            
            # Log
            PREDICTION_LOGGER.log_prediction(
                text=combined_text,
                prediction=str(prediction),
                probability=max_proba,
                true_label=ticket.true_label
            )
        
        return {
            "total": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/drift")
async def check_drift():
    """
    Verifica si hay drift en los datos recientes.
    """
    if DRIFT_DETECTOR is None:
        raise HTTPException(status_code=503, detail="Drift detector no inicializado")
    
    try:
        # Obtener predicciones recientes (últimas 24 horas)
        recent_predictions = PREDICTION_LOGGER.get_recent_predictions(hours=24)
        
        if recent_predictions.empty:
            return {
                "status": "no_data",
                "message": "No hay datos recientes para analizar"
            }
        
        # Preparar datos para drift detection
        recent_data = pd.DataFrame({
            'text_length': recent_predictions['text'].str.len(),
            'prediction': recent_predictions['prediction']
        })
        
        # Agregar processed_text si es necesario
        if 'text' in recent_predictions.columns:
            recent_data['processed_text'] = recent_predictions['text']
        
        # Detectar drift
        drift_results = DRIFT_DETECTOR.detect_data_drift(recent_data)
        
        # Si hay labels verdaderos, detectar concept drift
        if 'true_label' in recent_predictions.columns and recent_predictions['true_label'].notna().any():
            concept_drift = DRIFT_DETECTOR.detect_concept_drift(
                predictions=recent_predictions['prediction'].tolist(),
                true_labels=recent_predictions['true_label'].tolist()
            )
            drift_results['concept_drift'] = concept_drift
        
        return {
            "timestamp": datetime.now().isoformat(),
            "data_points_analyzed": len(recent_predictions),
            "drift_detected": drift_results.get('drift_detected', False),
            "drift_score": drift_results.get('drift_score', 0.0),
            "details": drift_results
        }
        
    except Exception as e:
        logger.error(f"Error verificando drift: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/monitoring/metrics")
async def get_metrics():
    """
    Obtiene métricas de monitoreo recientes.
    """
    try:
        daily_metrics = PREDICTION_LOGGER.compute_daily_metrics()
        return daily_metrics
    except Exception as e:
        logger.error(f"Error obteniendo métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/monitoring/save-metrics")
async def save_metrics():
    """
    Guarda métricas diarias (llamar periódicamente, ej: cada hora).
    """
    try:
        metrics = PREDICTION_LOGGER.save_daily_metrics()
        return {
            "status": "saved",
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Error guardando métricas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

