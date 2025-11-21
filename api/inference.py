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
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

# Agregar raíz del proyecto al path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.append(str(project_root))

from utils.preprocessing_data import preprocess_text, load_config
from utils.monitoring import PredictionLogger, DriftDetector
from utils.database import (
    update_ticket_causa,
    update_tickets_batch,
    get_tickets_pending_prediction,
    get_ticket_by_number,
    get_ticket_text_fields,
    verify_connection as verify_db_connection,
    initialize_database
)

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

# CORS - Configurar orígenes permitidos desde variable de entorno
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
ALLOWED_ORIGINS = [origin.strip() for origin in ALLOWED_ORIGINS]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key"],
)

# ============================================================================
# AUTENTICACIÓN API
# ============================================================================

# Configuración de API Key
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Keys desde variables de entorno
API_KEY = os.getenv("API_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")

async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Verifica API key para endpoints normales"""
    if not API_KEY:
        # Si no hay API_KEY configurada, permitir acceso (desarrollo)
        return "development"

    if api_key == API_KEY:
        return "user"

    raise HTTPException(
        status_code=401,
        detail="API Key inválida o no proporcionada",
        headers={"WWW-Authenticate": "ApiKey"},
    )

async def verify_admin_key(api_key: str = Security(api_key_header)) -> str:
    """Verifica API key para endpoints administrativos"""
    if not ADMIN_API_KEY:
        raise HTTPException(status_code=500, detail="ADMIN_API_KEY no configurada")

    if api_key == ADMIN_API_KEY:
        return "admin"

    raise HTTPException(
        status_code=401,
        detail="Se requiere API Key de administrador",
        headers={"WWW-Authenticate": "ApiKey"},
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
    # Inicializar base de datos (opcional, no bloquea si falla)
    try:
        initialize_database()
    except Exception as e:
        logger.warning(f"No se pudo inicializar base de datos: {e}")
        logger.info("La API funcionará sin conexión a base de datos")

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class TicketPredictionRequest(BaseModel):
    """Request para predicción de ticket individual con BD"""
    ticket_id: str = Field(
        ...,
        description="ID del ticket (debe coincidir con columna 'number' en BD)",
        min_length=1,
        max_length=100
    )
    short_description: str = Field(
        ...,
        description="Descripción corta del ticket",
        min_length=1,
        max_length=10000
    )
    close_notes: Optional[str] = Field(None, description="Notas de cierre del ticket", max_length=50000)

class BatchTicketPredictionRequest(BaseModel):
    """Request para predicción en batch con BD"""
    tickets: List[TicketPredictionRequest] = Field(
        ...,
        description="Lista de tickets a predecir y actualizar en BD",
        min_items=1,
        max_items=100
    )

class UpdateDBRequest(BaseModel):
    """Request para actualizar ticket en BD"""
    ticket_number: str = Field(..., description="Número del ticket en BD")
    short_description: str = Field(..., description="Descripción corta del ticket")
    close_notes: Optional[str] = Field(None, description="Notas de cierre del ticket")

# Modelos legacy (mantener para compatibilidad)
class PredictionRequest(BaseModel):
    """Request para predicción individual (sin BD)"""
    short_description: str = Field(
        ...,
        description="Descripción corta del ticket",
        min_length=1,
        max_length=10000
    )
    close_notes: Optional[str] = Field(None, description="Notas de cierre del ticket", max_length=50000)
    true_label: Optional[str] = Field(None, description="Label verdadero (para evaluación)", max_length=100)

class BatchPredictionRequest(BaseModel):
    """Request para predicción en batch (sin BD)"""
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
async def predict(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
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

@app.post("/predict/ticket")
async def predict_ticket(
    request: TicketPredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Predice un ticket individual y actualiza automáticamente en la BD.
    
    El ticket_id del JSON debe coincidir con la columna 'number' en la BD.
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
                proba_dict = {str(prediction): 1.0}
                max_proba = 1.0
        else:
            proba_dict = {str(prediction): 1.0}
            max_proba = 1.0
        
        # Actualizar en BD (ticket_id se mapea a columna 'number')
        update_result = update_ticket_causa(
            ticket_number=request.ticket_id,  # ticket_id del JSON → columna 'number' en BD
            causa=str(prediction),
            confidence=max_proba,
            metadata={
                'probabilities': proba_dict,
                'predicted_at': datetime.now().isoformat(),
                'model_name': MODEL_METADATA.get('model_name') if MODEL_METADATA else None
            }
        )
        
        # Log en background
        background_tasks.add_task(
            PREDICTION_LOGGER.log_prediction,
            text=combined_text,
            prediction=str(prediction),
            probability=max_proba,
            true_label=None
        )
        
        return {
            "ticket_id": request.ticket_id,
            "prediction": str(prediction),
            "probability": max_proba,
            "probabilities": proba_dict,
            "database_update": update_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción de ticket: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/tickets/batch")
async def predict_tickets_batch(
    request: BatchTicketPredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Predice múltiples tickets en batch y actualiza automáticamente en la BD.
    
    Cada ticket_id del JSON debe coincidir con la columna 'number' en la BD.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        results = []
        updates = []
        
        for ticket in request.tickets:
            # Preprocesar texto
            clean_short = preprocess_text(ticket.short_description)
            clean_close = preprocess_text(ticket.close_notes or "")
            combined_text = clean_short + " " + clean_close
            
            if not combined_text.strip():
                results.append({
                    "ticket_id": ticket.ticket_id,
                    "status": "error",
                    "error": "Texto vacío después de preprocesamiento",
                    "prediction": None
                })
                continue
            
            # Predecir
            prediction = MODEL.predict([combined_text])[0]
            
            # Obtener probabilidades
            if hasattr(MODEL, 'predict_proba'):
                probas = MODEL.predict_proba([combined_text])[0]
                max_proba = float(np.max(probas))
                classes = MODEL.classes_ if hasattr(MODEL, 'classes_') else None
                if classes is not None:
                    proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, probas)}
                else:
                    proba_dict = {str(prediction): max_proba}
            else:
                max_proba = 1.0
                proba_dict = {str(prediction): max_proba}
            
            # Agregar a resultados
            results.append({
                "ticket_id": ticket.ticket_id,
                "prediction": str(prediction),
                "probability": max_proba,
                "probabilities": proba_dict,
                "status": "success"
            })
            
            # Agregar a updates para BD
            updates.append({
                'ticket_number': ticket.ticket_id,  # ticket_id → columna 'number' en BD
                'causa': str(prediction),
                'confidence': max_proba,
                'metadata': {
                    'probabilities': proba_dict,
                    'processed_at': datetime.now().isoformat()
                }
            })
            
            # Log en background
            background_tasks.add_task(
                PREDICTION_LOGGER.log_prediction,
                text=combined_text,
                prediction=str(prediction),
                probability=max_proba,
                true_label=None
            )
        
        # Actualizar en BD en batch
        batch_result = None
        if updates:
            batch_result = update_tickets_batch(updates)
            
            # Agregar información de BD a resultados
            success_tickets = {r['ticket_number']: r for r in batch_result['success']}
            failed_tickets = {r['ticket_number']: r for r in batch_result['failed']}
            
            for result in results:
                ticket_id = result['ticket_id']
                if ticket_id in success_tickets:
                    result['database_update'] = {'success': True}
                elif ticket_id in failed_tickets:
                    result['database_update'] = {'success': False, 'error': failed_tickets[ticket_id].get('error')}
        
        return {
            "total": len(results),
            "processed": len([r for r in results if r.get('status') == 'success']),
            "failed": len([r for r in results if r.get('status') == 'error']),
            "results": results,
            "batch_update_summary": {
                "success": len(batch_result['success']) if batch_result else 0,
                "failed": len(batch_result['failed']) if batch_result else 0
            } if batch_result else None,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(
    request: BatchPredictionRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Predice múltiples tickets en batch (sin actualizar BD - legacy endpoint).
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

# ============================================================================
# ENDPOINTS DE BASE DE DATOS
# ============================================================================

@app.post("/predict/from-db/{ticket_number}")
async def predict_from_db(ticket_number: str, background_tasks: BackgroundTasks):
    """
    Obtiene un ticket de la BD, predice su causa y la actualiza automáticamente.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Obtener ticket de la BD
        ticket = get_ticket_by_number(ticket_number)
        
        if not ticket:
            raise HTTPException(status_code=404, detail=f"Ticket {ticket_number} no encontrado")
        
        # Extraer campos de texto
        short_description, close_notes = get_ticket_text_fields(ticket)
        
        if not short_description and not close_notes:
            raise HTTPException(status_code=400, detail="Ticket sin texto para predecir")
        
        # Preprocesar y predecir
        clean_short = preprocess_text(short_description)
        clean_close = preprocess_text(close_notes or "")
        combined_text = clean_short + " " + clean_close
        
        if not combined_text.strip():
            raise HTTPException(status_code=400, detail="Texto vacío después de preprocesamiento")
        
        # Predecir
        prediction = MODEL.predict([combined_text])[0]
        
        # Obtener probabilidades
        if hasattr(MODEL, 'predict_proba'):
            probas = MODEL.predict_proba([combined_text])[0]
            max_proba = float(np.max(probas))
            classes = MODEL.classes_ if hasattr(MODEL, 'classes_') else None
            if classes is not None:
                proba_dict = {str(cls): float(prob) for cls, prob in zip(classes, probas)}
            else:
                proba_dict = {str(prediction): max_proba}
        else:
            max_proba = 1.0
            proba_dict = {str(prediction): max_proba}
        
        # Actualizar en BD
        update_result = update_ticket_causa(
            ticket_number=ticket_number,
            causa=str(prediction),
            confidence=max_proba,
            metadata={
                'probabilities': proba_dict,
                'predicted_at': datetime.now().isoformat(),
                'model_name': MODEL_METADATA.get('model_name') if MODEL_METADATA else None
            }
        )
        
        # Log en background
        background_tasks.add_task(
            PREDICTION_LOGGER.log_prediction,
            text=combined_text,
            prediction=str(prediction),
            probability=max_proba,
            true_label=None
        )
        
        return {
            "ticket_number": ticket_number,
            "prediction": str(prediction),
            "probability": max_proba,
            "probabilities": proba_dict,
            "database_update": update_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en predicción desde BD: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/update-db")
async def predict_and_update_db(
    request: UpdateDBRequest,
    background_tasks: BackgroundTasks
):
    """
    Predice y actualiza directamente en la BD (similar a actualizar_causa_ticket).
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Preprocesar y predecir
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
            max_proba = float(np.max(probas))
        else:
            max_proba = 1.0
        
        # Actualizar en BD
        update_result = update_ticket_causa(
            ticket_number=request.ticket_number,
            causa=str(prediction),
            confidence=max_proba
        )
        
        # Log en background
        background_tasks.add_task(
            PREDICTION_LOGGER.log_prediction,
            text=combined_text,
            prediction=str(prediction),
            probability=max_proba,
            true_label=None
        )
        
        return {
            "success": update_result['success'],
            "ticket_number": request.ticket_number,
            "prediction": str(prediction),
            "probability": max_proba,
            "database_update": update_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error en predict y update: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/process-pending")
async def process_pending_tickets(
    limit: int = 100,
    background_tasks: BackgroundTasks = None
):
    """
    Procesa tickets pendientes de predicción desde la BD.
    Obtiene tickets sin causa, predice y actualiza en batch.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    try:
        # Obtener tickets pendientes
        tickets = get_tickets_pending_prediction(limit=limit)
        
        if not tickets:
            return {
                "message": "No hay tickets pendientes",
                "processed": 0,
                "results": []
            }
        
        # Procesar cada ticket
        updates = []
        results = []
        
        for ticket in tickets:
            ticket_number = ticket.get('number')
            if not ticket_number:
                continue
            
            # Extraer texto
            short_description, close_notes = get_ticket_text_fields(ticket)
            
            if not short_description and not close_notes:
                results.append({
                    "ticket_number": ticket_number,
                    "status": "skipped",
                    "reason": "Sin texto para predecir"
                })
                continue
            
            # Preprocesar y predecir
            clean_short = preprocess_text(short_description)
            clean_close = preprocess_text(close_notes or "")
            combined_text = clean_short + " " + clean_close
            
            if not combined_text.strip():
                results.append({
                    "ticket_number": ticket_number,
                    "status": "skipped",
                    "reason": "Texto vacío después de preprocesamiento"
                })
                continue
            
            # Predecir
            prediction = MODEL.predict([combined_text])[0]
            
            if hasattr(MODEL, 'predict_proba'):
                probas = MODEL.predict_proba([combined_text])[0]
                max_proba = float(np.max(probas))
            else:
                max_proba = 1.0
            
            # Agregar a updates
            updates.append({
                'ticket_number': ticket_number,
                'causa': str(prediction),
                'confidence': max_proba,
                'metadata': {
                    'processed_at': datetime.now().isoformat()
                }
            })
            
            # Log en background
            if background_tasks:
                background_tasks.add_task(
                    PREDICTION_LOGGER.log_prediction,
                    text=combined_text,
                    prediction=str(prediction),
                    probability=max_proba,
                    true_label=None
                )
        
        # Actualizar en batch
        if updates:
            batch_result = update_tickets_batch(updates)
            
            return {
                "message": f"Procesados {len(updates)} tickets",
                "total_pending": len(tickets),
                "processed": len(batch_result['success']),
                "failed": len(batch_result['failed']),
                "results": batch_result
            }
        else:
            return {
                "message": "No se pudo procesar ningún ticket",
                "total_pending": len(tickets),
                "processed": 0,
                "results": results
            }
        
    except Exception as e:
        logger.error(f"Error procesando tickets pendientes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/db/health")
async def db_health():
    """
    Verifica la conexión a la base de datos.
    """
    try:
        is_connected = verify_db_connection()
        return {
            "database_connected": is_connected,
            "status": "healthy" if is_connected else "unhealthy"
        }
    except Exception as e:
        return {
            "database_connected": False,
            "status": "error",
            "error": str(e)
        }

@app.get("/db/tickets/pending")
async def get_pending_tickets(
    limit: int = 100,
    api_key: str = Depends(verify_api_key)
):
    """
    Obtiene tickets pendientes de predicción.
    """
    try:
        tickets = get_tickets_pending_prediction(limit=limit)
        return {
            "total": len(tickets),
            "tickets": tickets
        }
    except Exception as e:
        logger.error(f"Error obteniendo tickets pendientes: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# ENDPOINTS ADMINISTRATIVOS
# ============================================================================

@app.post("/admin/reload-model")
async def reload_model(api_key: str = Depends(verify_admin_key)):
    """
    Recarga el modelo en memoria sin reiniciar la API.
    Requiere ADMIN_API_KEY.

    Uso:
        curl -X POST http://localhost:8000/admin/reload-model \
             -H "X-API-Key: tu_admin_api_key"
    """
    global MODEL, MODEL_METADATA, DRIFT_DETECTOR, REFERENCE_DATA

    try:
        # Verificar que el modelo existe
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Modelo no encontrado en {MODEL_PATH}"
            )

        # Guardar referencia al modelo anterior (para rollback si falla)
        old_model = MODEL
        old_metadata = MODEL_METADATA

        # Cargar nuevo modelo
        logger.info("Recargando modelo...")
        new_model = joblib.load(MODEL_PATH)

        # Cargar nueva metadata
        new_metadata = None
        if MODEL_METADATA_PATH.exists():
            with open(MODEL_METADATA_PATH, 'r') as f:
                new_metadata = json.load(f)

        # Verificar que el modelo funciona con una predicción de prueba
        test_text = "prueba de predicción"
        try:
            _ = new_model.predict([test_text])
        except Exception as e:
            logger.error(f"Nuevo modelo falló en prueba: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Nuevo modelo falló validación: {str(e)}"
            )

        # Actualizar variables globales
        MODEL = new_model
        MODEL_METADATA = new_metadata

        # Recargar DriftDetector si hay datos de referencia
        if REFERENCE_DATA is not None:
            DRIFT_DETECTOR = DriftDetector(REFERENCE_DATA)

        logger.info(f"Modelo recargado exitosamente: {new_metadata.get('model_name') if new_metadata else 'unknown'}")

        return {
            "status": "success",
            "message": "Modelo recargado exitosamente",
            "model_name": new_metadata.get("model_name") if new_metadata else None,
            "f1_score": new_metadata.get("f1_score") if new_metadata else None,
            "reloaded_at": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recargando modelo: {e}")
        raise HTTPException(status_code=500, detail=f"Error recargando modelo: {str(e)}")


@app.get("/admin/model-info")
async def get_model_info(api_key: str = Depends(verify_admin_key)):
    """Obtiene información del modelo actualmente cargado"""
    return {
        "model_loaded": MODEL is not None,
        "model_path": str(MODEL_PATH),
        "metadata": MODEL_METADATA,
        "drift_detector_ready": DRIFT_DETECTOR is not None,
        "model_file_exists": MODEL_PATH.exists(),
        "model_file_modified": datetime.fromtimestamp(
            MODEL_PATH.stat().st_mtime
        ).isoformat() if MODEL_PATH.exists() else None
    }

if __name__ == "__main__":
    import uvicorn
    # Render.com usa la variable PORT, si no existe usa 8000 por defecto
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

