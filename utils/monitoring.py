"""
Sistema de monitoreo y detección de drift para el modelo de clasificación.

Este módulo implementa:
- Detección de Data Drift (cambios en distribución de datos de entrada)
- Detección de Concept Drift (degradación de performance del modelo)
- Almacenamiento de predicciones para análisis
- Alertas cuando se detectan problemas

Autor: Sistema MLOps
Fecha: 2024
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from collections import defaultdict

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Detector de drift en datos y modelo"""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """
        Inicializa el detector de drift.
        
        Args:
            reference_data: DataFrame con datos de referencia (training data)
            threshold: Umbral de p-value para considerar drift (default: 0.05)
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_reference_stats()
        
    def _compute_reference_stats(self) -> Dict[str, Any]:
        """Calcula estadísticas de referencia"""
        stats = {
            'text_length_mean': self.reference_data['text_length'].mean(),
            'text_length_std': self.reference_data['text_length'].std(),
            'text_length_median': self.reference_data['text_length'].median(),
            'vocab_size': len(set(' '.join(self.reference_data['processed_text']).split())),
            'class_distribution': self.reference_data['label'].value_counts(normalize=True).to_dict()
        }
        return stats
    
    def detect_data_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta data drift comparando datos actuales con referencia.
        
        Args:
            current_data: DataFrame con datos actuales a comparar
            
        Returns:
            Dict con resultados de detección de drift
        """
        results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'checks': {}
        }
        
        # Check 1: Distribución de longitud de texto
        if 'text_length' in current_data.columns:
            current_lengths = current_data['text_length']
            ref_mean = self.reference_stats['text_length_mean']
            ref_std = self.reference_stats['text_length_std']
            
            # Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(
                self.reference_data['text_length'],
                current_lengths
            )
            
            results['checks']['text_length'] = {
                'p_value': float(p_value),
                'drift': p_value < self.threshold,
                'current_mean': float(current_lengths.mean()),
                'reference_mean': float(ref_mean)
            }
            
            if p_value < self.threshold:
                results['drift_detected'] = True
                results['drift_score'] += 0.3
        
        # Check 2: Distribución de clases (si hay labels)
        if 'label' in current_data.columns and 'label' in self.reference_data.columns:
            current_dist = current_data['label'].value_counts(normalize=True).to_dict()
            ref_dist = self.reference_stats['class_distribution']
            
            # Chi-square test
            all_classes = set(list(ref_dist.keys()) + list(current_dist.keys()))
            ref_counts = [ref_dist.get(c, 0) * len(self.reference_data) for c in all_classes]
            current_counts = [current_dist.get(c, 0) * len(current_data) for c in all_classes]
            
            if sum(current_counts) > 0:
                chi2, p_value = stats.chisquare(current_counts, f_exp=ref_counts)
                
                results['checks']['class_distribution'] = {
                    'p_value': float(p_value),
                    'drift': p_value < self.threshold,
                    'current_distribution': current_dist,
                    'reference_distribution': ref_dist
                }
                
                if p_value < self.threshold:
                    results['drift_detected'] = True
                    results['drift_score'] += 0.4
        
        # Check 3: Vocabulario (nuevas palabras)
        if 'processed_text' in current_data.columns:
            current_text = ' '.join(current_data['processed_text'].astype(str))
            current_vocab = set(current_text.split())
            ref_vocab_size = self.reference_stats['vocab_size']
            
            # Porcentaje de palabras nuevas
            # (simplificado - en producción usar embedding similarity)
            new_words_ratio = len(current_vocab) / max(ref_vocab_size, 1)
            
            # Si el vocabulario crece más del 20%, hay posible drift
            vocab_drift = new_words_ratio > 1.2
            
            results['checks']['vocabulary'] = {
                'drift': vocab_drift,
                'vocab_ratio': float(new_words_ratio),
                'current_vocab_size': len(current_vocab),
                'reference_vocab_size': ref_vocab_size
            }
            
            if vocab_drift:
                results['drift_detected'] = True
                results['drift_score'] += 0.3
        
        results['drift_score'] = min(results['drift_score'], 1.0)
        return results
    
    def detect_concept_drift(
        self, 
        predictions: List[str], 
        true_labels: Optional[List[str]] = None,
        prediction_probas: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Detecta concept drift basado en métricas de performance.
        
        Args:
            predictions: Lista de predicciones del modelo
            true_labels: Lista de labels verdaderos (opcional, para evaluación)
            prediction_probas: Probabilidades de predicción (opcional)
            
        Returns:
            Dict con resultados de concept drift
        """
        results = {
            'drift_detected': False,
            'performance_metrics': {},
            'confidence_issues': False
        }
        
        # Si tenemos labels verdaderos, calculamos métricas
        if true_labels is not None and len(true_labels) > 0:
            try:
                acc = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
                prec = precision_score(true_labels, predictions, average='weighted', zero_division=0)
                rec = recall_score(true_labels, predictions, average='weighted', zero_division=0)
                
                results['performance_metrics'] = {
                    'accuracy': float(acc),
                    'f1_score': float(f1),
                    'precision': float(prec),
                    'recall': float(rec)
                }
                
                # Concept drift si F1 baja de 0.85 (ajustable)
                if f1 < 0.85:
                    results['drift_detected'] = True
                    results['drift_reason'] = f'F1 score bajo: {f1:.4f}'
                    
            except Exception as e:
                logger.warning(f"Error calculando métricas: {e}")
        
        # Check de confianza en predicciones
        if prediction_probas is not None:
            max_probas = np.max(prediction_probas, axis=1)
            avg_confidence = float(np.mean(max_probas))
            low_confidence_ratio = float(np.sum(max_probas < 0.5) / len(max_probas))
            
            results['confidence_metrics'] = {
                'average_confidence': avg_confidence,
                'low_confidence_ratio': low_confidence_ratio
            }
            
            # Si más del 30% tiene baja confianza, hay problema
            if low_confidence_ratio > 0.3 or avg_confidence < 0.6:
                results['confidence_issues'] = True
                if not results['drift_detected']:
                    results['drift_detected'] = True
                    results['drift_reason'] = f'Baja confianza: {avg_confidence:.4f}'
        
        return results


class PredictionLogger:
    """Logger para almacenar predicciones y métricas"""
    
    def __init__(self, log_dir: str = "monitoring/logs"):
        """
        Inicializa el logger de predicciones.
        
        Args:
            log_dir: Directorio donde se guardan los logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.log_dir / "predictions.jsonl"
        self.metrics_file = self.log_dir / "daily_metrics.json"
        
    def log_prediction(
        self,
        text: str,
        prediction: str,
        probability: float,
        true_label: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        """
        Registra una predicción individual.
        
        Args:
            text: Texto de entrada
            prediction: Predicción del modelo
            probability: Probabilidad de la predicción
            true_label: Label verdadero (si está disponible)
            timestamp: Timestamp de la predicción
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'text': text[:200],  # Limitar tamaño
            'prediction': prediction,
            'probability': float(probability),
            'true_label': true_label,
            'correct': true_label == prediction if true_label else None
        }
        
        # Append to JSONL file
        with open(self.predictions_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def get_recent_predictions(self, hours: int = 24) -> pd.DataFrame:
        """
        Obtiene predicciones recientes.
        
        Args:
            hours: Número de horas hacia atrás
            
        Returns:
            DataFrame con predicciones
        """
        if not self.predictions_file.exists():
            return pd.DataFrame()
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        predictions = []
        with open(self.predictions_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    entry_time = datetime.fromisoformat(entry['timestamp'])
                    if entry_time >= cutoff_time:
                        predictions.append(entry)
                except:
                    continue
        
        if not predictions:
            return pd.DataFrame()
        
        df = pd.DataFrame(predictions)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def compute_daily_metrics(self) -> Dict[str, Any]:
        """
        Calcula métricas diarias agregadas.
        
        Returns:
            Dict con métricas del día
        """
        df = self.get_recent_predictions(hours=24)
        
        if df.empty:
            return {
                'date': datetime.now().isoformat(),
                'total_predictions': 0,
                'metrics': {}
            }
        
        metrics = {
            'date': datetime.now().isoformat(),
            'total_predictions': len(df),
            'average_confidence': float(df['probability'].mean()) if 'probability' in df.columns else 0.0,
            'predictions_by_class': df['prediction'].value_counts().to_dict() if 'prediction' in df.columns else {}
        }
        
        # Si hay labels verdaderos, calcular accuracy
        if 'true_label' in df.columns and df['true_label'].notna().any():
            correct = df['true_label'] == df['prediction']
            metrics['accuracy'] = float(correct.mean())
            metrics['total_labeled'] = int(correct.sum())
        
        return metrics
    
    def save_daily_metrics(self):
        """Guarda métricas diarias en archivo JSON"""
        metrics = self.compute_daily_metrics()
        
        # Cargar métricas existentes
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = []
        
        # Agregar nuevas métricas
        all_metrics.append(metrics)
        
        # Mantener solo últimos 30 días
        all_metrics = all_metrics[-30:]
        
        # Guardar
        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, ensure_ascii=False)
        
        return metrics

