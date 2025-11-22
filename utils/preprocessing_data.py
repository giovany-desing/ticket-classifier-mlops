"""
Módulo de preprocesamiento de texto para clasificación de tickets.

Este módulo optimiza el preprocesamiento mediante:
- Carga única de recursos (stopwords, stemmer, regex)
- Manejo robusto de valores nulos
- Logging detallado
- Configuración externalizada

Autor: [Tu nombre]
Fecha: 2024
"""

import re
import string
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import yaml
import logging
from functools import lru_cache

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# ============================================================================
# CONFIGURACIÓN DE NLTK
# ============================================================================

# Descargar recursos NLTK solo si no existen
def _download_nltk_resources():
    """Descarga recursos de NLTK necesarios"""
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords'
    }
    
    for name, path in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)

_download_nltk_resources()

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CARGA DE CONFIGURACIÓN
# ============================================================================

@lru_cache(maxsize=1)
def load_config(config_path: str = 'config.yaml', validate: bool = True) -> dict:
    """
    Carga y valida configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración (relativa a la raíz del proyecto)
        validate: Si True, valida la configuración con Pydantic schemas
        
    Returns:
        Diccionario con la configuración validada
        
    Raises:
        FileNotFoundError: Si no se encuentra el archivo de configuración
        ValidationError: Si la configuración es inválida (cuando validate=True)
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    config_file = project_root / config_path
    
    if not config_file.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_file}\n"
            f"Current dir: {current_dir}\n"
            f"Project root: {project_root}"
        )
    
    with open(config_file, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)
    
    # Validar con Pydantic si está habilitado
    if validate:
        try:
            from utils.config_schema import validate_config
            validated_config = validate_config(config_dict)
            # Convertir de vuelta a dict para mantener compatibilidad
            config_dict = validated_config.dict()
            logger.info(f"✅ Config loaded and validated from: {config_file}")
        except ImportError:
            logger.warning("⚠️  config_schema not found, skipping validation")
            logger.info(f"✓ Config loaded from: {config_file}")
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            logger.error("   Fix config.yaml or set validate=False to skip validation")
            raise
    else:
        logger.info(f"✓ Config loaded from: {config_file} (without validation)")
    
    return config_dict

# ============================================================================
# INICIALIZACIÓN DE RECURSOS GLOBALES
# ============================================================================

# Cargar configuración
CONFIG = load_config()

# Extraer configuración de preprocesamiento
PREPROCESS_CONFIG = CONFIG.get('preprocessing', {})

# Cargar stopwords en español
STOPWORDS_ES = set(stopwords.words(PREPROCESS_CONFIG.get('language', 'spanish')))

# Cargar stopwords personalizadas desde config
STOPWORDS_CUSTOM = set(PREPROCESS_CONFIG.get('custom_stopwords', []))

# Unir ambos conjuntos
STOPWORDS_TOTAL = STOPWORDS_ES.union(STOPWORDS_CUSTOM)

# Inicializar stemmer
STEMMER = SnowballStemmer(PREPROCESS_CONFIG.get('language', 'spanish'))

# Compilar expresiones regulares (para mejor performance)
REGEX_PATTERNS = {
    'brackets': re.compile(r'\[.*?\]'),
    'url': re.compile(r'https?://\S+|www\.\S+'),
    'html': re.compile(r'<.*?>+'),
    'punctuation': re.compile('[%s]' % re.escape(string.punctuation)),
    'newline': re.compile(r'\n'),
    'numbers': re.compile(r'\w*\d\w*'),
    'non_ascii': re.compile(r'[^\x00-\x7F]+'),
    'whitespace': re.compile(r'\s+')
}

logger.info(
    f"✓ Preprocessing initialized:\n"
    f"  - Language: {PREPROCESS_CONFIG.get('language', 'spanish')}\n"
    f"  - Stopwords (ES): {len(STOPWORDS_ES)}\n"
    f"  - Stopwords (Custom): {len(STOPWORDS_CUSTOM)}\n"
    f"  - Total stopwords: {len(STOPWORDS_TOTAL)}\n"
    f"  - Stemming: {PREPROCESS_CONFIG.get('apply_stemming', True)}"
)

# ============================================================================
# FUNCIONES DE PREPROCESAMIENTO
# ============================================================================

def clean(text: Optional[str]) -> str:
    """
    Realiza limpieza básica del texto.
    
    Operaciones:
    - Conversión a minúsculas
    - Eliminación de URLs, HTML, puntuación
    - Eliminación de números y caracteres especiales
    - Normalización de espacios en blanco
    
    Args:
        text: Texto a limpiar (puede ser None)
        
    Returns:
        Texto limpio. Retorna string vacío si input es None/NaN
        
    Examples:
        >>> clean("Visita https://example.com! ¡Ahora!")
        'visita ahora'
        
        >>> clean(None)
        ''
    """
    # Manejo de valores nulos/vacíos
    if pd.isna(text) or text is None or text == '':
        return ''
    
    # Asegurar que sea string y convertir a minúsculas
    text = str(text).lower()
    
    # Aplicar todas las regex compiladas
    text = REGEX_PATTERNS['brackets'].sub('', text)
    text = REGEX_PATTERNS['url'].sub('', text)
    text = REGEX_PATTERNS['html'].sub('', text)
    text = REGEX_PATTERNS['punctuation'].sub('', text)
    text = REGEX_PATTERNS['newline'].sub(' ', text)
    text = REGEX_PATTERNS['numbers'].sub('', text)
    text = REGEX_PATTERNS['non_ascii'].sub('', text)
    
    # Normalizar espacios múltiples a uno solo
    text = REGEX_PATTERNS['whitespace'].sub(' ', text)
    
    # Eliminar espacios al inicio y final
    return text.strip()


def tokenize_and_stem(text: str, apply_stemming: Optional[bool] = None) -> str:
    """
    Tokeniza, elimina stopwords y aplica stemming.
    
    Args:
        text: Texto limpio a procesar
        apply_stemming: Si True, aplica stemming. Si None, usa valor del config
        
    Returns:
        Texto tokenizado y procesado
        
    Examples:
        >>> tokenize_and_stem("los clientes tienen problemas graves")
        'client problem grav'
    """
    if not text or text == '':
        return ''
    
    # Usar configuración si no se especifica
    if apply_stemming is None:
        apply_stemming = PREPROCESS_CONFIG.get('apply_stemming', True)
    
    try:
        # Tokenizar en palabras
        tokens = word_tokenize(text, language=PREPROCESS_CONFIG.get('language', 'spanish'))
        
        # Filtrar: solo palabras alfabéticas y no stopwords
        tokens_filtered = [
            token for token in tokens
            if token.isalpha() and token.lower() not in STOPWORDS_TOTAL
        ]
        
        # Aplicar stemming si está habilitado
        if apply_stemming:
            tokens_filtered = [STEMMER.stem(token.lower()) for token in tokens_filtered]
        
        return " ".join(tokens_filtered)
    
    except Exception as e:
        logger.warning(f"Error in tokenize_and_stem: {e}. Returning empty string.")
        return ''


def preprocess_text(text: Optional[str], apply_stemming: Optional[bool] = None) -> str:
    """
    Pipeline completo de preprocesamiento.
    
    Combina limpieza y tokenización en un solo paso.
    
    Args:
        text: Texto original a procesar
        apply_stemming: Si True, aplica stemming. Si None, usa valor del config
        
    Returns:
        Texto completamente procesado
        
    Examples:
        >>> preprocess_text("¡Hola! Tengo un problema: https://example.com")
        'problem'
    """
    # Paso 1: Limpieza
    cleaned_text = clean(text)
    
    # Paso 2: Tokenización y stemming
    processed_text = tokenize_and_stem(cleaned_text, apply_stemming=apply_stemming)
    
    return processed_text


