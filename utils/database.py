"""
Módulo de conexión y operaciones con base de datos PostgreSQL (Supabase).

Este módulo:
- Se conecta a Supabase (PostgreSQL en la nube)
- Actualiza predicciones en la tabla tickets_fiducia
- Obtiene tickets pendientes de predicción
- Maneja errores de forma robusta
- Integra con el sistema de monitoreo

Autor: Sistema MLOps
Fecha: 2024
"""

import os
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Callable
from functools import lru_cache, wraps

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("supabase-py no está instalado. Instala con: pip install supabase")

logger = logging.getLogger(__name__)

# ============================================================================
# RETRY LOGIC CON EXPONENTIAL BACKOFF
# ============================================================================

def retry_with_exponential_backoff(
    max_retries: int = 4,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple = (Exception,)
):
    """
    Decorador para reintentar una función con exponential backoff.
    
    Args:
        max_retries: Número máximo de reintentos (default: 4)
        initial_delay: Delay inicial en segundos (default: 1.0)
        max_delay: Delay máximo en segundos (default: 60.0)
        exponential_base: Base para el cálculo exponencial (default: 2.0)
        exceptions: Tupla de excepciones a capturar (default: todas)
    
    Returns:
        Decorador que aplica retry logic
    
    Ejemplo:
        @retry_with_exponential_backoff(max_retries=3)
        def update_database():
            # código que puede fallar
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retries += 1
                    
                    if retries >= max_retries:
                        logger.error(
                            f"❌ {func.__name__} falló después de {max_retries} intentos. "
                            f"Último error: {e}"
                        )
                        raise
                    
                    # Calcular delay con exponential backoff
                    wait_time = min(delay * (exponential_base ** (retries - 1)), max_delay)
                    
                    logger.warning(
                        f"⚠️  {func.__name__} falló (intento {retries}/{max_retries}). "
                        f"Error: {e}. Reintentando en {wait_time:.1f}s..."
                    )
                    
                    time.sleep(wait_time)
            
            # Nunca debería llegar aquí, pero por si acaso
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Variables de entorno (más seguro que hardcodear)
# ⚠️ IMPORTANTE: En producción, usa variables de entorno, no hardcodees la key
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    logger.warning("SUPABASE_URL o SUPABASE_KEY no configuradas. Base de datos no disponible.")

# Nombre de la tabla
TABLE_NAME = "tickets_fiducia"

# ============================================================================
# CLIENTE DE SUPABASE
# ============================================================================

@lru_cache(maxsize=1)
def get_supabase_client() -> Optional[Client]:
    """
    Obtiene cliente de Supabase (con cache para reutilización).
    
    Returns:
        Cliente de Supabase o None si hay error
    """
    if not SUPABASE_AVAILABLE:
        logger.error("supabase-py no está instalado")
        return None
    
    if not SUPABASE_KEY:
        logger.error("SUPABASE_KEY no configurada en variables de entorno")
        return None
    
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✓ Cliente de Supabase creado")
        return client
    except Exception as e:
        logger.error(f"Error creando cliente de Supabase: {e}")
        return None

# ============================================================================
# FUNCIONES DE VERIFICACIÓN
# ============================================================================

def verify_connection() -> bool:
    """
    Verifica la conexión a Supabase.
    
    Returns:
        True si la conexión es exitosa, False en caso contrario
    """
    try:
        client = get_supabase_client()
        if not client:
            return False
        
        # Intentar una consulta simple
        response = client.table(TABLE_NAME).select("number").limit(1).execute()
        logger.info("✓ Conexión a Supabase verificada")
        return True
    except Exception as e:
        logger.error(f"Error verificando conexión a Supabase: {e}")
        return False

def verify_causa_column() -> bool:
    """
    Verifica si la columna 'causa' existe en la tabla.
    
    Returns:
        True si la columna existe, False en caso contrario
    """
    try:
        client = get_supabase_client()
        if not client:
            return False
        
        # Intentar consultar la columna causa
        response = client.table(TABLE_NAME).select("causa, number").limit(1).execute()
        logger.info("✓ Columna 'causa' existe en la tabla")
        return True
    except Exception as e:
        logger.warning(f"La columna 'causa' no existe o hay error: {e}")
        return False

# ============================================================================
# FUNCIONES DE ACTUALIZACIÓN
# ============================================================================

@retry_with_exponential_backoff(
    max_retries=4,
    initial_delay=1.0,
    exceptions=(Exception,)
)
def _execute_update_ticket(
    client: Client,
    ticket_number: str,
    update_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Ejecuta la actualización del ticket en Supabase con retry automático.
    
    Esta función está separada para aplicar retry solo a la operación de BD,
    no a la validación de credenciales.
    
    Args:
        client: Cliente de Supabase
        ticket_number: Número del ticket
        update_data: Datos a actualizar
        
    Returns:
        Dict con resultado de la operación
        
    Raises:
        Exception: Si falla después de todos los reintentos
    """
    logger.info(f"Ejecutando UPDATE en tabla {TABLE_NAME} para ticket number='{ticket_number}'")
    response = client.table(TABLE_NAME).update(update_data).eq("number", ticket_number).execute()
    
    # Verificar si se actualizó algún registro
    if response.data and len(response.data) > 0:
        logger.info(f"✅ Ticket {ticket_number} actualizado exitosamente en BD")
        logger.debug(f"Datos actualizados: {response.data[0]}")
        return {
            'success': True,
            'message': f'Causa actualizada para ticket {ticket_number}',
            'ticket_number': ticket_number,
            'causa': update_data.get('causa'),
            'updated_at': update_data['updated_at'],
            'data': response.data[0]
        }
    else:
        logger.warning(f"⚠️  Ticket {ticket_number} no encontrado en la tabla {TABLE_NAME}")
        logger.warning(f"   Verifica que el ticket_id '{ticket_number}' coincida con la columna 'number' en la BD")
        
        # Intentar verificar si el ticket existe con otro formato
        try:
            check_response = client.table(TABLE_NAME).select("number").eq("number", ticket_number).limit(1).execute()
            if not check_response.data or len(check_response.data) == 0:
                logger.warning(f"   No existe ningún registro con number='{ticket_number}' en la BD")
        except Exception as e:
            logger.debug(f"   No se pudo verificar existencia del ticket: {e}")
        
        return {
            'success': False,
            'error': f'Ticket {ticket_number} no encontrado en la base de datos',
            'ticket_number': ticket_number,
            'suggestion': f'Verifica que el ticket_id "{ticket_number}" coincida exactamente con la columna "number" en la tabla {TABLE_NAME}'
        }

def update_ticket_causa(
    ticket_number: str,
    causa: str,
    confidence: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Actualiza la causa de un ticket en Supabase con retry automático.
    
    Implementa exponential backoff para manejar:
    - Timeouts de red
    - Rate limiting de Supabase
    - Conexiones temporalmente caídas
    
    Args:
        ticket_number: Número del ticket (campo 'number')
        causa: Causa predicha (puede ser string o int)
        confidence: Confianza de la predicción (opcional)
        metadata: Metadatos adicionales (opcional)
        
    Returns:
        Dict con resultado de la operación
    """
    logger.info(f"Actualizando causa para ticket: {ticket_number}")
    
    # Verificar credenciales antes de intentar conectar
    if not SUPABASE_URL or not SUPABASE_KEY:
        error_msg = "SUPABASE_URL o SUPABASE_KEY no configuradas en variables de entorno"
        logger.error(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'ticket_number': ticket_number,
            'suggestion': 'Configura SUPABASE_URL y SUPABASE_KEY en las variables de entorno de Render'
        }
    
    try:
        client = get_supabase_client()
        if not client:
            error_msg = 'No se pudo crear cliente de Supabase. Verifica credenciales.'
            logger.error(f"❌ {error_msg}")
            return {
                'success': False,
                'error': error_msg,
                'ticket_number': ticket_number,
                'suggestion': 'Verifica que SUPABASE_URL y SUPABASE_KEY sean correctas'
            }
        
        # Preparar datos de actualización (solo columnas esenciales)
        update_data = {
            "causa": str(causa),  # Asegurar que sea string
            "updated_at": datetime.now().isoformat()
        }
        
        # Nota: prediction_confidence y prediction_metadata están comentadas
        # porque estas columnas son opcionales y pueden no existir en todas las tablas.
        # Si necesitas estas columnas, créalas en Supabase primero:
        #   ALTER TABLE tickets_fiducia ADD COLUMN IF NOT EXISTS prediction_confidence FLOAT;
        #   ALTER TABLE tickets_fiducia ADD COLUMN IF NOT EXISTS prediction_metadata JSONB;
        # 
        # Luego descomenta estas líneas:
        # if confidence is not None:
        #     update_data["prediction_confidence"] = float(confidence)
        # if metadata:
        #     update_data["prediction_metadata"] = metadata
        
        # Ejecutar UPDATE con retry automático
        return _execute_update_ticket(client, ticket_number, update_data)
        
    except Exception as e:
        logger.error(f"❌ Error actualizando ticket {ticket_number}: {e}")
        return {
            'success': False,
            'error': str(e),
            'ticket_number': ticket_number
        }

def update_tickets_batch(
    updates: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Actualiza múltiples tickets en batch.
    
    Args:
        updates: Lista de dicts con formato:
            [{'ticket_number': 'INC123', 'causa': 'Tipo1', 'confidence': 0.95}, ...]
        
    Returns:
        Dict con resultados de la operación
    """
    logger.info(f"Actualizando {len(updates)} tickets en batch")
    
    results = {
        'success': [],
        'failed': [],
        'total': len(updates)
    }
    
    for update in updates:
        ticket_number = update.get('ticket_number')
        causa = update.get('causa')
        confidence = update.get('confidence')
        metadata = update.get('metadata')
        
        if not ticket_number or not causa:
            results['failed'].append({
                'ticket_number': ticket_number,
                'error': 'ticket_number y causa son requeridos'
            })
            continue
        
        result = update_ticket_causa(
            ticket_number=ticket_number,
            causa=causa,
            confidence=confidence,
            metadata=metadata
        )
        
        if result['success']:
            results['success'].append(result)
        else:
            results['failed'].append(result)
    
    logger.info(f"✓ Batch completado: {len(results['success'])} exitosos, {len(results['failed'])} fallidos")
    
    return results

# ============================================================================
# FUNCIONES DE CONSULTA
# ============================================================================

def get_tickets_pending_prediction(
    limit: int = 100,
    where_clause: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Obtiene tickets pendientes de predicción (sin causa o causa nula).
    
    Args:
        limit: Número máximo de tickets a retornar
        where_clause: Filtros adicionales (opcional)
        
    Returns:
        Lista de tickets pendientes
    """
    try:
        client = get_supabase_client()
        if not client:
            logger.error("No se pudo conectar a Supabase")
            return []
        
        # Construir query
        query = client.table(TABLE_NAME).select("*")
        
        # Filtrar tickets sin causa o con causa nula
        query = query.is_("causa", "null").limit(limit)
        
        # Aplicar filtros adicionales si existen
        if where_clause:
            for key, value in where_clause.items():
                query = query.eq(key, value)
        
        response = query.execute()
        
        tickets = response.data if response.data else []
        logger.info(f"✓ Encontrados {len(tickets)} tickets pendientes de predicción")
        
        return tickets
        
    except Exception as e:
        logger.error(f"Error obteniendo tickets pendientes: {e}")
        return []

def get_ticket_by_number(ticket_number: str) -> Optional[Dict[str, Any]]:
    """
    Obtiene un ticket por su número.
    
    Args:
        ticket_number: Número del ticket
        
    Returns:
        Dict con datos del ticket o None si no existe
    """
    try:
        client = get_supabase_client()
        if not client:
            return None
        
        response = client.table(TABLE_NAME).select("*").eq("number", ticket_number).execute()
        
        if response.data and len(response.data) > 0:
            return response.data[0]
        else:
            return None
            
    except Exception as e:
        logger.error(f"Error obteniendo ticket {ticket_number}: {e}")
        return None

# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

def get_ticket_text_fields(ticket: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extrae los campos de texto de un ticket para predicción.
    
    Args:
        ticket: Dict con datos del ticket
        
    Returns:
        Tuple (short_description, close_notes)
    """
    short_description = ticket.get('short_description', '') or ticket.get('description', '') or ''
    close_notes = ticket.get('close_notes', '') or ticket.get('notes', '') or ''
    
    return short_description, close_notes

def create_causa_column_if_not_exists() -> bool:
    """
    Crea la columna 'causa' si no existe.
    Nota: Esto requiere permisos de administrador en Supabase.
    
    Returns:
        True si la columna existe o fue creada, False en caso contrario
    """
    if verify_causa_column():
        return True
    
    logger.warning("⚠️  La columna 'causa' no existe")
    logger.info("💡 Ejecuta este SQL en Supabase para crear la columna:")
    logger.info("   ALTER TABLE tickets_fiducia ADD COLUMN IF NOT EXISTS causa VARCHAR;")
    logger.info("   ALTER TABLE tickets_fiducia ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ;")
    logger.info("   ALTER TABLE tickets_fiducia ADD COLUMN IF NOT EXISTS prediction_confidence FLOAT;")
    logger.info("   ALTER TABLE tickets_fiducia ADD COLUMN IF NOT EXISTS prediction_metadata JSONB;")
    
    return False

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

def initialize_database() -> bool:
    """
    Inicializa y verifica la conexión a la base de datos.
    
    Returns:
        True si la inicialización fue exitosa
    """
    logger.info("Inicializando conexión a base de datos...")
    
    if not verify_connection():
        logger.error("❌ No se pudo conectar a Supabase")
        return False
    
    if not verify_causa_column():
        logger.warning("⚠️  Columna 'causa' no existe. Usa create_causa_column_if_not_exists() para ver instrucciones")
    
    logger.info("✓ Base de datos inicializada correctamente")
    return True

# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("🧪 PRUEBA DE CONEXIÓN A SUPABASE")
    print("=" * 80)
    print()
    
    # Verificar conexión
    if verify_connection():
        print("✅ Conexión exitosa")
    else:
        print("❌ Error de conexión")
        exit(1)
    
    # Verificar columna causa
    if verify_causa_column():
        print("✅ Columna 'causa' existe")
    else:
        print("⚠️  Columna 'causa' no existe")
        create_causa_column_if_not_exists()
    
    # Ejemplo de actualización
    print()
    print("📝 Ejemplo de actualización:")
    resultado = update_ticket_causa(
        ticket_number='INC1353571',
        causa='Tipo1',
        confidence=0.95
    )
    print(f"Resultado: {resultado}")
    
    # Obtener tickets pendientes
    print()
    print("📋 Tickets pendientes de predicción:")
    tickets = get_tickets_pending_prediction(limit=5)
    print(f"Encontrados: {len(tickets)} tickets")

