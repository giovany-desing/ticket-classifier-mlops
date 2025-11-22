"""
Ejemplo de uso del módulo de base de datos.

Este archivo muestra cómo usar las funciones de database.py
para integrar con Supabase/PostgreSQL.
"""

from utils.database import (
    update_ticket_causa,
    update_tickets_batch,
    get_tickets_pending_prediction,
    get_ticket_by_number,
    verify_connection,
    initialize_database
)

# Ejemplo 1: Actualizar un ticket individual
def ejemplo_actualizar_ticket():
    """Ejemplo de actualización de un ticket"""
    resultado = update_ticket_causa(
        ticket_number='INC1353571',
        causa='Tipo1',
        confidence=0.95
    )
    
    if resultado['success']:
        print(f"✅ Ticket actualizado: {resultado['message']}")
    else:
        print(f"❌ Error: {resultado['error']}")

# Ejemplo 2: Actualizar múltiples tickets en batch
def ejemplo_batch_update():
    """Ejemplo de actualización en batch"""
    updates = [
        {'ticket_number': 'INC123', 'causa': 'Tipo1', 'confidence': 0.95},
        {'ticket_number': 'INC124', 'causa': 'Tipo2', 'confidence': 0.87},
        {'ticket_number': 'INC125', 'causa': 'Tipo1', 'confidence': 0.92}
    ]
    
    resultado = update_tickets_batch(updates)
    print(f"✅ Actualizados: {len(resultado['success'])}")
    print(f"❌ Fallidos: {len(resultado['failed'])}")

# Ejemplo 3: Obtener tickets pendientes
def ejemplo_tickets_pendientes():
    """Ejemplo de obtener tickets pendientes"""
    tickets = get_tickets_pending_prediction(limit=10)
    
    print(f"📋 Tickets pendientes: {len(tickets)}")
    for ticket in tickets:
        print(f"  - {ticket.get('number')}: {ticket.get('short_description', '')[:50]}...")

# Ejemplo 4: Obtener un ticket específico
def ejemplo_obtener_ticket():
    """Ejemplo de obtener un ticket"""
    ticket = get_ticket_by_number('INC1353571')
    
    if ticket:
        print(f"✅ Ticket encontrado: {ticket.get('number')}")
        print(f"   Descripción: {ticket.get('short_description', '')[:100]}")
    else:
        print("❌ Ticket no encontrado")

if __name__ == "__main__":
    # Verificar conexión
    if verify_connection():
        print("✅ Conexión exitosa")
        
        # Ejemplos
        ejemplo_actualizar_ticket()
        ejemplo_tickets_pendientes()
        ejemplo_obtener_ticket()
    else:
        print("❌ Error de conexión")





















