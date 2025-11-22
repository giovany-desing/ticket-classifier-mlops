"""
Script para diagnosticar problemas con la API y Supabase.
Ejecutar: python test_api_connection.py
"""

import requests
import json
import os
from supabase import create_client

# ============================================================================
# 1. VERIFICAR VARIABLES DE ENTORNO
# ============================================================================
print("=" * 80)
print("1. VERIFICANDO VARIABLES DE ENTORNO")
print("=" * 80)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"✓ SUPABASE_URL: {'✅ Configurada' if SUPABASE_URL else '❌ NO configurada'}")
print(f"✓ SUPABASE_KEY: {'✅ Configurada' if SUPABASE_KEY else '❌ NO configurada'}")

if SUPABASE_URL:
    print(f"  URL: {SUPABASE_URL}")

# ============================================================================
# 2. PROBAR CONEXIÓN DIRECTA A SUPABASE
# ============================================================================
print("\n" + "=" * 80)
print("2. PROBANDO CONEXIÓN DIRECTA A SUPABASE")
print("=" * 80)

if SUPABASE_URL and SUPABASE_KEY:
    try:
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Verificar que la tabla existe
        print("Intentando consultar tabla tickets_fiducia...")
        response = client.table("tickets_fiducia").select("number").limit(1).execute()
        print(f"✅ Conexión exitosa a Supabase")
        print(f"   Registros encontrados: {len(response.data)}")
        
        # Verificar si el ticket específico existe
        print(f"\nBuscando ticket INC1353571...")
        ticket_response = client.table("tickets_fiducia").select("*").eq("number", "INC1353571").execute()
        
        if ticket_response.data and len(ticket_response.data) > 0:
            ticket = ticket_response.data[0]
            print(f"✅ Ticket encontrado!")
            print(f"   number: {ticket.get('number')}")
            print(f"   causa: {ticket.get('causa')}")
            print(f"   updated_at: {ticket.get('updated_at')}")
            print(f"   short_description: {ticket.get('short_description', 'N/A')[:50]}...")
            
            # Verificar columnas
            print(f"\n✓ Columnas disponibles en el ticket:")
            for key in ticket.keys():
                print(f"   - {key}")
        else:
            print(f"❌ Ticket INC1353571 NO encontrado en la tabla")
            print(f"   El ticket_id debe coincidir EXACTAMENTE con la columna 'number'")
            
            # Mostrar algunos números de ejemplo
            print(f"\n📋 Mostrando primeros 5 números de tickets en la BD:")
            sample = client.table("tickets_fiducia").select("number").limit(5).execute()
            for item in sample.data:
                print(f"   - {item.get('number')}")
        
        # Intentar actualización de prueba
        print(f"\n" + "=" * 80)
        print("3. PROBANDO ACTUALIZACIÓN DIRECTA EN SUPABASE")
        print("=" * 80)
        
        from datetime import datetime
        
        update_data = {
            "causa": "TEST_MANUAL",
            "updated_at": datetime.now().isoformat()
        }
        
        print(f"Intentando UPDATE en ticket INC1353571...")
        update_response = client.table("tickets_fiducia").update(update_data).eq("number", "INC1353571").execute()
        
        if update_response.data and len(update_response.data) > 0:
            print(f"✅ Actualización exitosa!")
            print(f"   Datos actualizados: {json.dumps(update_response.data[0], indent=2, default=str)}")
        else:
            print(f"❌ No se actualizó ningún registro")
            print(f"   Posibles causas:")
            print(f"   1. El ticket INC1353571 no existe en la tabla")
            print(f"   2. La columna 'number' no coincide con 'INC1353571'")
            print(f"   3. Permisos insuficientes en Supabase")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"\nSugerencias:")
        print(f"1. Verifica que SUPABASE_URL y SUPABASE_KEY sean correctas")
        print(f"2. Verifica que la tabla 'tickets_fiducia' existe en Supabase")
        print(f"3. Verifica los permisos de la API key en Supabase")
else:
    print("❌ No se pueden hacer pruebas sin SUPABASE_URL y SUPABASE_KEY")
    print("\nConfigura las variables de entorno:")
    print("  export SUPABASE_URL='https://tu-proyecto.supabase.co'")
    print("  export SUPABASE_KEY='tu-anon-key'")

# ============================================================================
# 4. PROBAR API EN RENDER
# ============================================================================
print("\n" + "=" * 80)
print("4. PROBANDO API EN RENDER.COM")
print("=" * 80)

API_URL = "https://ticket-classifier-api-imbu.onrender.com"

try:
    # Health check de DB
    print("Verificando estado de la base de datos en la API...")
    db_health = requests.get(f"{API_URL}/db/health", timeout=10)
    print(f"Status: {db_health.status_code}")
    print(f"Respuesta: {json.dumps(db_health.json(), indent=2)}")
    
    db_status = db_health.json()
    if not db_status.get('database_connected'):
        print("\n⚠️  LA API NO ESTÁ CONECTADA A SUPABASE")
        print("Verifica las variables de entorno en Render Dashboard:")
        print("  - SUPABASE_URL")
        print("  - SUPABASE_KEY")
    
    # Probar predicción
    print(f"\n" + "-" * 80)
    print("Probando endpoint /predict/ticket...")
    
    payload = {
        "ticket_id": "INC1353571",
        "short_description": "No puedo acceder al sistema",
        "close_notes": "Se reseteo la contraseña"
    }
    
    response = requests.post(
        f"{API_URL}/predict/ticket",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=30
    )
    
    print(f"Status: {response.status_code}")
    print(f"Respuesta completa:")
    print(json.dumps(response.json(), indent=2))
    
    result = response.json()
    if 'database_update' in result:
        db_update = result['database_update']
        if db_update.get('success'):
            print(f"\n✅ ACTUALIZACIÓN EXITOSA EN BD")
        else:
            print(f"\n❌ ERROR EN ACTUALIZACIÓN BD:")
            print(f"   Error: {db_update.get('error')}")
            print(f"   Sugerencia: {db_update.get('suggestion', 'N/A')}")
    
except requests.exceptions.Timeout:
    print("❌ Timeout - La API tardó demasiado en responder")
except requests.exceptions.RequestException as e:
    print(f"❌ Error de conexión: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 80)
print("DIAGNÓSTICO COMPLETO")
print("=" * 80)
print("\nSi ves errores arriba, revisa:")
print("1. Variables de entorno en Render (SUPABASE_URL, SUPABASE_KEY)")
print("2. Que el ticket INC1353571 exista en la tabla tickets_fiducia")
print("3. Que la columna 'number' coincida exactamente con 'INC1353571'")
print("4. Que las columnas 'causa' y 'updated_at' existan en la tabla")
print("5. Permisos de la API key en Supabase (debe poder UPDATE)")

