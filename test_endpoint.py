"""
Script para probar el endpoint /predict/ticket
"""

import requests
import json

API_URL = "https://ticket-classifier-api-imbu.onrender.com"

# Datos de prueba
payload = {
    "ticket_id": "INC1353571",
    "short_description": "No puedo acceder al sistema",
    "close_notes": "Se reseteo la contraseña"
}

print("=" * 80)
print("PROBANDO ENDPOINT /predict/ticket")
print("=" * 80)
print(f"\nURL: {API_URL}/predict/ticket")
print(f"\nPayload:")
print(json.dumps(payload, indent=2))

try:
    # Hacer request
    response = requests.post(
        f"{API_URL}/predict/ticket",
        json=payload,
        headers={
            "Content-Type": "application/json"
        },
        timeout=30
    )
    
    print(f"\n{'=' * 80}")
    print(f"RESPUESTA")
    print(f"{'=' * 80}")
    print(f"\nStatus Code: {response.status_code}")
    print(f"\nHeaders:")
    for key, value in response.headers.items():
        print(f"  {key}: {value}")
    
    print(f"\nBody:")
    try:
        print(json.dumps(response.json(), indent=2))
    except:
        print(response.text)
    
    # Analizar resultado
    if response.status_code == 200:
        print(f"\n✅ ÉXITO - Predicción realizada")
        result = response.json()
        if result.get('database_update', {}).get('success'):
            print(f"✅ Base de datos actualizada correctamente")
        else:
            print(f"⚠️  Warning: {result.get('database_update', {}).get('error', 'Error desconocido')}")
    elif response.status_code == 422:
        print(f"\n❌ ERROR 422 - Datos inválidos")
        print(f"\nDetalles del error:")
        error_detail = response.json()
        print(json.dumps(error_detail, indent=2))
    elif response.status_code == 401:
        print(f"\n❌ ERROR 401 - API Key requerida")
    else:
        print(f"\n❌ ERROR {response.status_code}")
        
except requests.exceptions.Timeout:
    print("\n❌ ERROR: Timeout - El servidor tardó demasiado en responder")
except requests.exceptions.ConnectionError:
    print("\n❌ ERROR: No se pudo conectar al servidor")
except Exception as e:
    print(f"\n❌ ERROR: {e}")

print("\n" + "=" * 80)
print("INSTRUCCIONES PARA POSTMAN")
print("=" * 80)
print("""
1. Método: POST
2. URL: https://ticket-classifier-api-imbu.onrender.com/predict/ticket
3. Headers:
   - Content-Type: application/json
4. Body (raw JSON):
{
  "ticket_id": "INC1353571",
  "short_description": "No puedo acceder al sistema",
  "close_notes": "Se reseteo la contraseña"
}

IMPORTANTE:
- Usa Body → raw → JSON (NO form-data ni x-www-form-urlencoded)
- Asegúrate de que Content-Type sea application/json
- NO uses comillas simples, solo dobles
""")

