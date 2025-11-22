# 🔧 Configuración de Supabase para API de Clasificación

## ❌ Problema Actual

El error `Could not find the 'prediction_confidence' column` ocurre porque la tabla no tiene las columnas necesarias.

## ✅ Solución

### Paso 1: Ejecutar SQL en Supabase

1. Ve a **Supabase Dashboard**: https://app.supabase.com
2. Selecciona tu proyecto
3. Ve a **SQL Editor** (menú izquierdo)
4. Crea una **New Query**
5. Copia y pega el siguiente SQL:

```sql
-- Columnas esenciales (REQUERIDAS)
ALTER TABLE tickets_fiducia 
ADD COLUMN IF NOT EXISTS causa VARCHAR;

ALTER TABLE tickets_fiducia 
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Trigger para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_tickets_fiducia_updated_at ON tickets_fiducia;
CREATE TRIGGER update_tickets_fiducia_updated_at 
    BEFORE UPDATE ON tickets_fiducia 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Verificar que funcionó
SELECT column_name, data_type 
FROM information_schema.columns
WHERE table_name = 'tickets_fiducia'
  AND column_name IN ('number', 'causa', 'updated_at');
```

6. Haz clic en **Run** (o presiona `Ctrl/Cmd + Enter`)

### Paso 2: Verificar Variables de Entorno en Render

Ve a tu servicio en Render.com → **Environment**

Asegúrate de tener configuradas:

```bash
SUPABASE_URL=https://tu-proyecto.supabase.co
SUPABASE_KEY=tu-anon-key-aqui
```

Para obtener estas credenciales:
- Ve a Supabase Dashboard → **Settings** → **API**
- Copia **Project URL** → `SUPABASE_URL`
- Copia **anon public** key → `SUPABASE_KEY`

### Paso 3: Reiniciar el Servicio en Render

1. Ve a Render Dashboard
2. Selecciona tu servicio `ticket-classifier-api-imbu`
3. Haz clic en **Manual Deploy** → **Deploy latest commit**
4. O simplemente haz un cambio en el código y haz push

### Paso 4: Probar

```bash
curl -X POST https://ticket-classifier-api-imbu.onrender.com/predict/ticket \
  -H "Content-Type: application/json" \
  -d '{
    "ticket_id": "INC1353571",
    "short_description": "No puedo acceder al sistema",
    "close_notes": "Se reseteo la contraseña"
  }'
```

Deberías ver una respuesta como:

```json
{
  "ticket_id": "INC1353571",
  "prediction": "Accesos",
  "probability": 0.95,
  "database_update": {
    "success": true,
    "message": "Causa actualizada para ticket INC1353571",
    "updated_at": "2025-11-22T01:40:00.000Z"
  },
  "timestamp": "2025-11-22T01:40:00.000Z"
}
```

### Paso 5: Verificar en Supabase

1. Ve a Supabase Dashboard → **Table Editor**
2. Selecciona la tabla `tickets_fiducia`
3. Busca el ticket `INC1353571`
4. Verifica que los campos `causa` y `updated_at` se hayan actualizado

## 🐛 Diagnóstico de Problemas

### Verificar conexión a BD:
```bash
curl https://ticket-classifier-api-imbu.onrender.com/db/health
```

Respuesta esperada:
```json
{
  "database_connected": true,
  "status": "healthy",
  "table_name": "tickets_fiducia"
}
```

Si `database_connected: false`:
- Verifica las variables de entorno en Render
- Verifica que la API key tenga permisos correctos en Supabase

### Ver logs en tiempo real:
1. Ve a Render Dashboard
2. Selecciona tu servicio
3. Haz clic en **Logs**
4. Busca mensajes con `✅` (éxito) o `❌` (error)

## 📊 Columnas Opcionales (Avanzado)

Si quieres guardar información adicional (confianza, metadatos), ejecuta también:

```sql
ALTER TABLE tickets_fiducia 
ADD COLUMN IF NOT EXISTS prediction_confidence FLOAT;

ALTER TABLE tickets_fiducia 
ADD COLUMN IF NOT EXISTS prediction_metadata JSONB;
```

Y descomenta las líneas en `utils/database.py` (líneas 176-180).

## ✅ Checklist Final

- [ ] Columnas `causa` y `updated_at` creadas en Supabase
- [ ] Variables `SUPABASE_URL` y `SUPABASE_KEY` configuradas en Render
- [ ] Servicio reiniciado en Render
- [ ] Endpoint `/db/health` retorna `database_connected: true`
- [ ] Ticket de prueba se actualiza correctamente
- [ ] Cambios visibles en Supabase Table Editor

## 🆘 ¿Necesitas Ayuda?

Si sigues teniendo problemas:

1. **Revisa los logs en Render** - Busca mensajes que digan "Error actualizando ticket"
2. **Verifica que el ticket existe** - El `ticket_id` debe coincidir EXACTAMENTE con la columna `number` en Supabase
3. **Verifica permisos** - La API key debe tener permisos de UPDATE en la tabla

### Comando de diagnóstico local:
```bash
python test_api_connection.py
```

Este script te dirá exactamente dónde está el problema.

