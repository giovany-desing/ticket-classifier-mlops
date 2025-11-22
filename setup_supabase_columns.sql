-- ============================================================================
-- Script para configurar columnas necesarias en tabla tickets_fiducia
-- Ejecutar en: Supabase Dashboard → SQL Editor → New Query
-- ============================================================================

-- 1. COLUMNAS ESENCIALES (REQUERIDAS)
-- Estas son las columnas que el sistema actualiza en cada predicción

-- Columna para la predicción (causa)
ALTER TABLE tickets_fiducia 
ADD COLUMN IF NOT EXISTS causa VARCHAR;

-- Columna para timestamp de actualización
ALTER TABLE tickets_fiducia 
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();

-- Actualizar updated_at automáticamente cuando cambie cualquier campo
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Crear trigger si no existe
DROP TRIGGER IF EXISTS update_tickets_fiducia_updated_at ON tickets_fiducia;
CREATE TRIGGER update_tickets_fiducia_updated_at 
    BEFORE UPDATE ON tickets_fiducia 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 2. COLUMNAS OPCIONALES (RECOMENDADAS)
-- Estas columnas guardan información adicional útil para análisis
-- Descomenta las siguientes líneas si quieres usar estas funcionalidades
-- ============================================================================

-- Confianza de la predicción (0-1)
-- ALTER TABLE tickets_fiducia 
-- ADD COLUMN IF NOT EXISTS prediction_confidence FLOAT;

-- Metadatos de la predicción (JSON con probabilidades, modelo usado, etc.)
-- ALTER TABLE tickets_fiducia 
-- ADD COLUMN IF NOT EXISTS prediction_metadata JSONB;

-- ============================================================================
-- 3. VERIFICACIÓN
-- ============================================================================

-- Verificar que las columnas existen
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'tickets_fiducia'
  AND column_name IN ('number', 'causa', 'updated_at', 'prediction_confidence', 'prediction_metadata', 'short_description', 'close_notes')
ORDER BY column_name;

-- Verificar que hay datos
SELECT COUNT(*) as total_tickets FROM tickets_fiducia;

-- Ver estructura de un ticket de ejemplo
SELECT 
    number,
    causa,
    updated_at,
    short_description,
    close_notes
FROM tickets_fiducia 
LIMIT 1;

