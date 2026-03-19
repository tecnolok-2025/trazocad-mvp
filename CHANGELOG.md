## 69.1.0
- Blindaje de arranque para Render con `healthCheckPath: /health`.
- Persistencia lazy: no intenta conectar PostgreSQL/Neon durante el import del módulo.
- Imports pesados diferidos hasta el momento de procesar un archivo.
- Mantiene continuidad, reintento y corrección visual de la v69.

# Changelog

## 69.1.0
- Recuperación y reintento robusto de tareas interrumpidas.
- Persistencia opcional del archivo original en la metadata del job para reanudar/reintentar.
- OCR en modo seguro por presupuesto de tiempo/regiones.
- Reconstrucción estructural base para cerrar microcortes.
