# TrazoCad v73

Versión enfocada en **orquestación adaptativa**, OCR por regiones y saneamiento del DXF.

## Qué mejora esta versión
- prioriza terminar sin caerse: ajusta OCR y nube de puntos según presión de memoria
- OCR regional por bloques documentales, no sobre todo el plano
- saneamiento de líneas espurias en DXF, especialmente cerca de rótulo y zonas documentales
- nube de puntos más densa en rótulo, notas y título, y más liviana en zonas vacías
- salida visual preservada desde el original normalizado

## Flujo principal
1. subir imagen
2. elegir tipo/hoja/orientación
3. usar ayuda guiada opcional
4. procesar
5. abrir o descargar DXF/PDF/JPG/PNG

## Criterio operativo
- el OCR sigue siendo opcional y regional
- el sistema baja exigencia si detecta presión de memoria
- la salida visual y el DXF base tienen prioridad sobre OCR de baja confianza

## Recomendación
Para medir estabilidad primero probá sin OCR extra. Después activá OCR regional solo cuando necesites texto de rótulo o notas.
