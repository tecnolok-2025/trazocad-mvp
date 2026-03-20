# TrazoCad v71

Versión enfocada en **preservación documental**: la presentación visual parte del plano original normalizado, con tratamiento separado para rótulo, notas y título del plano.

TrazoCad es una aplicación de Tecno Logisti-K SA para digitalizar croquis, fotos o planos escaneados y generar DXF, PDF, JPG y PNG.

## Enfoque de esta versión
- OCR dirigido por regiones para cotas, textos y rótulos
- reconstrucción raster más inteligente para cerrar pequeños cortes y reforzar trazos débiles
- PDF más aprovechado en hoja
- DXF con capas separadas de geometría, cotas, textos y rótulo cuando el OCR reconoce contenido

## Flujo principal
1. subir imagen
2. elegir tipo/hoja/orientación
3. usar ayuda guiada opcional
4. procesar
5. abrir o descargar DXF/PDF/JPG/PNG

## Notas guiadas
La interfaz incluye una temática y una acción sugerida que se convierten en notas internas del proceso. También puede agregarse una instrucción libre.

## OCR
La versión incorpora RapidOCR por regiones. Si el motor OCR no está disponible, la app sigue funcionando y conserva los textos/rótulos en las salidas raster y en el DXF de nube de puntos.


## Notas de la v71

- El OCR queda desactivado por defecto para priorizar estabilidad en Render.
- Solo se activa si el usuario elige una acción OCR desde la interfaz.
- La imagen de presentación fue suavizada para evitar líneas negras demasiado agresivas y preservar mejor el rótulo.


## Corrección de la v71
- Cache busting de estáticos para evitar que el navegador muestre una versión vieja.
- Cabeceras no-store en HTML y respuestas JSON de versión/infra/health.
- Alineación completa de numeración visible en README y UI.
