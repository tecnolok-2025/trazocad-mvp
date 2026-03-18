# TrazoCad v65 — Release profesional final

TrazoCad es una aplicación de Tecno Logisti-K SA (TLK) para digitalizar croquis, fotos o planos escaneados y generar salidas técnicas limpias.

## Alcance actual

La v65 consolida una línea de trabajo profesional centrada en:

- carga de imagen
- procesamiento asíncrono robusto
- recuperación de estado por job
- persistencia con fallback seguro a SQLite
- DXF vectorizado limpio
- DXF nube de puntos con fallback raster
- PDF de presentación por hoja solicitada
- JPG y PNG de salida
- apertura externa de DXF cuando existe URL pública
- apertura directa de PDF

## Qué no incluye esta versión

Esta release no expone en la interfaz:

- mesa de revisión asistida
- OCR operativo como función visible
- métricas visuales complejas
- visor DXF modal interno
- paneles técnicos residuales

## Requisitos

- Python 3.12
- dependencias de `requirements.txt`

## Ejecución local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Rutas principales:

- `/` interfaz principal
- `/manual` manual corto
- `/version` identidad de la versión
- `/infra` estado básico de infraestructura

## Flujo principal

1. Subir una imagen compatible (`png`, `jpg`, `jpeg`, `bmp`, `tif`, `tiff`)
2. Elegir tipo de dibujo, tamaño de hoja y orientación
3. Procesar
4. Consultar el estado del job
5. Abrir o descargar resultados

## Endpoints principales

- `GET /`
- `GET /manual`
- `GET /health`
- `GET /version`
- `GET /infra`
- `GET /abrir/dxf/{job_id}`
- `GET /abrir/pdf/{job_id}`
- `POST /api/process`
- `GET /api/process/{job_id}/status`

## Salidas esperadas

- DXF vectorizado
- DXF nube de puntos
- PDF
- JPG
- PNG

## Notas de esta release

- El PDF respeta el tamaño de hoja solicitado y encuadra el contenido sin deformación.
- La nube de puntos intenta usar una base raster limpia antes de caer al muestreo geométrico.
- Si PostgreSQL/Neon falla al arrancar en Render, la app cae automáticamente a SQLite local para no perder disponibilidad.


## Enfoque v65
- mayor fidelidad visual en PDF/JPG/PNG
- mejor preservación de rótulos, cotas y notas como imagen presentada
- DXF nube de puntos con base raster más fiel
- OCR sigue siendo opcional y no es todavía la salida principal
