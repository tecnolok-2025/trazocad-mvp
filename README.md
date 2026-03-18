# TrazoCad v61 — Release profesional

TrazoCad es una aplicación de Tecno Logisti-K SA (TLK) para digitalizar croquis, fotos o planos escaneados y generar salidas técnicas limpias.

## Alcance actual

La v61 consolida una línea de trabajo profesional centrada en:

- carga de imagen
- procesamiento asíncrono
- DXF vectorizado limpio
- DXF nube de puntos con fallback raster
- PDF de presentación por hoja solicitada
- JPG y PNG de salida
- apertura externa de DXF
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
- `GET /abrir/dxf?url=...`
- `GET /abrir/pdf?url=...`
- `POST /api/process`
- `GET /api/process/{job_id}/status`

## Salidas esperadas

- DXF vectorizado
- DXF nube de puntos
- PDF
- JPG
- PNG

## Notas de esta release

- El PDF ya respeta el tamaño de hoja solicitado y encuadra el contenido sin deformación.
- La nube de puntos intenta usar una base raster limpia antes de caer al muestreo geométrico.
- La optimización profunda del pipeline todavía puede seguir mejorándose en una etapa posterior de performance.
