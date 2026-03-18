from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from urllib.parse import quote
from uuid import uuid4

from PIL import Image
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.services.dxf_exporter import export_to_dxf, export_to_point_cloud_dxf
from app.services.image_pipeline import process_drawing
from app.services.persistence import persistence
from app.services.report_generator import build_report_pdf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
OUTPUT_DIR = DATA_DIR / 'outputs'
STATIC_DIR = BASE_DIR / 'app' / 'static'
VERSION_FILE = BASE_DIR / 'VERSION'
APP_VERSION = VERSION_FILE.read_text(encoding='utf-8').strip() if VERSION_FILE.exists() else '6.1.0'

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
SUPPORTED_SHEETS = {'A4', 'A3', 'A2', 'A1'}
SUPPORTED_DRAWING_TYPES = {'arquitectura', 'mecanico', 'electrico', 'civil', 'layout industrial'}
SUPPORTED_ORIENTATION = {'AUTO', 'VERTICAL', 'HORIZONTAL'}
MAX_UPLOAD_MB = int(os.getenv('TRAZOCAD_MAX_UPLOAD_MB', '25'))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

app = FastAPI(
    title='TrazoCad',
    version=APP_VERSION,
    description='TrazoCad release profesional: digitalización técnica simple con salidas DXF, PDF, JPG y PNG.',
)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
app.mount('/outputs', StaticFiles(directory=OUTPUT_DIR), name='outputs')

executor = ThreadPoolExecutor(max_workers=max(int(os.getenv('TRAZOCAD_WORKERS', '2')), 1))
job_lock = Lock()
job_store: dict[str, dict] = {}

STAGE_MAP = {
    'cola': {'percent': 2, 'detail': 'La tarea está en cola y lista para iniciar.'},
    'preparando_archivo': {'percent': 8, 'detail': 'Validando el archivo y preparando el área de trabajo.'},
    'reconstruyendo_base': {'percent': 24, 'detail': 'Corrigiendo base, perspectiva y contraste sin deformar el plano.'},
    'separando_texto_y_dibujo': {'percent': 42, 'detail': 'Separando zonas de dibujo para dejar una base geométrica más limpia.'},
    'reconociendo_texto': {'percent': 58, 'detail': 'Consolidando capas internas de análisis sin exponer OCR en la interfaz.'},
    'vectorizando_geometria': {'percent': 72, 'detail': 'Detectando líneas, contornos y trazos útiles para las salidas técnicas.'},
    'generando_salida': {'percent': 90, 'detail': 'Generando DXF, PDF y variantes raster finales.'},
    'finalizando': {'percent': 97, 'detail': 'Ordenando resultados y preparando acciones de descarga.'},
    'completado': {'percent': 100, 'detail': 'Proceso terminado correctamente.'},
    'error': {'percent': 100, 'detail': 'La tarea terminó con error.'},
}


def _runtime_version_payload() -> dict[str, str]:
    return {
        'producto': 'TrazoCad',
        'empresa': 'Tecno Logisti-K SA (TLK)',
        'version': APP_VERSION,
        'linea': 'release profesional',
        'rama': os.getenv('RENDER_GIT_BRANCH', 'local'),
        'commit': os.getenv('RENDER_GIT_COMMIT', 'sin-dato'),
        'repositorio': os.getenv('RENDER_GIT_REPO_SLUG', 'sin-dato'),
        'servicio': os.getenv('RENDER_SERVICE_NAME', 'local'),
        'persistencia': persistence.provider,
    }


def _public_base_url() -> str | None:
    raw = os.getenv('RENDER_EXTERNAL_URL') or os.getenv('PUBLIC_BASE_URL')
    return raw.rstrip('/') if raw else None


def _public_url_for(path: str) -> str:
    base = _public_base_url()
    return f'{base}{path}' if base else path


def _sharecad_url(file_path: str) -> str:
    return f'https://sharecad.org/cadframe/load?url={quote(_public_url_for(file_path), safe="")}'


def _set_job_stage(job_id: str, stage: str, extra: dict | None = None) -> None:
    stage_meta = STAGE_MAP.get(stage, {'percent': 0, 'detail': stage.replace('_', ' ')})
    snapshot = None
    with job_lock:
        job = job_store.get(job_id)
        if not job:
            return
        job['stage'] = stage
        job['progress'] = stage_meta['percent']
        job['message'] = stage_meta['detail']
        job['updated_at'] = time.time()
        if extra:
            job.update(extra)
        snapshot = dict(job)
    if snapshot:
        persistence.save_job(snapshot)


def _validate_process_request(file_name: str, file_bytes: bytes, sheet_size: str, drawing_type: str, sheet_orientation: str) -> tuple[str, str, str]:
    suffix = Path(file_name or 'plano').suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail='Formato no soportado. Usá PNG, JPG, BMP o TIFF.')
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail='El archivo llegó vacío.')
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f'El archivo supera el límite permitido de {MAX_UPLOAD_MB} MB.')

    sheet = str(sheet_size or 'A3').upper()
    if sheet not in SUPPORTED_SHEETS:
        raise HTTPException(status_code=400, detail='Tamaño de hoja inválido.')

    drawing = str(drawing_type or 'arquitectura').strip().lower()
    if drawing not in SUPPORTED_DRAWING_TYPES:
        raise HTTPException(status_code=400, detail='Tipo de dibujo inválido.')

    orientation = str(sheet_orientation or 'AUTO').upper()
    if orientation not in SUPPORTED_ORIENTATION:
        raise HTTPException(status_code=400, detail='Orientación inválida.')

    return suffix, sheet, orientation


def _export_raster_variants(source_png: Path, jpg_path: Path, png_path: Path) -> tuple[Path, Path]:
    png_path.write_bytes(source_png.read_bytes())
    with Image.open(source_png) as image:
        image.convert('RGB').save(jpg_path, quality=92)
    return jpg_path, png_path


def _result_summary(result: dict, drawing_type: str, sheet_size: str) -> dict[str, str | int | float]:
    return {
        'tipo_dibujo': drawing_type,
        'hoja': sheet_size,
        'orientacion_documento': str(result.get('document_orientation', 'sin-dato')),
        'lineas': int(result.get('detected_line_count', 0)),
        'contornos': int(result.get('detected_contour_count', 0)),
        'escala_mm_px': float(result.get('estimated_scale_mm_per_px', 0)),
        'precision_clase': str(result.get('precision_class', 'preliminar')),
    }


def _write_manifest(job_dir: Path, payload: dict) -> Path:
    manifest_path = job_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest_path


def _build_result_payload(job_id: str, file_name: str, sheet_size: str, drawing_type: str, result: dict) -> dict:
    job_dir = OUTPUT_DIR / job_id
    dxf_path = export_to_dxf(
        output_path=job_dir / 'trazocad_reconstruccion.dxf',
        geometry=result['geometry'],
        image_width=result['processed_width_px'],
        image_height=result['processed_height_px'],
        mm_per_px=result['estimated_scale_mm_per_px'],
    )
    dxf_points_path = export_to_point_cloud_dxf(
        output_path=job_dir / 'trazocad_nube_de_puntos.dxf',
        geometry=result['geometry'],
        image_width=result['processed_width_px'],
        image_height=result['processed_height_px'],
        mm_per_px=result['estimated_scale_mm_per_px'],
        raster_path=Path(result.get('output_files', {}).get('graphics_mask', '')) if result.get('output_files', {}).get('graphics_mask') else None,
    )
    report_path = build_report_pdf(
        output_path=job_dir / 'trazocad_plano.pdf',
        result=result,
        drawing_type=drawing_type,
        sheet_size=sheet_size,
        original_filename=file_name,
    )
    cleaned_source = Path(result.get('output_files', {}).get('cleaned_full_image') or result.get('output_files', {}).get('cleaned_image'))
    jpg_path, png_path = _export_raster_variants(
        cleaned_source,
        job_dir / 'trazocad_limpio.jpg',
        job_dir / 'trazocad_limpio.png',
    )

    payload = dict(result)
    payload.update(
        {
            'job_id': job_id,
            'brand': 'TrazoCad',
            'company': 'Tecno Logisti-K SA (TLK)',
            'version': APP_VERSION,
            'viewer_mode': 'externo',
            'runtime': _runtime_version_payload(),
            'sheet_size': sheet_size,
            'drawing_type': drawing_type,
            'summary': _result_summary(result, drawing_type, sheet_size),
            'downloads': {
                'dxf': f'/outputs/{job_id}/{dxf_path.name}',
                'dxf_nube_puntos': f'/outputs/{job_id}/{dxf_points_path.name}',
                'report': f'/outputs/{job_id}/{report_path.name}',
                'jpg': f'/outputs/{job_id}/{jpg_path.name}',
                'png': f'/outputs/{job_id}/{png_path.name}',
                'abrir_dxf': f'/abrir/dxf?url=/outputs/{job_id}/{dxf_path.name}',
                'abrir_pdf': f'/abrir/pdf?url=/outputs/{job_id}/{report_path.name}',
            },
        }
    )
    _write_manifest(job_dir, payload)
    return payload


def _start_job(job_id: str, payload: dict) -> None:
    upload_path: Path = payload['upload_path']
    file_name: str = payload['file_name']
    sheet_size: str = payload['sheet_size']
    drawing_type: str = payload['drawing_type']
    sheet_orientation: str = payload['sheet_orientation']
    notes: str = payload['notes']

    started_at = time.time()
    try:
        def progress(stage: str, extra: dict | None = None) -> None:
            _set_job_stage(job_id, stage, extra)

        progress('preparando_archivo')
        pipeline_result = process_drawing(
            input_path=upload_path,
            output_dir=OUTPUT_DIR / job_id,
            sheet_size=sheet_size,
            drawing_type=drawing_type,
            reference_mm=None,
            notes=notes,
            sheet_orientation=sheet_orientation,
            calibration={'mode': 'AUTO'},
            progress_callback=progress,
        )
        progress('generando_salida')
        response_payload = _build_result_payload(
            job_id=job_id,
            file_name=file_name,
            sheet_size=sheet_size,
            drawing_type=drawing_type,
            result=pipeline_result.to_dict(),
        )
        duration = round(time.time() - started_at, 2)
        progress('finalizando', {'elapsed_seconds': duration})
        with job_lock:
            job = job_store.get(job_id, {})
            job.update(
                {
                    'updated_at': time.time(),
                    'state': 'done',
                    'stage': 'completado',
                    'progress': 100,
                    'message': STAGE_MAP['completado']['detail'],
                    'elapsed_seconds': duration,
                    'result': response_payload,
                    'error': None,
                }
            )
            job_store[job_id] = job
        persistence.save_job(job)
    except Exception as exc:
        with job_lock:
            job = job_store.get(job_id, {})
            job.update(
                {
                    'updated_at': time.time(),
                    'state': 'error',
                    'stage': 'error',
                    'progress': 100,
                    'message': f'Se produjo un error durante el proceso: {exc}',
                    'error': str(exc),
                    'traceback': traceback.format_exc(limit=5),
                }
            )
            job_store[job_id] = job
        persistence.save_job(job)


@app.get('/')
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / 'index.html')


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'TrazoCad', 'company': 'Tecno Logisti-K SA (TLK)', 'version': APP_VERSION}


@app.get('/manual')
def manual() -> FileResponse:
    return FileResponse(STATIC_DIR / 'manual.html')


@app.get('/abrir/dxf')
def abrir_dxf(url: str):
    return RedirectResponse(_sharecad_url(url))


@app.get('/abrir/pdf')
def abrir_pdf(url: str):
    return RedirectResponse(_public_url_for(url))


@app.get('/version')
def version() -> JSONResponse:
    return JSONResponse(_runtime_version_payload())


@app.get('/infra')
def infra() -> JSONResponse:
    return JSONResponse(
        {
            'producto': 'TrazoCad',
            'version': APP_VERSION,
            'servidor': {
                'plataforma': 'Render',
                'servicio': os.getenv('RENDER_SERVICE_NAME', 'local'),
                'rama': os.getenv('RENDER_GIT_BRANCH', 'local'),
                'commit': os.getenv('RENDER_GIT_COMMIT', 'sin-dato'),
                'base_publica': _public_base_url() or 'no configurada',
                'nota': 'La línea profesional sigue recomendando revisar tiempos de proceso en planes free.',
            },
            'persistencia': persistence.stats(),
            'alcance': 'Release profesional: flujo asíncrono, DXF limpio, DXF nube de puntos independiente, PDF de presentación y descargas directas.',
        }
    )


@app.post('/api/process')
async def process_file(
    file: UploadFile = File(...),
    sheet_size: str = Form('A3'),
    drawing_type: str = Form('arquitectura'),
    notes: str = Form(''),
    sheet_orientation: str = Form('AUTO'),
) -> JSONResponse:
    file_name = file.filename or 'plano'
    raw = await file.read()
    suffix, sheet, orientation = _validate_process_request(file_name, raw, sheet_size, drawing_type, sheet_orientation)

    job_id = uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f'{job_id}{suffix}'
    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(raw)

    with job_lock:
        job_store[job_id] = {
            'job_id': job_id,
            'state': 'queued',
            'stage': 'cola',
            'progress': STAGE_MAP['cola']['percent'],
            'message': STAGE_MAP['cola']['detail'],
            'file_name': file_name,
            'sheet_size': sheet,
            'drawing_type': drawing_type,
            'result': None,
            'error': None,
            'created_at': time.time(),
            'updated_at': time.time(),
        }
        persistence.save_job(job_store[job_id])

    executor.submit(
        _start_job,
        job_id,
        {
            'upload_path': upload_path,
            'output_dir': output_dir,
            'file_name': file_name,
            'sheet_size': sheet,
            'drawing_type': drawing_type,
            'sheet_orientation': orientation,
            'notes': notes[:800],
        },
    )
    return JSONResponse(
        {
            'job_id': job_id,
            'state': 'queued',
            'stage': 'cola',
            'progress': STAGE_MAP['cola']['percent'],
            'message': STAGE_MAP['cola']['detail'],
            'version': APP_VERSION,
        }
    )


@app.get('/api/process/{job_id}/status')
def process_status(job_id: str) -> JSONResponse:
    with job_lock:
        job_mem = job_store.get(job_id)
    job_db = persistence.load_job(job_id)
    if job_mem and job_db:
        job = job_db if float(job_db.get('updated_at') or 0) > float(job_mem.get('updated_at') or 0) else job_mem
    else:
        job = job_mem or job_db

    if not job:
        output_dir = OUTPUT_DIR / job_id
        if output_dir.exists():
            return JSONResponse(
                {
                    'job_id': job_id,
                    'state': 'recovering',
                    'stage': 'cola',
                    'progress': 5,
                    'message': 'La tarea existe en disco, pero el servidor todavía está reconstruyendo su estado.',
                    'version': APP_VERSION,
                    'updated_at': time.time(),
                    'warning': 'Recuperación de estado en progreso.',
                }
            )
        raise HTTPException(status_code=404, detail='No se encontró la tarea solicitada.')

    payload = {
        'job_id': job_id,
        'state': job.get('state', 'queued'),
        'stage': job.get('stage', 'cola'),
        'progress': job.get('progress', 0),
        'message': job.get('message', 'Sin novedades por el momento.'),
        'version': APP_VERSION,
        'updated_at': job.get('updated_at', time.time()),
        'persistencia': persistence.provider,
        'elapsed_seconds': job.get('elapsed_seconds'),
    }
    if job.get('state') == 'done':
        payload['result'] = job.get('result')
    if job.get('state') == 'error':
        payload['error'] = job.get('error') or 'Ocurrió un error inesperado.'
    return JSONResponse(payload)


@app.get('/estado-producto')
def product_state() -> dict:
    return {
        'producto': 'TrazoCad',
        'empresa': 'Tecno Logisti-K SA (TLK)',
        'version': APP_VERSION,
        'etapa': 'Release profesional',
        'estado': 'operativo',
        'cierre': 'Flujo principal reducido a carga, proceso, apertura y descarga de archivos técnicos.',
        'capacidades_clave': [
            'proceso asíncrono con seguimiento de estado',
            'DXF vectorizado limpio',
            'DXF nube de puntos con base raster independiente',
            'PDF de presentación por hoja solicitada',
            'JPG y PNG limpios',
            'apertura externa de DXF y apertura directa de PDF',
        ],
        'criterios': [
            'simplicidad de interfaz',
            'no deformación',
            'orientación correcta',
            'presentación profesional del plano',
        ],
    }
