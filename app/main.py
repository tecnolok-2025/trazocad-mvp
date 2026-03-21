from __future__ import annotations

import base64
import json
import os
import shutil
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

from app.services.persistence import persistence

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
OUTPUT_DIR = DATA_DIR / 'outputs'
STATIC_DIR = BASE_DIR / 'app' / 'static'
VERSION_FILE = BASE_DIR / 'VERSION'
APP_VERSION = VERSION_FILE.read_text(encoding='utf-8').strip() if VERSION_FILE.exists() else '78.1.0'

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
SUPPORTED_SHEET_SIZES = {'A4', 'A3', 'A2', 'A1'}
SUPPORTED_DRAWING_TYPES = {'arquitectura', 'mecanico', 'electrico', 'civil', 'layout industrial'}
SUPPORTED_ORIENTATIONS = {'AUTO', 'VERTICAL', 'HORIZONTAL'}
MAX_UPLOAD_MB = int(os.getenv('MAX_UPLOAD_MB', os.getenv('TRAZOCAD_MAX_UPLOAD_MB', '25')))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_EMBEDDED_UPLOAD_MB = int(os.getenv('TRAZOCAD_EMBEDDED_UPLOAD_MB', '8'))
MAX_EMBEDDED_UPLOAD_BYTES = MAX_EMBEDDED_UPLOAD_MB * 1024 * 1024
EXPECTED_OUTPUTS = {
    'dxf': 'trazocad_reconstruccion.dxf',
    'dxf_nube_puntos': 'trazocad_nube_de_puntos.dxf',
    'report': 'trazocad_plano.pdf',
    'jpg': 'trazocad_limpio.jpg',
    'png': 'trazocad_limpio.png',
}

app = FastAPI(
    title='TrazoCad',
    version=APP_VERSION,
    description='TrazoCad 78.1 base documental y segmentación: separación entre documento, rótulo, texto y geometría como base del rediseño del motor DXF.',
)
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
app.mount('/outputs', StaticFiles(directory=OUTPUT_DIR), name='outputs')


@app.middleware('http')
async def disable_cache_for_ui(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path in {'/', '/manual', '/version', '/infra', '/health'} or path.startswith('/static/'):
        for k, v in _no_cache_headers().items():
            response.headers[k] = v
    return response


executor = ThreadPoolExecutor(max_workers=max(int(os.getenv('TRAZOCAD_WORKERS', '1')), 1))
job_lock = Lock()
job_store: dict[str, dict] = {}

STAGE_MAP = {
    'cola': {'percent': 2, 'detail': 'La tarea está en cola y lista para iniciar.'},
    'preparando_archivo': {'percent': 8, 'detail': 'Validando el archivo y preparando el área de trabajo.'},
    'reconstruyendo_base': {'percent': 24, 'detail': 'Corrigiendo base, perspectiva y contraste sin deformar el plano.'},
    'separando_texto_y_dibujo': {'percent': 42, 'detail': 'Separando zonas de dibujo para dejar una base geométrica más limpia.'},
    'reconociendo_texto': {'percent': 58, 'detail': 'Reconociendo textos, cotas y rótulos por regiones priorizadas del plano.'},
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
        'linea': 'base documental y segmentación 78.1',
        'rama': os.getenv('RENDER_GIT_BRANCH', 'local'),
        'commit': os.getenv('RENDER_GIT_COMMIT', 'sin-dato'),
        'repositorio': os.getenv('RENDER_GIT_REPO_SLUG', 'sin-dato'),
        'servicio': os.getenv('RENDER_SERVICE_NAME', 'local'),
        'persistencia': persistence.provider,
        'max_upload_mb': str(MAX_UPLOAD_MB),
    }




def _no_cache_headers() -> dict[str, str]:
    return {
        'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
        'Pragma': 'no-cache',
        'Expires': '0',
    }

def _public_base_url() -> str | None:
    raw = os.getenv('RENDER_EXTERNAL_URL') or os.getenv('PUBLIC_BASE_URL')
    return raw.rstrip('/') if raw else None


def _public_url_for(path: str) -> str:
    base = _public_base_url()
    return f'{base}{path}' if base else path


def _sharecad_url(file_path: str) -> str:
    return f'https://sharecad.org/cadframe/load?url={quote(_public_url_for(file_path), safe="")}'


def _job_output_dir(job_id: str) -> Path:
    return OUTPUT_DIR / job_id


def _output_relpath(job_id: str, filename: str) -> str:
    return f'/outputs/{job_id}/{filename}'


def _downloads_payload(job_id: str) -> dict[str, str]:
    downloads = {key: _output_relpath(job_id, filename) for key, filename in EXPECTED_OUTPUTS.items()}
    downloads['abrir_dxf'] = f'/abrir/dxf/{job_id}'
    downloads['abrir_pdf'] = f'/abrir/pdf/{job_id}'
    return downloads


def _missing_outputs(job_id: str) -> list[str]:
    output_dir = _job_output_dir(job_id)
    return [key for key, filename in EXPECTED_OUTPUTS.items() if not (output_dir / filename).exists()]


def _preferred_clean_image(result: dict) -> Path:
    output_files = result.get('output_files', {}) if isinstance(result, dict) else {}
    for key in ('presentation_image', 'original_preview', 'cleaned_full_image', 'cleaned_image', 'graphics_mask', 'original_preview'):
        raw = output_files.get(key)
        if raw and Path(raw).exists():
            return Path(raw)
    fallback = _job_output_dir(result.get('job_id', '')) / 'limpio.png'
    if fallback.exists():
        return fallback
    raise FileNotFoundError('No se encontró una imagen limpia de salida para generar JPG, PNG y PDF.')


def _result_summary(result: dict, drawing_type: str, sheet_size: str) -> dict[str, str | int | float]:
    return {
        'tipo_dibujo': drawing_type,
        'hoja': sheet_size,
        'orientacion_hoja': str(result.get('sheet_orientation', 'AUTO')),
        'orientacion_documento': str(result.get('document_orientation', 'sin-dato')),
        'lineas': int(result.get('detected_line_count', 0)),
        'contornos': int(result.get('detected_contour_count', 0)),
        'textos_ocr': int(result.get('recognized_text_count', 0)),
        'escala_mm_px': float(result.get('estimated_scale_mm_per_px', 0)),
        'precision_clase': str(result.get('precision_class', 'preliminar')),
    }


def _write_manifest(job_dir: Path, payload: dict) -> Path:
    manifest_path = job_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest_path


def _recover_done_payload(job_id: str, job: dict | None = None) -> dict | None:
    if _missing_outputs(job_id):
        return None

    recovered = dict(job or {})
    existing_result = recovered.get('result') or {}
    result = dict(existing_result)
    result.update(
        {
            'job_id': job_id,
            'brand': 'TrazoCad',
            'company': 'Tecno Logisti-K SA (TLK)',
            'version': APP_VERSION,
            'viewer_mode': 'externo',
            'runtime': _runtime_version_payload(),
            'downloads': _downloads_payload(job_id),
        }
    )
    recovered.update(
        {
            'job_id': job_id,
            'state': 'done',
            'stage': 'completado',
            'progress': 100,
            'message': STAGE_MAP['completado']['detail'],
            'result': result,
            'error': None,
            'updated_at': time.time(),
        }
    )
    if 'created_at' not in recovered:
        recovered['created_at'] = recovered['updated_at']
    persistence.save_job(recovered)
    return recovered


def _elapsed_fields(job: dict) -> dict[str, float]:
    now = time.time()
    created_at = float(job.get('created_at') or now)
    updated_at = float(job.get('updated_at') or now)
    return {
        'created_at': created_at,
        'updated_at': updated_at,
        'elapsed_seconds': max(0.0, now - created_at),
        'last_update_age_seconds': max(0.0, now - updated_at),
    }


def _job_meta_payload(upload_path: Path, output_dir: Path, file_name: str, sheet_size: str, drawing_type: str, sheet_orientation: str, notes: str) -> dict[str, str]:
    return {
        'upload_path': str(upload_path),
        'output_dir': str(output_dir),
        'file_name': file_name,
        'sheet_size': sheet_size,
        'drawing_type': drawing_type,
        'sheet_orientation': sheet_orientation,
        'notes': notes,
    }


def _maybe_embed_upload(raw_bytes: bytes, suffix: str) -> dict[str, str] | None:
    if not raw_bytes or len(raw_bytes) > MAX_EMBEDDED_UPLOAD_BYTES:
        return None
    return {
        'encoding': 'base64',
        'suffix': suffix,
        'size_bytes': str(len(raw_bytes)),
        'data': base64.b64encode(raw_bytes).decode('ascii'),
    }


def _restore_upload_from_meta(meta: dict) -> Path | None:
    upload_raw = meta.get('upload_path')
    upload_path = Path(upload_raw) if upload_raw else None
    if upload_path and upload_path.exists():
        return upload_path
    blob = meta.get('upload_blob')
    if not blob or blob.get('encoding') != 'base64' or not upload_path:
        return None
    try:
        upload_path.parent.mkdir(parents=True, exist_ok=True)
        upload_path.write_bytes(base64.b64decode(blob['data'].encode('ascii')))
        return upload_path
    except Exception:
        return None


def _clean_output_dir(job_id: str) -> None:
    output_dir = _job_output_dir(job_id)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    for child in output_dir.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except Exception:
            pass


def _resume_job_if_possible(job_id: str, job: dict) -> bool:
    if job.get('state') in {'done', 'error'}:
        return False
    meta = job.get('meta') or {}
    upload_raw = meta.get('upload_path')
    if not upload_raw:
        return False
    upload_path = _restore_upload_from_meta(meta)
    if not upload_path or not upload_path.exists():
        return False
    with job_lock:
        current = job_store.get(job_id)
        if current and current.get('state') in {'queued', 'running'}:
            return True
        resumed = dict(job)
        resumed['state'] = 'running'
        resumed['stage'] = job.get('stage') or 'cola'
        resumed['message'] = 'El servidor retomó la tarea después de una interrupción temporal.'
        resumed['updated_at'] = time.time()
        job_store[job_id] = resumed
    persistence.save_job(resumed)
    executor.submit(_start_job, job_id, meta)
    return True


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
        if stage not in {'cola', 'completado', 'error'}:
            job['state'] = 'running'
        if extra:
            job.update(extra)
        snapshot = dict(job)
    if snapshot:
        persistence.save_job(snapshot)


def _export_raster_variants(source_png: Path, jpg_path: Path, png_path: Path) -> tuple[Path, Path]:
    png_path.write_bytes(source_png.read_bytes())
    with Image.open(source_png) as image:
        image.convert('RGB').save(jpg_path, quality=92)
    return jpg_path, png_path


def _build_result_payload(job_id: str, file_name: str, sheet_size: str, drawing_type: str, result: dict) -> dict:
    from app.services.dxf_exporter import export_to_dxf, export_to_point_cloud_dxf
    from app.services.report_generator import build_report_pdf

    output_dir = _job_output_dir(job_id)
    cleaned_source = _preferred_clean_image(result)

    dxf_path = export_to_dxf(
        output_path=output_dir / EXPECTED_OUTPUTS['dxf'],
        geometry=result['geometry'],
        image_width=result['processed_width_px'],
        image_height=result['processed_height_px'],
        mm_per_px=result['estimated_scale_mm_per_px'],
        raster_path=cleaned_source,
    )
    dxf_points_path = export_to_point_cloud_dxf(
        output_path=output_dir / EXPECTED_OUTPUTS['dxf_nube_puntos'],
        geometry=result['geometry'],
        image_width=result['processed_width_px'],
        image_height=result['processed_height_px'],
        mm_per_px=result['estimated_scale_mm_per_px'],
        raster_path=cleaned_source,
        step_px=float(result.get('geometry', {}).get('runtime_profile', {}).get('point_cloud_step_px', 10.0)),
    )
    report_path = build_report_pdf(
        output_path=output_dir / EXPECTED_OUTPUTS['report'],
        result=result,
        drawing_type=drawing_type,
        sheet_size=sheet_size,
        original_filename=file_name,
    )
    jpg_path, png_path = _export_raster_variants(
        cleaned_source,
        output_dir / EXPECTED_OUTPUTS['jpg'],
        output_dir / EXPECTED_OUTPUTS['png'],
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
            'downloads': _downloads_payload(job_id),
            'artifacts': {
                'dxf': {'name': dxf_path.name, 'bytes': dxf_path.stat().st_size},
                'dxf_nube_puntos': {'name': dxf_points_path.name, 'bytes': dxf_points_path.stat().st_size},
                'report': {'name': report_path.name, 'bytes': report_path.stat().st_size},
                'jpg': {'name': jpg_path.name, 'bytes': jpg_path.stat().st_size},
                'png': {'name': png_path.name, 'bytes': png_path.stat().st_size},
            },
        }
    )
    _write_manifest(output_dir, payload)
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
        from app.services.image_pipeline import process_drawing

        pipeline_result = process_drawing(
            input_path=upload_path,
            output_dir=_job_output_dir(job_id),
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
        missing = _missing_outputs(job_id)
        if missing:
            raise RuntimeError(f'La tarea terminó sin generar todas las salidas esperadas: {", ".join(missing)}.')
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
                    'traceback': traceback.format_exc(limit=6),
                }
            )
            job_store[job_id] = job
        persistence.save_job(job)
    finally:
        try:
            if upload_path.exists():
                upload_path.unlink()
        except OSError:
            pass


@app.get('/')
def home() -> FileResponse:
    return FileResponse(STATIC_DIR / 'index.html', headers=_no_cache_headers())


@app.get('/manual')
def manual() -> FileResponse:
    return FileResponse(STATIC_DIR / 'manual.html', headers=_no_cache_headers())


@app.get('/health')
def health() -> dict[str, str]:
    return {'status': 'ok', 'service': 'TrazoCad', 'company': 'Tecno Logisti-K SA (TLK)', 'version': APP_VERSION}


@app.get('/version')
def version() -> JSONResponse:
    return JSONResponse(_runtime_version_payload(), headers=_no_cache_headers())


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
                'nota': 'La release privilegia estabilidad y usa PostgreSQL/Neon cuando está disponible y mantiene un espejo local SQLite para recuperar estado ante reinicios o fallos transitorios.',
            },
            'persistencia': persistence.stats(),
            'alcance': 'Release 71.0: numeración visible corregida, estáticos con cache busting y no-store para evitar servir UI vieja; mantiene OCR opt-in y procesamiento seguro.',
        }
    , headers=_no_cache_headers())


@app.get('/abrir/dxf/{job_id}')
def abrir_dxf(job_id: str):
    target = _job_output_dir(job_id) / EXPECTED_OUTPUTS['dxf']
    if not target.exists():
        raise HTTPException(status_code=404, detail='No se encontró el DXF solicitado para esta tarea.')
    if _public_base_url():
        return RedirectResponse(_sharecad_url(_output_relpath(job_id, EXPECTED_OUTPUTS['dxf'])))
    return FileResponse(target, filename=target.name, media_type='application/dxf')


@app.get('/abrir/pdf/{job_id}')
def abrir_pdf(job_id: str):
    target = _job_output_dir(job_id) / EXPECTED_OUTPUTS['report']
    if not target.exists():
        raise HTTPException(status_code=404, detail='No se encontró el PDF solicitado para esta tarea.')
    return RedirectResponse(_public_url_for(_output_relpath(job_id, EXPECTED_OUTPUTS['report'])))


@app.post('/api/process')
async def process_file(
    file: UploadFile = File(...),
    sheet_size: str = Form('A3'),
    drawing_type: str = Form('arquitectura'),
    notes: str = Form(''),
    sheet_orientation: str = Form('AUTO'),
) -> JSONResponse:
    filename = (file.filename or '').strip()
    suffix = Path(filename or 'plano').suffix.lower()
    if not filename:
        raise HTTPException(status_code=400, detail='No se recibió ningún archivo para procesar.')
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail='Formato no soportado. Usá PNG, JPG, BMP o TIFF.')

    normalized_sheet_size = sheet_size.upper().strip()
    if normalized_sheet_size not in SUPPORTED_SHEET_SIZES:
        raise HTTPException(status_code=400, detail='Tamaño de hoja no válido. Usá A4, A3, A2 o A1.')

    normalized_drawing_type = drawing_type.strip().lower()
    if normalized_drawing_type not in SUPPORTED_DRAWING_TYPES:
        raise HTTPException(status_code=400, detail='Tipo de dibujo no válido.')

    normalized_orientation = sheet_orientation.upper().strip() or 'AUTO'
    if normalized_orientation not in SUPPORTED_ORIENTATIONS:
        raise HTTPException(status_code=400, detail='Orientación no válida. Usá AUTO, VERTICAL u HORIZONTAL.')

    normalized_notes = notes.strip()
    if len(normalized_notes) > 600:
        raise HTTPException(status_code=400, detail='Las notas son demasiado largas. Usá hasta 600 caracteres.')

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail='El archivo llegó vacío. Probá nuevamente con una imagen válida.')
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f'El archivo supera el máximo permitido de {MAX_UPLOAD_MB} MB.')

    job_id = uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f'{job_id}{suffix}'
    output_dir = _job_output_dir(job_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_path.write_bytes(raw_bytes)

    meta = _job_meta_payload(
        upload_path=upload_path,
        output_dir=output_dir,
        file_name=filename or upload_path.name,
        sheet_size=normalized_sheet_size,
        drawing_type=normalized_drawing_type,
        sheet_orientation=normalized_orientation,
        notes=normalized_notes,
    )
    embedded_upload = _maybe_embed_upload(raw_bytes, suffix)
    if embedded_upload:
        meta['upload_blob'] = embedded_upload

    initial_job = {
        'job_id': job_id,
        'state': 'queued',
        'stage': 'cola',
        'progress': STAGE_MAP['cola']['percent'],
        'message': STAGE_MAP['cola']['detail'],
        'file_name': filename or upload_path.name,
        'meta': meta,
        'result': None,
        'error': None,
        'created_at': time.time(),
        'updated_at': time.time(),
    }
    with job_lock:
        job_store[job_id] = initial_job
    persistence.save_job(initial_job)

    executor.submit(_start_job, job_id, dict(initial_job['meta']))
    return JSONResponse(
        {
            'job_id': job_id,
            'state': 'queued',
            'stage': 'cola',
            'progress': STAGE_MAP['cola']['percent'],
            'message': STAGE_MAP['cola']['detail'],
            'version': APP_VERSION,
            'max_upload_mb': MAX_UPLOAD_MB,
            'downloads': _downloads_payload(job_id),
        }
    )


@app.post('/api/process/{job_id}/retry')
def retry_process(job_id: str) -> JSONResponse:
    with job_lock:
        existing = job_store.get(job_id)
    job = existing or persistence.load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail='No se encontró la tarea original para reintentarla.')
    if job.get('state') in {'queued', 'running'}:
        return JSONResponse({'job_id': job_id, 'state': job.get('state'), 'message': 'La tarea ya está activa.'})
    if job.get('state') == 'done' and not _missing_outputs(job_id):
        recovered = _recover_done_payload(job_id, job=job) or job
        return JSONResponse({'job_id': job_id, 'state': 'done', 'result': recovered.get('result'), 'message': 'La tarea ya estaba completada.'})

    meta = dict(job.get('meta') or {})
    upload_path = _restore_upload_from_meta(meta)
    if not upload_path or not upload_path.exists():
        raise HTTPException(status_code=409, detail='No quedó una copia recuperable del archivo original para reintentar esta tarea. Conviene volver a subir el plano.')

    _clean_output_dir(job_id)
    restarted = dict(job)
    restarted.update({
        'state': 'queued',
        'stage': 'cola',
        'progress': STAGE_MAP['cola']['percent'],
        'message': 'Se relanzó automáticamente la tarea después de una interrupción.',
        'error': None,
        'result': None,
        'updated_at': time.time(),
    })
    with job_lock:
        job_store[job_id] = restarted
    persistence.save_job(restarted)
    executor.submit(_start_job, job_id, meta)
    return JSONResponse({'job_id': job_id, 'state': 'queued', 'stage': 'cola', 'progress': STAGE_MAP['cola']['percent'], 'message': restarted['message'], 'version': APP_VERSION})


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
        recovered = _recover_done_payload(job_id)
        if recovered:
            job = recovered
        else:
            output_dir = _job_output_dir(job_id)
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
            return JSONResponse(
                {
                    'job_id': job_id,
                    'state': 'missing',
                    'stage': 'interrumpida',
                    'progress': 0,
                    'message': 'El servidor perdió el estado de la tarea antes de completarla. Esto suele pasar por un reinicio del proceso o una caída transitoria de persistencia.',
                    'version': APP_VERSION,
                    'updated_at': time.time(),
                    'error': 'La tarea ya no está disponible para continuar en este servidor. Se puede intentar un reintento automático si existe una copia del archivo original.',
                }
            )
    elif job.get('state') == 'done' and not job.get('result'):
        recovered = _recover_done_payload(job_id, job=job)
        if recovered:
            job = recovered

    if job and job.get('state') in {'queued', 'running'} and job_id not in job_store:
        if _resume_job_if_possible(job_id, job):
            job = persistence.load_job(job_id) or job
            job['state'] = 'running'
            job['message'] = 'El servidor retomó la tarea después de una interrupción temporal.'

    payload = {
        'job_id': job_id,
        'state': job.get('state', 'queued'),
        'stage': job.get('stage', 'cola'),
        'progress': job.get('progress', 0),
        'message': job.get('message', 'Sin novedades por el momento.'),
        'version': APP_VERSION,
        'persistencia': persistence.provider,
        **_elapsed_fields(job),
    }
    if job.get('state') == 'done':
        payload['result'] = job.get('result')
    if job.get('state') == 'error':
        payload['error'] = job.get('error') or 'Ocurrió un error inesperado.'
    if job.get('state') == 'missing':
        payload['error'] = job.get('error') or 'La tarea ya no está disponible para continuar en este servidor.'
    return JSONResponse(payload)


@app.get('/estado-producto')
def product_state() -> dict:
    return {
        'producto': 'TrazoCad',
        'empresa': 'Tecno Logisti-K SA (TLK)',
        'version': APP_VERSION,
        'estado': 'release 71.0 de preservación documental',
        'criterio': [
            'interfaz simple',
            'DXF limpio',
            'PDF de mayor fidelidad visual',
            'no deformación',
            'apertura sin visor DXF interno',
        ],
        'salidas': list(EXPECTED_OUTPUTS.keys()),
    }
