from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app, EXPECTED_OUTPUTS

sample = ROOT / 'examples' / 'sample_croquis.png'
client = TestClient(app)

# Error de formato
with sample.open('rb') as fh:
    invalid = client.post(
        '/api/process',
        files={'file': ('sample.txt', fh, 'text/plain')},
        data={'sheet_size': 'A3', 'drawing_type': 'arquitectura'},
    )
assert invalid.status_code == 400, invalid.text

# Flujo feliz
with sample.open('rb') as fh:
    response = client.post(
        '/api/process',
        files={'file': ('sample_croquis.png', fh, 'image/png')},
        data={
            'sheet_size': 'A3',
            'drawing_type': 'arquitectura',
            'sheet_orientation': 'AUTO',
            'notes': 'Prueba automatica local',
        },
    )

response.raise_for_status()
payload = response.json()
job_id = payload['job_id']
print({'job_id': job_id, 'initial_state': payload['state'], 'progress': payload['progress']})

final_payload = None
for attempt in range(120):
    status = client.get(f'/api/process/{job_id}/status')
    status.raise_for_status()
    status_payload = status.json()
    print({
        'attempt': attempt + 1,
        'state': status_payload['state'],
        'stage': status_payload['stage'],
        'progress': status_payload['progress'],
        'elapsed_seconds': round(status_payload.get('elapsed_seconds') or 0, 2),
    })
    if status_payload['state'] == 'done':
        final_payload = status_payload
        break
    if status_payload['state'] == 'error':
        raise SystemExit(status_payload.get('error', 'La tarea terminó con error.'))
    time.sleep(1)
else:
    raise SystemExit('Timeout esperando la finalización del job.')

result = final_payload['result']
downloads = result['downloads']
for key, filename in EXPECTED_OUTPUTS.items():
    route = downloads[key]
    artifact = client.get(route)
    artifact.raise_for_status()
    assert artifact.content, f'Salida vacía para {key}'
    assert filename in route, f'Nombre inesperado para {key}: {route}'

pdf_open = client.get(downloads['abrir_pdf'], follow_redirects=False)
assert pdf_open.status_code in (302, 307), pdf_open.text

dxf_open = client.get(downloads['abrir_dxf'], follow_redirects=False)
assert dxf_open.status_code in (200, 302, 307), dxf_open.text

manifest = ROOT / 'data' / 'outputs' / job_id / 'manifest.json'
assert manifest.exists(), 'Falta manifest.json'
manifest_payload = json.loads(manifest.read_text(encoding='utf-8'))
assert manifest_payload.get('summary'), 'El manifest quedó sin resumen'
assert manifest_payload['summary']['hoja'] == 'A3'
print({'downloads': downloads, 'summary': result.get('summary', {})})
