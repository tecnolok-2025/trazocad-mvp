from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from app.main import app

sample = ROOT / 'examples' / 'sample_croquis.png'
client = TestClient(app)
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
    print({'attempt': attempt + 1, 'state': status_payload['state'], 'stage': status_payload['stage'], 'progress': status_payload['progress']})
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
required = ['dxf', 'dxf_nube_puntos', 'report', 'jpg', 'png']
for key in required:
    rel = downloads[key]
    relative = rel.removeprefix('/outputs/')
    path = ROOT / 'data' / 'outputs' / relative
    assert path.exists(), f'Falta el archivo esperado: {path}'

manifest = ROOT / 'data' / 'outputs' / job_id / 'manifest.json'
assert manifest.exists(), 'Falta manifest.json'
print({'downloads': downloads, 'summary': result.get('summary', {})})
