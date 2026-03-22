from __future__ import annotations

from typing import Any
import time
import cv2
import numpy as np


def _normalize_text(value: str) -> str:
    return ' '.join(str(value).replace('\n', ' ').split())


def _clamp_box(box: dict[str, int], w: int, h: int) -> dict[str, int]:
    x = max(0, min(w - 1, int(box.get('x', 0))))
    y = max(0, min(h - 1, int(box.get('y', 0))))
    bw = max(1, int(box.get('w', 0)))
    bh = max(1, int(box.get('h', 0)))
    x2 = max(x + 1, min(w, x + bw))
    y2 = max(y + 1, min(h, y + bh))
    return {'x': x, 'y': y, 'w': x2 - x, 'h': y2 - y}


def _prepare_roi(roi: np.ndarray, region_type: str) -> np.ndarray:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    scale = 2.0 if region_type in {'rotulo', 'notas', 'titulo'} else 1.6
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray, (0, 0), 0.8)
    sharp = cv2.addWeighted(gray, 1.25, blur, -0.25, 0)
    th = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    if region_type in {'rotulo', 'notas', 'titulo'}:
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)), iterations=1)
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def _split_region_lines(box: dict[str, int], image_shape: tuple[int, int, int], max_parts: int = 4) -> list[dict[str, int]]:
    h = box['h']
    w = box['w']
    if h < 18 or w < 40:
        return [box]
    parts = min(max_parts, max(1, h // 22))
    stride = h / float(parts)
    out = []
    for i in range(parts):
        y = int(round(box['y'] + i * stride))
        y2 = int(round(box['y'] + (i + 1) * stride))
        out.append({'x': box['x'], 'y': y, 'w': box['w'], 'h': max(8, y2 - y)})
    return out


def build_region_boxes(segmented: dict[str, Any], image_shape: tuple[int, int, int], extra_boxes: list[dict[str, int]] | None = None) -> list[dict[str, Any]]:
    h, w = image_shape[:2]
    regions: list[dict[str, Any]] = []
    mapping = [
        ('title_block', 'rotulo', 3),
        ('notes_region', 'notas', 2),
        ('title_region', 'titulo', 2),
    ]
    for key, region_type, priority in mapping:
        box = segmented.get(key)
        if not box:
            continue
        box = _clamp_box(box, w, h)
        regions.append({**box, 'region_type': region_type, 'priority': priority})
        if region_type in {'notas', 'titulo'}:
            for part in _split_region_lines(box, image_shape):
                regions.append({**_clamp_box(part, w, h), 'region_type': region_type, 'priority': priority + 1})
    for box in extra_boxes or []:
        c = _clamp_box(box, w, h)
        regions.append({**c, 'region_type': 'texto_general', 'priority': 5})
    # dedupe and sort
    dedup = []
    seen = set()
    for r in sorted(regions, key=lambda d: (d['priority'], -(d['w'] * d['h']))):
        key = (round(r['x']/8), round(r['y']/8), round(r['w']/8), round(r['h']/8), r['region_type'])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


def extract_text_by_regions(image_bgr: np.ndarray, segmented: dict[str, Any], directives: dict[str, bool], runtime_profile: dict[str, Any], extra_boxes: list[dict[str, int]] | None = None) -> tuple[list[dict[str, Any]], str, str | None, list[dict[str, Any]]]:
    if not runtime_profile.get('ocr_allowed', directives.get('force_ocr', False)):
        return [], 'desactivado por presupuesto', 'El OCR regional se omitió para preservar memoria y estabilidad.', []
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore
    except Exception:
        return [], 'no disponible', 'No se encontró rapidocr-onnxruntime para OCR regional.', []
    try:
        engine = RapidOCR()
    except Exception as exc:
        return [], 'falló', f'No se pudo inicializar OCR regional: {exc}', []

    regions = build_region_boxes(segmented, image_bgr.shape, extra_boxes)
    if not regions:
        return [], 'sin regiones', None, []
    max_regions = 3 if runtime_profile.get('memory_pressure') == 'critical' else 6 if runtime_profile.get('memory_pressure') == 'high' else 10
    pixel_budget = int(image_bgr.shape[0] * image_bgr.shape[1] * (0.22 if runtime_profile.get('memory_pressure') == 'high' else 0.35))
    started = time.monotonic()
    texts: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_pixels = 0
    for idx, reg in enumerate(regions[:max_regions]):
        x, y, bw, bh = reg['x'], reg['y'], reg['w'], reg['h']
        roi = image_bgr[y:y+bh, x:x+bw]
        if roi.size == 0:
            continue
        pix = int(roi.shape[0] * roi.shape[1])
        if used_pixels + pix > pixel_budget and used_pixels > 0:
            warnings.append('OCR regional reducido por presupuesto de memoria/cálculo.')
            break
        if time.monotonic() - started > 8.0:
            warnings.append('OCR regional recortado por tiempo para sostener estabilidad.')
            break
        used_pixels += pix
        prep = _prepare_roi(roi, reg['region_type'])
        try:
            result = engine(prep)
            items = result[0] if isinstance(result, tuple) else result
        except Exception as exc:
            warnings.append(f'OCR parcial falló en una región: {exc}')
            items = []
        overlays.append(reg)
        if not items:
            continue
        best_text = ''
        best_score = 0.0
        for item in items:
            try:
                txt = _normalize_text(item[1])
                score = float(item[2])
            except Exception:
                continue
            if txt and score > best_score:
                best_text = txt
                best_score = score
        if best_text and best_score >= 0.30:
            texts.append({
                'text': best_text,
                'score': round(best_score, 3),
                'x': x,
                'y': y,
                'w': bw,
                'h': bh,
                'region_type': reg['region_type'],
                'ocr_source': 'regional',
            })
    # dedupe keeping best score
    dedup: list[dict[str, Any]] = []
    for item in sorted(texts, key=lambda d: (-d['score'], d['y'], d['x'])):
        dup = False
        for keep in dedup:
            ix1 = max(item['x'], keep['x']); iy1 = max(item['y'], keep['y'])
            ix2 = min(item['x']+item['w'], keep['x']+keep['w']); iy2 = min(item['y']+item['h'], keep['y']+keep['h'])
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2-ix1)*(iy2-iy1)
            area = max(item['w']*item['h'], 1)
            if inter / area > 0.45:
                dup = True
                break
        if not dup:
            dedup.append(item)
    warning = '; '.join(dict.fromkeys(warnings)) if warnings else None
    return dedup, 'RapidOCR regional', warning, overlays
